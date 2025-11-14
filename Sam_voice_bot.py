import torch
import numpy as np
import sounddevice as sd
import queue, threading, time
import soundfile as sf
from faster_whisper import WhisperModel
import requests
from gtts import gTTS
import pygame
import io
import collections
from pydub import AudioSegment

# ===================== CONFIG =====================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512
VAD_SILENCE_TIMEOUT = 2.5
MIN_SEGMENT_SEC = 0.5
API_URL = "https://Sam.chatbot/"

# Smart endpointing / merging tuning
SPEECH_END_PADDING = 0.4
MERGE_WINDOW_SEC = 2.0
CONFIDENCE_THRESHOLD = -0.7
speech_start_time = None

VAD_SILENCE_TIMEOUT = 1.8
SPEECH_END_PADDING = 0.4
VAD_START_THRESHOLD_FRAMES = 10
VAD_STOP_THRESHOLD_FRAMES = 56  # ~1.8s at 31 chunks/sec; adjust for your needs
AI_RESUME_TIMEOUT = 1.8
VAD_PROB_THRESHOLD = 0.7  # NEW: Higher threshold to ignore low-confidence noise

# Silence smoothing
silence_counter = 0
silence_threshold_frames = 10

# AI state
is_playing_ai = False
is_responding = False
interrupted_audio_data = None
interrupted_pos = 0

# Transcription + dedup state
is_transcribing = False
ongoing_text = ""
last_transcription_time = 0.0
last_user_text = ""
last_ai_reply = None

# Echo Canceller state (new robust approach)
# We keep a ring buffer of recent playback PCM samples that we can align against.
PLAYBACK_RING_SAMPLES = SAMPLE_RATE * 6     # keep up to 6 seconds of playback for alignment if necessary
playback_ringbuffer = collections.deque(maxlen=PLAYBACK_RING_SAMPLES)

# Parameters for delay-tolerant echo estimation
MAX_ECHO_SEARCH_MS = 500      # how far back (ms) to search for an echo match; tune up to 1000 ms if needed
ALPHA_SMOOTH = 0.85           # smoothing for estimated gain (0..1) higher = smoother but slower
MIN_REF_ENERGY = 1e-6         # threshold to avoid dividing by zero in LS
alpha_smoothed = 0.0          # running smoothed scalar gain

# VAD state for thresholds
speech_frame_counter = 0

# ===================== LOAD MODELS =====================
print("Loading models...")
vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
(_, _, _, VADIterator, _) = utils
# Note: We're not using VADIterator anymore for finer control over threshold
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# ===================== STATE =====================
audio_q = queue.Queue()
is_talking = False
last_speech_time = time.time()
speech_buffer = np.zeros(0, dtype=np.float32)
segment_index = 0

# ===================== ECHO CANCELLATION UTILITIES =====================
def aec_feed_playback(play_chunk: np.ndarray):
    """
    Feed AI playback PCM samples (float32 in -1..1) into the reference ring buffer.
    Call this when you generate TTS PCM (before starting playback) so the AEC has the reference.
    """
    if play_chunk is None or len(play_chunk) == 0:
        return
    # Ensure numpy float32 array
    arr = play_chunk.astype(np.float32)
    playback_ringbuffer.extend(arr.tolist())

def aec_process(mic_chunk: np.ndarray) -> np.ndarray:
    """
    Delay-tolerant echo removal:
      - find best alignment of mic_chunk in recent playback buffer using cross-correlation
      - compute scalar least-squares gain alpha that maps ref -> echo
      - subtract alpha * ref from mic_chunk
    Returns cleaned chunk (float32).
    """
    global alpha_smoothed

    # If no playback data yet, return unchanged
    if len(playback_ringbuffer) < 16:
        return mic_chunk

    # Convert ring buffer to numpy (only the part we need)
    mic_len = len(mic_chunk)
    max_search_samples = int((MAX_ECHO_SEARCH_MS / 1000.0) * SAMPLE_RATE)
    # Ensure at least mic_len in buffer
    rb = np.frombuffer(np.array(playback_ringbuffer, dtype=np.float32), dtype=np.float32)
    total_rb = len(rb)
    if total_rb < mic_len:
        return mic_chunk

    # Determine search start index — we will search the last (mic_len + max_search) samples
    search_len = min(total_rb - mic_len, max_search_samples)
    # If nothing to search, align with tail
    if search_len <= 0:
        # Use the most recent matching-length tail
        ref = rb[-mic_len:]
        # fallback alpha estimation
        denom = np.dot(ref, ref) + MIN_REF_ENERGY
        alpha = np.dot(mic_chunk, ref) / denom
        # clamp alpha to reasonable range
        alpha = float(np.clip(alpha, 0.0, 2.0))
        # smooth alpha
        alpha_smoothed = ALPHA_SMOOTH * alpha_smoothed + (1 - ALPHA_SMOOTH) * alpha
        return mic_chunk - alpha_smoothed * ref

    # Build the search array: last (search_len + mic_len) samples
    start_idx = total_rb - mic_len - search_len
    search_arr = rb[start_idx : start_idx + search_len + mic_len]

    # Cross-correlate mic_chunk with search_arr to find best alignment
    # We compute correlation via direct np.correlate (valid mode positions where ref window has full length)
    try:
        # reverse mic for correlation compatibility
        corr = np.correlate(search_arr, mic_chunk, mode='valid')
        # corr length = search_len + 1
        best = int(np.argmax(np.abs(corr)))
        ref_start = start_idx + best
        ref = rb[ref_start : ref_start + mic_len]
        if len(ref) < mic_len:
            # pad if necessary (shouldn't normally happen)
            ref = np.pad(ref, (0, mic_len - len(ref)), 'constant')
    except Exception:
        # fallback: use tail
        ref = rb[-mic_len:]

    # Normalize / compute least squares scalar: alpha = dot(mic, ref) / dot(ref, ref)
    denom = np.dot(ref, ref) + MIN_REF_ENERGY
    alpha = float(np.dot(mic_chunk, ref) / denom)

    # Clamp alpha to avoid absurd gains; typical echo gain between 0..1.5
    alpha = float(np.clip(alpha, 0.0, 2.0))

    # Smooth alpha over time to avoid jumps
    alpha_smoothed = ALPHA_SMOOTH * alpha_smoothed + (1.0 - ALPHA_SMOOTH) * alpha

    # Subtract estimated echo
    cleaned = mic_chunk - alpha_smoothed * ref

    # Optional: small gain limiting to avoid underflow/overflow
    # cleaned = np.clip(cleaned, -1.0, 1.0)

    return cleaned.astype(np.float32)

# ===================== AUDIO OUTPUT INIT =====================
pygame.mixer.init()

# ===================== CALLBACK =====================
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio stream warning:", status)
    audio_q.put(indata.copy())

# ===================== STREAM =====================
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=CHANNELS,
    dtype='float32',
    callback=audio_callback
)
stream.start()
print("Listening for speech...")

# ===================== TTS (Speak Function) =====================
def speak_text(text):
    global is_playing_ai, interrupted_audio_data, interrupted_pos

    """Convert text to speech and play it, allowing interruption & safe resuming."""
    try:
        tts = gTTS(text=text, lang="en")
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # Keep safe copy for resume
        interrupted_audio_data = mp3_fp.getvalue()
        interrupted_pos = 0
        is_playing_ai = True

        # === Decode MP3 to PCM safely for echo cancellation ===
        try:
            pcm_buf = io.BytesIO(interrupted_audio_data)
            audio_segment = AudioSegment.from_file(pcm_buf, format="mp3")
            audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
            pcm_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            # Feed playback PCM to AEC ringbuffer
            aec_feed_playback(pcm_data)
        except Exception as e:
            # Do not fail playback — AEC feed is best-effort
            print("AEC decode skipped (pydub/ffmpeg error):", e)

        # Play audio (separate BytesIO to avoid ffmpeg/pygame race)
        play_buf = io.BytesIO(interrupted_audio_data)
        pygame.mixer.music.load(play_buf, "mp3")
        pygame.mixer.music.play()

        # Monitor playback, allow interruption by user speech
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            if is_talking:
                interrupted_pos = pygame.mixer.music.get_pos()
                pygame.mixer.music.stop()
                is_playing_ai = False
                print(" AI interrupted — waiting to see if user actually speaks...")
                break
    except Exception as e:
        print("TTS Error:", e)
    finally:
        if not pygame.mixer.music.get_busy():
            is_playing_ai = False
            interrupted_audio_data = None
            interrupted_pos = 0

# ===================== API CALL =====================
def get_response_from_ai(text):
    """Send transcribed text to Gen AI API with dedupe and lock."""
    global last_user_text, last_ai_reply, is_responding
    print(f"API called with: {text!r} | is_responding={is_responding}")

    user_text = text.strip().lower()
    if not user_text:
        return None

    # prevent double replies
    if is_responding:
        print("AI still responding — skipping new request.")
        return None

    if user_text == last_user_text:
        print("Duplicate user text detected — skipping API call.")
        return last_ai_reply

    is_responding = True
    last_user_text = user_text

    payload = {
        "user_id": "janmeet1233",
        "client_id": "44ihRG38UX24DKeFzE15FbbPZfCgz3rh",
        "text": user_text
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            reply = data.get("response") or data.get("reply") or data.get("text") or str(data)
            print(f"AI: {reply}\n")
            last_ai_reply = reply
            # spawn TTS in background thread so caller thread isn't blocked (keeps behavior similar to before)
            t = threading.Thread(target=speak_text, args=(reply,), daemon=True)
            t.start()
            return reply
        else:
            print("API error:", resp.status_code, resp.text)
    except Exception as e:
        print("Request failed:", e)
    finally:
        is_responding = False
    return None

# ===================== TRANSCRIBE =====================
def transcribe_segment(segment):
    """Whisper + semantic endpointing + dedupe check."""
    global ongoing_text, last_transcription_time, last_user_text

    print(f"\nTranscribing... ({len(segment)/SAMPLE_RATE:.2f}s)")
    segments, info = whisper_model.transcribe(segment, beam_size=1, language="en")

    full_text = ""
    avg_conf = -10.0
    count = 0

    for s in segments:
        txt = s.text or ""
        full_text += txt.strip() + " "
        if hasattr(s, "avg_logprob"):
            avg_conf = (avg_conf + s.avg_logprob) / 2 if count > 0 else s.avg_logprob
        count += 1

    full_text = full_text.strip()
    print(f"You: {full_text}")
    print(f"avg_logprob: {avg_conf:.2f}")

    if not full_text:
        print("No valid speech detected in segment.")
        return

    # skip duplicates
    if full_text.lower().strip() == last_user_text.lower().strip():
        print("Skipping duplicate transcription output (same as last).")
        return

    # Merge consecutive transcripts
    now = time.time()
    if (now - last_transcription_time) <= MERGE_WINDOW_SEC and ongoing_text:
        ongoing_text = (ongoing_text + " " + full_text).strip()
    else:
        ongoing_text = full_text
    last_transcription_time = now

    # Decide if it's a final thought
    complete_sentence = full_text.endswith((".", "?", "!", "…"))
    confident = avg_conf > CONFIDENCE_THRESHOLD
    long_text = len(full_text.split()) >= 8

    likely_complete = complete_sentence or confident or long_text

    if likely_complete:
        print("Linguistic endpoint reached — finalizing:", ongoing_text)
        final_text = ongoing_text
        ongoing_text = ""
        get_response_from_ai(final_text)
    else:
        print("Partial speech — waiting for continuation:", ongoing_text)

# ===================== LISTENER LOOP =====================
def listener_loop():
    global is_talking, last_speech_time, speech_buffer, segment_index
    global is_playing_ai, interrupted_pos, silence_counter, is_transcribing, interrupted_audio_data
    global speech_start_time, speech_frame_counter
    silence_frame_counter = 0  
    while True:
        if not audio_q.empty():
            chunk = audio_q.get().flatten()
            # Apply AEC to chunk before VAD/transcription
            try:
                cleaned = aec_process(chunk)
            except Exception as e:
                # On unexpected AEC failure, fallback to raw chunk
                print("AEC processing error:", e)
                cleaned = chunk

            # keep the rest of your logic identical: use 'cleaned' instead of raw chunk
            chunk_tensor = torch.from_numpy(cleaned).unsqueeze(0)
            try:
                prob = vad_model(chunk_tensor, SAMPLE_RATE).item()
                speech = prob > VAD_PROB_THRESHOLD
            except Exception as e:
                print("VAD error:", e)
                speech = False
                prob = 0.0
            speech_buffer = np.concatenate((speech_buffer, cleaned))[-SAMPLE_RATE * 10:]
            if speech:
                # voice activity detected
                silence_counter = 0
                silence_frame_counter = 0  # Reset silence count
                speech_frame_counter += 1
                last_speech_time = time.time()
                if speech_frame_counter >= VAD_START_THRESHOLD_FRAMES and not is_talking:
                    print(" Voice detected (confirmed after threshold)")
                    is_talking = True
                    speech_start_time = time.time()
                    if is_playing_ai:
                        interrupted_pos = pygame.mixer.music.get_pos()
                        pygame.mixer.music.stop()
                        is_playing_ai = False
                        print("AI interrupted — confirmed real user speech")
            else:
                speech_frame_counter = 0
                silence_counter += 1
                if is_talking:
                    silence_frame_counter += 1
                # Detect speech end using consecutive silence frames
                if (
                    is_talking
                    and not is_transcribing
                    and silence_frame_counter >= VAD_STOP_THRESHOLD_FRAMES
                ):
                    dur = (time.time() - speech_start_time) if speech_start_time else (len(speech_buffer) / SAMPLE_RATE)
                    print(f"Speech ended → checking ({dur:.2f}s actual speech)...")
                    if dur >= MIN_SEGMENT_SEC:
                        print(f"Valid segment → transcribing ({dur:.2f}s)...")
                        segment = speech_buffer.copy()
                        is_transcribing = True
                        def safe_transcribe(seg, idx):
                            global is_transcribing
                            time.sleep(0.3)  # Debounce
                            try:
                                sf.write(f"segment_{idx}.wav", seg, SAMPLE_RATE)
                                rms_energy = np.sqrt(np.mean(segment ** 2))
                                if rms_energy < 0.002 or len(segment) < SAMPLE_RATE * 0.4:
                                    print(f"Low-energy or too-short segment (RMS={rms_energy:.4f}), skipping transcription.")
                                    is_transcribing = False
                                    return
                                transcribe_segment(seg)
                            finally:
                                is_transcribing = False
                        threading.Thread(target=safe_transcribe, args=(segment, segment_index), daemon=True).start()
                        segment_index += 1
                        # FIX: Clear old interrupted data (prevents resume after real speech)
                        interrupted_audio_data = None
                        interrupted_pos = 0
                    else:
                        print(f"Segment too short ({dur:.2f}s), skipping.")
                    is_talking = False
                    speech_buffer = np.zeros(0, dtype=np.float32)
                    speech_start_time = None  # Reset
                    silence_frame_counter = 0  
                
                elif (
                    not is_talking
                    and not is_playing_ai
                    and interrupted_audio_data is not None
                    and (time.time() - last_speech_time) > AI_RESUME_TIMEOUT
                ):
                    print("No real speech detected — resuming AI playback...")
                    try:
                        audio_stream = io.BytesIO(interrupted_audio_data)
                        pygame.mixer.music.load(audio_stream, "mp3")
                        start_sec = max(0.0, interrupted_pos / 1000.0)
                        pygame.mixer.music.play(start=start_sec)
                        is_playing_ai = True
                        interrupted_audio_data = None
                        interrupted_pos = 0
                    except Exception as e:
                        print("Resume error:", e)
        time.sleep(0.01)

# ===================== START =====================
threading.Thread(target=listener_loop, daemon=True).start()

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Exiting...")
    stream.stop()
    stream.close()

