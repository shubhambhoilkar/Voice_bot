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
 
# ===================== CONFIG =====================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512
VAD_SILENCE_TIMEOUT = 2.5
MIN_SEGMENT_SEC = 0.5
API_URL = "localhost:/Sam_voicebot"

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
    """Convert text to speech and play it, allowing interruption & safe resuming."""
    global is_playing_ai, interrupted_audio_data, interrupted_pos
    try:
        tts = gTTS(text=text, lang="en")
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # Keep safe copy for resume
        interrupted_audio_data = mp3_fp.getvalue()
        interrupted_pos = 0
        is_playing_ai = True

        pygame.mixer.music.load(io.BytesIO(interrupted_audio_data), "mp3")
        pygame.mixer.music.play()

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
            speak_text(reply)
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
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
            try:
                prob = vad_model(chunk_tensor, SAMPLE_RATE).item()
                speech = prob > VAD_PROB_THRESHOLD
                # print(f"VAD prob: {prob:.3f} → speech={speech}")  # Debug print (remove if too spammy)
            except Exception as e:
                print("VAD error:", e)
                speech = False
                prob = 0.0
            speech_buffer = np.concatenate((speech_buffer, chunk))[-SAMPLE_RATE * 10:]
            if speech:
                print("Voice activity detected")
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
                # Detect speech end using consecutive silence frames (robust to brief noises)
                if (
                    is_talking
                    and not is_transcribing
                    and silence_frame_counter >= VAD_STOP_THRESHOLD_FRAMES
                ):
                    # Accurate dur (speech time only, ignores trailing silence)
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
                        # NO clear here → allows resume in next chunk
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
