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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio

# ===================== CONFIG =====================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512
MIN_SEGMENT_SEC = 0.5
API_URL = "localhost:/Sam_voicebot"

# Smart endpointing / merging tuning
SPEECH_END_PADDING = 0.4
MERGE_WINDOW_SEC = 2.0
CONFIDENCE_THRESHOLD = -0.7
speech_start_time = None

VAD_SILENCE_TIMEOUT = 1.8
VAD_START_THRESHOLD_FRAMES = 12  # Balanced threshold
VAD_STOP_THRESHOLD_FRAMES = 56
AI_RESUME_TIMEOUT = 1.8
VAD_PROB_THRESHOLD = 0.75  # Balanced threshold (not too high)

# AI state
is_playing_ai = False
is_responding = False
interrupted_audio_data = None
interrupted_pos = 0
ai_speaking_start_time = None
ai_finished_time = None  # NEW: Track when AI finished

# Transcription + dedup state
is_transcribing = False
ongoing_text = ""
last_transcription_time = 0.0
last_user_text = ""
last_ai_reply = None

# Echo Canceller state
AEC_FILTER_LEN = 4096
AEC_LEARNING_RATE = 0.3
AEC_EPS = 1e-8
aec_weights = np.zeros(AEC_FILTER_LEN, dtype=np.float32)
playback_ringbuffer = collections.deque(maxlen=AEC_FILTER_LEN * 6)
ai_audio_reference = None

# VAD state
speech_frame_counter = 0
silence_frame_counter = 0

# Thread safety locks
transcription_lock = threading.Lock()
api_lock = threading.Lock()
playback_lock = threading.Lock()
stop_event = threading.Event()

# ===================== LOAD MODELS =====================
print("Loading models...")
publish_queue = queue.Queue()   # send events to websocket 

def publish(event_type: str, text: str = "", meta: dict | None = None):
    payload = {"type": event_type, "text": text}
    if meta:
        payload["meta"] =meta
    try:
        publish_queue.put_nowait(payload)
    except Exception:
        pass

#Publish Startup status
publish("status", "loading_models")

vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

publish("status", "models_loaded")
print("Model Loaded.")

# ===================== STATE =====================
audio_q = queue.Queue()
is_talking = False
last_speech_time = time.time()
speech_buffer = np.zeros(0, dtype=np.float32)
segment_index = 0

# ==================== IMPROVED ECHO CANCELLATION =========================
def aec_feed_playback(play_chunk: np.ndarray):
    """Feed AI playback audio to echo canceller."""
    with playback_lock:
        playback_ringbuffer.extend(play_chunk.tolist())

def aec_process(mic_chunk: np.ndarray) -> np.ndarray:
    """Enhanced adaptive echo cancellation."""
    global aec_weights

    with playback_lock:
        if len(playback_ringbuffer) < AEC_FILTER_LEN:
            return mic_chunk

        x = np.array(list(playback_ringbuffer)[-AEC_FILTER_LEN:], dtype=np.float32)
    
    d = mic_chunk.astype(np.float32)

    # Predict echo
    y_est = np.dot(aec_weights, x[-len(aec_weights):])
    y = np.full_like(d, y_est)
    e = d - y

    # Adaptive NLMS update with double-talk detection
    mic_power = np.mean(d ** 2)
    echo_power = np.mean(y ** 2)
    
    # Only adapt when echo is dominant
    if echo_power > mic_power * 0.5:
        norm_x = np.dot(x, x) + AEC_EPS
        update = (AEC_LEARNING_RATE / norm_x) * e.mean() * x
        
        if len(update) != len(aec_weights):
            update = np.resize(update, len(aec_weights))
        
        aec_weights += update

    return e

def detect_ai_voice_in_mic(mic_chunk: np.ndarray) -> tuple[bool, float]:
    """
    Detect if microphone is picking up AI's own voice.
    Returns (is_echo, confidence) where:
    - is_echo: True if ONLY AI echo detected (no user speech)
    - confidence: How strong the echo correlation is (0-1)
    """
    global ai_audio_reference, is_playing_ai, ai_speaking_start_time, ai_finished_time
    
    # Check if AI recently finished (grace period for echo tail)
    if ai_finished_time is not None:
        if (time.time() - ai_finished_time) < 0.5:  # 500ms grace period
            return True, 0.9  # Still consider it echo
        else:
            ai_finished_time = None  # Clear after grace period
    
    if not is_playing_ai or ai_audio_reference is None:
        return False, 0.0
    
    # Quick check: if pygame says nothing is playing, no echo
    if not pygame.mixer.music.get_busy():
        return False, 0.0
    
    try:
        # Calculate mic energy first
        mic_energy = np.sqrt(np.mean(mic_chunk ** 2))
        
        # Lowered threshold for strong user speech (was 0.12, now 0.08) to catch normal volumes sooner
        if mic_energy > 0.08:
            # ADDED: Debug print for user speech detection
            print(f"👤 Potential user speech detected (energy: {mic_energy:.4f}) - treating as non-echo")
            return False, 0.0  # Not pure echo - user is speaking!
        
        # Normalize mic signal
        mic_normalized = mic_chunk.copy()
        if np.max(np.abs(mic_normalized)) > 0:
            mic_normalized = mic_normalized / np.max(np.abs(mic_normalized))
        else:
            return False, 0.0  # Silent mic
        
        # Get recent playback snippet
        ref_len = len(mic_chunk)
        if len(ai_audio_reference) < ref_len:
            ai_snippet = ai_audio_reference
        else:
            # Get the most recent played portion
            elapsed_time = time.time() - ai_speaking_start_time
            sample_pos = int(elapsed_time * SAMPLE_RATE)
            start_pos = max(0, sample_pos - ref_len)
            end_pos = min(len(ai_audio_reference), sample_pos)
            
            if end_pos > start_pos:
                ai_snippet = ai_audio_reference[start_pos:end_pos]
                
                # Pad if needed
                if len(ai_snippet) < ref_len:
                    ai_snippet = np.pad(ai_snippet, (0, ref_len - len(ai_snippet)))
                else:
                    ai_snippet = ai_snippet[:ref_len]
            else:
                return False, 0.0
        
        # Normalize AI snippet
        if np.max(np.abs(ai_snippet)) > 0:
            ai_snippet = ai_snippet / np.max(np.abs(ai_snippet))
        else:
            return False, 0.0
        
        # Compute correlation
        if len(mic_normalized) == len(ai_snippet):
            corr = np.correlate(mic_normalized, ai_snippet, mode='valid')
            corr_value = float(np.max(np.abs(corr))) if len(corr) > 0 else 0.0
            
            print(f"🔍 Echo check: energy={mic_energy:.4f}, corr={corr_value:.3f}")
            
            # This reduces false positives for user speech (which may have some incidental correlation)
            if corr_value > 0.85 and mic_energy < 0.03:  # Was 0.75 and <0.05; tightened for stricter echo
                return True, corr_value
            
            # Further narrowed moderate band to avoid catching soft user speech
            if 0.6 < corr_value < 0.85 and 0.005 < mic_energy < 0.025:  # Was 0.5-0.75 and 0.01-0.08
                return True, corr_value
            
            return False, corr_value
                
    except Exception as e:
        print(f"Echo detection error: {e}")
    
    return False, 0.0


# ===================== AUDIO OUTPUT INIT =====================
pygame.mixer.init(frequency=SAMPLE_RATE, channels=1)

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
print("🎤 Listening for speech...")

# ===================== TTS (Speak Function) =====================
def speak_text(text):
    global is_playing_ai, interrupted_audio_data, interrupted_pos
    global ai_audio_reference, ai_speaking_start_time, ai_finished_time

    """Convert text to speech and play it, allowing interruption."""
    try:
        print(f"🔊 AI speaking: {text[:50]}...")
        publish("bot_start", text[:200])
        tts = gTTS(text=text, lang="en")
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        interrupted_audio_data = mp3_fp.getvalue()
        interrupted_pos = 0
        
        # Extract PCM for echo cancellation and reference
        try: 
            pcm_buf = io.BytesIO(interrupted_audio_data)
            audio_segment = AudioSegment.from_file(pcm_buf, format="mp3")
            audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
            pcm_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            
            # Feed to AEC
            aec_feed_playback(pcm_data)
            
            # Store reference for detection
            with playback_lock:
                ai_audio_reference = pcm_data
            
        except Exception as e:
            print(f"AEC/reference decode error: {e}")
            publish("error", f"AEC/reference decode error: {e}")

        is_playing_ai = True
        ai_speaking_start_time = time.time()
        ai_finished_time = None  # Clear any previous finish time

        play_buf = io.BytesIO(interrupted_audio_data)
        pygame.mixer.music.load(play_buf, "mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            if is_talking:  # Real user speech detected
                interrupted_pos = pygame.mixer.music.get_pos()
                pygame.mixer.music.stop()
                is_playing_ai = False
                print("⚠️ AI interrupted by user")
                break
                
    except Exception as e:
        print(f"TTS Error: {e}")
        publish("error",f"TTS Error: {e}")
    finally:
        if not pygame.mixer.music.get_busy():
            is_playing_ai = False
            ai_finished_time = time.time()  # Mark when AI finished
            with playback_lock:
                ai_audio_reference = None
            interrupted_audio_data = None   
            interrupted_pos = 0
            print("✅ AI finished speaking")
            publish("bot_end", "")

# ===================== API CALL =====================
def get_response_from_ai(text):
    """Send transcribed text to Gen AI API with dedupe and lock."""
    global last_user_text, last_ai_reply, is_responding
    
    with api_lock:
        print(f"📤 API called with: {text!r}")
        publish("api_call", text[:200])

        user_text = text.strip().lower()
        if not user_text:
            return None

        if is_responding:
            print("⏸️ AI still responding — skipping new request")
            publish("status", "ai_busy")
            return None

        if user_text == last_user_text:
            print("🔁 Duplicate detected — skipping API call")
            return last_ai_reply

        is_responding = True
        last_user_text = user_text

    payload = {
        "user_id": "user1",
        "client_id": "44ihRG38UX24DKeFzE15FbbPZfCgz3rh",
        "text": user_text
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            reply = data.get("response") or data.get("reply") or data.get("text") or str(data)
            print(f"🤖 AI: {reply}\n")
            
            with api_lock:
                last_ai_reply = reply
            
            publish("bot", reply[:1000])
            speak_text(reply)
            return reply
        else:
            print(f"❌ API error: {resp.status_code}")
            publish("error", f"Request failed: {e}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    finally:
        with api_lock:
            is_responding = False
    return None

# ===================== TRANSCRIBE =====================
def transcribe_segment(segment):
    """Whisper + semantic endpointing + dedupe check."""
    global ongoing_text, last_transcription_time, last_user_text

    with transcription_lock:
        print(f"🎯 Transcribing... ({len(segment)/SAMPLE_RATE:.2f}s)")
        publish("status", f"transcribing {len(segment)/SAMPLE_RATE: .2f}s")
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
        
        if not full_text:
            print("❌ No valid speech detected")
            return

        print(f"👤 You: {full_text} (confidence: {avg_conf:.2f})")
        publish("user", full_text)

        # Skip duplicates
        if full_text.lower().strip() == last_user_text.lower().strip():
            print("🔁 Duplicate transcription, skipping")
            return

        # Merge consecutive transcripts
        now = time.time()
        if (now - last_transcription_time) <= MERGE_WINDOW_SEC and ongoing_text:
            ongoing_text = (ongoing_text + " " + full_text).strip()
        else:
            ongoing_text = full_text
        last_transcription_time = now

        # Decide if complete
        complete_sentence = full_text.endswith((".", "?", "!", "…"))
        confident = avg_conf > CONFIDENCE_THRESHOLD
        long_text = len(full_text.split()) >= 8

        likely_complete = complete_sentence or confident or long_text

        if likely_complete:
            print(f"✅ Complete utterance: {ongoing_text}")
            final_text = ongoing_text
            ongoing_text = ""
            threading.Thread(target=get_response_from_ai, args=(final_text,), daemon=True).start()
        else:
            print(f"⏳ Partial speech, waiting: {ongoing_text}")

# ===================== LISTENER LOOP =====================
def listener_loop():
    global is_talking, last_speech_time, speech_buffer, segment_index
    global is_playing_ai, interrupted_pos, is_transcribing, interrupted_audio_data
    global speech_start_time, speech_frame_counter, silence_frame_counter
    
    consecutive_speech_over_ai = 0  # NEW: Track sustained user speech during AI playback
    
    # while True:
    while not stop_event.is_set():
        if not audio_q.empty():
            chunk = audio_q.get().flatten()
            
            # Check for AI echo and get confidence
            is_echo, echo_confidence = detect_ai_voice_in_mic(chunk)
            
            # Apply AEC regardless
            chunk = aec_process(chunk)

            # VAD processing
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
            try:
                prob = vad_model(chunk_tensor, SAMPLE_RATE).item()
                speech = prob > VAD_PROB_THRESHOLD
            except Exception as e:
                print(f"VAD error: {e}")
                speech = False
                prob = 0.0
            
            speech_buffer = np.concatenate((speech_buffer, chunk))[-SAMPLE_RATE * 10:]
            
            if speech:
                silence_frame_counter = 0
                
                # Only skip frame if it's pure echo (not user speech)
                if is_echo and echo_confidence > 0.7:
                    consecutive_speech_over_ai = 0
                    # ADDED: Debug print for skipped echo
                    print(f"🔇 Skipping echo frame (conf: {echo_confidence:.3f})")
                    continue  # Skip pure echo frames
                
                # Real speech detected
                speech_frame_counter += 1
                last_speech_time = time.time()
                
                # ADDED: Debug print for real speech frames during AI
                if is_playing_ai:
                    print(f"👤 Real speech frame during AI (consec: {consecutive_speech_over_ai + 1})")
                
                # User speech detection logic
                if not is_talking:
                    if is_playing_ai:
                        # Track consecutive speech frames during AI playback
                        consecutive_speech_over_ai += 1
                        
                        # Reduced threshold for interruption (from 10 to 7) for even faster response
                        interruption_threshold = 10
                        
                        if consecutive_speech_over_ai >= interruption_threshold:
                            print(f"✋ User interruption confirmed! (frames: {consecutive_speech_over_ai})")
                            publish("event", "user_interruption_confirmed")
                            is_talking = True
                            speech_start_time = time.time()
                            interrupted_pos = pygame.mixer.music.get_pos()
                            pygame.mixer.music.stop()
                            is_playing_ai = False
                            consecutive_speech_over_ai = 0
                    else:
                        # Normal detection when AI not speaking
                        if speech_frame_counter >= VAD_START_THRESHOLD_FRAMES:
                            print("🗣️ User speech started")
                            publish("event", "user_speech_started")
                            is_talking = True
                            speech_start_time = time.time()
                            consecutive_speech_over_ai = 0
            else:
                speech_frame_counter = 0
                consecutive_speech_over_ai = 0  # Reset on silence
                
                if is_talking:
                    silence_frame_counter += 1
                
                # Detect speech end
                if (
                    is_talking
                    and not is_transcribing
                    and silence_frame_counter >= VAD_STOP_THRESHOLD_FRAMES
                ):
                    dur = (time.time() - speech_start_time) if speech_start_time else (len(speech_buffer) / SAMPLE_RATE)
                    print(f"🛑 Speech ended ({dur:.2f}s)")
                    publish("event","user_speech_ended", meta={"duration_s": dur})
                    
                    if dur >= MIN_SEGMENT_SEC:
                        segment = speech_buffer.copy()
                        is_transcribing = True
                        
                        def safe_transcribe(seg, idx):
                            global is_transcribing
                            time.sleep(0.3)
                            try:
                                sf.write(f"segment_{idx}.wav", seg, SAMPLE_RATE)
                                rms_energy = np.sqrt(np.mean(seg ** 2))
                                
                                if rms_energy < 0.002 or len(seg) < SAMPLE_RATE * 0.4:
                                    print(f"⚠️ Low-energy segment (RMS={rms_energy:.4f}), skipping")
                                    publish("Warning", "low_energy_segment")
                                    return
                                
                                transcribe_segment(seg)
                            except Exception as e:
                                print(f"Transcription error: {e}")
                                publish("error",f"Trascription error: {e}")
                            finally:
                                is_transcribing = False
                        
                        threading.Thread(target=safe_transcribe, args=(segment, segment_index), daemon=True).start()
                        segment_index += 1
                        interrupted_audio_data = None
                        interrupted_pos = 0
                    else:
                        print(f"⚠️ Segment too short ({dur:.2f}s)")
                    
                    is_talking = False
                    speech_buffer = np.zeros(0, dtype=np.float32)
                    speech_start_time = None
                    silence_frame_counter = 0
                
                # Resume AI playback
                elif (
                    not is_talking
                    and not is_playing_ai
                    and interrupted_audio_data is not None
                    and (time.time() - last_speech_time) > AI_RESUME_TIMEOUT
                ):
                    print("▶️ Resuming AI playback...")
                    publish("event", "ai_resuming")
                    try:
                        audio_stream = io.BytesIO(interrupted_audio_data)
                        pygame.mixer.music.load(audio_stream, "mp3")
                        start_sec = max(0.0, interrupted_pos / 1000.0)
                        pygame.mixer.music.play(start=start_sec)
                        is_playing_ai = True
                        ai_speaking_start_time = time.time() - start_sec
                        interrupted_audio_data = None
                        interrupted_pos = 0
                    except Exception as e:
                        print(f"Resume error: {e}")
        
        time.sleep(0.01)

# ===================== END CALL =================
def keyboard_listner():
    input("Press ENTER to Stop the Voice bot...\n")
    stop_event.set()

# ===================== START =====================
threading.Thread(target=listener_loop, daemon=True).start()

# --- Fastapi Websocket app ===
app = FastAPI()

@app.get("/")
async def root():
    html = "<html><body><h3>Voice Bot Websocket is running. Use a websocket client at /ws.</h3></body></html>"
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Websocket client connected.")
    try:
        while True:
            event = await asyncio.to_thread(publish_queue.get)
            try:
                await websocket.send_json(event)
            except Exception as e:
                print(f"Webscoket send error: {e}")
                break
    
    except WebSocketDisconnect:
            print("WebSocket send error: {e}")
            # time.sleep(0.5)
    except Exception as e:
        print(f"Websocket error: {e}")

@app.post("/stop")
def stop_bot():
    stop_event.set()
    return {"status": "stopping"}


if __name__ == "__main__":
    print("Starting FastAPI Uvicorn server on port 9900...")
    publish("status","server_starting")

    uvicorn.run("voicebotfix:app", host="localhost", port= 9900, reload=False)
