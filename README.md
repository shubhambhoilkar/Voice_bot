* 🎙️ AI Voice Bot — Real-Time Speech Interaction with Whisper + TTS + Echo Cancellation *

  - A production-grade real-time voice assistant built using:
  - Whisper (Faster-Whisper) for speech-to-text
  - gTTS for text-to-speech
  - Custom LLM API for intelligent responses
  - Adaptive Echo Cancellation to avoid self-triggering when using speakers
  - Silero VAD for robust voice activity detection
  - Interruption support (user can talk over the AI and it will pause/resume)

# 📌 Key Features 
# 🔊 1. Real-time streaming speech-to-text

Continuous audio capture from microphone, processed in small blocks with Silero VAD and Whisper.

🗣️ 2. Natural-sounding text-to-speech

The bot uses Google TTS to speak responses with smooth playback and optional interruption handling.

🤖 3. AI conversation via custom API

User speech is sent to your custom LLM backend for contextual and intelligent replies.

🛑 4. Full interruption handling

If the user talks during playback, the bot pauses, processes the speech, and optionally resumes.

🔇 5. Echo cancellation (delay-tolerant)

Removes the bot's own voice from the microphone input — even when using external speakers.

📁 6. Automatic segmenting & merging

Speech is segmented with linguistic endpointing for natural conversation flow.

🧱 7. Plug-and-play structure

Modular design with clear separation:
```bash
VAD → Whisper → API → TTS → Playback → Echo Cancellation → VAD → ...

```

# 🧩 Project Structure
```bash
.
├── update_voice_bot_1.py    # Main voice bot file
├── README.md                # Documentation
├── segment_*.wav            # (Optional) Saved speech segments for debugging
└── requirements.txt         # Recommended dependencies

```
# Key Components Inside the Script
audio_callback():  	
Captures small audio frames from mic

listener_loop():
Main loop: AEC → VAD → segmentation → Whisper → API  

transcribe_segment():
Whisper transcription + linguistic endpointing

speak_text():
TTS + echo feed + interruption support

get_response_from_ai():
Sends user text to AI backend

aec_process():
Adaptive echo cancellation engine

aec_feed_playback();
Feeds speaker PCM into reference ringbuffer

# ⚙️ System Workflow
Below is the complete path of audio and data through the system.

# 1️⃣ Audio Capture → VAD → Speech Buffering

 - Microphone audio (16 kHz, mono) streams in as small blocks (512 samples).
 - Each chunk is preprocessed by echo cancellation (explained later).
 - Silero VAD decides whether the chunk contains speech.

Results:
 - Continuous user speech is merged into a "speech buffer".
 - Silence after speech triggers transcription.

# 2️⃣ Whisper Transcription
When speech ends, a final segment is sent to Faster-Whisper.
Features:
  - Automatic confidence evaluation
  - Partial merging of consecutive utterances
  - Endpoint detection based on:
    - Confidence
    - Punctuation
    - Number of words
    - Silence timing
   
# 4️⃣ TTS & Playback
 - gTTS converts AI response → MP3
 - Decoded PCM is fed into echo cancellation
 - MP3 is played using pygame

#Interruption support
If VAD detects user speech while playing audio:
 - Playback pauses
 - User input is processed
 - Audio optionally resumes

# 🔇 Echo Cancellation Workflow (Detailed)

This is the heart of the system and is often the hardest part to get right.

The implementation uses a delay-tolerant echo estimator, designed to cancel the bot's own speech picked up by the microphone when using speakers.
Pure NLMS adaptive filters in Python are unstable in real-time; this method is lighter, predictable, and stable.
