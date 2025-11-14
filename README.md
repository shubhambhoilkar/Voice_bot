# Voice_bot
Simple offiline Voice Bot using openai for LLM

Workflow for Voice Bot:

#code
Microphone → VAD → buffer speech → silence detected?
                 ↳ no → keep listening
                 ↳ yes → Whisper → transcript text
                                     ↓
                                LLM API call
                                     ↓
                                gTTS synthesize
                                     ↓
                               Pygame playback
      ↘ if user speaks (VAD detects)
         stop playback (barge-in)
         ↳ after silence → resume playback
