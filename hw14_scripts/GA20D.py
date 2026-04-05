# GA20D.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20D.py

import torch
from genaibook.core import generate_long_audio
long_audio = generate_long_audio()
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
outputs = pipe(
    long_audio,
    generate_kwargs={"task": "transcribe"},
    chunk_length_s=5,
    batch_size=8,
    return_timestamps=True,
)
print(outputs)