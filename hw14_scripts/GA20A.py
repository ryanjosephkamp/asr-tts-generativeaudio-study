# GA20A.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20A.py

import torch
from datasets import load_dataset
from datasets import Audio
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

from transformers import pipeline
asr = pipeline("automatic-speech-recognition", 
    model="openai/whisper-small",
    max_new_tokens=200,
)

example = minds.shuffle()[0]
transcription = asr(example["audio"]["array"])
print(transcription)
print(example["english_transcription"])
