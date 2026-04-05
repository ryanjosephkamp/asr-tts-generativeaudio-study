# GA20H.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20H.py

import torch
from transformers import AutoModel, AutoProcessor
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

inputs = processor(
    text=[
        """Hello, my name is Suno. And, uh — and I like pizza. [laughs]
        But I also have other interests such as playing tic tac toe."""
    ],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)
import scipy
scipy.io.wavfile.write("GA20Hout1.wav", rate=16_000, data=speech_values.numpy().squeeze())

