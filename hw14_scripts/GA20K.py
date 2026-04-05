# GA20K.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20K.py

import torch
from transformers import AutoModel, AutoProcessor
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")
voice_preset = "v2/en_speaker_5"
inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)
speech_values = model.generate(**inputs, do_sample=True)
import scipy
scipy.io.wavfile.write("GA20Kout2.wav", rate=16_000, data=speech_values.numpy().squeeze())

