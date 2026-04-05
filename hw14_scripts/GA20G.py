# GA20G.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20G.py

import torch
from transformers import VitsModel, VitsTokenizer, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")
with torch.inference_mode():
    outputs = model(inputs["input_ids"])

array = outputs.waveform[0].numpy()

import scipy
scipy.io.wavfile.write("GA20Gout1.wav", rate=16_000, data=array)
