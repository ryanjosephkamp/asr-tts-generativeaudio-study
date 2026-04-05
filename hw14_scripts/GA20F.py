# GA20F.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20F.py

import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
from genaibook.core import get_speaker_embeddings

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

inputs = processor(text="There are llamas all around.", return_tensors="pt")
speaker_embeddings = torch.tensor(get_speaker_embeddings()).unsqueeze(0)

with torch.inference_mode():
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.imshow(np.rot90(np.array(spectrogram)))
plt.show()

from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
with torch.inference_mode():
    # Alternatively
    # model.generate_speech(
    #   inputs["input_ids"],
    #   speaker_embeddings,
    #   vocoder=vocoder)
    speech = vocoder(spectrogram)
array = speech.numpy()

import scipy
scipy.io.wavfile.write("GA20Fout2.wav", rate=16_000, data=array)

