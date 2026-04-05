# GA20E.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20E.py

import torch
from datasets import load_dataset
from datasets import Audio
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
example = minds.shuffle()[0]
array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

from transformers import SpeechT5ForSpeechToText, SpeechT5Processor

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

inputs = processor(
    audio=array, sampling_rate=sampling_rate, return_tensors="pt"
)
with torch.inference_mode():
    predicted_ids = model.generate(**inputs, max_new_tokens=70)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
print(example["english_transcription"])
