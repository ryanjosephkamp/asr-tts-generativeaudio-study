# GA20B.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20B.py

import torch
from datasets import load_dataset
from datasets import Audio
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
example = minds.shuffle()[0]
array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
inputs = wav2vec2_processor(array, sampling_rate=sampling_rate, return_tensors="pt")
with torch.inference_mode():
    outputs = wav2vec2_model(**inputs)
predicted_ids = torch.argmax(outputs.logits, dim=-1)
transcription = wav2vec2_processor.batch_decode(predicted_ids)
print(transcription)
print(example["english_transcription"])
