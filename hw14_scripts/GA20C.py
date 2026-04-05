# GA20C.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA20C.py

import torch
from datasets import load_dataset
from datasets import Audio
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
example = minds.shuffle()[0]
array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

from transformers import WhisperForConditionalGeneration, WhisperProcessor
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
inputs = whisper_processor(array, sampling_rate=sampling_rate, return_tensors="pt")
with torch.inference_mode():
    generated_ids = whisper_model.generate(**inputs)
transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=False)
print(transcription)
print(example["english_transcription"])
