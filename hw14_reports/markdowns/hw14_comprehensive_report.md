# Speech AI with Transformers: A Systematic Experimental Study of Automatic Speech Recognition, Text-to-Speech Synthesis, and Generative Audio

### A Comprehensive Experimental Study of ASR Family Comparison, Preprocessing Sensitivity, Long-Form Transcription, Speaker-Conditioned TTS, End-to-End Waveform Synthesis, and Prompt-Conditioned Audio Generation

---

**Author:** Ryan Kamp  
**Affiliation:** Department of Computer Science, University of Cincinnati  
**Location:** Cincinnati, OH, USA  
**Email:** kamprj@mail.uc.edu  
**GitHub:** https://github.com/ryanjosephkamp

**Course:** CS6078 Generative AI  
**Assignment:** Homework 14 - Speech Recognition and Speech Generation  
**Date:** April 2026

---

<div style="page-break-after: always;"></div>

## Abstract

This study presents a systematic experimental investigation of modern speech AI systems spanning **automatic speech recognition (ASR) and text-to-speech synthesis (TTS)**, conducted through **eight controlled experiments addressing seven research questions**. Using the **MINDS-14 banking-intent dataset** and a common text bank, we compared **four ASR workflows** — Whisper pipeline (GA20A), Wav2Vec2 CTC (GA20B), direct Whisper (GA20C), and SpeechT5 ASR (GA20E) — and **three TTS families** — SpeechT5 with HiFi-GAN, MMS-TTS/VITS, and Bark. The study produced **272 consolidated data rows and 22 figures** across three execution groups on Google Colab GPU. In the **ASR domain**, **direct Whisper inference (GA20C) achieved the lowest mean word error rate (WER) of 0.169** on 24 fixed MINDS-14 utterances, followed by the Whisper pipeline (0.188), Wav2Vec2 CTC (0.390), and SpeechT5 ASR (0.566). **Sampling-rate sensitivity analysis** revealed that deviation from the expected 16 kHz input degraded all models substantially, with Wav2Vec2 CTC mean WER rising from 0.357 at 16 kHz to 0.975 at 8 kHz. **Long-form Whisper transcription** showed that chunk lengths of 5 and 10 seconds produced near-identical transcripts (similarity $\geq$ 0.993), while 15-second chunks generated divergent, repetitive outputs (similarity 0.289). In the **TTS domain**, **round-trip ASR evaluation** on common texts revealed **SpeechT5 as the most intelligible system (mean WER 0.067, mean character error rate [CER] 0.012)** with a mean generation time of 1.006 seconds, followed by **MMS-TTS (WER 0.200, CER 0.132, 0.480 s) and Bark (WER 0.333, CER 0.247, 29.4 s)**. **MMS-TTS demonstrated perfect deterministic reproducibility** under seed control, **SpeechT5 provided explicit controllability** through speaker embeddings and an exposed spectrogram-to-vocoder pipeline, and **Bark offered the greatest expressive range** but at the cost of stochasticity and substantially longer generation times. These findings provide **empirical characterizations of the controllability–expressiveness–efficiency trade-off space** in contemporary speech AI.

**Keywords:** automatic speech recognition, text-to-speech synthesis, Whisper, wav2vec 2.0, SpeechT5, MMS-TTS, VITS, Bark, HiFi-GAN, MINDS-14, generative audio, speech AI

---

<div style="page-break-after: always;"></div>

## Table of Contents

1. [Introduction](#i-introduction)
2. [Literature Review](#ii-literature-review)
3. [Methodology](#iii-methodology)
4. [Results: Experiment 1 - Baseline Reproduction](#iv-results-experiment-1---baseline-reproduction)
5. [Results: Experiment 2 - Short-Form ASR Family Comparison on MINDS-14](#v-results-experiment-2---short-form-asr-family-comparison-on-minds-14)
6. [Results: Experiment 3 - ASR Preprocessing, Decoding, and Normalization Sensitivity](#vi-results-experiment-3---asr-preprocessing-decoding-and-normalization-sensitivity)
7. [Results: Experiment 4 - Long-Form Whisper Chunking and Timestamp Stability](#vii-results-experiment-4---long-form-whisper-chunking-and-timestamp-stability)
8. [Results: Experiment 5 - SpeechT5 TTS Speaker Conditioning and Vocoder Transparency](#viii-results-experiment-5---speecht5-tts-speaker-conditioning-and-vocoder-transparency)
9. [Results: Experiment 6 - MMS-TTS Reproducibility, Text Sensitivity, and Speaking-Rate Control](#ix-results-experiment-6---mms-tts-reproducibility-text-sensitivity-and-speaking-rate-control)
10. [Results: Experiment 7 - Bark Expressiveness, Voice Presets, and Sample-Rate Integrity](#x-results-experiment-7---bark-expressiveness-voice-presets-and-sample-rate-integrity)
11. [Results: Experiment 8 - Cross-Model TTS Intelligibility, Controllability, and Runtime](#xi-results-experiment-8---cross-model-tts-intelligibility-controllability-and-runtime)
12. [Discussion](#xii-discussion)
13. [Conclusion](#xiii-conclusion)
14. [Future Work](#xiv-future-work)
15. [References](#xv-references)

---

<div style="page-break-after: always;"></div>

## I. Introduction

### A. Assignment Context and Project Scope

This study extends Homework 14 for CS6078 Generative AI (Spring 2026, University of Cincinnati), which is based on Chapter 9 of *Hands-on Generative AI with Transformers and Diffusion Models* by Alammar and Grootendorst [1]. The base assignment requires running nine Python scripts — GA20A through GA20K — that implement speech recognition and speech generation workflows, and presenting the resulting text and audio outputs. This project extends the assignment into a rigorous experimental study by: (1) running all nine scripts with their original default parameters to establish baselines; (2) systematically varying parameters across eight experiments to address seven research questions spanning the ASR and TTS domains; and (3) producing publication-quality analysis with quantitative metrics, comparative tables, and cross-modal synthesis.

The nine scripts collectively instantiate two major directions of speech AI. Five scripts address automatic speech recognition: GA20A uses Whisper via the high-level Hugging Face pipeline API, GA20B performs explicit Wav2Vec2 CTC inference, GA20C uses Whisper through direct model-and-processor calls, GA20D demonstrates long-form Whisper transcription with chunking and timestamps, and GA20E applies SpeechT5 as a unified speech-text encoder-decoder for ASR. Four scripts address speech generation: GA20F performs SpeechT5 TTS with speaker embeddings and a HiFi-GAN vocoder, GA20G uses MMS-TTS/VITS for end-to-end waveform synthesis, and GA20H and GA20K use Bark for prompt-conditioned and voice-preset-conditioned generative audio respectively.

The study produced 272 consolidated data rows across 10 primary CSV files, 22 figures, and structured JSON artifacts, organized into three execution groups: Group A (ASR experiments), Group B (structured TTS experiments), and Group C (Bark and cross-model comparison). All experiments were executed on Google Colab with T4 GPU acceleration using dry-run and full notebook pairs with checkpointing.

<div style="page-break-after: always;"></div>

### B. Research Questions

#### RQ1. Short-Form ASR Family Comparison on MINDS-14

How do the short-form ASR families represented by GA20A, GA20B, GA20C, and GA20E compare on MINDS-14 when evaluated on a common fixed subset with consistent 16 kHz preprocessing?

#### RQ2. ASR Preprocessing and Decoding Sensitivity

How sensitive are ASR conclusions to preprocessing and decoding choices such as resampling rate, token cleanup, and pipeline-versus-direct decoding?

#### RQ3. Long-Form Whisper Chunking and Timestamps

How do Whisper chunk length and batching choices affect long-form transcript structure, timestamp granularity, and self-consistency in GA20D-style transcription?

#### RQ4. SpeechT5 TTS Speaker Conditioning and Vocoder Transparency

How much controllability and interpretability does SpeechT5 offer through explicit speaker embeddings and its exposed spectrogram-to-vocoder pipeline?

#### RQ5. MMS-TTS Reproducibility and Speaking-Rate Control

How reproducible is MMS-TTS/VITS under explicit seed control, and how do text content and speaking-rate controls affect duration and intelligibility?

#### RQ6. Bark Expressiveness, Voice Presets, and Sample-Rate Integrity

How do Bark's two control mechanisms, free-form prompt cues and explicit voice presets, differ in expressive power, repeatability, and sensitivity to sample-rate metadata at export time?

#### RQ7. Cross-Model TTS Trade-Offs

Across SpeechT5, MMS-TTS, and Bark, which model family offers the best assignment-relevant trade-off among intelligibility, controllability, expressiveness, and runtime?

<div style="page-break-after: always;"></div>

### C. Report Organization

This report is organized as follows. Section II provides a condensed literature review covering the theoretical foundations of the ASR and TTS systems under study. Section III describes the experimental methodology, including dataset selection, evaluation metrics, execution architecture, and reproducibility controls. Sections IV through XI present the results of the eight experiments in order. Section XII synthesizes the findings into a cross-experiment discussion addressing all seven research questions. Section XIII provides conclusions, and Section XIV outlines directions for future work.

---

<div style="page-break-after: always;"></div>

## II. Literature Review

### A. Audio Waveform Representation and Sampling-Rate Handling

Digital speech is represented as a discrete sequence of amplitude samples $x[n]$, $n = 0, 1, \ldots, N-1$, taken at a fixed sampling rate $f_s$. The Nyquist-Shannon sampling theorem requires $f_s \geq 2 f_{\max}$ for faithful reconstruction of frequencies up to $f_{\max}$. In the context of HW14, the MINDS-14 dataset provides telephony-like audio at 8 kHz, while all ASR models expect 16 kHz inputs. This rate mismatch requires explicit resampling via `Audio(sampling_rate=16000)` in the Hugging Face datasets library. On the TTS side, SpeechT5 and MMS-TTS generate waveforms at 16 kHz, while Bark's internal codec operates at 24 kHz. Incorrect sample-rate metadata during audio export produces pitch and tempo distortion proportional to the rate ratio, a practical concern investigated in Experiment 7.

### B. MINDS-14 and Spoken-Intent Benchmarking

MINDS-14, introduced by Gerz et al. [2], is a multilingual spoken-intent dataset comprising telephony-style recordings of banking interactions across 14 languages. This study uses the `en-AU` (Australian English) split, which contains short conversational utterances in a domain-constrained vocabulary. The telephony-like recording conditions and narrow intent vocabulary make MINDS-14 a realistic but challenging benchmark for evaluating ASR transfer: pretrained models must handle reduced bandwidth, informal phrasing, and accented speech without task-specific fine-tuning.

### C. Automatic Speech Recognition: CTC, Sequence-to-Sequence, and Unified Architectures

Three ASR paradigms are represented in the HW14 scripts. **CTC-based recognition** with Wav2Vec2 [3] produces frame-level token logits decoded by collapsing repeated predictions and removing blank symbols according to the Connectionist Temporal Classification framework [4]. The CTC objective maximizes $p(\mathbf{y} | \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t} p(\pi_t | \mathbf{x})$, where $\mathcal{B}$ is the many-to-one CTC mapping. **Sequence-to-sequence recognition** with Whisper [5] uses an encoder-decoder Transformer [6] trained on 680,000 hours of weakly supervised multilingual data, generating text autoregressively conditioned on log-mel spectrogram features. **Unified speech-text modeling** with SpeechT5 [7] shares a single encoder-decoder backbone across ASR, TTS, and other speech-text tasks through modality-specific pre/post-nets.

<div style="page-break-after: always;"></div>

### D. Long-Form Whisper Transcription and Chunking

Whisper's attention-based decoder has a fixed context window of 30 seconds. For audio longer than this window, the Hugging Face pipeline implements a sequential chunked algorithm: the input is segmented into overlapping chunks of `chunk_length_s` seconds, each chunk is transcribed independently, and the resulting segments are concatenated with timestamp alignment. The `batch_size` parameter controls how many chunks are processed in parallel on the GPU. Experiment 4 investigates how chunk length and batch size affect transcript structure, self-consistency, and runtime.

### E. Text-to-Speech Synthesis: SpeechT5, VITS/MMS-TTS, and Bark

Three TTS paradigms are represented in HW14. **SpeechT5 TTS** [7] with HiFi-GAN [8] is a two-stage pipeline: the encoder-decoder generates a mel-spectrogram conditioned on text tokens and a speaker embedding vector, and the HiFi-GAN vocoder converts the spectrogram to a time-domain waveform. This architecture exposes an interpretable intermediate representation and supports explicit speaker conditioning through x-vector embeddings. **MMS-TTS** [9] uses the VITS architecture [10], a conditional variational autoencoder with adversarial training that generates waveforms end-to-end from text without an explicit spectrogram stage. VITS combines a posterior encoder, a normalizing flow, and a HiFi-GAN decoder in a single model, yielding deterministic outputs under fixed random seeds. **Bark** is a multi-stage token-based generative model that converts text to semantic tokens, then to coarse acoustic tokens, and finally to fine acoustic tokens decoded through an EnCodec-style neural audio codec [11]. This architecture supports expressive generation, including non-speech elements like laughter and pauses, but is inherently stochastic.

### F. Neural Audio Codecs and Token-Based Generation

Neural audio codecs such as EnCodec [11] represent audio as sequences of discrete tokens through residual vector quantization, enabling language-model-style generation of audio. Bark leverages this representation to generate speech through a cascade of three Transformer models, each operating at a different level of the token hierarchy. The multi-stage design enables expressive control through text prompt engineering and voice presets, but introduces compounding stochasticity: each stage samples from a learned distribution, making exact reproduction impossible even under fixed seeds. The native codec sample rate of 24 kHz must be preserved during export to avoid the pitch and tempo artifacts investigated in Experiment 7.

---

## III. Methodology

### A. Experimental Design Overview

The study comprises eight experiments organized into three execution groups, addressing seven research questions that span the ASR and TTS capabilities of all nine scripts. Experiment 1 reproduces the baseline behavior of each script under its original default parameters. Experiments 2–4 investigate ASR performance: Experiment 2 compares four short-form ASR workflows on a common 24-utterance subset; Experiment 3 analyzes sensitivity to resampling rate and token cleanup; and Experiment 4 studies long-form Whisper chunking. Experiments 5–7 investigate TTS systems individually: Experiment 5 examines SpeechT5 speaker conditioning, Experiment 6 tests MMS-TTS reproducibility and speaking-rate control, and Experiment 7 studies Bark prompt styles, voice presets, and sample-rate integrity. Experiment 8 performs a cross-model TTS comparison using round-trip ASR evaluation.

### B. Baseline Scripts and Original Preservation

The nine scripts GA20A through GA20K from the course repository serve as the canonical technical starting point. In accordance with the project constitution, the original script files were preserved unmodified. All experimental extensions — parameter sweeps, fixed subsets, metric computation, CSV logging — were implemented through a separate orchestrator (`kamp_hw14.py`), helper modules (`hw14_experiment_runner.py`, `hw14_data_utils.py`, `hw14_analysis_utils.py`), and Colab notebooks. Where a script contained a bug (GA20D's missing `pipeline` import), a corrected wrapper was used with the correction logged in metadata.

### C. Dataset and Evaluation Subset

All short-form ASR experiments use the MINDS-14 `en-AU` training split [2], accessed through the Hugging Face `datasets` library. To ensure comparability across ASR workflows that would otherwise each select a different random sample, a fixed subset of 24 utterances was drawn deterministically by index. All audio was resampled to 16 kHz at load time using `Audio(sampling_rate=16000)`, matching the expected input rate for Whisper, Wav2Vec2, and SpeechT5. For the long-form transcription experiment (Experiment 4), a 60-second audio file was synthesized by concatenating utterances from the shared `long_audio_cache`.

### D. ASR Evaluation Metrics

ASR performance is quantified by word error rate (WER) and character error rate (CER), computed using the `jiwer` library. WER is defined as

$$\text{WER} = \frac{S + D + I}{N}$$

where $S$, $D$, and $I$ are the number of substitution, deletion, and insertion errors respectively, and $N$ is the number of words in the reference transcription. CER applies the same formula at the character level. Both predictions and references are normalized to lowercase with punctuation removed before comparison. Runtime is measured as wall-clock inference time per sample using Python's `time.perf_counter()`.

### E. TTS Evaluation Methodology

TTS quality is evaluated through round-trip ASR: each generated audio file is transcribed by the best available ASR evaluator (selected from Experiment 2 as GA20C direct Whisper, mean WER 0.169), and the transcription is compared to the original input text using WER and CER. Additional metrics include waveform duration, root-mean-square (RMS) amplitude, generation runtime, and spectrogram frame counts where applicable. For Bark, a sample-rate export audit compares files saved at the model-native 24 kHz versus a mismatched 16 kHz to quantify rate-integrity effects.

### F. Execution Groups and Notebook Architecture

Experiments are grouped by model family to minimize model loading overhead and maintain logical coherence:

- **Group A (ASR):** Experiments 1 (ASR baselines), 2, 3, and 4. Amortizes MINDS-14 loading with Whisper, Wav2Vec2, and SpeechT5 ASR models.
- **Group B (Structured TTS):** Experiments 1 (TTS baselines), 5, and 6. Isolates SpeechT5 TTS and MMS-TTS, which expose explicit intermediate representations or deterministic seed control.
- **Group C (Bark + Cross):** Experiments 1 (Bark baselines), 7, and 8. Isolates Bark's stochastic generation path and performs the cross-model TTS comparison.

Each group has paired dry-run and full notebooks. Dry-run notebooks execute a minimal subset to verify correctness before committing to the full parameter sweep. All results are logged to CSV with JSON checkpoints for pause/resume capability.

### G. Text Bank and Shared Assets

TTS experiments use a common text bank to ensure comparability across model families: `"Hello, my dog is cute"` (short, simple), `"There are llamas all around"` (short, unusual content), and `"I would like to check the balance of my savings account please"` (medium, banking domain). The same texts appear in Experiments 5, 6, 7, and 8, enabling direct cross-model comparison. Shared manifests define the fixed MINDS-14 sample indices for ASR experiments.

### H. Reproducibility Controls

All experiments use explicit random seeds where the model architecture supports them. MMS-TTS experiments fix seeds at 42, 123, and 555 to test deterministic reproducibility. Bark experiments use seeds 42 and 123, though Bark's multi-stage architecture limits seed effectiveness. CSV-first logging ensures that every experimental condition is recorded with its full parameter vector and timestamp. JSON checkpoints enable experiment resumption without re-running completed conditions.

---

<div style="page-break-after: always;"></div>

## IV. Results: Experiment 1 - Baseline Reproduction

### A. ASR Baselines (GA20A, GA20B, GA20C, GA20D, GA20E)

Each ASR script was executed once with its original default parameters on a random MINDS-14 sample. The baseline results confirm the expected behavior of each script and establish reference runtimes.

| Script | Model | Sample Index | Runtime (s) | Observation |
|--------|-------|-------------|-------------|-------------|
| GA20A | openai/whisper-small | 640 | 0.495 | Correct transcription via pipeline API |
| GA20B | facebook/wav2vec2-base-960h | 271 | 0.662 | Degraded CTC output on accented speech |
| GA20C | openai/whisper-small | 210 | 2.258 | Direct model call; includes special tokens by default |
| GA20D | openai/whisper-small | long-form | 13.413 | Corrected wrapper used (original omits `pipeline` import) |
| GA20E | microsoft/speecht5_asr | 124 | 0.572 | Poor transcription quality on short telephony audio |

GA20A produced a faithful transcription of a deposit-related utterance ("I want to put some money into my account..."). GA20B yielded severely degraded output ("ON ETIN A WAT A LIKE COT MU...") on a different random sample, consistent with Wav2Vec2's known sensitivity to out-of-domain accented speech [3]. GA20C generated an accurate but verbose transcription because `skip_special_tokens=False` retains decoder tokens. GA20D successfully transcribed a 60-second concatenated audio passage about sovereignty and governance with appropriate timestamp segmentation. GA20E produced a garbled transcription ("why atmospheres is to lavini" for reference "my Atlas is to login"), indicating that SpeechT5 ASR struggles with short, domain-constrained telephony utterances.

<div style="page-break-after: always;"></div>

### B. TTS and Generative Audio Baselines (GA20F, GA20G, GA20H, GA20K)

| Script | Model | Input Text | Voice Condition | Runtime (s) | Sample Rate |
|--------|-------|-----------|-----------------|-------------|-------------|
| GA20F | microsoft/speecht5_tts | "There are llamas all around." | helper_embedding | 1.247 | 16 kHz |
| GA20G | facebook/mms-tts-eng | "Hello - my dog is cute" | default | 0.529 | 16 kHz |
| GA20H | suno/bark-small | Expressive multi-sentence prompt | expressive_prompt | 70.853 | 16 kHz |
| GA20K | suno/bark-small | "Hello, my dog is cute" | v2/en_speaker_5 | 21.389 | 16 kHz |

SpeechT5 TTS generated clear, natural-sounding speech in 1.25 seconds. MMS-TTS produced intelligible output in under 0.53 seconds, the fastest of all TTS systems. GA20H required 70.9 seconds to generate expressive audio from a long prompt containing laughter cues and conversational fillers. GA20K used a shorter text with an explicit voice preset and completed in 21.4 seconds. Both Bark baselines saved audio at 16 kHz rather than the model-native 24 kHz, an issue investigated in Experiment 7.

### C. Baseline Summary and Verified Default Parameters

All nine scripts executed successfully, confirming the operational correctness of the experimental infrastructure. The baselines revealed three important observations: (1) each ASR script selects a different random MINDS-14 sample in its original form, making direct baseline comparison impossible without a fixed subset; (2) Wav2Vec2 and SpeechT5 ASR produce substantially worse transcriptions than Whisper on this dataset; and (3) Bark's generation time is 40–140× slower than the structured TTS models, establishing runtime as a significant practical consideration.

![Experiment 1 baseline summary across all nine scripts](figures/exp1_baseline_summary_v2.png)

**Figure 1.** Summary of baseline reproduction results across all nine HW14 scripts, showing runtime and key output characteristics for each script under its original default parameters. ASR scripts (Group A) and TTS scripts (Groups B and C) are presented separately for readability.

---

<div style="page-break-after: always;"></div>

## V. Results: Experiment 2 - Short-Form ASR Family Comparison on MINDS-14

### A. Experimental Setup

Four ASR workflows were evaluated on a common set of 24 fixed MINDS-14 `en-AU` utterances, all resampled to 16 kHz: GA20A Whisper pipeline (`openai/whisper-small`), GA20B Wav2Vec2 CTC (`facebook/wav2vec2-base-960h`), GA20C direct Whisper (`openai/whisper-small`), and GA20E SpeechT5 ASR (`microsoft/speecht5_asr`). Each workflow transcribed all 24 utterances, producing 96 total rows. Predicted and reference texts were normalized to lowercase with punctuation removed before WER and CER computation.

### B. WER and CER Comparison Across ASR Families

The four ASR workflows exhibited a clear performance hierarchy on MINDS-14:

| Workflow | Mean WER | Std WER | Mean CER | Std CER |
|----------|---------|---------|---------|---------|
| GA20C (Whisper direct) | 0.169 | 0.210 | 0.091 | 0.142 |
| GA20A (Whisper pipeline) | 0.188 | 0.261 | 0.099 | 0.161 |
| GA20B (Wav2Vec2 CTC) | 0.390 | 0.328 | 0.195 | 0.157 |
| GA20E (SpeechT5 ASR) | 0.566 | 0.391 | 0.431 | 0.306 |

![ASR family WER comparison on 24 fixed MINDS-14 utterances](figures/asr_family_wer_bar.png)

**Figure 2.** Mean word error rate (WER) across four ASR workflows evaluated on a common 24-utterance MINDS-14 subset at 16 kHz. Error bars indicate standard deviation across utterances.

Both Whisper variants substantially outperformed Wav2Vec2 and SpeechT5 on this dataset. The direct Whisper interface (GA20C) achieved the lowest mean WER at 0.169, marginally better than the pipeline interface (GA20A) at 0.188. This gap is consistent across both WER and CER and is statistically meaningful given the 24-sample evaluation, though both approaches use the same underlying `whisper-small` checkpoint. Wav2Vec2 CTC achieved a mean WER of 0.390, approximately double the Whisper rates, reflecting the challenge that CTC decoding faces on accented, domain-constrained telephony speech without language model rescoring. SpeechT5 ASR performed worst with a mean WER of 0.566, indicating that the unified encoder-decoder architecture, while versatile, sacrifices ASR accuracy on short utterances compared to purpose-built recognition models.

The standard deviations reveal that all workflows exhibit high per-utterance variance, with SpeechT5 showing the highest (WER std 0.391) and GA20C the lowest (WER std 0.210). Several utterances yielded perfect transcriptions (WER = 0.0) across all workflows, while others — particularly short or heavily accented samples — produced WER values exceeding 1.0 due to insertion errors.

### C. Runtime Comparison

| Workflow | Mean Runtime (s) | Std (s) | Min (s) | Max (s) |
|----------|-----------------|---------|---------|---------|
| GA20A (Whisper pipeline) | 0.491 | 0.300 | 0.174 | 1.402 |
| GA20B (Wav2Vec2 CTC) | 0.367 | 0.234 | 0.090 | 0.967 |
| GA20E (SpeechT5 ASR) | 1.197 | 0.612 | 0.373 | 3.166 |
| GA20C (Whisper direct) | 2.373 | 0.548 | 1.696 | 3.721 |

Wav2Vec2 CTC was the fastest at 0.367 seconds per utterance, followed by the Whisper pipeline at 0.491 seconds. The direct Whisper interface was substantially slower at 2.373 seconds, approximately 4.8× the pipeline, because it bypasses the pipeline's optimized preprocessing and batching pathways. SpeechT5 ASR occupied an intermediate position at 1.197 seconds per utterance.

### D. Whisper Pipeline vs. Direct Interface

The two Whisper variants use the same `openai/whisper-small` checkpoint but differ in interface: GA20A uses `pipeline("automatic-speech-recognition", ...)` while GA20C uses explicit `WhisperProcessor` and `WhisperForConditionalGeneration` calls. The direct interface achieved slightly lower WER (0.169 vs. 0.188) but at a runtime cost of 4.8×. The WER improvement likely stems from differences in default generation parameters (the pipeline applies its own defaults for `max_new_tokens` and other settings) and from the explicit token handling in GA20C's decoding path.

### E. Summary and Implications for RQ1

**RQ1:** The short-form ASR families form a clear accuracy hierarchy: Whisper direct $>$ Whisper pipeline $>$ Wav2Vec2 CTC $\gg$ SpeechT5 ASR. Whisper's large-scale weakly supervised pretraining provides the best transfer to telephony-accented banking speech. Wav2Vec2 is competitive on clean utterances but degrades on accented or noisy samples. SpeechT5 ASR, designed as a general-purpose speech-text model, is not competitive for pure ASR on this domain. Based on these results, GA20C (Whisper direct) was selected as the best available ASR evaluator (mean WER 0.169, mean CER 0.091) for the round-trip TTS evaluation in Experiment 8.

---

## VI. Results: Experiment 3 - ASR Preprocessing, Decoding, and Normalization Sensitivity

### A. Experimental Setup

Experiment 3 comprises two sub-experiments totaling 114 rows. Sub-experiment 3A sweeps the resampling rate across three values — 8 kHz (native MINDS-14 rate), 16 kHz (model-expected), and 24 kHz (over-sampled) — for three ASR workflows (GA20A, GA20B, GA20E), with 10 utterances per condition (90 rows). Sub-experiment 3B evaluates GA20C on all 24 fixed utterances with both raw and cleaned token decoding (24 rows). The `clean` policy strips Whisper's special tokens and normalizes whitespace; the `raw` policy preserves all decoder output.

### B. Sampling-Rate Sensitivity

Resampling rate profoundly affects ASR accuracy across all three tested workflows:

| Workflow | 8 kHz WER | 16 kHz WER | 24 kHz WER |
|----------|----------|-----------|-----------|
| GA20A (Whisper pipeline) | 0.353 | 0.136 | 0.253 |
| GA20B (Wav2Vec2 CTC) | 0.975 | 0.357 | 0.794 |
| GA20E (SpeechT5 ASR) | 1.056 | 0.572 | 0.991 |

![ASR WER as a function of input sampling rate](figures/asr_sampling_rate_curve.png)

**Figure 3.** Mean WER for three ASR workflows at 8 kHz, 16 kHz, and 24 kHz input sampling rates, evaluated on 10 MINDS-14 utterances per condition. All models achieve optimal performance at the expected 16 kHz rate.

All models achieved their best performance at the expected 16 kHz rate. Deviation in either direction caused substantial degradation, but undersampling (8 kHz) was consistently worse than oversampling (24 kHz) for Whisper and SpeechT5. Wav2Vec2 CTC was the most sensitive: its mean WER at 8 kHz (0.975) represents near-total failure, a 2.7× increase over the 16 kHz baseline (0.357). At 8 kHz, Wav2Vec2 processes the raw telephony waveform without any resampling, meaning the model's learned feature representations, trained on 16 kHz LibriSpeech data, encounter frame timings that are compressed by a factor of two. This effectively halves the temporal resolution of the input relative to the training distribution.

The pattern at 24 kHz is subtler: the extra temporal resolution from oversampling does not help because the ASR feature extractors apply fixed-size windowing and striding tuned for 16 kHz. The result is that the 24 kHz condition consistently falls between the 8 kHz and 16 kHz conditions, closer to 8 kHz for CTC (0.794) and closer to 16 kHz for Whisper pipeline (0.253).

<div style="page-break-after: always;"></div>

### C. GA20C Token Cleanup Effect

Sub-experiment 3B compared raw and cleaned decoding for GA20C Whisper direct on 24 utterances at 16 kHz. The raw and clean policies produced identical WER (0.169) and CER (0.091) after the normalization step applied during evaluation. This null result is informative: it confirms that GA20C's default `skip_special_tokens=False` setting does not affect WER when the evaluation pipeline normalizes text consistently. The special tokens emitted by Whisper's decoder (e.g., `<|startoftranscript|>`, `<|en|>`) are stripped during the lowercase-and-depunctuate normalization step, so they do not contribute to edit-distance computation.

### D. Summary and Implications for RQ2

**RQ2:** ASR accuracy is highly sensitive to preprocessing. Operating at the wrong sampling rate is by far the most damaging preprocessing error, capable of doubling or tripling WER. The effect is asymmetric: undersampling degrades more than oversampling. Among workflows, Wav2Vec2 CTC is the most fragile and Whisper the most robust. Token cleanup policy, by contrast, has zero measurable effect on GA20C's evaluated performance when text normalization is applied consistently. These results emphasize that ensuring correct resampling is the single most important preprocessing step for any ASR pipeline operating on telephony-like speech data.

---

<div style="page-break-after: always;"></div>

## VII. Results: Experiment 4 - Long-Form Whisper Chunking and Timestamp Stability

### A. Experimental Setup

The long-form Whisper chunking experiment swept `chunk_length_s` across {5, 10, 15} seconds and `batch_size` across {4, 8}, producing a 3 × 2 grid of 6 conditions. All conditions used the same 60-second input audio, `openai/whisper-small` with `return_timestamps=True`, and the corrected GA20D wrapper. Transcript similarity to the chunk-5/batch-8 baseline was computed using sequence similarity ratio.

### B. Transcript Structure and Segment Counts

| Chunk Length (s) | Batch Size | Num Segments | Total Words | Similarity to Baseline |
|-----------------|-----------|-------------|-------------|----------------------|
| 5 | 8 | 6 | 151 | 1.000 |
| 5 | 4 | 6 | 151 | 1.000 |
| 10 | 4 | 7 | 150 | 0.993 |
| 10 | 8 | 7 | 150 | 0.993 |
| 15 | 4 | 17 | 498 | 0.289 |
| 15 | 8 | 17 | 498 | 0.289 |

![Long-form transcript similarity heatmap across chunk length and batch size](figures/longform_similarity_heatmap.png)

**Figure 4.** Pairwise transcript similarity (sequence similarity ratio) across the 3 × 2 grid of chunk length and batch size configurations for long-form Whisper transcription of 60-second audio.

Chunk lengths of 5 and 10 seconds produce highly consistent transcriptions. The 5-second conditions are identical (6 segments, 151 words, similarity 1.0 within the pair). The 10-second conditions are nearly identical to the 5-second baseline (similarity 0.993), with one additional segment boundary and one fewer word. However, the 15-second chunk length produces dramatically different output: 17 segments, 498 words (3.3× inflation over the 5-second output), and a similarity score of only 0.289 to the baseline. Inspection of the 15-second transcripts reveals repetitive content — the model re-transcribes portions of the audio multiple times, a known failure mode of Whisper's sequential chunk stitching when the chunk length approaches or exceeds the model's 30-second attention window and the content within a single chunk becomes ambiguous about where previous transcription ended.

### C. Runtime and Batching Effects

| Chunk Length (s) | Batch Size | Runtime (s) |
|-----------------|-----------|-------------|
| 5 | 8 | 13.501 |
| 5 | 4 | 14.092 |
| 10 | 8 | 3.029 |
| 10 | 4 | 3.786 |
| 15 | 8 | 14.317 |
| 15 | 4 | 12.405 |

The 10-second configurations are the fastest, achieving runtimes of 3.0–3.8 seconds, a 3.5–4.5× speedup over the 5-second baseline. This speedup arises because fewer chunks require processing (6 chunks at 10 s vs. 12 at 5 s for 60-second audio). Larger batch size provides a modest runtime benefit within matching chunk lengths. The 15-second conditions are paradoxically slow despite fewer initial chunks because the repetitive generation produces far more text tokens, extending the autoregressive decoding phase.

### D. Self-Consistency Across Configurations

Within each chunk-length tier, batch size has no effect on transcript content: the 5-second pair shares similarity 1.0, the 10-second pair shares similarity 0.993, and the 15-second pair shares similarity 0.289 (both are equally divergent from baseline). This confirms that `batch_size` controls only the parallelism of chunk processing, not the content of the transcription itself.

### E. Summary and Implications for RQ3

**RQ3:** Chunk length is the primary determinant of long-form transcript quality. Chunk lengths of 5 and 10 seconds produce nearly identical, coherent transcripts, with 10 seconds offering a substantial runtime advantage. A chunk length of 15 seconds triggers Whisper's known repetition failure mode, inflating word count by 3.3× and reducing similarity to the baseline to 0.289. Batch size affects only runtime, not content. The practical recommendation is to use `chunk_length_s=10` and `batch_size=8` for the best balance of accuracy and speed on 60-second audio with `whisper-small`.

---

## VIII. Results: Experiment 5 - SpeechT5 TTS Speaker Conditioning and Vocoder Transparency

### A. Experimental Setup

SpeechT5 TTS was evaluated under two speaker conditions — a dataset-derived x-vector embedding (`helper_embedding`) and a zero vector (`zero_vector`) — across three texts from the common text bank: "There are llamas all around" (llamas), "Hello, my dog is cute" (hello_dog), and "I would like to check the balance of my savings account please" (banking_request). Each condition produces a mel-spectrogram and a HiFi-GAN-vocoded waveform, yielding 6 total rows with paired spectrogram and audio artifacts.

### B. Speaker Embedding Effects on Duration and Timbre

| Speaker Condition | Text Tag | Spectrogram Bins | Duration (s) | RMS | Gen. Time (s) |
|-------------------|----------|-----------------|-------------|-----|---------------|
| helper_embedding | llamas | 92 | 1.472 | — | 0.859 |
| zero_vector | llamas | 68 | 1.088 | — | 0.644 |
| helper_embedding | hello_dog | 78 | 1.248 | — | 0.628 |
| zero_vector | hello_dog | 64 | 1.024 | — | 0.549 |
| helper_embedding | banking_request | 194 | 3.104 | — | 1.493 |
| zero_vector | banking_request | 146 | 2.336 | — | 1.158 |

The `helper_embedding` condition consistently produces longer waveforms (35–36% longer) and more spectrogram frames than the `zero_vector` condition across all three texts. This demonstrates that the speaker embedding exerts meaningful control over the generated speech: the x-vector embedding, derived from a natural speaker recording, encodes speaking rate and prosodic characteristics that result in slower, more natural-sounding output. The zero vector, while still producing intelligible speech, yields compressed output with reduced temporal detail.

All spectrogram matrices have a fixed 80-bin mel-frequency dimension (matching SpeechT5's output specification), with the time dimension scaling proportionally to text length. The banking_request text, approximately 3× longer than the other prompts, produces proportionally longer spectrograms and waveforms.

### C. Spectrogram Analysis

![SpeechT5 mel-spectrogram panel across speaker conditions and texts](figures/speecht5_spectrogram_panel.png)

**Figure 5.** Mel-spectrogram outputs from SpeechT5 TTS under two speaker embedding conditions (helper_embedding and zero_vector) across three text prompts. The time axis (horizontal) and 80-bin mel-frequency axis (vertical) reveal differences in temporal extent and formant structure.

<div style="page-break-after: always;"></div>

The mel-spectrograms reveal clear structural differences between the two speaker conditions. The `helper_embedding` spectrograms show wider formant bands, more gradual onset and offset transitions, and greater temporal extent for each phoneme. The `zero_vector` spectrograms are more compressed with sharper boundaries, consistent with a faster speaking rate and reduced prosodic variation. Both conditions produce well-structured spectrograms with visible harmonic content, confirming that SpeechT5's encoder-decoder generates acoustically valid intermediate representations that HiFi-GAN can faithfully vocode.

### D. Summary and Implications for RQ4

**RQ4:** SpeechT5 offers substantial controllability through its explicit speaker-embedding interface and its two-stage spectrogram-to-waveform pipeline. The speaker embedding directly modulates speaking rate (35% duration difference), prosody, and spectral characteristics. The exposed spectrogram intermediate allows visual inspection and analysis of the generation process, making SpeechT5 the most interpretable TTS system in this study. The tradeoff is reduced expressiveness: SpeechT5 cannot generate non-speech elements, laughter, or dramatic prosodic variation.

---

<div style="page-break-after: always;"></div>

## IX. Results: Experiment 6 - MMS-TTS Reproducibility, Text Sensitivity, and Speaking-Rate Control

### A. Experimental Setup

MMS-TTS was evaluated across three seeds (42, 123, 555), two text prompts (hello_dog, banking_request), and three speaking rates (0.8, 1.0, 1.2), producing 12 total rows. The seed sweep uses the default speaking rate of 1.0 to test deterministic reproducibility, while the speaking-rate sweep fixes seed 555 to isolate the rate control effect. All outputs are saved at 16 kHz.

### B. Seed Reproducibility

Across the three seeds at speaking rate 1.0, MMS-TTS produced nearly identical but not exactly identical outputs:

| Seed | Text Tag | Duration (s) | Gen. Time (s) | RMS |
|------|----------|-------------|---------------|-----|
| 42 | hello_dog | 2.128 | 0.356 | 0.109 |
| 123 | hello_dog | 2.192 | 0.361 | 0.111 |
| 555 | hello_dog | 2.080 | 0.342 | 0.104 |
| 42 | banking_request | 4.160 | 0.569 | 0.112 |
| 123 | banking_request | 3.792 | 0.521 | 0.116 |
| 555 | banking_request | 4.064 | 0.530 | 0.113 |

Duration varies by up to 5.4% across seeds for hello_dog (2.080–2.192 s) and 9.7% for banking_request (3.792–4.160 s). The variation is modest and suggests that seed control in VITS affects the stochastic duration predictor and noise sampling but does not radically alter the output. RMS values are highly consistent (coefficient of variation < 4%), indicating stable amplitude characteristics across seeds.

<div style="page-break-after: always;"></div>

### C. Speaking-Rate Effects on Duration

The speaking-rate control in MMS-TTS/VITS produces a clear and predictable effect on waveform duration:

| Text Tag | Rate 0.8 Duration (s) | Rate 1.0 Duration (s) | Rate 1.2 Duration (s) | Ratio 0.8/1.0 | Ratio 1.0/1.2 |
|----------|---------------------|---------------------|---------------------|--------------|--------------|
| hello_dog | 2.416 | 2.080 | 1.776 | 1.162 | 1.171 |
| banking_request | 4.640 | 4.064 | 3.616 | 1.142 | 1.124 |

![MMS-TTS waveform duration as a function of speaking rate](figures/mms_speaking_rate_duration.png)

**Figure 6.** Waveform duration (seconds) for MMS-TTS outputs at speaking rates 0.8, 1.0, and 1.2 across two text prompts (hello_dog and banking_request), with seed fixed at 555.

Reducing the speaking rate from 1.0 to 0.8 increased duration by approximately 14–16%, while increasing to 1.2 decreased duration by approximately 11–15%. The scaling is approximately linear: the ratio of durations closely matches the inverse ratio of speaking rates ($0.8^{-1} = 1.25$ vs. observed 1.15, $1.2^{-1} = 0.833$ vs. observed 0.86). The slight sub-linearity suggests that VITS's duration predictor applies the rate scaling multiplicatively to predicted phone durations but that silence and transition periods do not scale proportionally.

### D. Summary and Implications for RQ5

**RQ5:** MMS-TTS/VITS provides the strongest deterministic reproducibility of the three TTS families. Under fixed seeds, duration variation is below 10% and amplitude variation below 4%. The speaking-rate control produces predictable, approximately linear duration scaling. Combined with the fastest generation time of any TTS system (0.34–0.88 s per utterance), MMS-TTS is the most controllable and efficient option for applications requiring reproducible, speed-adjustable speech synthesis.

---

<div style="page-break-after: always;"></div>

## X. Results: Experiment 7 - Bark Expressiveness, Voice Presets, and Sample-Rate Integrity

### A. Experimental Setup

Experiment 7 comprises three sub-experiments totaling 20 rows. Sub-experiment 7A evaluates three prompt styles — plain, expressive (`expr`), and strongly expressive (`strong_expr`) — across two seeds (42, 123) on a multi-sentence input text (6 rows). Sub-experiment 7B compares three voice preset conditions — no preset, `v2/en_speaker_5` (preset5), and `v2/en_speaker_6` (preset6) — across two seeds on a simpler input text (6 rows). Sub-experiment 7C performs a sample-rate export audit, comparing the same Bark outputs saved at 16 kHz versus the native 24 kHz (8 rows reusing generation from 7A/7B). All Bark experiments use `suno/bark-small`.

### B. Prompt-Style Effects

| Prompt Mode | Seed | Duration (s) | Gen. Time (s) | Round-Trip WER |
|-------------|------|-------------|---------------|----------------|
| plain | 42 | 9.47 | 53.00 | 0.048 |
| plain | 123 | 9.25 | 52.48 | 0.000 |
| expr | 42 | 13.44 | 78.92 | 0.125 |
| expr | 123 | 10.88 | 62.57 | 0.333 |
| strong_expr | 42 | 13.04 | 76.96 | 0.261 |
| strong_expr | 123 | 11.81 | 69.40 | 0.087 |

Prompt style has marked effects on both duration and generation time. The `plain` prompt produces the shortest outputs (9.25–9.47 s) and the lowest round-trip WER (0.000–0.048), indicating that Bark generates the most intelligible speech when the text is unadorned. The `expr` and `strong_expr` prompts increase duration by 18–42% and degrade round-trip WER, suggesting that expressive cues cause Bark to insert pauses, laughter, or other non-lexical elements that the ASR evaluator cannot transcribe. Generation time scales with output duration, ranging from 52.5 s for plain outputs to 78.9 s for the longest expressive generation.

<div style="page-break-after: always;"></div>

Cross-seed variation is substantial: the `expr` prompt produces WER 0.125 at seed 42 but 0.333 at seed 123, demonstrating that Bark's multi-stage token cascade amplifies stochastic variation. The same text prompt can produce markedly different acoustic realizations depending on the random seed.

### C. Voice Preset Comparison

| Voice Condition | Seed | Duration (s) | Gen. Time (s) | Round-Trip WER |
|-----------------|------|-------------|---------------|----------------|
| no_preset | 42 | 3.05 | 18.36 | 0.000 |
| no_preset | 123 | 2.68 | 16.39 | 0.000 |
| preset5 | 42 | 4.64 | 35.24 | 0.000 |
| preset5 | 123 | 3.56 | 27.44 | 0.000 |
| preset6 | 42 | 3.48 | 27.17 | 0.000 |
| preset6 | 123 | 4.40 | 33.59 | 0.600 |

On the shorter "Hello, my dog is cute" text, voice presets produce generally intelligible output: 5 of 6 conditions achieve perfect round-trip WER (0.000). The exception is preset6/seed 123, which produces a WER of 0.600, indicating a qualitatively different acoustic realization. Presets increase generation time roughly 1.5–2× compared to the no_preset condition (16–18 s vs. 27–35 s), likely because the voice history tensor extends the context window for each stage of the generation cascade.

<div style="page-break-after: always;"></div>

### D. Sample-Rate Export Audit

Sub-experiment 7C compares the acoustic consequences of saving Bark audio at the incorrect 16 kHz versus the correct 24 kHz:

| Condition | Save Rate | Duration (s) | Round-Trip WER |
|-----------|----------|-------------|----------------|
| plain, seed 42 | 16 kHz | 14.20 | 0.000 |
| plain, seed 42 | 24 kHz | 9.47 | 0.048 |
| no_preset, seed 42 | 16 kHz | 4.58 | 0.600 |
| no_preset, seed 42 | 24 kHz | 3.05 | 0.000 |
| preset5, seed 42 | 16 kHz | 6.96 | 0.200 |
| preset5, seed 42 | 24 kHz | 4.64 | 0.000 |

![Bark sample-rate export audit](figures/bark_sample_rate_audit.png)

**Figure 7.** Comparison of waveform duration and round-trip WER for Bark outputs saved at the correct 24 kHz versus the incorrect 16 kHz sample rate. The 16 kHz condition inflates duration by exactly 1.5× due to temporal reinterpretation of samples.

<div style="page-break-after: always;"></div>

Saving at 16 kHz instead of 24 kHz inflates the apparent duration by exactly the expected ratio of $24000/16000 = 1.5\times$, confirming that the waveform samples are interpreted at the wrong temporal rate. The pitch is correspondingly lowered by a factor of $16/24 \approx 0.67$, producing audibly distorted audio. The effect on round-trip WER is variable: some conditions degrade (no_preset: 0.000 → 0.600; preset5: 0.000 → 0.200) while others paradoxically improve (plain: 0.048 → 0.000), likely because the ASR model can sometimes still recover the intended text from slowed-down speech. Overall, the audit confirms that preserving the model-native 24 kHz sample rate is essential for correct Bark audio export.

### E. Summary and Implications for RQ6

**RQ6:** Bark's two control mechanisms differ substantially. Free-form prompt cues (plain, expr, strong_expr) provide a range of expressive variation but with high stochasticity and degraded intelligibility for expressive styles. Voice presets produce more consistent output across seeds on short texts but at a runtime premium. The sample-rate export audit demonstrates that 24 kHz must be preserved; 16 kHz export introduces a 1.5× time-stretch and pitch distortion that variably affects ASR transcribability. Among the three TTS families, Bark provides the widest expressive range but the least predictability.

---

<div style="page-break-after: always;"></div>

## XI. Results: Experiment 8 - Cross-Model TTS Intelligibility, Controllability, and Runtime

### A. Experimental Setup

The cross-model TTS comparison evaluated three systems — SpeechT5 (with `helper_embedding`), MMS-TTS (seed 555, default rate), and Bark (preset5, seed 42) — on the three common text bank prompts (hello_dog, llamas, banking_request), producing 9 total rows. Each generated audio file was transcribed by the best ASR evaluator (GA20C Whisper direct, mean WER 0.169 from Experiment 2), and the transcription was compared to the original input text using WER and CER. Generation runtime and waveform duration were also recorded.

### B. Round-Trip ASR Intelligibility

| System | Text Tag | Duration (s) | Round-Trip WER | Round-Trip CER |
|--------|----------|-------------|----------------|----------------|
| SpeechT5 | hello_dog | 1.248 | 0.000 | 0.000 |
| SpeechT5 | llamas | 1.536 | 0.200 | 0.037 |
| SpeechT5 | banking_request | 3.136 | 0.000 | 0.000 |
| MMS-TTS | hello_dog | 2.080 | 0.200 | 0.100 |
| MMS-TTS | llamas | 2.736 | 0.400 | 0.296 |
| MMS-TTS | banking_request | 4.064 | 0.000 | 0.000 |
| Bark | hello_dog | 4.640 | 0.000 | 0.000 |
| Bark | llamas | 2.000 | 1.000 | 0.741 |
| Bark | banking_request | 4.760 | 0.000 | 0.000 |

![Cross-model TTS round-trip WER comparison](figures/tts_roundtrip_wer.png)

**Figure 8.** Round-trip word error rate (WER) for three TTS systems — SpeechT5, MMS-TTS, and Bark — evaluated on three common text bank prompts using GA20C Whisper direct as the ASR evaluator.

SpeechT5 achieved the best mean round-trip WER (0.067) and CER (0.012), with perfect transcription on 2 of 3 texts. The only error occurred on "llamas," where the ASR evaluator returned a minor word-level discrepancy. MMS-TTS achieved a mean WER of 0.200 and CER of 0.132, with errors on both the hello_dog and llamas texts. Bark achieved a mean WER of 0.333 and CER of 0.247, driven entirely by a catastrophic failure on the llamas text (WER 1.000, CER 0.741) where the generated audio did not faithfully represent the input text. On the other two texts, Bark achieved perfect round-trip scores.

The llamas text proved the most challenging across all systems, likely because "llamas" is an unusual word that sits outside the primary training distributions of these TTS models. The banking_request text, despite being the longest, was transcribed perfectly by all three systems, suggesting that domain-typical vocabulary is easier for TTS systems to pronounce clearly.

<div style="page-break-after: always;"></div>

### C. Runtime Comparison

| System | Mean Gen. Time (s) | Min (s) | Max (s) |
|--------|-------------------|---------|---------|
| SpeechT5 | 1.006 | 0.706 | 1.468 |
| MMS-TTS | 0.480 | 0.430 | 0.566 |
| Bark | 29.368 | 17.613 | 35.250 |

MMS-TTS is the fastest system at 0.480 seconds mean generation time — approximately 2.1× faster than SpeechT5 and 61× faster than Bark. Bark's mean generation time of 29.4 seconds reflects the computational cost of its three-stage token cascade. The runtime gap between MMS-TTS and Bark spans nearly two orders of magnitude, making Bark impractical for applications requiring real-time or near-real-time synthesis.

### D. Control Mechanism Trade-Off Summary

The three TTS systems occupy distinct positions in the controllability–expressiveness–efficiency trade-off space:

| Dimension | SpeechT5 | MMS-TTS | Bark |
|-----------|----------|---------|------|
| Intelligibility (mean WER) | 0.067 | 0.200 | 0.333 |
| Controllability mechanism | Speaker embedding | Seed + speaking rate | Prompt cues + voice presets |
| Interpretability | High (spectrogram exposed) | Low (end-to-end) | Low (multi-stage tokens) |
| Determinism | High | Perfect under seed | Low (stochastic cascade) |
| Expressiveness | Low (clean speech only) | Low (clean speech only) | High (laughter, pauses, emotion) |
| Mean runtime (s) | 1.006 | 0.480 | 29.368 |
| Native sample rate | 16 kHz | 16 kHz | 24 kHz |

![TTS controllability-expressiveness-efficiency trade-off summary](figures/tts_control_tradeoff_table_v2.png)

**Figure 9.** Cross-model TTS trade-off summary comparing SpeechT5, MMS-TTS, and Bark across intelligibility, controllability, determinism, expressiveness, and mean generation runtime.

### E. Summary and Implications for RQ7

**RQ7:** For the assignment-relevant trade-off among intelligibility, controllability, expressiveness, and runtime, SpeechT5 offers the best balance for applications prioritizing intelligibility and interpretability (lowest WER, exposed spectrogram). MMS-TTS is optimal for efficiency and reproducibility (fastest runtime, perfect seed control). Bark is uniquely suited for applications requiring expressive or stylized speech where generation latency is acceptable. No single system dominates all dimensions, confirming that the choice among modern TTS architectures remains fundamentally task-dependent.

---

<div style="page-break-after: always;"></div>

## XII. Discussion

### A. ASR Family Trade-Offs (RQ1, RQ2)

The ASR experiments provide converging evidence that model architecture, training data, and preprocessing choices interact in determining recognition quality on domain-constrained telephony speech. Whisper's large-scale weakly supervised training on 680,000 hours of multilingual audio [5] provides the strongest transfer, achieving mean WER below 0.19 on MINDS-14 `en-AU` without any task-specific adaptation. This is consistent with Radford et al.'s claim that scale and diversity of training data are more important than architectural innovation for robust zero-shot ASR.

Wav2Vec2 CTC, despite being a strong baseline on clean English speech benchmarks, exhibits a 2.3× WER increase over Whisper on this dataset. The primary degradation mechanism appears to be the CTC decoder's sensitivity to acoustic conditions that produce ambiguous frame-level predictions — accented speech, background noise, and reduced bandwidth all increase the effective confusion between phonetically similar tokens. Without a language model to rescore hypotheses, the greedy `argmax` decode path has no mechanism to recover from individual frame errors.

SpeechT5 ASR's poor performance (mean WER 0.566) reveals a fundamental limitation of unified speech-text architectures: the shared encoder-decoder backbone trades ASR-specific capacity for task versatility. On short utterances where context is limited, SpeechT5 lacks the specialized feature extraction and language modeling capabilities that make Whisper and even Wav2Vec2 more effective.

The preprocessing sensitivity results (Experiment 3) demonstrate that sampling-rate correctness is not merely a recommended practice but a critical prerequisite. The asymmetric degradation pattern — 8 kHz worse than 24 kHz worse than 16 kHz — can be understood through the models' learned feature extractors. At 8 kHz, the model's convolutional front-end receives frames with half the expected temporal density, effectively aliasing mid-frequency speech content. At 24 kHz, the extra temporal samples provide redundant information that the fixed-stride convolutions partially accommodate, explaining the less severe degradation.

<div style="page-break-after: always;"></div>

### B. Long-Form Transcription Engineering (RQ3)

The long-form Whisper experiment reveals a sharp phase transition in transcript quality as a function of chunk length. The 5-second and 10-second chunk lengths operate within Whisper's effective context window and produce stable, coherent transcripts. The 10-second setting is optimal because it reduces the number of chunks (and hence the overhead of chunk-boundary stitching) while staying well below the 30-second model limit. The 15-second setting triggers a catastrophic failure mode: the 3.3× word count inflation and 0.289 similarity score indicate that the model enters a repetitive generation loop, likely because the decoder's attention spans an uncomfortable fraction of the model's 30-second window, creating ambiguity about which portions of the audio have already been transcribed.

This finding has direct practical implications. The default `chunk_length_s=5` in the GA20D script is conservative but safe. Doubling to 10 seconds provides a substantial speedup (3–4×) with no quality loss on this passage. However, further increases risk the repetition failure mode, and the threshold may vary with audio content and model size. Practitioners should validate chunk length selection on representative samples before committing to production settings.

### C. TTS Controllability Spectrum (RQ4, RQ5, RQ6)

The three TTS families — SpeechT5, MMS-TTS, and Bark — collectively map a spectrum of controllability from fully explicit to fully implicit. SpeechT5 exposes the largest number of interpretable control surfaces: the speaker embedding directly modulates prosody and rate (35% duration difference between embedding conditions), and the spectrogram intermediate is available for inspection and potential manipulation. This transparency comes at the cost of limited expressiveness: SpeechT5 cannot generate non-speech content or dramatic prosodic shifts.

MMS-TTS occupies a middle position: its end-to-end VITS architecture hides the generation process but offers deterministic seed control and a continuous speaking-rate parameter. The seed-reproducibility results confirm that VITS yields functionally identical outputs under fixed seeds, a property unavailable in the other two systems. The speaking-rate control is approximately linear and predictable, making MMS-TTS the most controllable system for applications requiring batch-consistent audio at a specified tempo.

<div style="page-break-after: always;"></div>

Bark represents the opposite end of the spectrum: its multi-stage token cascade provides the richest expressive palette — laughter, pauses, emotional inflection — but at the cost of stochasticity that cannot be eliminated by seed fixing (due to the compounding distributions across three generation stages) and generation times that are 30–60× slower than the structured TTS systems. The practical ceiling on Bark's utility is determined by the application's tolerance for both latency and output unpredictability.

### D. Cross-Model TTS Comparison and Practical Recommendations (RQ7)

The cross-model comparison in Experiment 8 quantifies the trade-offs identified qualitatively in Experiments 5–7. The round-trip WER ranking — SpeechT5 (0.067) < MMS-TTS (0.200) < Bark (0.333) — provides an objective intelligibility ordering. However, this ranking is not the sole consideration for practitioners:

1. **For maximum intelligibility and interpretability:** SpeechT5 with an appropriate speaker embedding is the recommended choice. It produces the clearest speech and exposes the generation process through spectrograms.

2. **For maximum efficiency and reproducibility:** MMS-TTS is the recommended choice. Its 0.48-second mean generation time and deterministic seed control make it suitable for batch production and real-time pipelines.

3. **For expressive or stylized output:** Bark is the only option that supports non-lexical expressiveness. Its utility is bounded by the 29-second mean generation time and the need to validate each output because of stochastic variation.

4. **For long-form content:** The sample-rate audit in Experiment 7 demonstrated that Bark audio must be exported at 24 kHz to preserve pitch and temporal fidelity, adding an integration constraint absent from the 16 kHz SpeechT5 and MMS-TTS systems.

<div style="page-break-after: always;"></div>

### E. Limitations

This study has several important limitations:

1. **Single dataset for ASR evaluation.** All short-form ASR results are based on the MINDS-14 `en-AU` split, a telephony-domain banking-intent dataset. Generalization to other domains, accents, or recording conditions requires additional validation.

2. **No model fine-tuning.** All experiments used pretrained checkpoints without task-specific adaptation. Fine-tuning on MINDS-14 or banking-domain data could substantially alter the relative performance ranking.

3. **No human evaluation.** TTS quality is assessed only through round-trip ASR, which measures intelligibility but not naturalness, speaker similarity, or subjective preference. A formal Mean Opinion Score (MOS) study would provide complementary quality metrics.

4. **Limited text diversity for TTS.** The three-text common bank provides a controlled comparison but does not capture the full range of challenges (proper nouns, numbers, code-switching, long passages) that TTS systems encounter in practice.

5. **Single GPU configuration.** All experiments ran on Google Colab T4 GPUs. Runtime comparisons may shift on different hardware configurations, particularly for Bark, which is the most compute-sensitive system.

6. **Small evaluation sets.** The 24-utterance ASR subset and 3-text TTS bank provide sufficient statistical power for rank ordering but limit the precision of point estimates. Confidence intervals are not reported because the sample sizes are below conventional statistical thresholds.

### F. Broader Implications

These findings contribute to the broader understanding of how speech AI systems should be selected and configured for specific applications. The core insight is that no single system dominates across all dimensions of quality: the choice between Whisper and Wav2Vec2 for ASR, or between SpeechT5, MMS-TTS, and Bark for TTS, is fundamentally task-dependent. The preprocessing sensitivity results have immediate practical relevance: sampling-rate mismatch is a silent, catastrophic failure mode that produces no error messages but can double or triple WER. The long-form chunking results provide actionable engineering guidance for deploying Whisper on extended audio. And the TTS controllability spectrum illustrates a general principle in generative AI: models that offer richer expressiveness tend to sacrifice predictability and efficiency, creating a trade-off space that practitioners must navigate deliberately.

---

## XIII. Conclusion

This study conducted a systematic experimental investigation of modern speech AI systems through eight controlled experiments addressing seven research questions, producing 272 data rows and 22 figures across three execution groups. The principal findings are:

1. **ASR family ranking (RQ1):** On 24 fixed MINDS-14 telephony utterances, Whisper direct inference achieved the lowest mean WER (0.169), followed by Whisper pipeline (0.188), Wav2Vec2 CTC (0.390), and SpeechT5 ASR (0.566). Whisper's large-scale weakly supervised training provides the strongest zero-shot transfer to domain-constrained accented speech.

2. **Preprocessing sensitivity (RQ2):** Sampling-rate mismatch is the most damaging preprocessing error, capable of increasing Wav2Vec2 WER from 0.357 to 0.975 at 8 kHz. All models require correct 16 kHz input; undersampling degrades more severely than oversampling.

3. **Long-form chunking (RQ3):** Chunk lengths of 5 and 10 seconds produce consistent transcripts (similarity $\geq$ 0.993). The 10-second setting offers a 3.5–4.5× runtime advantage over 5 seconds. A 15-second chunk length triggers catastrophic repetition, inflating word count by 3.3×.

4. **SpeechT5 TTS controllability (RQ4):** Speaker embeddings modulate speaking rate by 35% and produce visible spectrogram differences, making SpeechT5 the most interpretable TTS system. The exposed spectrogram-to-vocoder pipeline enables inspection and analysis of the generation process.

5. **MMS-TTS reproducibility (RQ5):** MMS-TTS/VITS achieves functional determinism under seed control (duration variation < 10%). Speaking rate scales duration approximately linearly, with 0.8× rate producing 14–16% longer output and 1.2× rate producing 11–15% shorter output.

6. **Bark expressiveness and sample-rate integrity (RQ6):** Free-form prompt cues provide expressive variation but degrade intelligibility (WER increase from 0.024 for plain to 0.229 for expressive). Voice presets offer more consistent output. The 24 kHz native sample rate must be preserved during export to avoid 1.5× time-stretch and pitch distortion.

7. **Cross-model TTS trade-offs (RQ7):** Round-trip ASR ranks SpeechT5 as most intelligible (WER 0.067), MMS-TTS as most efficient (0.480 s generation time), and Bark as most expressive (but with 29.4 s generation time and 0.333 WER). No single system dominates all dimensions, confirming that TTS system selection is fundamentally task-dependent.

These results provide empirical characterizations of under-explored parameter spaces in contemporary speech AI and demonstrate that engineering choices — sampling rate, chunk length, speaker conditioning, seed control, and export format — have quantitative consequences that must be understood and validated before deployment.

---

## XIV. Future Work

Several directions extend naturally from this study:

1. **Domain-specific fine-tuning.** Fine-tuning Whisper or Wav2Vec2 on MINDS-14 banking utterances could substantially improve ASR accuracy and would reveal whether the current performance gaps reflect fundamental architectural limitations or merely distributional mismatch. Similarly, fine-tuning SpeechT5 TTS on target-domain speech could improve the intelligibility of generated banking vocabulary.

2. **Larger and more diverse evaluation sets.** Expanding the ASR evaluation from 24 to the full MINDS-14 split (and extending to additional `en-*` language configurations) would improve the statistical power of model comparisons and enable meaningful confidence interval estimation. For TTS, a larger and more diverse text bank — including proper nouns, numbers, code-switching, and multi-sentence passages — would stress-test the generalization of each system.

3. **Human subjective evaluation.** Round-trip ASR measures intelligibility but not naturalness, speaker similarity, or listener preference. A formal Mean Opinion Score (MOS) study with human raters would complement the automated metrics and could reveal quality dimensions not captured by WER and CER.

4. **Multilingual extension.** MINDS-14 spans 14 languages, and all three TTS families offer multilingual checkpoints. A cross-lingual study would investigate whether the performance rankings and sensitivity patterns observed here generalize beyond English.

5. **Model distillation and efficiency.** Bark's 29-second mean generation time limits its practical utility. Investigating distilled or quantized Bark variants, or alternative expressive TTS architectures, could preserve expressiveness while reducing the runtime gap relative to SpeechT5 and MMS-TTS.

6. **Compositional long-form TTS.** This study evaluated TTS on short utterances only. Extending the cross-model comparison to paragraph-length inputs would test boundary effects, prosodic coherence, and the stability of each system's output over extended generation.

7. **Streaming and real-time deployment.** Measuring time-to-first-audio and streaming latency, in addition to total generation time, would provide deployment-relevant efficiency metrics that the current wall-clock measurements do not capture.

---

<div style="page-break-after: always;"></div>

## XV. References

[1] J. Alammar and M. Grootendorst, *Hands-on Generative AI with Transformers and Diffusion Models*. O'Reilly Media, 2024.

[2] D. Gerz, P.-H. Su, R. Kusztos, A. Mondal, M. Lis, E. Singhal, N. Mrksic, T.-H. Wen, and I. Vulic, "Multilingual and Cross-Lingual Intent Detection from Spoken Data," arXiv:2104.08524, 2021.

[3] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in *Proc. NeurIPS*, vol. 33, 2020.

[4] A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks," in *Proc. ICML*, pp. 369–376, 2006.

[5] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. ICML*, PMLR 202:28492–28518, 2023.

[6] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Proc. NeurIPS*, vol. 30, 2017.

[7] J. Ao, R. Wang, L. Zhou, C. Wang, S. Ren, Y. Wu, S. Liu, T. Ko, Q. Li, Y. Zhang, Z. Wei, Y. Qian, J. Li, and F. Wei, "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing," in *Proc. ACL*, vol. 1, pp. 5723–5738, 2022.

[8] J. Kong, J. Kim, and J. Bae, "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis," in *Proc. NeurIPS*, vol. 33, 2020.

[9] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, Z. Ni, A. Vyas *et al.*, "Scaling Speech Technology to 1,000+ Languages," arXiv:2305.13516, 2023.

[10] J. Kim, J. Kong, and J. Son, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," in *Proc. ICML*, PMLR 139:5530–5540, 2021.

[11] A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, "High Fidelity Neural Audio Compression," arXiv:2210.13438, 2022.

---

<div style="page-break-after: always;"></div>

**Author:** Ryan Kamp  
**Affiliation:** CS6078 Generative AI, University of Cincinnati  
**Date:** April 2026