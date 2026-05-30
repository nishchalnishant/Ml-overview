---
module: Specialized Domains
topic: Speech and Audio
subtopic: ""
status: unread
tags: [speech, audio, asr, tts, whisper, mel-spectrogram, ctc, specialized-domains]
---
# Speech and Audio Processing

---

## Table of Contents

1. [The Audio Signal: From Waveforms to Features](#1-the-audio-signal-from-waveforms-to-features)
2. [Mel Spectrograms and MFCCs](#2-mel-spectrograms-and-mfccs)
3. [Automatic Speech Recognition](#3-automatic-speech-recognition)
4. [CTC and Connectionist Temporal Classification](#4-ctc-and-connectionist-temporal-classification)
5. [Attention-Based ASR and Seq2Seq](#5-attention-based-asr-and-seq2seq)
6. [Whisper — Weakly Supervised ASR](#6-whisper--weakly-supervised-asr)
7. [Speaker Verification and Speaker Diarization](#7-speaker-verification-and-speaker-diarization)
8. [Text-to-Speech (TTS)](#8-text-to-speech-tts)
9. [Common Interview Questions](#9-common-interview-questions)

---

## 1. The Audio Signal: From Waveforms to Features

**The raw signal:** Audio is a pressure wave sampled at discrete time steps. A standard speech recording is sampled at 16kHz — 16,000 amplitude values per second. One second of speech is a sequence of 16,000 numbers in [-1, 1].

**Why raw waveforms are hard for neural networks:**
1. **High resolution:** 16,000 samples/second × 10 seconds = 160,000 time steps. At 100ms frames (the typical resolution for phoneme-level information), this is 100 steps — but each step must encode temporal patterns at multiple time scales simultaneously.
2. **Long-range dependencies:** The beginning of a word influences the end; speaker identity is global. Temporal dependencies span very different time scales.
3. **Irrelevant variation:** Phase shifts, DC offset, and absolute amplitude levels carry no linguistic information but dominate the raw signal.

**Solution:** Compute time-frequency representations that separate the signal into frequency bands over short windows. These representations discard irrelevant phase information and expose the frequency patterns that human auditory perception — and speech recognition systems — rely on.

### Short-Time Fourier Transform (STFT)

1. **Frame the signal:** Divide into overlapping windows of N samples (typically N=400 for 25ms at 16kHz). Overlap by 10ms (stride=160 samples) to capture transitions.
2. **Apply a window function (Hann window):** Tapers the signal to zero at the edges, reducing spectral leakage.
3. **Compute FFT:** For each frame, compute the discrete Fourier transform. Produces N/2+1 frequency bins.
4. **Take magnitude:** Discard phase (|FFT|). The magnitude spectrogram shows energy at each frequency over time.

```python
import librosa
import numpy as np

# Load audio
y, sr = librosa.load('speech.wav', sr=16000)

# STFT → magnitude spectrogram
D = np.abs(librosa.stft(y, n_fft=400, hop_length=160, win_length=400))
# D shape: (201, time_frames)
```

---

## 2. Mel Spectrograms and MFCCs

### Mel Scale

Human hearing is non-linearly sensitive to frequency — we perceive differences in pitch more finely at low frequencies than at high. The mel scale converts frequency to a perceptually uniform scale:

```
mel(f) = 2595 × log10(1 + f/700)
```

Below ~1kHz: roughly linear. Above ~1kHz: logarithmic. A piano note difference that sounds like "one step" maps to the same mel difference whether the base note is low or high.

### Mel Spectrogram

Apply a bank of triangular filters in the mel frequency domain to the STFT magnitude:

```python
# Compute mel spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr,
    n_fft=400,          # FFT window size
    hop_length=160,     # frame stride (10ms at 16kHz)
    n_mels=80,          # number of mel filter banks
    fmin=0,             # min frequency (Hz)
    fmax=8000           # max frequency (Hz)
)
# Convert to dB scale
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
# Shape: (80, time_frames)
```

**80 mel filter banks** is standard for modern speech systems (Whisper, wav2vec). Each filter bank captures energy in one perceptually-spaced frequency band.

### MFCCs (Mel-Frequency Cepstral Coefficients)

Apply the Discrete Cosine Transform (DCT) to the log mel spectrum. This de-correlates the mel filter bank outputs (which are correlated because adjacent filters overlap) and compresses the representation.

```python
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_mels=40)
# shape: (13, time_frames)
```

**13 MFCCs** are traditional in classical ASR (HMM-based systems). The first coefficient (C0) captures overall energy; higher coefficients capture fine spectral detail.

**MFCCs vs. mel spectrograms:**
- MFCCs: smaller feature size, decorrelated, good for classical HMM-based ASR
- Mel spectrograms: preserve more spectral detail, preferred as input to deep neural networks (no information discarded by DCT)

---

## 3. Automatic Speech Recognition

**The task:** Map an audio sequence to a text transcript. ASR systems must handle:
- Acoustic variability: different speakers, accents, microphones, noise levels
- Language variability: vocabulary out-of-vocabulary words, domain-specific terminology
- Temporal alignment: unknown relationship between audio frames and characters/words

### Classical HMM-DNN Systems

**Hidden Markov Model (HMM):** Models speech as a sequence of acoustic states (typically phonemes). Each state emits a probability distribution over acoustic features.

**Components:**
1. **Acoustic model:** P(audio | phoneme sequence) — maps audio features to phoneme likelihoods. Traditionally: Gaussian Mixture Models (GMMs); modern: Deep Neural Networks (DNNs)
2. **Pronunciation dictionary:** Maps words to phoneme sequences ("cat" → /k æ t/)
3. **Language model:** P(word sequence) — n-gram or neural LM

**Decoding:** Viterbi algorithm finds the most likely word sequence given all three components. Beam search explores top-k hypotheses.

**Why DNNs replaced GMMs:** DNNs share parameters across phonemes and can learn features from the raw mel spectrogram. They model context (a phoneme's acoustic realization depends on surrounding phonemes) more naturally via deeper representations.

### The Alignment Problem

**The core challenge:** An audio signal of length T frames must map to a text of length N characters/words, where T >> N. There is no explicit alignment in the training data — we know the transcript but not which frames correspond to which characters.

**Example:**
```
Audio frames: [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
Transcript:   "cat"                                     
```

We don't know that f1-f4 correspond to /k/, f5-f7 to /æ/, f8-f10 to /t/. CTC and attention-based models solve this alignment problem differently.

---

## 4. CTC and Connectionist Temporal Classification

**The problem (Graves et al., 2006):** Given input sequence of T frames and target sequence of length N < T, train without explicit frame-to-character alignment. The model must figure out the alignment from the training data alone.

**The core insight:** Define a probability distribution over all possible alignments (sequences of T labels including a special blank token ε), then marginalize over all valid alignments that "collapse" to the target transcript.

### CTC Mechanics

**Output:** For each frame t, the model outputs a probability distribution over the vocabulary + blank token ε. Shape: (T, |vocab| + 1).

**Collapsing rules:**
1. Merge consecutive identical non-blank labels
2. Remove blank tokens

**Examples of sequences that all collapse to "cat":**
```
ε ε k ε æ æ ε t ε → "cat" (after merge identical + remove blanks)
k k ε æ ε t ε ε → "cat"
ε k æ t → "cat"
```

**CTC loss:** Sum over all valid alignments:
$$P(y | x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p(\pi_t | x)$$

Computed efficiently via forward-backward algorithm (dynamic programming).

```python
import torch
import torch.nn as nn

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')

# log_probs: (T, N, C) — T=time, N=batch, C=classes (incl. blank)
# targets: concatenated target sequences
# input_lengths: lengths of each sequence in batch
# target_lengths: lengths of each target in batch

log_probs = torch.randn(50, 16, 29).log_softmax(2)  # 50 frames, 16 batch, 28 chars + blank
targets = torch.randint(1, 29, (16 * 20,))
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.full((16,), 20, dtype=torch.long)

loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

**Decoding at inference:**
- **Greedy:** Take argmax at each frame, then collapse. Fast but suboptimal.
- **Beam search:** Maintain top-k partial sequences, expand each with all possible next tokens. Optionally integrate an external language model.

**CTC properties:**
- Assumes conditional independence between output labels (factorizes over time) — cannot model output-to-output dependencies ("q" → high probability of "u")
- Frames can only attend to past context (unless using Bi-RNN/Transformer encoder)
- No explicit attention mechanism — alignment is implicit in the CTC loss

### wav2vec 2.0 and Pre-Trained CTC

**wav2vec 2.0 (Baevski et al., 2020):** Learn speech representations self-supervised on unlabeled audio, then fine-tune with CTC on labeled data.

**Pre-training:**
1. Encode raw audio waveform with a CNN → latent speech representations z_t
2. Quantize z_t to discrete speech units ĉ_t via a codebook (Gumbel softmax)
3. Mask spans of z_t (analogous to BERT's MLM); Transformer contextualizes masked representations
4. Contrastive loss: for each masked position, identify the true quantized unit from distractors

**Fine-tuning:** Add a linear layer on top of the Transformer; train with CTC loss on transcribed data. With only 10 minutes of labeled data + wav2vec 2.0 pre-training, achieves competitive WER — dramatic reduction in labeled data requirements.

---

## 5. Attention-Based ASR and Seq2Seq

**The problem with CTC:** CTC assumes output labels are conditionally independent given the input — it cannot directly model "given I predicted 'q', the next character is almost certainly 'u'." This limits performance, especially in noisy conditions.

**Attention-based encoder-decoder:** Full seq2seq model with attention. No independence assumption. The decoder can directly condition each output token on all encoder states and all previously generated tokens.

### Architecture

```
Audio → CNN frontend → Encoder (Transformer/LSTM) → encoder states h_1,...,h_T
                                                            ↓ cross-attention
[SOS] → Decoder Transformer → y_1
  y_1 → Decoder Transformer → y_2
  ...
```

**Encoder:** Deep Transformer or Conformer (see below) processes mel spectrogram frames.

**Decoder:** Autoregressive transformer. At each step:
1. Self-attention over previously generated tokens (causal mask)
2. Cross-attention over encoder states — attends to relevant audio frames
3. Feed-forward → next token probability

### Conformer (Gulati et al., 2020)

**The insight:** Transformers capture global context well (long-range attention) but miss local patterns (adjacent frames are correlated). CNNs capture local patterns but not global. Conformer combines both.

```
Conformer block:
  Half-step FFN
  → Multi-Head Self-Attention
  → Convolution module (depthwise separable conv)
  → Half-step FFN
  → Layer Norm
```

The convolution module between attention layers captures local temporal patterns — the periodicity of phonemes. Conformer became the dominant encoder architecture in production ASR (ESPnet, SpeechBrain, on-device models).

### CTC vs. Attention Tradeoffs

| | CTC | Attention Encoder-Decoder |
|---|---|---|
| Output dependencies | Conditional independence | Full autoregressive conditioning |
| Training | One forward pass, efficient | Teacher forcing; requires right-shifted targets |
| Decoding | Fast (frame-by-frame) | Slow (sequential token generation) |
| Noisy conditions | More robust (does not need output context) | More sensitive to noise |
| Long-form audio | Struggles (attention not naturally chunked) | Struggles with very long audio |

**Hybrid CTC/Attention:** Train with both CTC (auxiliary) and attention (primary) losses. CTC loss improves encoder training and monotonic alignment; attention decoder improves accuracy. Used in ESPnet and many production systems.

---

## 6. Whisper — Weakly Supervised ASR

**The core insight (Radford et al., 2022):** Instead of carefully curated transcribed speech data, use 680,000 hours of weakly supervised audio-transcript pairs from the web. Scale overwhelms noise.

**Architecture:**
- Input: 80-channel log-mel spectrogram, 30-second segments
- Encoder: transformer (input-level CNN for feature extraction, then 2D convolutional positional encoding, then transformer layers)
- Decoder: autoregressive transformer conditioned on encoder states

**Multi-task training via decoder prompts:** The decoder is prompted with task specification tokens that tell the model what to produce. No task-specific heads needed.

```
[SOT] [English] [Transcribe] [NOTIMESTAMPS]  → transcript
[SOT] [French] [Translate]   → English translation
[SOT] [Spanish] [Transcribe] [00:00:02.000]  → transcript with timestamps
```

**Why this works:** The decoder is a language model conditioned on audio. The task prompt controls whether the model transcribes or translates, what language to expect, and whether to include timestamps.

**Training scale:** 680K hours of audio. ~99% of this is weakly labeled (subtitles from the web, not careful human transcription). The scale compensates for label noise.

**Zero-shot capability:** Whisper achieves competitive WER on benchmarks without any fine-tuning. Especially robust to accents and noise.

**Limitations:**
- 30-second context window — handles long-form audio by sliding window, can miss context at boundaries
- Slower than streaming-optimized models (not designed for real-time)
- Hallucinations on silence or background noise — decoder generates plausible text even when no speech is present

```python
import whisper

model = whisper.load_model("large-v2")
result = model.transcribe("audio.wav", language="en", task="transcribe")
print(result["text"])
```

---

## 7. Speaker Verification and Speaker Diarization

### Speaker Verification

**Task:** Given a reference audio sample from a target speaker and a new audio clip, decide whether the new clip is from the same speaker.

**Approach:** Learn a speaker embedding space where the same speaker's utterances cluster together regardless of content.

**d-vectors (Deep speaker embeddings):**
1. Train a speaker classification model on many speakers
2. Extract the penultimate layer's activation as the speaker embedding (d-vector)
3. Verification: compare cosine similarity of d-vectors from reference and test utterances. Threshold to accept/reject.

**x-vectors (Snyder et al., 2018):**
- TDNN (Time Delay Neural Network) encoder with statistics pooling
- Statistics pooling: compute mean and standard deviation across all frames → fixed-size vector regardless of utterance length
- Fine-tuned with PLDA (Probabilistic Linear Discriminant Analysis) for back-end scoring

**Training objective:**
- Softmax over speaker IDs: standard classification, but embeddings generalize to unseen speakers
- Angular margin loss (ArcFace): add margin in angle space to enforce more discriminative embeddings:
  ```
  L = -log(exp(s·cos(θ_yi + m)) / (exp(s·cos(θ_yi + m)) + Σ_{j≠y_i} exp(s·cos(θ_j))))
  ```

**Evaluation:**
- Equal Error Rate (EER): the point where false acceptance rate = false rejection rate
- minDCF (minimum Detection Cost Function): weighted combination of false alarm and miss rates

### Speaker Diarization

**Task:** Given a multi-speaker audio, answer "who spoke when?" — segment the audio and label each segment with a speaker identity.

**Pipeline:**
1. **Speech Activity Detection (SAD):** Remove silence
2. **Segmentation:** Divide audio into short segments (1–2s) likely from a single speaker
3. **Embedding extraction:** Compute x-vector or d-vector for each segment
4. **Clustering:** Cluster embeddings by speaker identity (k-means, agglomerative hierarchical clustering, spectral clustering). Number of speakers often unknown → hyperparameter or estimated via eigenvalue analysis
5. **Resegmentation:** Viterbi re-alignment with speaker models derived from clustering

**Modern end-to-end diarization (EEND):** Directly predict per-frame speaker activity as a multi-label problem. Handles overlapping speech (two speakers talking simultaneously) which pipeline approaches miss.

---

## 8. Text-to-Speech (TTS)

**Task:** Convert text to natural-sounding speech. The output must be intelligible, natural, and ideally match a target speaker's voice.

### Parametric TTS Pipeline

Classical approach: text → linguistic features → acoustic model → vocoder → waveform.

1. **Text normalization:** "Dr. Smith" → "Doctor Smith," "$5.99" → "five dollars ninety-nine"
2. **Grapheme-to-phoneme (G2P):** Map text to phoneme sequences
3. **Acoustic model:** Predict speech parameters (mel spectrogram frames or acoustic features) from phonemes
4. **Vocoder:** Convert acoustic parameters to waveform

### Tacotron 2

**The core insight (Shen et al., 2018):** Train an end-to-end seq2seq model to generate mel spectrograms directly from character/phoneme sequences, bypassing explicit linguistic feature engineering.

**Architecture:**
- Encoder: character embedding + conv layers + BiLSTM → encoder states
- Attention: location-sensitive attention (prevents attention from repeating or skipping)
- Decoder: autoregressive LSTM predicts mel spectrogram frames
- Post-net: 5-layer CNN refines the mel spectrogram
- Vocoder: WaveNet (or WaveGlow, HiFi-GAN) converts mel to waveform

**Location-sensitive attention:** The attention mechanism is conditioned on the previous attention weights — prevents getting "stuck" at one position or jumping ahead erratically. Monotonic (left-to-right) attention constraint is also enforced.

### FastSpeech 2 (Non-Autoregressive)

**The problem with autoregressive TTS:** Tacotron generates mel frames one at a time. Slow inference. Attention can fail (repeating or skipping words).

**FastSpeech 2:** Replace autoregressive attention with explicit duration prediction:

1. **Duration predictor:** Predict how many mel frames each input phoneme should expand to
2. **Length regulator:** Expand phoneme representations according to predicted durations
3. **Mel decoder:** Transform the upsampled sequence into mel spectrogram (non-autoregressive — all frames in parallel)

```
Phoneme sequence (N) → encoder → duration predictor → expand → (T) mel frames → decoder → mel spectrogram
```

**What this buys:** Parallel generation (all mel frames at once) → 50–100× faster than autoregressive. More stable (no attention collapse). Duration control allows speaking rate adjustment.

**What breaks:** Duration predictor errors cause subtle prosody issues. Without attention, the model learns a rigid phoneme-to-frame mapping rather than flexible dynamic alignment.

### Neural Vocoders

**The vocoder's job:** Convert mel spectrogram to a waveform. Quality here directly determines perceived naturalness.

**WaveNet (Oord et al., 2016):** Dilated causal convolutions. Generates one sample at a time — high quality but extremely slow at inference (1 sec of audio takes minutes).

**WaveGlow (Prenger et al., 2018):** Normalizing flow model. Invertible transformations map Gaussian noise to waveform. Parallel generation; fast inference. Good quality.

**HiFi-GAN (Kong et al., 2020):** GAN-based vocoder. Generator produces waveform; multi-scale discriminators evaluate quality at multiple time resolutions. State-of-the-art quality with real-time inference.

```python
# HiFi-GAN vocoder usage (schematic)
from hifi_gan import Generator
import torch

generator = Generator(config).to('cuda')
mel_spectrogram = ...  # (1, 80, T)
with torch.no_grad():
    audio = generator(mel_spectrogram)  # (1, T * hop_length)
```

### Zero-Shot Voice Cloning

**YourTTS / VALL-E approach:** Given a short reference audio from an unseen speaker, generate speech that sounds like that speaker saying new text.

**VALL-E (Wang et al., 2023):** Frame TTS as a language modeling problem on audio codec tokens. Encode audio to discrete tokens (EnCodec neural codec); train a language model (similar to GPT) to predict audio tokens given text and a 3-second speaker reference. Achieves impressive voice cloning with 3 seconds of reference audio.

---

## 9. Common Interview Questions

### Q1: What is a mel spectrogram and why is it used instead of raw waveforms?

A mel spectrogram applies the Short-Time Fourier Transform to a waveform, converts the frequency axis to the mel scale (perceptually uniform), and takes the log of the resulting energy in each mel frequency band.

**Why not raw waveforms:** The mel spectrogram compresses 16,000 samples/second to ~100 frames/second, each with 80 dimensions — a 200× compression that preserves linguistically relevant information while discarding irrelevant phase information. The mel scale better matches human auditory perception. The log compression reduces the dynamic range, making the representation more robust to loudness variations.

**Why not MFCCs for deep learning:** MFCCs apply a DCT to the log mel spectrum, discarding information. For classical GMM-HMM systems, decorrelated features are better. For neural networks (which can decorrelate internally), mel spectrograms preserve more information.

---

### Q2: How does CTC solve the alignment problem?

CTC introduces a blank token ε and defines a collapsing function that merges consecutive identical labels and removes blanks. The CTC loss marginalizes over all T-length output sequences (with blanks) that collapse to the target transcript. The forward-backward algorithm computes this sum efficiently.

Key properties: the model outputs a distribution at each frame independently, which enables training without frame-level alignments. At inference, greedy decoding (argmax + collapse) is fast; beam search with a language model improves accuracy.

**Limitation:** CTC assumes label conditional independence — each output token is predicted independently given the input. It cannot model output-output dependencies (e.g., "q" → "u"). Attention-based models address this.

---

### Q3: What is the Conformer and why does it work better than a pure transformer for speech?

A pure transformer captures global dependencies via self-attention but misses local temporal patterns. For speech, adjacent frames are highly correlated — consonant articulations happen in 10–50ms windows. A convolutional module captures this local structure efficiently.

The Conformer interleaves attention layers (global context) with depthwise separable convolution modules (local patterns). The result: better acoustic representations than either pure attention or pure convolution at similar parameter counts. Conformer became the dominant encoder architecture for speech recognition.

---

### Q4: What is speaker diarization? How does it differ from speaker verification?

**Speaker verification:** Binary task — is this audio clip from speaker X? One-to-one comparison of a test utterance to a reference.

**Speaker diarization:** Segmentation + labeling — given a recording with multiple speakers, produce "Speaker A: 0:00–0:45, Speaker B: 0:45–1:20, Speaker A: 1:20–2:00..." with no pre-known speaker identities.

Diarization requires: speech activity detection (remove silence), segmentation (split audio into segments unlikely to contain speaker changes), embedding extraction (x-vector per segment), and clustering (group similar speaker embeddings). The challenge: unknown number of speakers and no reference recordings.

---

### Q5: How does Whisper handle multilingual ASR without language-specific heads?

Whisper uses the decoder's prompt tokens to specify the task. Different language identifiers ([English], [French], [Spanish]) in the prompt tell the decoder which language to generate. The [Translate] task token switches the decoder to translate to English. This is controlled by the same decoder parameters — no separate heads.

This works because the decoder is a language model conditioned on both the audio (via cross-attention to encoder) and the prompt (via self-attention to previous tokens). The task prompt programs the desired behavior without architectural changes.

---

## Key Papers

| Paper | Year | Contribution |
|---|---|---|
| Graves et al. | 2006 | CTC — connectionist temporal classification |
| Hochreiter & Schmidhuber | 1997 | LSTM — used in most early deep ASR systems |
| Gulati et al. | 2020 | Conformer — CNN + attention for speech |
| Baevski et al. | 2020 | wav2vec 2.0 — self-supervised speech pre-training |
| Radford et al. | 2022 | Whisper — large-scale weakly supervised ASR |
| Snyder et al. | 2018 | x-vectors — speaker embeddings with TDNN |
| Shen et al. | 2018 | Tacotron 2 — end-to-end neural TTS |
| Ren et al. | 2020 | FastSpeech 2 — non-autoregressive TTS |
| Kong et al. | 2020 | HiFi-GAN — high-fidelity neural vocoder |
| Wang et al. | 2023 | VALL-E — zero-shot voice cloning as LM |
