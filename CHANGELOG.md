# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- GUI UX improvements (planned for v0.2.0)
- Emotion preservation model support

## [0.1.0] - 2026-03-16
### Added
#### Speaker Anonymization & Privacy
- Real-time speaker anonymization via neural audio codec and language models
- Speaker anonymization via noise mixing on speaker embeddings (`--alpha`)
- Multiple reference audio support for stronger anonymization (`--ref_path` with multiple entries), blends speaker representations to make source speaker harder to trace
- Privacy evaluation pipeline (`anon/`)

#### Voice Conversion Model
- Dual AR voice conversion model with configurable delay (0-8 frames)
- ASR-based content encoder with BSQ tokenizer
- Speaker encoders (CampPlus + SparkTTS timbre encoder)
- Firefly-GAN vocoder with FSQ quantization

#### Inference
- Offline inference (`evaluations/infer_arvc.py`)
- Simulated streaming inference (`--simulate_streaming`)
- Real-time GUI (`evaluations/real-time-gui.py`)
- torch.compile support for real-time inference
- Pretrained checkpoints on HuggingFace Hub

#### Training
- Training code for VC model and ASR content encoder

#### GUI Improvements
- Warmup progress window for torch.compile (prevents frozen UI)
- Model loading progress window
- Path whitespace stripping and file existence validation
