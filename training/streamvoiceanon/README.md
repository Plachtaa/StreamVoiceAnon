# StreamVoiceAnon training

This directory contains the original StreamVoiceAnon trainers, configs, model modules, and dataloader from [v0.1.0](https://github.com/Plachtaa/StreamVoiceAnon/tree/e015100d6439ff9993aed5de3512f5240cc9ea84). The source follows the original [Apache 2.0 license](LICENSE). Set up the training environment separately from the inference package.

From the release repository root, create a Python 3.10 environment with the CUDA 12.4 PyTorch build and install the training dependencies.

```bash
cd training/streamvoiceanon
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Request access to the [Emilia dataset](https://huggingface.co/datasets/Amphion/Emilia-Dataset) and sign in with `hf auth login`. The original dataloader streams the training data. Download the frozen components into the training directory before launching either trainer.

```bash
hf download Plachta/StreamVoiceAnon \
    campplus_cn_common.bin asr_s2s_bsq_8192_causal_down_whisper.pth \
    firefly-gan-vq-fsq-8x1024-21hz-generator.pth spark_speaker_encoder.pth \
    --revision 3684dff8e8706b306e6d1f0217a1f79968bbb966 \
    --local-dir pretrained_checkpoints
```

## Voice-conversion model

Run from `training/streamvoiceanon` on a node with multiple GPUs. The trainer uses the pretrained content encoder and learns the voice-conversion model.

```bash
PYTHONPATH=. accelerate launch --multi_gpu trainers/arvc_trainer.py \
    --config_path configs/config_firefly_arvcasr_8192_delay0_8.yaml \
    --mixed-precision bf16
```

## Content encoder

Use the ASR trainer to train the content encoder. It loads the pretrained [Wav2Vec2-BERT model](https://huggingface.co/facebook/w2v-bert-2.0) as a feature extractor.

```bash
PYTHONPATH=. accelerate launch --multi_gpu trainers/asr_trainer.py \
    --config_path configs/config_asr_bsq_8192.yaml \
    --mixed-precision bf16
```

Set `log_dir` in the YAML to a separate experiment directory for each run. The original trainers resume from checkpoints in that directory and prune older checkpoints.
