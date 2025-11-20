## ⚠️This repository is currently under construction.⚠️


# StreamVoiceAnon
This repository contains the implementation of StreamVoiceAnon, a real-time voice anonymization / voice conversion model.
Relevant paper has been submitted to ICASSP 2026.  

### Installation
```bash
git clone https://github.com/Plachtaa/StreamVoiceAnon.git
cd StreamVoiceAnon
pip install -r requirements.txt
```

If running on Windows OS, please install the following:
```bash
pip install triton-windows==3.2.0.post13
```
Note that this is **compulsory** to run inference with RTF < 1.0

Full MacOS support is still under construction.

### Download Pretrained Models
```bash
hf download Plachta/StreamVoiceAnon --local-dir pretrained_checkpoints/
```

### Training
Below is an example command to launch single node multi-GPU training with streaming Emilia dataset from HuggingFace:
```bash
accelerate launch trainers/arvc_trainer.py --config_path configs/config_firefly_arvcasr_8192_delay0_8.yaml --mixed-precision bf16
```
To customize model config or training datasets, we encourage users to read config files or training code.

### Inference
Offline inference 
```bash
python evaluations/infer_arvc.py \
    --src_path <path_to_audio> \
    --ref_path <path_to_audio> \
    --out_dir <path_to_output_directory> \
    --delay 2 \  # Specify delay in number of frames (must have)
    --compile
```
Simulated online inference
```bash
python evaluations/infer_arvc.py \
    --src_path <path_to_audio> \
    --ref_path <path_to_audio> \
    --out_dir <path_to_output_directory> \
    --delay 2 \  # Specify delay in number of frames (must have)
    --compile \
    --simulate_streaming \
    --decode_chunk_frames 1 # how many frames for encoder & vocoder to process each time
```
This simulates a chunk-by-chunk online inference with specified chunk size. `src_path` (source audio) has no length limit here. `ref_path` (reference audio) will be truncated to some maximum length (if longer than that limit)

Real-time inference
```bash
python evaluations/real-time-gui.py
```
This UI uses the same behavior as simulated online inference. It uses `--compile` by default, so please ensure you have installed triton (as previously stated) before using it.

### TODO
 - [ ] Release privacy protection code
 - [ ] Release metrics for voice conversion & speaker anonymization
 - [x] Release training code (for VC model)
 - [ ] Release training code (for content encoder)
 - [ ] Release fine-tuning code
 - [ ] Full MacOS support
 - [ ] More to be added

### Acknowledgements
 - Co-author: https://github.com/paniquex
 - Computation resources: https://www.nscc.sg/
 - Real-time GUI: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
 - Speaker representations (1 of 2) https://huggingface.co/funasr/campplus
 - Speaker representations (2 of 2) https://github.com/SparkAudio/Spark-TTS
 - Speech acoustic codec https://huggingface.co/fishaudio/fish-speech-1.5
 - Idea: https://arxiv.org/html/2401.11053v1
 - VoicePrivacyChallenge: https://www.voiceprivacychallenge.org/
