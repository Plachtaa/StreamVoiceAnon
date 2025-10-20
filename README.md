# StreamVoiceAnon
This repository contains the implementation of StreamVoiceAnon, a real-time voice anonymization / voice conversion model.
Relevant paper has been submitted to ICASSP 2026.  
Training code will be released after the paper is accepted.

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
Although offline, the above command still uses a simulated streaming inference logic which processes audio chunk by chunk.

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
python real-time-gui.py
```
This UI uses the same behavior as simulated online inference. It uses `--compile` by default, so please ensure you have installed triton (as previously stated) before using it.

### TODO
 - [ ] Full MacOS support
 - [ ] Release metrics for voice conversion & speaker anonymization
 - [ ] Release fine-tuning code
 - [ ] More to be added