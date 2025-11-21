# Evaluation


## Privacy

### ASV
All scripts regarding the ASV training in [privacy/asv/asv_train](privacy/asv/asv_train), including parts of [privacy/asv/asv.py](privacy/asv/asv.py), are based on the 
[VoxCeleb recipe by SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb) and adapted for LibriSpeech.
The privacy evaluation metric is equal error rate (EER).

## Utility

### ASR
The ASR scripts are adapted from the Speechbrain recipe for LibriSpeech.
The utility evaluation metric is word error rate (WER).

### SER
The SER scripts are adapted from the Speechbrain recipe for IEMOCAP.
The utility evaluation metric is unweighted average recall (UAR).
