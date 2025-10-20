import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
import hydra
import yaml
from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path
from tqdm import tqdm
from evaluations.infer_arvc import InferenceWrapper

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference Wrapper")
    parser.add_argument("--config_path", type=str, default="configs/config_firefly_arvcasr_8192_delay0_8.yaml")
    parser.add_argument("--checkpoint_path", type=str,
                        default="runs/firefly_arvc_8192_delay4_0_8_grpo/DAR_epoch_00000_step_296060.pth")
    parser.add_argument("--meta-file", type=str, default="../../seedtts_testset/en/non_para_reconstruct_meta.lst")
    parser.add_argument("--output", type=str, default="../../seedtts_testset/en_output/vc_dynamic_delay1/")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--delay", type=int, default=8, help="Delay for the decoder (in frames), 0 means no delay")
    args = parser.parse_args()
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    infer_wrapper = InferenceWrapper(config_path, checkpoint_path, compile_ar=args.compile,)

    meta_file_path = args.meta_file
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    root_dir = os.path.dirname(meta_file_path)
    # open meta file
    with open(meta_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    target_names = []
    prompt_names = []
    source_names = []
    for lines in lines:
        target_name, prompt_trans, prompt_name, source_trans, source_name = lines.strip().split("|")
        target_names.append(target_name)
        prompt_names.append(prompt_name)
        source_names.append(source_name)

    for wav_idx, (source_name, prompt_name, target_name) in enumerate(
            tqdm(zip(source_names, prompt_names, target_names))):
        source_path = os.path.join(root_dir, source_name)
        prompt_path = os.path.join(root_dir, prompt_name)
        output_path = os.path.join(output_dir, target_name)
        if not output_path.endswith(".wav"):
            output_path += ".wav"
        vc_wav = infer_wrapper.infer(source_path, prompt_path, out_dir=None, output_path=output_path,
                                     top_p=args.top_p, temperature=args.temperature, delay=args.delay)

