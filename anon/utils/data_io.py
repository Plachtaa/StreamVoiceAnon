from pathlib import Path, PosixPath
from hyperpyyaml import load_hyperpyyaml, dump_hyperpyyaml
import json
import pandas as pd
import logging

import torchaudio
import io
import os
import subprocess

logger = logging.getLogger(__name__)

def read_kaldi_format(filename, return_as_dict=True, values_as_string=False):
    key_list = []
    value_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            splitted_line = line.split()
            if len(splitted_line) == 1:
                key_list.append(splitted_line[0].strip())
                value_list.append("")
            if len(splitted_line) == 2:
                key_list.append(splitted_line[0].strip())
                value_list.append(splitted_line[1].strip())
            elif len(splitted_line) > 2:
                key_list.append(splitted_line[0].strip())
                if values_as_string:
                    value_list.append(' '.join([x.strip() for x in splitted_line[1:]]))
                else:
                    value_list.append([x.strip() for x in splitted_line[1:]])
    if not return_as_dict:
        return key_list, value_list
    return dict(zip(key_list, value_list))


def load_wav_from_scp(wav, frame_offset: int = 0,  num_frames: int = -1):
    """Reads a wav.scp entry like kaldi with embeded unix command
    and returns a pytorch tensor like it was open with torchaudio.load()
    (within some tolerance due to numerical precision))

    Args:
        wav: a list containing the scp entry

    Returns:
        out_feats: torch.tensor array dtype float32 (default)
    """
    if isinstance(wav, list):
        wav = " ".join(str(x) for x in wav)
    if wav.strip().endswith("|"):
        devnull = open(os.devnull, "w")
        try:
            wav_read_process = subprocess.Popen(
                wav.strip()[:-1], stdout=subprocess.PIPE, shell=True, stderr=devnull
            )
            sample, sr = torchaudio.backend.soundfile_backend.load(
                io.BytesIO(wav_read_process.communicate()[0]),
                frame_offset=frame_offset, num_frames=num_frames
            )
        except Exception as e:
            raise IOError("Error processing wav file: {}\n{}".format(wav, e))
    else:
        sample, sr = torchaudio.backend.soundfile_backend.load(wav, frame_offset=frame_offset, num_frames=num_frames)

    return sample, sr

def read_table(filename, names, sep=' ', dtype=None):
    if isinstance(dtype, list):
        dtype = dict(zip(names, dtype))
    return pd.read_csv(filename, sep=sep, names=names, dtype=dtype)


def write_table(filename, table, sep=' ', save_header=False):
    table.to_csv(filename, sep=sep, index=False, header=save_header)


def read_matrix(filename):
    matrix = []
    with open(filename) as f:
        for line in f:
            matrix.append([str(value) for value in line.split()])
    return matrix


def save_kaldi_format(data, filename):
    if isinstance(data, list):
        if len(data) == 2:
            data = dict(zip(data[0], data[1]))
        elif isinstance(data[0], tuple) and len(data[0]) == 2:
            data = dict(data)
    with open(filename, 'w', encoding='utf-8') as f:
        for key, value in sorted(data.items(), key=lambda x: x[0]):
            if isinstance(value, list):
                value = ' '.join(value)
            try:
                #value = value.encode('utf-8')
                f.write(f'{key} {value}\n')
            except UnicodeEncodeError:
                logger.error(f'{key} {value}')
                raise


def parse_yaml(filename, overrides=None):
    with open(filename, 'r') as f:
        config = load_hyperpyyaml(f, overrides=overrides)

    # tranform strings denoting paths into pathlib.Path instances
    config = _transform_paths(config)
    return config


def save_yaml(config, filename):
    config = _transform_paths_back_to_strings(config)
    with open(filename, 'w') as f:
        #json.dump(f, config)
        dump_hyperpyyaml(config, f)


def _transform_paths(yaml_dict):
    path_tags = {'_dir', '_file', '_path', '_folder'}
    for key, value in yaml_dict.items():
        if value is None:
            continue
        if isinstance(value, dict):
            yaml_dict[key] = _transform_paths(value)
        elif any(tag in key for tag in path_tags) and value is not None:
            yaml_dict[key] = Path(value)
    return yaml_dict


def _transform_paths_back_to_strings(yaml_dict):
    for key, value in yaml_dict.items():
        if isinstance(value, dict):
            yaml_dict[key] = _transform_paths_back_to_strings(value)
        elif isinstance(value, PosixPath):
            yaml_dict[key] = str(value.absolute())
    return yaml_dict
