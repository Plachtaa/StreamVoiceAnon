import datasets
import torch
import librosa
import numpy as np
import random
import os
import json
from torch.utils.data import DataLoader, Dataset, ChainDataset, IterableDataset
from modules.audio import mel_spectrogram
from datasets.distributed import split_dataset_by_node
from text_utils.clean import clean_text, repetition_found
from text_utils.chn_text_norm.text import Text as ChnNormedText
duration_setting = {'min': 0.5, 'max': 30}
_punctuation = ':,.!?¡¿-…"«»“”'
# 全角标点
_punctuation += "，。、；：？！…“”‘’（）《》【】—～"
class ExceptionHandlingWrapper(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        iterator = iter(self.dataset)

        while True:
            try:
                item = next(iterator)
                yield item
            except StopIteration:
                break
            except Exception as e:
                print(f"Skipping item due to exception: {e}")
                continue
class UpsampleIterableDataset(IterableDataset):
    def __init__(self, dataset, factor=2):
        """
        Args:
            dataset (IterableDataset): 要上采样的可迭代数据集
            factor (int): 上采样倍数，默认2倍
        """
        assert isinstance(dataset, IterableDataset), "Dataset must be an instance of IterableDataset"
        self.dataset = dataset
        self.factor = factor

    def __iter__(self):
        for _ in range(self.factor):
            for data in self.dataset:
                yield data


class DatasetToIterableWrapper(IterableDataset):
    def __init__(self, dataset):
        """
        Args:
            dataset (Dataset): PyTorch标准Dataset，需要转换为IterableDataset
        """
        assert isinstance(dataset, Dataset), "Input dataset must be a Dataset instance"
        self.dataset = dataset

    def __iter__(self):
        # generate random indices for shuffling
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for idx in indices:
            yield self.dataset[idx]

class ProbabilisticChainDataset(IterableDataset):
    def __init__(self, datasets, sampling_probs):
        """
        Args:
            datasets (list of IterableDataset): 多个IterableDataset实例
            sampling_probs (list of float): 每个IterableDataset的抽样概率, 长度必须与datasets相同
        """
        assert len(datasets) == len(sampling_probs), "Datasets and sampling_probs must have the same length"
        assert all(isinstance(ds, IterableDataset) for ds in datasets), "All datasets must be IterableDataset instances"
        assert sum(sampling_probs) > 0, "Sum of sampling_probs must be greater than 0"

        self.datasets = datasets
        self.sampling_probs = [p / sum(sampling_probs) for p in sampling_probs]  # 归一化概率

    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]

        while True:
            selected_idx = random.choices(range(len(self.datasets)), weights=self.sampling_probs, k=1)[0]

            try:
                yield next(iterators[selected_idx])
            except StopIteration:
                iterators[selected_idx] = None
                self.datasets[selected_idx] = None
                self.sampling_probs[selected_idx] = 0
                if all(it is None for it in iterators):
                    break

class LocalDataset(Dataset):
    def __init__(self,
                 directorys=[],
                 upsample_rate=1,
                 sr=22050,
                 spect_params=None
                 ):
        mel_fn_args = {
            "n_fft": spect_params['n_fft'],
            "win_size": spect_params['win_length'],
            "hop_size": spect_params['hop_length'],
            "num_mels": spect_params['n_mels'],
            "sampling_rate": sr,
            "fmin": spect_params['fmin'],
            "fmax": None,
            "center": False
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
        self.sr = sr
        self.upsample_rate = upsample_rate
        self.wav_paths = []
        for directory in directorys:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".ogg") or file.endswith(".wav") or file.endswith(".mp3") or file.endswith(".m4a"):
                        self.wav_paths.append(os.path.join(root, file))
        random.shuffle(self.wav_paths)
        print(f"Found {len(self.wav_paths)} files in {directorys}.")

    def __len__(self):
        return len(self.wav_paths) * self.upsample_rate

    def split_dataset_by_node(self, rank=0, world_size=1):
        if world_size == 1:
            return self
        wav_paths = self.wav_paths[rank::world_size]
        return wav_paths


    def __getitem__(self, idx):
        idx = idx % len(self.wav_paths)
        wav_path = self.wav_paths[idx]
        try:
            speech, orig_sr = librosa.load(wav_path, sr=self.sr)
        except Exception as e:
            print(f"Failed to load wav file with error {e}")
            return self.__getitem__(random.randint(0, len(self)))
        if len(speech) < self.sr * duration_setting["min"] or len(speech) > self.sr * duration_setting["max"]:
            return self.__getitem__(random.randint(0, len(self)))
        return_dict = {
            'audio': speech,
            'sr': orig_sr
        }
        return return_dict

def build_emilia_preprocess(sr, spect_params=None):
    def emilia_preprocess(batch):
        orig_audio = batch['mp3']['array']
        orig_sr = batch['mp3']['sampling_rate']
        return_dict = {
            'audio': orig_audio,
            'sr': orig_sr,
            'text': batch['json']['text'],
            'language': batch['json']['language'],
        }
        return return_dict
    return emilia_preprocess

class PseudoDataset(IterableDataset):
    def __init__(self,
                 base_dataset,
                 spect_params=None,
                 sr=22050,
                 pad_to_multiple_of=None,
                 **dataset_config
                 ):
        self.base_dataset = base_dataset
        mel_fn_args = {
            "n_fft": spect_params['n_fft'],
            "win_size": spect_params['win_length'],
            "hop_size": spect_params['hop_length'],
            "num_mels": spect_params['n_mels'],
            "sampling_rate": sr,
            "fmin": spect_params['fmin'],
            "fmax": spect_params['fmax'],
            "center": False
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
        self.sr = sr
        self.prepend_language_token = dataset_config.get("prepend_language_token", False)
        self.language_token_offset = dataset_config.get("language_token_offset", 9000)
        self.min_size = dataset_config.get("min_duration", 0.2)
        self.max_size = dataset_config.get("max_duration", 45.0)
        self.pad_to_multiple_of = pad_to_multiple_of if pad_to_multiple_of is not None else 1
        from transformers import WhisperTokenizer
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")
        self.tokenizer_type = "whisper"

    def __iter__(self):
        # replace this with your own data loading
        for data in self.base_dataset:
            wave, orig_sr = data['audio'], data['sr']
            text, language = data['text'], data.get('language', 'en-us')
            try:
                text_ids = self.process_text(text, language)
            except ValueError as e:
                print(f"Skipping due to text processing error: {e}")
                continue
            if len(wave) > orig_sr * duration_setting['max'] or len(wave) < orig_sr * duration_setting['min']:
                print(f"Skipping due to duration {len(wave) / orig_sr} too long or too short")
                continue
            if orig_sr != self.sr:
                wave = librosa.resample(wave, orig_sr=orig_sr, target_sr=self.sr)
            if np.abs(wave).max() > 1.0:
                wave = wave / np.abs(wave).max()
            wave = torch.from_numpy(wave).float()
            if self.pad_to_multiple_of > 1:
                wave = torch.nn.functional.pad(
                    wave, (0, self.pad_to_multiple_of - (len(wave) % self.pad_to_multiple_of)),
                    mode='constant', value=0
                )
            mel = self.to_mel(wave[None]).squeeze(0)

            yield (
                "ANONYMOUS",
                mel,
                text_ids,
                None,
                wave,
                text,
            )
    def process_text(self, text, lang_id):
        lang_id = "en-us" if lang_id == "en" else lang_id
        lang_id = "fr" if lang_id == "fr-fr" else lang_id
        lang_id = "nl" if lang_id == "dutch" else lang_id
        lang_id = "zh-CN" if lang_id == "zh" else lang_id

        text = text.strip()
        text_wo_punct = "".join([c for c in text if c not in _punctuation])
        if repetition_found(text_wo_punct, length=4, tolerance=15):
            raise ValueError(f"Repetition found in text")
        text_cleaned = clean_text(text)
        if lang_id in ["zh-CN"]:
            text_cleaned = ChnNormedText(raw_text=text_cleaned).normalize()
        text_ids = self.tokenizer(text_cleaned).input_ids
        text_ids = torch.LongTensor(text_ids)
        return text_ids


def collate(batch):
    batch_size = len(batch)

    # sort by mel length
    lengths = [b[1].shape[1] for b in batch]
    batch_indexes = np.argsort(lengths)[::-1]
    batch = [batch[bid] for bid in batch_indexes]

    nmels = batch[0][1].size(0)
    max_mel_length = max([b[1].shape[1] for b in batch])
    max_text_length = max([b[2].shape[0] for b in batch])

    speaker_ids = [b[0] for b in batch]
    mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
    text_tensors = torch.zeros((batch_size, max_text_length)).long() + 50257  # eos of whisper tokenizer
    text_list = []

    input_lengths = torch.zeros(batch_size).long()
    output_lengths = torch.zeros(batch_size).long()
    waves = [None for _ in range(batch_size)]

    langs = torch.zeros((batch_size)).long()
    max_wave_length = max([len(b[-2]) for b in batch])
    wave_tensors = torch.zeros((batch_size, max_wave_length)).float()
    wave_lengths = torch.zeros(batch_size).long()

    for bid, (
            sid,
            mel,
            text_tensor,
            _,
            wave,
            text,
    ) in enumerate(batch):
        mel_size = mel.size(1)
        text_size = len(text_tensor)
        mels[bid, :, :mel_size] = mel
        text_tensors[bid, :text_size] = text_tensor
        text_list.append(text)
        input_lengths[bid] = text_size
        output_lengths[bid] = mel_size

        waves[bid] = wave

        # langs[bid] = lang_token

        wave_tensors[bid, : len(wave)] = torch.tensor(wave)
        wave_lengths[bid] = len(wave)

    return (
        text_list,
        text_tensors,
        input_lengths,
        mels,
        output_lengths,
        langs,
        wave_tensors,
        wave_lengths,
    )

def build_dataloader(
    rank=0,
    world_size=1,
    batch_size=32,
    num_workers=0,
    prefetch_factor=16,
    preprocess_params=None,
    epoch=0,
):
    sr = preprocess_params['sr']
    spect_params = preprocess_params['spect_params']
    emilia_preprocess = build_emilia_preprocess(sr, spect_params)
    emilia_dataset = datasets.load_dataset("amphion/Emilia-Dataset", data_files={"train": "Emilia/**/*.tar"}, streaming=True)['train'] # 548000k
    emilia_dataset = emilia_dataset.map(emilia_preprocess).shuffle(seed=epoch, buffer_size=5_000)
    emilia_dataset = split_dataset_by_node(emilia_dataset, rank=rank, world_size=world_size)
    emilia_dataset = ExceptionHandlingWrapper(emilia_dataset)

    base_dataset = ProbabilisticChainDataset([emilia_dataset],
                                                 [1.0])
    dataset = PseudoDataset(base_dataset, **preprocess_params)
    collate_fn = collate
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        # shuffle=True,
    )

    return data_loader

if __name__ == "__main__":
    import soundfile as sf
    from tqdm import tqdm
    target_dir = "../timbre_samples"
    os.makedirs(target_dir, exist_ok=True)
    # Create a DataLoader
    dataloader = build_dataloader(
        rank=0,
        world_size=1,
        batch_size=4,
        num_workers=4,
        prefetch_factor=2,
        preprocess_params={"sr": 22050, "spect_params": {"n_fft": 1024, "win_length": 1024, "hop_length": 256, "n_mels": 80, "fmin": 0, "fmax": None}},
    )

    for i, batch in enumerate(tqdm(dataloader)):
        # Train your model
        waves = batch[0]
        wave_lengths = batch[2]

        np_wave_list = []
        for idx, wave in enumerate(waves):
            np_wave_list.append(wave[:wave_lengths[idx]].numpy())

        for wave in np_wave_list:
            # randomly determine a file name
            file_name = f"{random.randint(0, 100000)}.wav"
            file_path = os.path.join(target_dir, file_name)
            # save the wave file
            sf.write(file_path, wave, 22050)

        if i == 10:
            break