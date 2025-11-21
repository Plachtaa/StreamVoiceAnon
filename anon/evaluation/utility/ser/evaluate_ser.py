import torch
import torchaudio
import tqdm
import pandas as pd
import warnings

from pathlib import Path
from sklearn.metrics import recall_score, accuracy_score
from speechbrain.pretrained.interfaces import foreign_class
from utils import read_kaldi_format, scan_checkpoint, setup_logger

logger = setup_logger(__name__)


class FoldSERDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = []
        utt2spk = read_kaldi_format(data_path / "utt2spk")
        for utt_id, wav_file in read_kaldi_format(data_path / "wav.scp").items():
            wav, sr = torchaudio.load(str(wav_file))
            wav_len = wav.shape
            spk = utt2spk[utt_id]
            data.append((utt_id, spk, wav, wav_len))

        # Sort the data based on audio length
        self.data = sorted(data, key=lambda x: x[3], reverse=True)

    def __getitem__(self, idx):
        wavname, spk, wav, wav_len = self.data[idx]
        return wavname, spk, wav, wav_len

    def __len__(self):
        return len(self.data)

@torch.no_grad()
def evaluate_ser(eval_datasets, eval_data_dir, models_path, anon_data_suffix, params, device):
    results_dir = params['results_dir']
    test_sets = eval_datasets + [f'{dataset}{anon_data_suffix}' for dataset in eval_datasets]
    results = []
    classifiers = {}
    logger.info(f"Emotion recognition on {','.join(test_sets)}")
    for test_set in tqdm.tqdm(test_sets):
        data_path = eval_data_dir / test_set
        dataset = FoldSERDataset(data_path)

        utt2emo = read_kaldi_format(data_path/  'utt2emo')
        for spkfold, fold in read_kaldi_format(data_path/  'spk2fold').items():
            if fold not in classifiers:
                model_dir = models_path/ f"fold_{fold}"
                model_dir = scan_checkpoint(model_dir, 'CKPT')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    classifiers[fold] = foreign_class(
                        source=model_dir,
                        savedir=model_dir,
                        run_opts={'device': device},
                        classname="CustomEncoderWav2vec2Classifier",
                        pymodule_file="custom_interface.py",
                    )
                classifiers[fold].hparams.label_encoder.ignore_len()

            hyp = []
            ref = []
            per_emo = {}
            for uttid, spkid, wav, wav_len in tqdm.tqdm(dataset):
                if spkid != spkfold:
                    continue

                out_prob, score, index, text_lab = classifiers[fold].classify_batch(wav)
                hyp += [classifiers[fold].hparams.label_encoder.lab2ind[text_lab[0]]]
                ref += [classifiers[fold].hparams.label_encoder.lab2ind[utt2emo[uttid]]]
                if utt2emo[uttid] not in per_emo:
                    per_emo[utt2emo[uttid]] = {"hyp": [], "ref": []}
                per_emo[utt2emo[uttid]]["hyp"].append([classifiers[fold].hparams.label_encoder.lab2ind[text_lab[0]]])
                per_emo[utt2emo[uttid]]["ref"].append([classifiers[fold].hparams.label_encoder.lab2ind[utt2emo[uttid]]])

            score = recall_score(y_true=ref, y_pred=hyp, average= "macro") * 100
            score = round(score, 3)
            score_per_emo = {}
            for k,v in per_emo.items():
                score_per_emo[f"ACC_{k}"] = round(accuracy_score(y_true=v["ref"], y_pred=v["hyp"]) * 100, 3)


            test_set_info = test_set.split('_')
            if len(test_set_info) == 1:
                test_set_info.append("_")
            results.append({'dataset': test_set_info[0], 'split': test_set_info[1], 'fold': fold,
                            'ser': 'anon' if anon_data_suffix in test_set else 'original', 'UAR': score, **score_per_emo})
            print(f'{test_set} fold: {fold} - UAR: {score}')
    results_df = pd.DataFrame(results)
    print(results_df)
    #result_mean = results_df.groupby(['dataset', 'split', 'ser'])['UAR'].agg(['mean', 'min', 'max'])
    result_mean = results_df.groupby(['dataset', 'split', 'ser']).agg({'UAR': ['mean'], **{k: ['mean'] for k in score_per_emo.keys()}})
    result_mean.reset_index(inplace=True)
    print(result_mean)

    results_df.to_csv(results_dir / f'results_folds{anon_data_suffix}.csv')
    result_mean.to_csv(results_dir / f'results{anon_data_suffix}.csv')
    return result_mean
