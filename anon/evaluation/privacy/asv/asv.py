# This code is partly based on
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/speaker_verification_plda.py
from pathlib import Path
import torch
from speechbrain.utils.metric_stats import EER
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd

from .utils import PLDAModel
from .speaker_extraction import SpeakerExtraction
from utils import write_table, read_kaldi_format, save_kaldi_format, setup_logger

logger = setup_logger(__name__)

class ASV:

    def __init__(self, model_dir, device, score_save_dir, distance='plda', plda_settings=None, vec_type='xvector'):
        self.device = device
        self.vec_type = vec_type
        self.model_dir = model_dir
        self.score_save_dir = score_save_dir
        self.distance = distance

        if plda_settings:
            self.plda_model_dir = plda_settings['model_dir']
            self.plda_train_data_dir = plda_settings['train_data_dir']
            self.plda_anon = plda_settings['anon']  # whether this model is trained on anon data or original
        else:
            if self.distance == 'plda':
                raise KeyError('PLDA settings must be given in config when using distance=plda!')
            self.plda_model_dir = None
            self.plda_train_data_dir = None
            self.plda_anon = None

        self.extractor = SpeakerExtraction(results_dir=self.score_save_dir / 'emb_xvect',
                                           devices=[self.device],
                                           settings={'vec_type': vec_type, 'emb_level': 'utt', 'emb_model_path': model_dir})

    def compute_trial_scores(self, trials, enrol_indices, test_indices, out_file, sim_scores):
        scores = []

        for enrol_id, test_id in trials:
            enrol_index = enrol_indices[enrol_id]
            test_index = test_indices[test_id]
            s = float(sim_scores[enrol_index, test_index])
            scores.append([enrol_id, test_id, s])

        score_data = pd.DataFrame(scores, columns=['enroll_id', 'trial_id', 'score'])
        write_table(filename=out_file, table=score_data, sep=' ')

        return score_data

    def _split_scores(self, scores, labels):
        positive_scores = []
        negative_scores = []

        for enrol_id, test_id, score in scores:
            label = labels[(enrol_id, test_id)]
            if label == 1:
                positive_scores.append(score)
            else:
                negative_scores.append(score)

        return positive_scores, negative_scores

    def select_data_for_plda(self, all_data_dir, selected_data_dir, out_dir):
        def change_id_format(data_dict):
            return {k.replace('--', '-'): v for k, v in data_dict.items()}

        df = pd.read_csv(selected_data_dir / 'train.csv', sep=',')
        selected_utts = set([change_id_format(segment.split('_')[0]) for segment in df['ID'].to_list()])

        wav_scp = read_kaldi_format(all_data_dir / 'wav.scp')
        utt2spk = change_id_format(read_kaldi_format(all_data_dir / 'utt2spk'))
        spk2gender = change_id_format(read_kaldi_format(all_data_dir / 'spk2gender'))

        selected_wav_scp = {utt: wav for utt, wav in wav_scp.items() if utt in selected_utts}
        selected_utt2spk = {utt: spk for utt, spk in utt2spk.items() if utt in selected_utts}
        selected_spk2gender = {spk: spk2gender[spk] for spk in set(selected_utt2spk.values())}

        out_dir.mkdir(parents=True, exist_ok=True)
        save_kaldi_format(selected_wav_scp, out_dir / 'wav.scp')
        save_kaldi_format(selected_utt2spk, out_dir / 'utt2spk')
        save_kaldi_format(selected_spk2gender, out_dir / 'spk2gender')

    def eer_compute(self, enrol_dir, test_dir, trial_runs_file):
        # Compute all enrol(spk level) and Test(utt level) embeddings
        # enroll vectors are the speaker-level average vectors
        enrol_all_dict = self.extractor.extract_speakers(dataset_path=Path(enrol_dir), emb_level='spk')
        test_all_dict = self.extractor.extract_speakers(dataset_path=Path(test_dir), emb_level='utt')

        enrol_vectors = []
        enrol_ids = []
        test_vectors = []
        test_ids = []
        trials = {}
        # 1462 1462-170142-0000 target
        # 1462 2412-153948-0004 nontarget
        with open(trial_runs_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                if temp[0] not in set(enrol_ids):
                    enrol_vectors.append(enrol_all_dict.get_embedding_for_identifier(temp[0]))
                    enrol_ids.append(temp[0])
                if temp[1] not in set(test_ids):
                    test_vectors.append(test_all_dict.get_embedding_for_identifier(temp[1]))
                    test_ids.append(temp[1])
                trials[(temp[0], temp[1])] = int(temp[2] == 'target')

        enrol_vectors = torch.stack(enrol_vectors)
        test_vectors = torch.stack(test_vectors)

        save_dir = Path(self.score_save_dir.parent, f'{Path(enrol_dir).name}-{Path(test_dir).name}')
        save_dir.mkdir(exist_ok=True)

        sim_scores, enrol_indices, test_indices = self.compute_distances(enrol_vectors=enrol_vectors,
                                                                         enrol_ids=enrol_ids,
                                                                         test_vectors=test_vectors, test_ids=test_ids)

        trial_scores = self.compute_trial_scores(trials=trials.keys(), enrol_indices=enrol_indices,
                                                 test_indices=test_indices, sim_scores=sim_scores,
                                                 out_file=save_dir / 'scores')

        positive_scores, negative_scores = self._split_scores(trial_scores.values, trials)
        del sim_scores


        # Final EER computation
        eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
        # min_dcf, th = minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores))
        with open(save_dir / 'EER', 'w') as f:
            f.write(str(eer))
        return eer

    def compute_distances(self, enrol_vectors, enrol_ids, test_vectors, test_ids):
        if self.distance == 'plda':
            """Computes the Equal Error Rate give the PLDA scores"""
            # Create ids, labels, and scoring list for EER evaluation
            if self.plda_model_dir.exists():
                self.plda = PLDAModel(train_embeddings=None, results_path=self.plda_model_dir)
            else:
                logger.info('Train PLDA model...')

                plda_data_dir = self.plda_train_data_dir
                if self.plda_anon:
                    plda_data_dir = Path(f'{plda_data_dir}_selected')
                    self.select_data_for_plda(all_data_dir=self.plda_train_data_dir,
                                              selected_data_dir=self.model_dir.parent,
                                              out_dir=plda_data_dir)
                logger.info(f'Using data under {plda_data_dir}')

                train_dict = self.extractor.extract_speakers(dataset_path=plda_data_dir, emb_level='utt')
                self.plda = PLDAModel(train_embeddings=train_dict, results_path=self.plda_model_dir)

            plda_score_object = self.plda.compute_distance(enrollment_vectors=enrol_vectors, enrollment_ids=enrol_ids,
                                                           trial_vectors=test_vectors, trial_ids=test_ids,
                                                           return_object=True)
            sim_scores = plda_score_object.scoremat
            enrol_indices = dict(zip(plda_score_object.modelset, range(len(enrol_ids))))
            test_indices = dict(zip(plda_score_object.segset, range(len(test_ids))))

        else:
            """Computes the Equal Error Rate give the cosine score"""
            sim_scores = 1 - cosine_distances(X=enrol_vectors.cpu(), Y=test_vectors.cpu())
            enrol_indices = dict(zip(enrol_ids, range(len(enrol_ids))))
            test_indices = dict(zip(test_ids, range(len(test_ids))))

        return sim_scores, enrol_indices, test_indices
