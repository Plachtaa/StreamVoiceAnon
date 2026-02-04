# This code is based on
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py
import os
import random
import torchaudio
import speechbrain as sb


class ASVDatasetGenerator:

    def __init__(self, hparams):
        self.sample_rate = hparams['sample_rate']
        self.sentence_len = hparams['sentence_len']
        self.snt_len_sample = int(self.sample_rate * self.sentence_len)

        self.data_folder = hparams['data_folder']
        self.train_annotation = hparams['train_annotation']
        self.dev_annotation = hparams['valid_annotation']

        self.random_chunk = hparams['random_chunk']
        self.batch_size = hparams['batch_size']
        self.save_folder = hparams['save_folder']

        self.lab_enc_file = os.path.join(self.save_folder, 'label_encoder.txt')

    def dataio_prep(self):
        "Creates the datasets and their data processing pipelines."

        # 1. Declarations:
        train_dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=self.train_annotation,
            replacements={'data_root': self.data_folder})

        dev_dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=self.dev_annotation,
            replacements={'data_root': self.data_folder})

        datasets = [train_dataset, dev_dataset]

        # 2. Define audio pipeline:
        @sb.utils.data_pipeline.takes('wav', 'start', 'stop', 'duration')
        @sb.utils.data_pipeline.provides('sig')
        def audio_pipeline(wav, start, stop, duration):
            if self.random_chunk:
                duration_sample = int(float(duration) * self.sample_rate)
                start = random.randint(0, duration_sample - self.snt_len_sample)
                stop = start + self.snt_len_sample
            else:
                start = int(start)
                stop = int(stop)
            num_frames = stop - start
            sig, fs = torchaudio.load(wav, num_frames=num_frames, frame_offset=start)
            sig = sig.transpose(0, 1).squeeze(1)
            return sig
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

        label_encoder = sb.dataio.encoder.CategoricalEncoder()
        label_encoder.ignore_len()
        # 3. Define text pipeline:
        @sb.utils.data_pipeline.takes('spk_id')
        @sb.utils.data_pipeline.provides('spk_id', 'spk_id_encoded')
        def label_pipeline(spk_id):
            yield spk_id
            spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
            yield spk_id_encoded

        sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

        # 3. Fit encoder:
        # Load or compute the label encoder (with multi-GPU DDP support)
        label_encoder.load_or_create(
            path=self.lab_enc_file, from_didatasets=[train_dataset], output_key='spk_id')

        # 4. Set output:
        sb.dataio.dataset.set_output_keys(datasets, ['id', 'sig', 'spk_id_encoded'])

        return train_dataset, dev_dataset
