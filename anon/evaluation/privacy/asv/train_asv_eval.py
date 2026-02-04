from copy import deepcopy
import torch

from .asv_train.train_speaker_embeddings import train_asv_speaker_embeddings

# For more information, see parse_arguments() in
# https://github.com/speechbrain/blob/develop/ speechbrain/speechbrain/core.py
run_opts = {
    'debug': False,
    'debug_batches': 2,
    'debug_epochs': 2,
    'debug_persistently': False,
    'device': 'cuda:0',
    'data_parallel_backend': False,
    'distributed_launch': False,
    'distributed_backend': 'nccl',
    'find_unused_parameters': False,
    'tqdm_colored_bar': False
}

from utils import setup_logger
logger = setup_logger(__name__)


def train_asv_eval(train_params, output_dir):
    backend = train_params.get('backend', 'speechbrain').lower()
    if backend == 'speechbrain':
        asv_train_speechbrain(train_params=train_params, output_dir=output_dir)
    else:
        raise ValueError(f'Unknown backend {backend} for ASR evaluation. Available backends: speechbrain.')


def asv_train_speechbrain(train_params, output_dir):
    logger.info(f'Train ASV model: {output_dir}')
    hparams = {
        'pretrained_path': str(train_params['pretrained_model']),
        'batch_size': train_params['batch_size'],
        'lr': train_params['lr'],
        'num_utt': train_params['num_utt'],
        'num_spk': train_params['num_spk'],
        'utt_selected_ways': train_params['utt_selection'],
        'number_of_epochs': train_params['epochs'],
        'data_folder': str(train_params['train_data_dir']),
        'output_folder': str(output_dir),
        'num_workers': train_params['num_workers']
    }

    config = train_params['train_config']

    if train_params['num_spk'] == 'ALL':
        hparams['out_n_neurons'] = 921
    else:
        hparams['out_n_neurons'] = int(train_params['num_spk'])
    sb_run_opts = deepcopy(run_opts)
    if torch.cuda.device_count() > 0:
        sb_run_opts['data_parallel_backend'] = True
    train_asv_speaker_embeddings(config, hparams, run_opts=sb_run_opts)
