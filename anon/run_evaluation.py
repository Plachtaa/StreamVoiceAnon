# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch, so we will read all arguments directly on startup
import os
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import multiprocessing
import time
import shutil
import itertools
from datetime import datetime

parser = ArgumentParser()
parser.add_argument('--config', default='config_eval.yaml')
parser.add_argument('--overwrite', type=str, default='{}')
parser.add_argument('--force_compute', type=str, default='False')
parser.add_argument('--gpu_ids', default='0')
args = parser.parse_args()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
else: # CUDA_VISIBLE_DEVICES more important than the gpu_ids arg
    args.gpu_ids = ",".join([ str(i) for i, _ in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(","))])

import torch

from evaluation import evaluate_asv, train_asv_eval, evaluate_asr, evaluate_ser
from utils import parse_yaml, scan_checkpoint, combine_asr_data, \
                   save_yaml, check_dependencies, setup_logger, check_kaldi_formart_data

logger = setup_logger(__name__)

def get_evaluation_steps(params):
    eval_steps = {}
    if 'privacy' in params:
        eval_steps['privacy'] = list(params['privacy'].keys())
    if 'utility' in params:
        eval_steps['utility'] = list(params['utility'].keys())

    if 'eval_steps' in params:
        param_eval_steps = params['eval_steps']

        for eval_part, eval_metrics in param_eval_steps.items():
            if eval_part not in eval_steps:
                raise KeyError(f'Unknown evaluation step {eval_part}, please specify in config')
            for metric in eval_metrics:
                if metric not in eval_steps[eval_part]:
                    raise KeyError(f'Unknown metric {metric}, please specify in config')
        return param_eval_steps
    return eval_steps


def save_result_summary(out_path, results_dict, config):
    out_dir = Path(os.path.dirname(out_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, out_dir / 'config.yaml')

    with open(out_path, 'w') as f:
        f.write(f'---- Time: {datetime.strftime(datetime.today(), "%d-%m-%y_%H:%M")} ----\n')
        if 'ser' in results_dict:
            f.write('\n')
            f.write('---- SER results ----\n')
            f.write(results_dict['ser'].sort_values(by=['dataset', 'split']).to_string())
            f.write('\n')
        if 'asv' in results_dict:
            f.write('\n')
            asv_results = results_dict['asv']
            # if EERs computed by ASV_eval, only keep the results of OO condition
            if 'orig' in os.path.basename(out_path):
                f.write('---- ASV_eval results ----\n')
                asv_results_filter = asv_results[((asv_results['enrollment'] == 'original') & (asv_results['trial'] == 'original'))]

            # if EERs computed by ASV_eval^anon, only keep the results of AA condition
            elif 'anon' in os.path.basename(out_path):
                f.write('---- ASV_eval^anon results ----\n')
                asv_results_filter = asv_results[((asv_results['enrollment'] == 'anon') & (asv_results['trial'] == 'anon'))]

            f.write(asv_results_filter.sort_values(by=['dataset', 'split']).to_string())
            f.write('\n')
        if 'asr' in results_dict:
            f.write('\n')
            f.write('---- ASR results ----\n')
            asr_results = results_dict['asr']
            asr_results_filter = asr_results[~(asr_results['dataset'] == 'IEMOCAP')]
            f.write(asr_results_filter.sort_values(by=['dataset', 'split']).to_string())
            f.write('\n')


if __name__ == '__main__':
    check_dependencies('requirements.txt')
    multiprocessing.set_start_method("fork",force=True)

    params = parse_yaml(Path(args.config), overrides=json.loads(args.overwrite))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    eval_data_dir = params['data_dir']
    anon_suffix = params['anon_data_suffix']
    eval_steps = get_evaluation_steps(params)
    check_kaldi_formart_data(params)
    results = {}

    # make sure given paths exist
    assert eval_data_dir.exists(), f'{eval_data_dir} does not exist'

    if 'privacy' in eval_steps:
        if 'asv' in eval_steps['privacy']:
            asv_params = params['privacy']['asv']
            if 'training' in asv_params:
                model_dir = params['privacy']['asv']['training']['model_dir']
                asv_train_params = asv_params['training']
                if not model_dir.exists() or asv_train_params.get('retrain', True) is True:
                    start_time = time.time()
                    logger.info('====================')
                    logger.info('Perform ASV training')
                    logger.info('====================')
                    if args.force_compute.lower() == "true":
                        shutil.rmtree(model_dir, ignore_errors=True)
                    train_asv_eval(train_params=asv_train_params, output_dir=model_dir)
                    logger.info("ASV training time: %f min ---" % (float(time.time() - start_time) / 60))
                    model_dir = scan_checkpoint(model_dir, 'CKPT')
                    shutil.copy(asv_train_params['train_config'], model_dir)
                    shutil.copy(asv_train_params['infer_config'], model_dir)

            if 'evaluation' in asv_params:
                logger.info('======================')
                logger.info('Perform ASV evaluation')
                logger.info('======================')
                model_dir = params['privacy']['asv']['evaluation']['model_dir']
                results_dir = params['privacy']['asv']['evaluation']['results_dir']
                if args.force_compute.lower() == "true":
                    for info in os.walk(results_dir / f"{params['privacy']['asv']['evaluation']['distance']}_out"):
                        dir, _, _ = info
                        if anon_suffix in dir:
                            shutil.rmtree(dir, ignore_errors=True)
                model_dir = scan_checkpoint(model_dir, 'CKPT+') or model_dir
                start_time = time.time()
                eval_data_name = params['privacy']['asv']['dataset_name']
                eval_pairs = []
                for name in eval_data_name:
                    for d in params['datasets']:
                        if d['name'] != name:
                            continue
                        if 'enrolls' not in d or 'trials' not in d:
                            raise ValueError(f"{name} is missing an ASV enrolls/trials split")
                        eval_pairs.extend([(f'{d["data"]}{enroll}', f'{d["data"]}{trial}')
                                          for enroll, trial in itertools.product(d['enrolls'], d['trials'])])
                asv_results = evaluate_asv(eval_datasets=eval_pairs, eval_data_dir=eval_data_dir,
                                           params=asv_params, device=device,  model_dir=model_dir,
                                           anon_data_suffix=anon_suffix)
                results['asv'] = asv_results
                logger.info("--- EER computation time: %f min ---" % (float(time.time() - start_time) / 60))

    if 'utility' in eval_steps:
        if 'ser' in eval_steps['utility']:
            logger.info('======================')
            logger.info('Perform SER evaluation')
            logger.info('======================')
            eval_data_name = params['utility']['ser']['dataset_name']
            ser_eval_params = params['utility']['ser']['evaluation']
            models_path = ser_eval_params['model_dir']

            eval_ser = []
            for name in eval_data_name:
                for d in params['datasets']:
                    if d['name'] != name:
                        continue
                    eval_ser.append(d['data'])
            start_time = time.time()
            ser_results = evaluate_ser(eval_ser, eval_data_dir, models_path, anon_data_suffix=anon_suffix, params=ser_eval_params, device=device)
            results['ser'] = ser_results
            logger.info("--- SER evaluation time: %f min ---" % (float(time.time() - start_time) / 60))

        if 'asr' in eval_steps['utility']:
            asr_params = params['utility']['asr']
            backend = asr_params.get('backend', 'speechbrain').lower()
            if 'evaluation' in asr_params:
                asr_eval_params = asr_params['evaluation']
                asr_eval_params["device"] = device

                start_time = time.time()
                logger.info('======================')
                logger.info('Perform ASR evaluation')
                logger.info('======================')
                eval_data_name = params['utility']['asr']['dataset_name']
                eval_asr = []
                for name in eval_data_name:
                    for d in params['datasets']:
                        if d['name'] != name:
                            continue
                        for suff in  ["", anon_suffix]:
                            splits_to_combine = []
                            if 'trials' in d:
                                splits_to_combine += list(map(lambda x: str(params['data_dir'])+"/"+d['data']+x+suff, d['trials']))
                                combine_asr_data(splits_to_combine, str(params['data_dir'])+"/"+d['name']+"_asr"+suff)

                        if 'trials' not in d and 'enrolls' not in d:
                            eval_asr.append(d['data'])
                        else:
                            eval_asr.append(d['name']+"_asr")

                if args.force_compute.lower() == "true":
                    results_dir = params['utility']['asr']['evaluation']['results_dir']
                    for d in eval_asr:
                        shutil.rmtree(Path(results_dir / Path(str(d) + str(anon_suffix))), ignore_errors=True)

                asr_results = evaluate_asr(eval_datasets=eval_asr, eval_data_dir=eval_data_dir,
                                           params=asr_eval_params,
                                           anon_data_suffix=anon_suffix, device=device, backend=backend)
                results['asr'] = asr_results
                logger.info("--- ASR evaluation time: %f min ---" % (float(time.time() - start_time) / 60))

    if results:
        now = datetime.strftime(datetime.today(), "%d-%m-%y_%H:%M")
        results_summary_path = params.get('results_summary_path', Path('exp', 'results_summary', now))
        save_result_summary(out_path=results_summary_path, results_dict=results, config=params)
