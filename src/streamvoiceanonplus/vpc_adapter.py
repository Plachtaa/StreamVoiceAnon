"""Thin subprocess adapter to the measured external VPC checkout; no training."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from .catalog import read, sha256


def verify_checkout(checkout):
    root = Path(checkout).resolve()
    pin = read('vpc-pin.json')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != pin['commit']:
        raise ValueError('External evaluator commit differs from the measurement pin')
    changed = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no', '--', 'anon'], text=True)
    if changed.strip():
        raise ValueError('External evaluator has tracked source changes')
    for name, expected in pin['files'].items():
        path = root / name
        if not path.is_file():
            raise ValueError('External evaluator source missing: ' + name)
        # The pinned files are text. Git can check them out with LF or CRLF.
        source = path.read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError('External evaluator source mismatch: ' + name)
    return {'commit': head, 'verified_source_files': len(pin['files']), 'training_enabled': False}


def template(data_root, weights_root, results_root):
    data, weights, results = [Path(p).resolve() for p in [data_root, weights_root, results_root]]
    return {'data_dir': str(data), 'anon_data_suffix': '_sva', 'results_summary_path': str(results / 'summary/results_orig.txt'), 'datasets': [{'name': 'IEMOCAP_dev', 'data': 'IEMOCAP_dev'}, {'name': 'libri_dev', 'data': 'libri_dev', 'enrolls': ['_enrolls'], 'trials': ['_trials_f', '_trials_m']}], 'eval_steps': {'privacy': ['asv'], 'utility': ['asr', 'ser']}, 'privacy': {'asv': {'dataset_name': ['libri_dev'], 'model_name': 'asv_orig', 'model_type': 'ecapa', 'evaluation': {'model_dir': str(weights / 'exp/asv_orig'), 'results_dir': str(results / 'asv_orig'), 'distance': 'cosine', 'plda': {}}}}, 'utility': {'ser': {'dataset_name': ['IEMOCAP_dev'], 'evaluation': {'model_dir': str(weights / 'exp/ser'), 'results_dir': str(results / 'ser')}}, 'asr': {'dataset_name': ['libri_dev'], 'evaluation': {'model_type': 'EncoderASR', 'model_dir': str(weights / 'exp/asr'), 'eval_batchsize': 8, 'results_dir': str(results / 'asr')}}}}


def validate_config(config, weights_root):
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if any(word in key.lower() for word in ['train', 'retrain', 'finetun']):
                    raise ValueError('Training configuration is disabled: ' + key)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(config)
    if {x['data'] for x in config['datasets']} != {'IEMOCAP_dev', 'libri_dev'}:
        raise ValueError('This adapter supports the fixed development sets')
    expected = template(config['data_dir'], weights_root, Path(config['results_summary_path']).parent.parent)
    if config != expected:
        raise ValueError('Use the generated, fixed development evaluation configuration')
    return expected


def verify_weights(root):
    for asset in read('evaluator-weights.json'):
        path = Path(root) / asset['file']
        if not path.is_file() or sha256(path) != asset['sha256']:
            raise ValueError('Missing or mismatched frozen evaluator weight: ' + asset['file'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pinned external VPC adapter — no training or attacker adaptation')
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--checkout', required=True)
    build = commands.add_parser('template')
    for key in ['data-root', 'weights-root', 'results-root']:
        build.add_argument('--' + key, required=True)
    run = commands.add_parser('run')
    for key in ['checkout', 'config', 'weights-root']:
        run.add_argument('--' + key, required=True)
    run.add_argument('--execute', action='store_true', help='Otherwise verify and print the command only')
    args = parser.parse_args(argv)
    if args.command == 'template':
        print(json.dumps(template(args.data_root, args.weights_root, args.results_root), indent=2))
        return
    proof = verify_checkout(args.checkout)
    if args.command == 'verify':
        print(json.dumps(proof, indent=2))
        return
    config_path = Path(args.config).resolve()
    config = validate_config(json.loads(config_path.read_text()), args.weights_root)
    verify_weights(args.weights_root)
    command = [sys.executable, 'run_evaluation.py', '--config', str(config_path), '--force_compute', 'False', '--gpu_ids', '0']
    if not args.execute:
        print(json.dumps({'external_command': command, 'cwd': str(Path(args.checkout).resolve() / 'anon'), 'config_validated': True, **proof}, indent=2))
        return
    results = Path(config['results_summary_path']).parent.parent
    if results.exists():
        raise FileExistsError('Use a new results directory; do not reuse another inference run')
    for folder in ['summary', 'ser', 'asr', 'asv_orig']:
        (results / folder).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=Path(args.checkout).resolve() / 'anon', check=True)


if __name__ == '__main__':
    main()
