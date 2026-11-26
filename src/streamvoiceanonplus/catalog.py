"""Model catalog and checkpoint verification."""
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / 'resources'
DEFAULT_MODEL = 'frame-distill'
MODEL_CHOICES = ('base', 'frame-distill', 'pool-distill')


def read(name):
    return json.loads((RESOURCES / name).read_text())


def model_info(model=DEFAULT_MODEL):
    for info in read('models.json')['models']:
        if model in (info['id'], info['selector']):
            return info
    raise ValueError('Model must be one of ' + ', '.join(MODEL_CHOICES))


def model_selector(model):
    return model_info(model)['selector']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utterance_seed(utterance_id):
    if not utterance_id:
        raise ValueError('A stable, non-empty utterance ID is required')
    message = '2026|' + utterance_id + '|development-evaluation'
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], 'big')


def verify_components(directory):
    assets = [a for a in read('components.json')['assets'] if a['filename'] != 'dual_ar_delay_0_8.pth']
    for asset in assets:
        filename = asset['filename']
        path = Path(directory[filename]) if isinstance(directory, Mapping) else Path(directory) / filename
        if not path.is_file() or sha256(path) != asset['sha256']:
            raise ValueError('Missing or mismatched component: ' + filename)
        asset['path'] = str(path.resolve())
    return assets
