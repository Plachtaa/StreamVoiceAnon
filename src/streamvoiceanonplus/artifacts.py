"""Export only inference tensors from the exact selected, trusted checkpoints."""
import json
from pathlib import Path
from .catalog import DEFAULT_MODEL, model_info, sha256

AUXILIARY = ('decoder.model.emotion_classifier.', 'decoder.model.feature_extraction_emo.', 'decoder.model.linear_emo.')
SEP = {'decoder.content_sep_prompt_token_embedding.weight', 'decoder.audio_sep_prompt_token_embedding.weight'}


def inference_state(state, model):
    """Drop only declared auxiliary heads/runtime caches; never filter by shape."""
    result, omitted = {}, []
    for original, value in state.items():
        key = original.removeprefix('module.')
        if key in result:
            raise ValueError('Duplicate tensor after prefix normalization: ' + key)
        if key.startswith(AUXILIARY) or ('.kv_cache.' in key and key.endswith(('.k_cache', '.v_cache'))):
            omitted.append(key)
        else:
            result[key] = value
    if model != 'base' and not SEP <= result.keys():
        raise ValueError('Selected Plus checkpoint is missing learned SEP embeddings')
    return result, omitted


def export_checkpoint(source, destination, model=DEFAULT_MODEL):
    info = model_info(model)
    model_id = info['id']
    source, destination = Path(source), Path(destination)
    if not source.is_file() or sha256(source) != info['checkpoint_sha256']:
        raise ValueError('Source hash does not match the selected checkpoint')
    if destination.exists() or destination.with_suffix('.receipt.json').exists():
        raise FileExistsError('Export destination/receipt already exists; preserve prior artifacts')
    import torch
    # Only the pinned source hash above permits loading resumable research state.
    raw = torch.load(source, map_location='cpu', weights_only=False)
    state, omitted = inference_state(raw.get('net', raw), model_id)
    if not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError('Inference export must contain tensors only')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    torch.save({k: v.detach().cpu().contiguous() for k, v in state.items()}, temporary)
    temporary.replace(destination)
    receipt = {'schema': 'sva-inference-export-v1', 'model': model_id, 'candidate_id': info['candidate_id'], 'source_sha256': info['checkpoint_sha256'], 'inference_sha256': sha256(destination), 'tensor_count': len(state), 'omitted_keys': omitted, 'SEP_retained': model_id == 'base' or SEP <= state.keys(), 'optimizer_scheduler_RNG_state_included': False, 'GPU_parity_validated': False}
    destination.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def verify_model_artifact(path, model=DEFAULT_MODEL):
    path = Path(path)
    info = model_info(model)
    checksum = sha256(path)
    if checksum == info['checkpoint_sha256']:
        return {'kind': 'pinned_research_checkpoint', 'sha256': checksum}
    receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    if receipt.get('schema') != 'sva-inference-export-v1' or receipt.get('model') != info['id'] or receipt.get('candidate_id') != info['candidate_id'] or receipt.get('source_sha256') != info['checkpoint_sha256'] or receipt.get('inference_sha256') != checksum:
        raise ValueError('Inference artifact receipt/hash does not match the selected model')
    return {'kind': 'inference_export', 'sha256': checksum, 'receipt': receipt}
