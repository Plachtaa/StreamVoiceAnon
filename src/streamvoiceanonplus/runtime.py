"""Measured file-streaming path, with packaged namespaces and strict loading."""
import importlib
import random
import tempfile
import threading
from pathlib import Path
from .catalog import DEFAULT_MODEL, RESOURCES, model_info, read, sha256, utterance_seed, verify_components
from .artifacts import inference_state, verify_model_artifact

_RNG_LOCK = threading.RLock()


class InferenceEngine:
    """One engine per serialized stream; complete file streaming returns 16-kHz audio.

    This wraps the measured three-frame processing path. It does not advertise a
    microphone callback, arbitrary incremental push API or measured live latency.
    """
    def __init__(self, checkpoint, components, model=DEFAULT_MODEL, compile_ar=True, *, compile_encoder=False, compile_decoder=False, fp16=False):
        import torch
        import yaml
        if not torch.cuda.is_available():
            raise RuntimeError('This reference profile requires a CUDA GPU; CPU inference is unvalidated')
        self.info = model_info(model)
        model_id = self.info['id']
        self.artifact = verify_model_artifact(checkpoint, model_id)
        self.component_audit = verify_components(components)
        self._lock = _RNG_LOCK
        self.compile_ar = compile_ar
        self.torch = torch
        backend = self.info['backend']
        from ._helper_loading import StrictHelpers
        native = importlib.import_module('streamvoiceanonplus._native.' + backend + '.evaluations.infer_arvc')
        artifact_kind = self.artifact['kind']

        class StrictWrapper(StrictHelpers, native.InferenceWrapper):
            def _load_checkpoint(wrapper, path):
                raw = torch.load(path, map_location='cpu', weights_only=artifact_kind == 'inference_export')
                state, omitted = inference_state(raw.get('net', raw), model_id)
                missing, unexpected = wrapper.model.load_state_dict(state, strict=False)
                if unexpected or any(not ('.kv_cache.' in k and k.endswith(('.k_cache', '.v_cache'))) for k in missing):
                    raise RuntimeError('Inference tensor mismatch: ' + repr((missing, unexpected)))
                wrapper.weight_audit['main'] = {'omitted': omitted, 'missing_runtime_caches': missing, 'unexpected': unexpected}

        config_dir = RESOURCES / 'configs' / backend
        model_yaml = 'firefly_arvc_bsq_8192_delay0_8.yaml' if backend == 'base' else 'modded_dac_arvc_bsq_8192_delay0_8.yaml'
        config = {'preprocess_params': {'sr': 44100, 'f0_enabled': False}, 'post_training_type': None if backend == 'base' else 'sft_sep_token', 'use_spk_embedding': True, 'emotion_loss_params': None, 'model_params': {'config_path': str(config_dir / model_yaml)}}
        helpers = {'style_encoder': ('campplus.yaml', 'campplus_cn_common.bin'), 'speech_tokenizer': ('causal-encoder-lfq-8192.yaml', 'asr_s2s_bsq_8192_causal_down_whisper.pth'), 'firefly': ('firefly_gan_vq.yaml', 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth'), 'timbre_encoder': ('sparktts_speaker_encoder.yaml', 'spark_speaker_encoder.pth')}
        verified_paths = {asset['filename']: asset['path'] for asset in self.component_audit}
        for key, (name, weight) in helpers.items():
            config[key] = {'config_path': str(config_dir / name), 'checkpoint_path': verified_paths[weight]}
        self._temporary = tempfile.TemporaryDirectory(prefix='sva-inference-config-')
        config_path = Path(self._temporary.name) / 'inference.yaml'
        config_path.write_text(yaml.safe_dump(config))
        # Native constructors do not initialize audit metadata; strict helper
        # loading records every exception (only deterministic Spark buffers).
        StrictWrapper.weight_audit = None
        instance = StrictWrapper.__new__(StrictWrapper)
        instance.weight_audit = {}
        with self._lock:
            StrictWrapper.__init__(instance, str(config_path), str(Path(checkpoint).resolve()), compile_ar=compile_ar, compile_encoder=compile_encoder, compile_decoder=compile_decoder, fp16=fp16)
        self.wrapper = instance

    def anonymize_file(self, source, reference, utterance_id):
        """Return mono float audio at 16kHz; seed/reset independently per file."""
        import numpy as np
        import torchaudio
        source, reference = Path(source), Path(reference)
        if not source.is_file() or not reference.is_file():
            raise FileNotFoundError('Source and reference must both exist')
        protocol = read('protocol.json')
        protocol['compile_ar'] = self.compile_ar
        seed = utterance_seed(utterance_id)
        with self._lock:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
            keys = ['delay', 'encode_window_frames', 'decode_window_frames', 'max_prompt_frames', 'max_seq_frames', 'buffer_frames', 'decode_chunk_frames']
            kwargs = {k: protocol[k] for k in keys}
            kwargs.update(alpha=protocol['alpha'], save_result=False)
            with self.torch.inference_mode():
                wave = self.wrapper.stream_infer(str(source), str(reference), **kwargs)
                audio = torchaudio.functional.resample(self.torch.from_numpy(wave.copy()), self.wrapper.sr, 16000).numpy()
            if audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) == 0:
                raise RuntimeError('Invalid generated waveform')
        return audio, {'model': self.info['id'], 'candidate_id': self.info['candidate_id'], 'artifact_sha256': self.artifact['sha256'], 'source_sha256': sha256(source), 'reference_sha256': sha256(reference), 'reference_matches_evaluation': sha256(reference) == protocol['reference_sha256'], 'utterance_id': utterance_id, 'seed': seed, 'sample_rate': 16000, 'protocol': protocol, 'weight_audit': self.wrapper.weight_audit}

    def close(self):
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
