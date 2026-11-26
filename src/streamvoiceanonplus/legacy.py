"""Compatibility entry points for StreamVoiceAnon file inference."""
import argparse
from pathlib import Path

from .catalog import DEFAULT_MODEL, MODEL_CHOICES, model_info, model_selector


COMPONENTS = {
    'speech_tokenizer': 'asr_s2s_bsq_8192_causal_down_whisper.pth',
    'style_encoder': 'campplus_cn_common.bin',
    'firefly': 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth',
    'timbre_encoder': 'spark_speaker_encoder.pth',
}
STREAM_DEFAULTS = {
    'encode_window_frames': 128,
    'decode_window_frames': 64,
    'max_prompt_frames': 256,
    'max_seq_frames': 768,
    'buffer_frames': 32,
    'decode_chunk_frames': 3,
}


def component_paths(config_path):
    import yaml

    with Path(config_path).open() as stream:
        config = yaml.safe_load(stream)
    try:
        return {filename: config[key]['checkpoint_path'] for key, filename in COMPONENTS.items()}
    except (KeyError, TypeError) as error:
        raise ValueError('The legacy configuration must specify all four shared checkpoint paths') from error


def references_for_model(model, ref_path, ref_crop_lengths, collate_type):
    if model == 'base':
        return ref_path, {'ref_crop_lengths': ref_crop_lengths, 'spk_emb_collate_type': collate_type}
    if isinstance(ref_path, (list, tuple)):
        if len(ref_path) != 1:
            raise ValueError('frame-distill and pool-distill require one reference. Use --model base for multiple references.')
        ref_path = ref_path[0]
    if ref_crop_lengths is not None or collate_type != 'concat_mel':
        raise ValueError('Reference cropping and collation require the baseline model')
    return ref_path, {}


def validate_controls(delay, alpha):
    if delay is not None and delay not in range(9):
        raise ValueError('delay must be between 0 and 8 frames')
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between 0 and 1')


class InferenceWrapper:
    """StreamVoiceAnon-style offline and file-streaming calls at the native sample rate."""

    def __init__(self, config_path=None, checkpoint_path=None, compile_encoder=False,
                 compile_decoder=False, compile_ar=False, fp16=False, *, model=DEFAULT_MODEL, components=None):
        from .runtime import InferenceEngine

        info = model_info(model)
        self.model_id = info['id']
        if components is None:
            components = component_paths(config_path) if config_path else 'pretrained_checkpoints'
        if checkpoint_path is None:
            name = 'dual_ar_delay_0_8.pth' if self.model_id == 'base' else info['checkpoint_file']
            checkpoint_path = Path('pretrained_checkpoints') / name
        self.engine = InferenceEngine(
            checkpoint_path, components, model=self.model_id, compile_ar=compile_ar,
            compile_encoder=compile_encoder, compile_decoder=compile_decoder, fp16=fp16,
        )
        self.sr = self.engine.wrapper.sr
        self.device = self.engine.wrapper.device

    def infer(self, src_path, ref_path, out_dir=None, output_path=None, delay=2,
              ref_crop_lengths=None, alpha=0.9, spk_emb_collate_type='concat_mel',
              save_result=True, **sampling_kwargs):
        validate_controls(delay, alpha)
        reference, native_options = references_for_model(self.model_id, ref_path, ref_crop_lengths, spk_emb_collate_type)
        with self.engine._lock, self.engine.torch.inference_mode():
            return self.engine.wrapper.infer(
                src_path, reference, out_dir=out_dir, output_path=output_path,
                delay=delay, alpha=alpha, save_result=save_result, **native_options, **sampling_kwargs,
            )

    def stream_infer(self, src_path, ref_path, out_dir=None, encode_window_frames=128,
                     decode_window_frames=64, max_prompt_frames=256, max_seq_frames=768,
                     buffer_frames=32, decode_chunk_frames=3, delay=2, ref_crop_lengths=None,
                     alpha=0.9, spk_emb_collate_type='concat_mel', save_result=True):
        validate_controls(delay, alpha)
        reference, native_options = references_for_model(self.model_id, ref_path, ref_crop_lengths, spk_emb_collate_type)
        with self.engine._lock, self.engine.torch.inference_mode():
            return self.engine.wrapper.stream_infer(
                src_path, reference, out_dir=out_dir, encode_window_frames=encode_window_frames,
                decode_window_frames=decode_window_frames, max_prompt_frames=max_prompt_frames,
                max_seq_frames=max_seq_frames, buffer_frames=buffer_frames, decode_chunk_frames=decode_chunk_frames,
                delay=delay, alpha=alpha, save_result=save_result, **native_options,
            )

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='StreamVoiceAnon+ file inference using StreamVoiceAnon flags')
    parser.add_argument('--src_path', required=True)
    parser.add_argument('--ref_path', nargs='+', required=True)
    parser.add_argument('--out_dir', default='audio_outputs')
    parser.add_argument('--model', type=model_selector, choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--components', help='Shared checkpoint directory. Overrides paths in --config_path.')
    parser.add_argument('--config_path', help='Legacy YAML with shared checkpoint paths')
    parser.add_argument('--compile', action='store_true', help='Compile the autoregressive decoder')
    parser.add_argument('--simulate_streaming', action='store_true')
    parser.add_argument('--delay', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--ref_crop_lengths', nargs='+', type=float)
    for field, default in STREAM_DEFAULTS.items():
        parser.add_argument('--' + field, type=int, default=default)
    args = parser.parse_args(argv)
    try:
        validate_controls(args.delay, args.alpha)
        references_for_model(args.model, args.ref_path, args.ref_crop_lengths, 'concat_mel')
    except ValueError as error:
        parser.error(str(error))
    reference = args.ref_path[0] if len(args.ref_path) == 1 else args.ref_path
    options = dict(out_dir=args.out_dir, delay=args.delay, alpha=args.alpha, ref_crop_lengths=args.ref_crop_lengths)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with InferenceWrapper(args.config_path, args.checkpoint_path, compile_ar=args.compile,
                          model=args.model, components=args.components) as wrapper:
        if args.simulate_streaming:
            options.update({field: getattr(args, field) for field in STREAM_DEFAULTS})
            wrapper.stream_infer(args.src_path, reference, **options)
        else:
            wrapper.infer(args.src_path, reference, **options)
