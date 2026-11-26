"""Inference-first CLI; --help and catalog commands need no model dependencies."""
import argparse
import json
from pathlib import Path
from .catalog import DEFAULT_MODEL, MODEL_CHOICES, model_info, model_selector, read, utterance_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description='StreamVoiceAnon and StreamVoiceAnon+ inference')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('models', help='Show the three frozen checkpoints and mean development and test scores')
    plan = sub.add_parser('plan', help='Inspect frozen inference settings; performs no inference')
    plan.add_argument('--utterance-id', default='example')
    infer = sub.add_parser('infer', help='Run measured file streaming on a CUDA worker')
    for key in ['checkpoint', 'components', 'source', 'reference', 'output', 'utterance-id']:
        infer.add_argument('--' + key, required=True)
    infer.add_argument('--no-compile', action='store_true', help='Disable autoregressive compilation')
    export = sub.add_parser('export', help='Export a tensor-only inference checkpoint')
    export.add_argument('--source', required=True)
    export.add_argument('--output', required=True)
    for command in (plan, infer, export):
        command.add_argument('--model', type=model_selector, choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    if args.command == 'models':
        print(json.dumps(read('models.json'), indent=2))
    elif args.command == 'plan':
        print(json.dumps({'model': model_info(args.model), 'protocol': read('protocol.json'), 'seed': utterance_seed(args.utterance_id), 'executes_inference': False}, indent=2))
    elif args.command == 'export':
        from .artifacts import export_checkpoint
        print(json.dumps(export_checkpoint(args.source, args.output, args.model), indent=2))
    else:
        output = Path(args.output)
        receipt = output.with_suffix('.json')
        if output.exists() or receipt.exists():
            parser.error('Output audio/receipt already exists; choose a new destination')
        from .runtime import InferenceEngine
        import soundfile as sf
        with InferenceEngine(args.checkpoint, args.components, model=args.model, compile_ar=not args.no_compile) as engine:
            audio, provenance = engine.anonymize_file(args.source, args.reference, args.utterance_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, 16000, subtype='PCM_16')
        receipt.write_text(json.dumps(provenance, indent=2) + '\n')
        print(str(output))
