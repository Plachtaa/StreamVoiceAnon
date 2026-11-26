import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamvoiceanonplus.catalog import model_info, read, sha256, utterance_seed
from streamvoiceanonplus.cli import main
from streamvoiceanonplus.artifacts import SEP, inference_state, export_checkpoint, verify_model_artifact
from streamvoiceanonplus.vpc_adapter import template, validate_config, verify_checkout


class ReleaseTests(unittest.TestCase):
    def test_exact_frozen_selection_and_default(self):
        catalog = read('models.json')
        self.assertEqual(catalog['default'], 'frame')
        self.assertEqual(catalog['default_selector'], 'frame-distill')
        self.assertEqual({m['candidate_id'] for m in catalog['models']}, {'base-original-a09-full-dev', 'highlr15-frame-u0350', 'histlr-pool-u0700'})
        self.assertEqual(model_info('frame')['checkpoint_sha256'], '63b610aabea3cb7c4b48704399d49b84aa2a2728e6cba5247e49839245b775e4')
        self.assertEqual(model_info('pool')['checkpoint_sha256'], 'cb337c1b418ad93ee7fe16ed95f91ed635edf39c6d539483b2946c94854b5222')
        self.assertEqual(model_info('frame-distill'), model_info('frame'))
        self.assertEqual(model_info('pool-distill'), model_info('pool'))
        self.assertEqual([m['metric_scope'] for m in catalog['models']], ['dev_test_mean'] * 3)

    def test_reported_scores_for_the_frozen_checkpoints(self):
        expected = {
            'base': (4.554023001011869, 39.841, 46.9065),
            'frame-distill': (6.0031422851826015, 51.2236, 47.26125),
            'pool-distill': (5.474406147213136, 48.3183, 46.10275),
        }
        for selector, values in expected.items():
            with self.subTest(model=selector):
                scores = model_info(selector)['metrics']
                for metric, value in zip(('WER', 'UAR', 'EER_L'), values):
                    self.assertAlmostEqual(scores[metric], value)
                self.assertAlmostEqual(scores['EER_L'], (scores['EER_f'] + scores['EER_m']) / 2)
                self.assertLess(abs(sum(scores['recalls'].values()) / 4 - scores['UAR']), 0.001)

    def test_cli_plan_does_not_load_models(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            main(['plan', '--utterance-id', 'example'])
        result = json.loads(stream.getvalue())
        self.assertEqual(result['model']['id'], 'frame')
        self.assertEqual(result['model']['selector'], 'frame-distill')
        self.assertFalse(result['executes_inference'])
        self.assertEqual(result['protocol']['mixing_scope'], 'both')
        self.assertEqual(result['protocol']['alpha'], 0.9)
        for name in ('pool-distill', 'pool'):
            with self.subTest(model=name), contextlib.redirect_stdout(io.StringIO()) as stream:
                main(['plan', '--model', name])
                self.assertEqual(json.loads(stream.getvalue())['model']['selector'], 'pool-distill')

    def test_unknown_model_and_empty_id_fail(self):
        with self.assertRaises(ValueError):
            model_info('pilot')
        with self.assertRaises(ValueError):
            utterance_seed('')
        self.assertEqual(utterance_seed('example'), utterance_seed('example'))
        self.assertNotEqual(utterance_seed('example'), utterance_seed('example2'))

    def test_export_keeps_sep_and_unknown_inference_keys(self):
        weights = {key: object() for key in SEP}
        weights.update({'module.decoder.model.weight': object(), 'decoder.model.emotion_classifier.weight': object(), 'decoder.model.layer.kv_cache.k_cache': object(), 'unexpected.weight': object()})
        result, omitted = inference_state(weights, 'frame')
        self.assertTrue(SEP <= result.keys())
        self.assertIn('unexpected.weight', result)  # strict runtime must reject it
        self.assertIn('decoder.model.weight', result)
        self.assertEqual(len(omitted), 2)
        for key in SEP:
            self.assertIs(result[key], weights[key])

    def test_missing_sep_and_prefix_collisions_fail(self):
        with self.assertRaises(ValueError):
            inference_state({'weight': object()}, 'pool')
        with self.assertRaises(ValueError):
            inference_state({'weight': object(), 'module.weight': object()}, 'base')

    def test_wrong_source_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'wrong.pth'
            source.write_bytes(b'this must never be deserialized')
            with self.assertRaisesRegex(ValueError, 'Source hash'):
                export_checkpoint(source, Path(directory) / 'frame.pth')

    def test_receipt_binds_source_model_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.pth'
            path.write_bytes(b'fixture; tensor loading is a separate runtime gate')
            info = model_info('frame')
            receipt = {'schema': 'sva-inference-export-v1', 'model': 'frame', 'candidate_id': info['candidate_id'], 'source_sha256': info['checkpoint_sha256'], 'inference_sha256': sha256(path)}
            path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(verify_model_artifact(path)['kind'], 'inference_export')
            self.assertEqual(verify_model_artifact(path, 'frame-distill'), verify_model_artifact(path, 'frame'))
            with self.assertRaises(ValueError):
                verify_model_artifact(path, 'pool-distill')
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_model_artifact(path)

    def test_pinned_vpc_config_and_no_training_boundary(self):
        config = template('/data', '/weights', '/results')
        self.assertEqual(validate_config(config, '/weights'), config)
        config['privacy']['asv']['training'] = {'retrain': True}
        with self.assertRaisesRegex(ValueError, 'Training'):
            validate_config(config, '/weights')

    def test_vpc_rejects_heldout_model_change_and_plda(self):
        for kind in ['test', 'model', 'plda']:
            config = template('/data', '/weights', '/results')
            if kind == 'test':
                config['datasets'][0]['data'] = 'IEMOCAP_test'
            elif kind == 'model':
                config['privacy']['asv']['evaluation']['model_dir'] = '/other/attacker'
            else:
                config['privacy']['asv']['evaluation']['distance'] = 'plda'
            with self.assertRaises(ValueError):
                validate_config(config, '/weights')

    def test_vpc_verification_accepts_lf_and_crlf_and_checks_content(self):
        source = b'print("pinned evaluator")\n'
        pin = {'commit': 'pinned', 'files': {'anon/eval.py': hashlib.sha256(source).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'anon/eval.py'
            path.parent.mkdir()
            with patch('streamvoiceanonplus.vpc_adapter.read', return_value=pin), patch(
                'streamvoiceanonplus.vpc_adapter.subprocess.check_output', side_effect=['pinned', ''] * 3
            ):
                for text in (source, source.replace(b'\n', b'\r\n')):
                    with self.subTest(source=text):
                        path.write_bytes(text)
                        self.assertEqual(verify_checkout(directory)['verified_source_files'], 1)
                path.write_bytes(source.replace(b'pinned', b'changed'))
                with self.assertRaisesRegex(ValueError, 'External evaluator source mismatch'):
                    verify_checkout(directory)

    def test_no_data_weights_or_training_programs_in_source(self):
        root = Path(__file__).resolve().parents[1]
        forbidden_suffixes = ('.wav', '.flac', '.pth', '.pt', '.ckpt', '.parquet', '.csv.gz')
        forbidden_folders = {'trainers', 'optimizers', 'dataloaders', 'data_prep', 'anon'}
        if (root / '.git').exists():
            tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'])
            paths = [Path(os.fsdecode(name)) for name in tracked.split(b'\0') if name]
        else:
            # Source archives have no Git index. Exclude the directories created by installation and inference.
            local_directories = {'.venv', 'build', 'dist', 'pretrained_checkpoints', 'audio_outputs'}
            paths = []
            for directory, subdirectories, files in os.walk(root):
                relative = Path(directory).relative_to(root)
                if relative == Path('.'):
                    subdirectories[:] = [name for name in subdirectories if name not in local_directories]
                paths.extend(relative / name for name in files)
        for path in paths:
            self.assertFalse(path.name.lower().endswith(forbidden_suffixes), path)
            self.assertNotIn(path.parts[0], forbidden_folders, path)


if __name__ == '__main__':
    unittest.main()
