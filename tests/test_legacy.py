from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from streamvoiceanonplus import InferenceWrapper
from streamvoiceanonplus.legacy import main


class LegacyInferenceTests(unittest.TestCase):
    def make_engine(self):
        engine = Mock()
        engine._lock = nullcontext()
        engine.torch.inference_mode.side_effect = nullcontext
        engine.wrapper.sr = 44100
        return engine

    @patch('streamvoiceanonplus.runtime.InferenceEngine')
    def test_old_python_streaming_call_routes_to_frame(self, engine_factory):
        engine = self.make_engine()
        engine_factory.return_value = engine
        with InferenceWrapper(None, 'frame.pth', compile_ar=True, components='shared') as wrapper:
            self.assertEqual(wrapper.sr, 44100)
            result = wrapper.stream_infer('source.wav', ['ref.wav'], 'out', save_result=False)
        self.assertIs(result, engine.wrapper.stream_infer.return_value)
        self.assertEqual(engine_factory.call_args.kwargs['model'], 'frame')
        self.assertEqual(engine.wrapper.stream_infer.call_args.args, ('source.wav', 'ref.wav'))
        kwargs = engine.wrapper.stream_infer.call_args.kwargs
        self.assertEqual(kwargs['out_dir'], 'out')
        self.assertEqual(kwargs['alpha'], 0.9)
        self.assertEqual(kwargs['decode_chunk_frames'], 3)
        self.assertFalse(kwargs['save_result'])
        engine.close.assert_called_once()

    @patch('streamvoiceanonplus.runtime.InferenceEngine')
    def test_baseline_keeps_multi_reference_and_crop(self, engine_factory):
        engine = self.make_engine()
        engine_factory.return_value = engine
        with InferenceWrapper(model='base') as wrapper:
            wrapper.infer('input.wav', ['one.wav', 'two.wav'], ref_crop_lengths=[2, 3])
        self.assertEqual(engine_factory.call_args.args[0], Path('pretrained_checkpoints/dual_ar_delay_0_8.pth'))
        self.assertEqual(engine.wrapper.infer.call_args.args[1], ['one.wav', 'two.wav'])
        self.assertEqual(engine.wrapper.infer.call_args.kwargs['ref_crop_lengths'], [2, 3])

    @patch('streamvoiceanonplus.runtime.InferenceEngine')
    def test_frame_rejects_multiple_refs_before_inference(self, engine_factory):
        engine_factory.return_value = self.make_engine()
        with InferenceWrapper(checkpoint_path='frame.pth') as wrapper:
            with self.assertRaisesRegex(ValueError, 'require one reference'):
                wrapper.stream_infer('input.wav', ['one.wav', 'two.wav'])
        engine_factory.return_value.wrapper.stream_infer.assert_not_called()

    @patch('streamvoiceanonplus.legacy.InferenceWrapper')
    def test_original_cli_flags_select_streaming(self, wrapper_factory):
        wrapper = wrapper_factory.return_value.__enter__.return_value
        for name in ('frame-distill', 'frame', 'pool-distill', 'pool'):
            with self.subTest(model=name), TemporaryDirectory() as output:
                main(['--src_path', 'in.wav', '--ref_path', 'reference.wav', '--out_dir', output,
                      '--model', name, '--checkpoint_path', 'checkpoint.pth', '--simulate_streaming',
                      '--compile', '--alpha', '0.8'])
                model = name if name.endswith('-distill') else name + '-distill'
                self.assertEqual(wrapper_factory.call_args.kwargs['model'], model)
                self.assertTrue(wrapper_factory.call_args.kwargs['compile_ar'])
                self.assertEqual(wrapper.stream_infer.call_args.args, ('in.wav', 'reference.wav'))
                self.assertEqual(wrapper.stream_infer.call_args.kwargs['alpha'], 0.8)
        wrapper.infer.assert_not_called()
