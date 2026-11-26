from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from streamvoiceanonplus.gui import arguments, default_chunk_frames, model_paths
from streamvoiceanonplus.live import LiveInference


class DesktopTests(unittest.TestCase):
    def test_selection_and_legacy_checkpoint_override(self):
        self.assertEqual(arguments([]).model, 'frame-distill')
        old = arguments(['--checkpoint_path', 'original.pth'])
        self.assertEqual(old.model, 'base')
        self.assertEqual(model_paths(old, 'base')[0], Path('original.pth'))
        self.assertEqual(model_paths(old, 'pool-distill')[0], Path('pretrained_checkpoints/pool.pth'))
        plus = arguments(['--model', 'frame', '--components', 'weights', '--checkpoint_path', 'custom.pth'])
        self.assertEqual(plus.model, 'frame-distill')
        self.assertEqual(model_paths(plus, 'frame-distill'), (Path('custom.pth'), Path('weights')))
        self.assertEqual(model_paths(plus, 'base')[0], Path('weights/dual_ar_delay_0_8.pth'))
        self.assertEqual([default_chunk_frames(name) for name in ('base', 'frame-distill', 'pool-distill')], [1, 3, 3])

    def test_live_chunk_indexes_reset_and_only_plus_receives_them(self):
        for model in ('base', 'frame', 'pool'):
            with self.subTest(model=model):
                inference = LiveInference.__new__(LiveInference)
                inference.wrapper = Mock()
                tensor = Mock()
                output = SimpleNamespace(shape=(6144,))
                torch = Mock()
                torch.inference_mode = nullcontext
                torch.from_numpy.return_value.to.return_value.unsqueeze.return_value = tensor
                inference.engine = SimpleNamespace(torch=torch, _lock=nullcontext(), info={'id': model})
                inference.np = Mock()
                wave = inference.wrapper.process_one_chunk.return_value
                wave.reshape.return_value.float.return_value.cpu.return_value.numpy.return_value = output
                inference.chunk_samples = 6144
                inference.chunk_frames = 3
                inference.delay = 2
                inference.alpha = 0.9
                inference.reference = tensor
                inference.reset()
                self.assertIs(inference.process(output), output)
                self.assertIs(inference.process(output), output)
                calls = inference.wrapper.process_one_chunk.call_args_list
                self.assertEqual([call.kwargs for call in calls], [{}, {}] if model == 'base' else [{'chunk_idx': 0}, {'chunk_idx': 1}])
                inference.reset()
                self.assertEqual(inference.chunk_index, 0)
                inference.wrapper.prefill_prompt.assert_called_with(tensor, max_prompt_frames=256, delay=2, alpha=0.9)
