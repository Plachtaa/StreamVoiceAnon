"""Process consecutive audio blocks with the released streaming models."""

from pathlib import Path

from .legacy import STREAM_DEFAULTS, validate_controls


class LiveInference:
    def __init__(self, engine, reference, *, delay, alpha, chunk_frames):
        import librosa
        import numpy as np

        validate_controls(delay, alpha)
        if chunk_frames not in range(1, 9):
            raise ValueError('Chunk size must be between 1 and 8 frames')
        self.engine = engine
        self.wrapper = engine.wrapper
        self.rate = self.wrapper.sr
        self.chunk_samples = chunk_frames * self.wrapper.SAMPLES_PER_FRAME
        self.chunk_frames = chunk_frames
        self.delay = delay
        self.alpha = alpha
        self.chunk_index = 0
        self.np = np
        if not Path(reference).is_file():
            raise ValueError('Choose a reference audio file')
        waveform, _ = librosa.load(reference, sr=self.rate, mono=True)
        minimum = STREAM_DEFAULTS['decode_window_frames'] * self.wrapper.SAMPLES_PER_FRAME
        if len(waveform) < minimum or not np.isfinite(waveform).all() or not np.any(waveform):
            raise ValueError('Choose a reference with at least three seconds of speech')
        self.reference = engine.torch.from_numpy(waveform).to(self.wrapper.device).unsqueeze(0)

    def reset(self):
        self.chunk_index = 0
        options = {name: STREAM_DEFAULTS[name] for name in ('encode_window_frames', 'decode_window_frames',
                                                           'max_seq_frames', 'buffer_frames')}
        with self.engine._lock, self.engine.torch.inference_mode():
            self.wrapper.prefill_prompt(self.reference, max_prompt_frames=STREAM_DEFAULTS['max_prompt_frames'],
                                        delay=self.delay, alpha=self.alpha)
            self.wrapper.setup_stream_caches(decode_chunk_frames=self.chunk_frames, **options)

    def process(self, audio):
        if audio.shape != (self.chunk_samples,):
            raise ValueError(f'Expected one mono block of {self.chunk_samples} samples')
        options = {'chunk_idx': self.chunk_index} if self.engine.info['id'] != 'base' else {}
        torch = self.engine.torch
        with self.engine._lock, torch.inference_mode():
            block = torch.from_numpy(audio).to(self.wrapper.device).unsqueeze(0)
            wave = self.wrapper.process_one_chunk(block, **options).reshape(-1).float().cpu().numpy()
        self.chunk_index += 1
        if wave.shape != audio.shape or not self.np.isfinite(wave).all():
            raise RuntimeError('The model returned an invalid audio block')
        return wave

    def warm_up(self, cancelled):
        self.reset()
        audio = self.np.zeros(self.chunk_samples, dtype=self.np.float32)
        for _ in range(self.delay // self.chunk_frames + 3):
            if cancelled.is_set():
                return
            self.process(audio)
        self.reset()
