"""Desktop controls for the StreamVoiceAnon release models."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event
from time import perf_counter

from .catalog import DEFAULT_MODEL, MODEL_CHOICES, model_info, model_selector, read
from .gui_guide import GuideStep, VisualGuide
from .legacy import component_paths, validate_controls


MODEL_LABELS = {
    'base': 'StreamVoiceAnon',
    'frame-distill': 'StreamVoiceAnon+ frame-distill',
    'pool-distill': 'StreamVoiceAnon+ pool-distill',
}
PRESETS = {'Release': (0.9, 2), 'Max Privacy': (0.0, 2), 'Balanced': (0.5, 2),
           'Max Quality': (1.0, 4), 'Low Latency': (0.7, 1)}
PRESET_DESCRIPTIONS = {
    'Release': 'Release settings with α 0.90 and two delay frames.',
    'Max Privacy': 'Strongest noise mixing in the speaker embeddings.',
    'Balanced': 'Equal contributions from the reference and noise embeddings.',
    'Max Quality': 'Full reference embeddings and four delay frames.',
    'Low Latency': 'One delay frame with α 0.70.',
    'Custom': 'Manually adjusted conversion settings.',
}
MODEL_RATE = 'Model rate (44100 Hz)'
DEVICE_RATE = 'Input device rate'
ALL_BACKENDS = 'All backends'
MUTED = ('#59687f', '#a6afc3')
CONVERSION_TAB = 'Voice Conversion'
DEVICES_TAB = 'Audio Devices'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='StreamVoiceAnon desktop microphone inference')
    parser.add_argument('--model', type=model_selector, choices=MODEL_CHOICES)
    parser.add_argument('--components', type=Path, default=Path('pretrained_checkpoints'))
    parser.add_argument('--checkpoint_path', type=Path)
    parser.add_argument('--config_path', type=Path, help='Read shared component paths from a legacy configuration')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')
    parser.add_argument('--no-compile', action='store_true', help='Use eager autoregressive decoding')
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = 'base' if args.checkpoint_path or args.config_path else DEFAULT_MODEL
    return args


def model_paths(args, selector):
    model = model_info(selector)
    components = component_paths(args.config_path) if args.config_path else args.components
    checkpoint = args.checkpoint_path if selector == args.model else None
    if checkpoint is None:
        filename = 'dual_ar_delay_0_8.pth' if selector == 'base' else model['checkpoint_file']
        checkpoint = args.components / filename
    return checkpoint, components


def default_chunk_frames(selector):
    return 1 if selector == 'base' else read('protocol.json')['decode_chunk_frames']


@dataclass(frozen=True)
class StreamSettings:
    model: str
    reference: Path
    input_device: int
    output_device: int
    use_device_rate: bool
    wasapi_exclusive: bool
    delay: int
    alpha: float
    chunk_frames: int


def resample_block(wave, length, resampler=None):
    import numpy as np

    if resampler is not None:
        import torch
        wave = resampler(torch.from_numpy(wave)).numpy()
    if len(wave) < length:
        wave = np.pad(wave, (0, length - len(wave)))
    return wave[:length]


def stream_audio(args, settings, sounddevice, cancelled, updates):
    import numpy as np
    import torch
    from torchaudio.transforms import Resample

    from .live import LiveInference
    from .runtime import InferenceEngine

    if not torch.cuda.is_available():
        raise RuntimeError('The desktop app requires a CUDA GPU')
    torch.cuda.set_device(args.gpu)
    checkpoint, components = model_paths(args, settings.model)
    updates.put(('status', 'Loading model…'))
    with InferenceEngine(checkpoint, components, model=settings.model, compile_ar=not args.no_compile) as engine:
        if cancelled.is_set():
            return
        live = LiveInference(engine, settings.reference, delay=settings.delay,
                             alpha=settings.alpha, chunk_frames=settings.chunk_frames)
        updates.put(('status', 'Warming up…'))
        live.warm_up(cancelled)
        if cancelled.is_set():
            return

        source = sounddevice.query_devices(settings.input_device, 'input')
        target = sounddevice.query_devices(settings.output_device, 'output')
        rate = int(source['default_samplerate']) if settings.use_device_rate else live.rate
        frames = round(live.chunk_samples * rate / live.rate)
        in_channels = min(2, source['max_input_channels'])
        out_channels = min(2, target['max_output_channels'])
        to_model = Resample(rate, live.rate) if rate != live.rate else None
        to_device = Resample(live.rate, rate) if rate != live.rate else None
        extra_settings = None
        if settings.wasapi_exclusive:
            apis = sounddevice.query_hostapis()
            if not all('WASAPI' in apis[device['hostapi']]['name'] for device in (source, target)):
                raise ValueError('WASAPI exclusive mode requires two WASAPI devices')
            extra_settings = sounddevice.WasapiSettings(exclusive=True)
        # Blocking read/write runs outside the GUI thread and the PortAudio callback.
        with sounddevice.Stream(device=(settings.input_device, settings.output_device), samplerate=rate,
                                blocksize=frames, channels=(in_channels, out_channels), dtype='float32',
                                extra_settings=extra_settings) as audio:
            updates.put(('status', f'Streaming at {rate:,} Hz'))
            while not cancelled.is_set():
                captured, overflow = audio.read(frames)
                if cancelled.is_set():
                    break
                start = perf_counter()
                mono = captured.mean(axis=1, dtype=np.float32)
                block = resample_block(mono, live.chunk_samples, to_model)
                result = resample_block(live.process(block), frames, to_device)
                elapsed = (perf_counter() - start) * 1000
                underflow = audio.write(np.repeat(result[:, None], out_channels, axis=1))
                updates.put(('timing', f'{elapsed:.0f} ms per block' + (' · audio dropped' if overflow or underflow else '')))


class DesktopApp:
    def __init__(self, root, args, ctk, sounddevice):
        self.root, self.args, self.ctk, self.sounddevice = root, args, ctk, sounddevice
        self.controls = []
        self.inputs, self.outputs = {}, {}
        self.updates = SimpleQueue()
        self.cancelled = Event()
        # PyTorch's CUDA graph state is thread-local. Keep the same worker across model changes.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.worker = None
        self.closing = False
        self.root.title('StreamVoiceAnon+ · Real-time anonymization')
        self.root.geometry('900x620')
        self.root.minsize(800, 580)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build()
        self.refresh_devices()
        self.root.after(100, self.poll)

    def _build(self):
        ctk = self.ctk
        self._build_sidebar()
        self.tabs = ctk.CTkTabview(self.root, corner_radius=10, anchor='nw', command=self._change_tab)
        self.tabs.grid(row=0, column=1, sticky='nsew', padx=15, pady=(8, 12))
        self._build_conversion(self.tabs.add(CONVERSION_TAB))
        self._build_devices(self.tabs.add(DEVICES_TAB))
        self._build_status()
        self.apply_preset('Release')
        self.guide = VisualGuide(self.root, self.tabs, ctk, [
            GuideStep(self.model, 'Choose a model', 'Select StreamVoiceAnon, frame-distill, or pool-distill. '
                      'Changing models restores that model’s default chunk size.', CONVERSION_TAB),
            GuideStep(self.reference, 'Select reference audio', 'Browse for a recording with at least three '
                      'seconds of speech. The model uses it as the target voice.', CONVERSION_TAB),
            GuideStep(self.preset, 'Choose a preset', 'Use Release to restore the default alpha, delay, and '
                      'chunk size. Moving a slider switches the preset to Custom.'),
            GuideStep(self.host_api, 'Configure audio devices', 'Choose your microphone and output device. '
                      'On Windows, select WASAPI to use exclusive mode with compatible devices.', DEVICES_TAB),
            GuideStep(self.start_button, 'Start conversion', 'Start loads the model and warms up the decoder. '
                      'Once streaming, speak into the microphone. Stop before switching models.'),
            GuideStep(self.alpha, 'Adjust embedding mixing', 'Lower alpha mixes in more noise. At 1.00, '
                      'the model uses the reference style and timbre embeddings without noise.', CONVERSION_TAB),
            GuideStep(self.chunk_size, 'Set chunk size and delay', 'Chunk size controls how many frames are '
                      'processed together. Delay sets the decoder look-ahead in frames.', CONVERSION_TAB),
            GuideStep(self.status_bar, 'Check streaming status', 'While streaming, the status bar shows the '
                      'audio sample rate and inference time per chunk.'),
        ])

    def _build_sidebar(self):
        ctk = self.ctk
        sidebar = ctk.CTkFrame(self.root, corner_radius=0, fg_color=('#e7ebf7', '#1a1a2e'))
        sidebar.grid(row=0, column=0, rowspan=2, sticky='nsew')
        sidebar.grid_columnconfigure(0, minsize=200, weight=1)
        sidebar.grid_rowconfigure(7, weight=1)
        ctk.CTkLabel(sidebar, text='Stream\nVoice Anon+', justify='left',
                     font=ctk.CTkFont(size=24, weight='bold')).grid(row=0, padx=20, pady=(23, 5), sticky='w')
        ctk.CTkLabel(sidebar, text='Real-time speaker anonymization', wraplength=160, justify='left',
                     text_color=MUTED).grid(row=1, padx=20, pady=(0, 16), sticky='w')
        self.start_button = ctk.CTkButton(sidebar, text='Start', height=40, fg_color=('#147551', '#2fa572'),
                                          hover_color='#1a7a50', font=ctk.CTkFont(size=15, weight='bold'),
                                          command=self.start)
        self.start_button.grid(row=2, padx=20, pady=(3, 7), sticky='ew')
        self.stop_button = ctk.CTkButton(sidebar, text='Stop', height=33, state='disabled',
                                         fg_color=('#b9403c', '#d35b58'), hover_color='#a83432', command=self.stop)
        self.stop_button.grid(row=3, padx=20, sticky='ew')
        self._sidebar_label(sidebar, 4, 'Preset')
        self.preset = ctk.CTkOptionMenu(sidebar, values=list(PRESETS) + ['Custom'],
                                        width=160, command=self.apply_preset)
        self.preset.grid(row=5, padx=20, sticky='ew')
        self.preset_description = ctk.CTkLabel(sidebar, text='', wraplength=160, anchor='w',
                                               justify='left', text_color=MUTED)
        self.preset_description.grid(row=6, padx=20, pady=(5, 0), sticky='w')
        self.controls.append(self.preset)

        self._sidebar_label(sidebar, 8, 'Theme')
        self.theme = ctk.CTkOptionMenu(sidebar, values=['Dark', 'Light', 'System'],
                                       width=160, command=self._change_theme)
        self.theme.set(ctk.get_appearance_mode())
        self.theme.grid(row=9, padx=20, sticky='ew')
        self.tutorial_button = ctk.CTkButton(
            sidebar, text='Show Tutorial', height=34, fg_color='transparent', border_width=1,
            border_color=('#8a7550', '#aa8b4b'), hover_color=('#f3e3b5', '#353045'),
            text_color=('#664608', '#f0c040'), command=lambda: self.guide.start(),
        )
        self.tutorial_button.grid(row=10, padx=20, pady=(14, 22), sticky='ew')

    def _sidebar_label(self, pane, row, title):
        self.ctk.CTkLabel(pane, text=title.upper(), text_color=MUTED,
                          font=self.ctk.CTkFont(size=11, weight='bold')).grid(
            row=row, padx=20, pady=(18, 4), sticky='sw')

    def _build_conversion(self, pane):
        ctk = self.ctk
        pane.grid_columnconfigure(1, weight=1)
        self.model = self._option(pane, 0, 'Model', list(MODEL_LABELS.values()), self.change_model)
        self.model.set(MODEL_LABELS[self.args.model])
        self._section(pane, 1, 'Reference Audio')
        self.reference = ctk.CTkEntry(pane, placeholder_text='Choose a reference recording with at least 3 s of speech')
        self.reference.grid(row=2, column=0, columnspan=2, sticky='ew', padx=(16, 8), pady=(0, 8))
        pick = ctk.CTkButton(pane, text='Browse', width=82, command=self.choose_reference)
        pick.grid(row=2, column=2, sticky='e', padx=(0, 16), pady=(0, 8))
        self.controls.extend((self.reference, pick))

        self._section(pane, 3, 'Anonymization')
        self.alpha, self.alpha_value = self._slider(pane, 4, 'Embedding mix (α)', 0, 1, 100)
        self._hint(pane, 5, 'Lower α mixes in more noise. At α = 1, the reference embeddings are used unchanged.')

        self._section(pane, 6, 'Performance')
        self.chunk_size, self.chunk_value = self._slider(pane, 7, 'Chunk size', 1, 8, 7)
        self.delay, self.delay_value = self._slider(pane, 8, 'Delay', 0, 8, 8)
        self._hint(pane, 9, 'Chunk size sets how many frames are processed together. Delay adds decoder look-ahead.')

    def _build_devices(self, pane):
        ctk = self.ctk
        pane.grid_columnconfigure(1, weight=1)
        self._section(pane, 0, 'Audio Devices')
        self.host_api = self._option(pane, 1, 'Audio backend', [ALL_BACKENDS], self.refresh_devices)
        self.input = self._option(pane, 2, 'Input device', ['No input devices'])
        self.output = self._option(pane, 3, 'Output device', ['No output devices'])
        refresh = ctk.CTkButton(pane, text='Refresh Devices', width=136, command=self.refresh_devices)
        refresh.grid(row=4, column=1, sticky='w', pady=(3, 8))
        self.controls.append(refresh)
        self.input.configure(command=self._update_exclusive)
        self.output.configure(command=self._update_exclusive)
        self.exclusive = ctk.CTkCheckBox(pane, text='WASAPI exclusive mode', state='disabled')
        self.exclusive.grid(row=5, column=1, sticky='w', pady=(3, 7))
        self._hint(pane, 6, 'Exclusive mode is available when both selected devices use Windows WASAPI.')

        self._section(pane, 7, 'Sample Rate')
        self.rate = ctk.StringVar(value=MODEL_RATE)
        for row, label in enumerate((MODEL_RATE, DEVICE_RATE), start=8):
            radio = ctk.CTkRadioButton(pane, text=label, variable=self.rate, value=label)
            radio.grid(row=row, column=0, columnspan=3, sticky='w', padx=16, pady=5)
            self.controls.append(radio)

    def _build_status(self):
        ctk = self.ctk
        self.status_bar = ctk.CTkFrame(self.root, corner_radius=8, fg_color=('#e2eafa', '#16213e'))
        self.status_bar.grid(row=1, column=1, sticky='ew', padx=15, pady=(0, 10))
        self.status_bar.grid_columnconfigure(1, weight=1)
        self.status_dot = ctk.CTkLabel(self.status_bar, text='●', text_color=MUTED)
        self.status_dot.grid(row=0, column=0, padx=(14, 6), pady=9)
        self.status = ctk.CTkLabel(self.status_bar, text='Ready', anchor='w', wraplength=330, justify='left')
        self.status.grid(row=0, column=1, sticky='ew', pady=6)
        self.timing = ctk.CTkLabel(self.status_bar, text='Inference  —', text_color=MUTED)
        self.timing.grid(row=0, column=2, sticky='e', padx=(8, 15))
        self.sample_rate = ctk.CTkLabel(self.status_bar, text='Sample rate  —', text_color=MUTED)
        self.sample_rate.grid(row=0, column=3, sticky='e', padx=(0, 15))

    def _section(self, pane, row, title):
        ctk = self.ctk
        section = ctk.CTkFrame(pane, fg_color='transparent')
        section.grid(row=row, column=0, columnspan=3, sticky='ew', padx=16, pady=(14, 7))
        ctk.CTkLabel(section, text=title, font=ctk.CTkFont(size=16, weight='bold')).pack(side='left')
        ctk.CTkFrame(section, height=1, corner_radius=0, fg_color=('#c2c9da', '#454954')).pack(
            side='left', fill='x', expand=True, padx=(12, 0))

    def _hint(self, pane, row, text):
        self.ctk.CTkLabel(pane, text=text, text_color=MUTED, wraplength=480, justify='left', anchor='w').grid(
            row=row, column=0, columnspan=3, sticky='ew', padx=16, pady=(0, 5))

    def _slider(self, pane, row, title, lower, upper, steps):
        self.ctk.CTkLabel(pane, text=title).grid(row=row, column=0, sticky='w', padx=(16, 13), pady=5)
        slider = self.ctk.CTkSlider(pane, from_=lower, to=upper, number_of_steps=steps, command=self.mark_custom)
        slider.grid(row=row, column=1, sticky='ew', pady=5)
        value = self.ctk.CTkLabel(pane, text='', width=70, anchor='e')
        value.grid(row=row, column=2, sticky='e', padx=(8, 16))
        self.controls.append(slider)
        return slider, value

    def _option(self, pane, row, title, values, callback=None):
        self.ctk.CTkLabel(pane, text=title).grid(row=row, column=0, sticky='w', padx=(16, 13), pady=5)
        menu = self.ctk.CTkOptionMenu(pane, values=values, command=callback, width=250, dynamic_resizing=False)
        menu.grid(row=row, column=1, columnspan=2, sticky='ew', padx=(0, 16), pady=5)
        self.controls.append(menu)
        return menu

    def _change_tab(self):
        self.guide.tab_changed()

    def _change_theme(self, theme):
        self.ctk.set_appearance_mode(theme)
        self.guide.refresh()

    def selector(self):
        return next(key for key, label in MODEL_LABELS.items() if label == self.model.get())

    def change_model(self, _label):
        self.chunk_size.set(default_chunk_frames(self.selector()))
        self._update_slider_values()

    def apply_preset(self, name):
        if name in PRESETS:
            alpha, delay = PRESETS[name]
            self.alpha.set(alpha)
            self.delay.set(delay)
            self.change_model(self.model.get())
        self.preset.set(name)
        self.preset_description.configure(text=PRESET_DESCRIPTIONS[name])

    def mark_custom(self, _value):
        self.apply_preset('Custom')
        self._update_slider_values()

    def _update_slider_values(self):
        self.alpha_value.configure(text=f'{self.alpha.get():.2f}')
        self.chunk_value.configure(text=f'{self.chunk_size.get():.0f} frames')
        self.delay_value.configure(text=f'{self.delay.get():.0f} frames')

    def choose_reference(self):
        from tkinter.filedialog import askopenfilename

        picked = askopenfilename(parent=self.root, filetypes=[('Audio', '*.wav *.flac *.mp3'), ('All files', '*')])
        if picked:
            self.reference.delete(0, 'end')
            self.reference.insert(0, picked)

    def refresh_devices(self, _backend=None):
        devices = self.sounddevice.query_devices()
        apis = self.sounddevice.query_hostapis()
        backend_names = [api['name'] for api in apis]
        self.host_api.configure(values=[ALL_BACKENDS, *backend_names])
        if self.host_api.get() not in backend_names:
            self.host_api.set(ALL_BACKENDS)
        selected_backend = self.host_api.get()
        self.device_backends = [backend_names[device['hostapi']] for device in devices]
        items = [(f'{i}: {device["name"]} ({backend})', i, device)
                 for i, (device, backend) in enumerate(zip(devices, self.device_backends))
                 if selected_backend in (ALL_BACKENDS, backend)]
        self.inputs = {label: i for label, i, device in items if device['max_input_channels'] > 0}
        self.outputs = {label: i for label, i, device in items if device['max_output_channels'] > 0}
        for menu, choices, default, empty in (
            (self.input, self.inputs, self.sounddevice.default.device[0], 'No input devices'),
            (self.output, self.outputs, self.sounddevice.default.device[1], 'No output devices'),
        ):
            selected = menu.get()
            options = list(choices) or [empty]
            menu.configure(values=options)
            preferred = next((label for label, index in choices.items() if index == default), options[0])
            menu.set(selected if selected in choices else preferred)
        self._update_exclusive()

    def _update_exclusive(self, _device=None):
        indices = (self.inputs.get(self.input.get()), self.outputs.get(self.output.get()))
        available = all(i is not None and 'WASAPI' in self.device_backends[i] for i in indices)
        if not available:
            self.exclusive.deselect()
        self.exclusive.configure(state='normal' if available and self.worker is None else 'disabled')

    def settings(self):
        path = Path(self.reference.get().strip())
        if not path.is_file():
            raise ValueError('Choose a reference audio file')
        if self.input.get() not in self.inputs or self.output.get() not in self.outputs:
            raise ValueError('Select an input and an output audio device')
        alpha, delay = round(self.alpha.get(), 2), round(self.delay.get())
        validate_controls(delay, alpha)
        return StreamSettings(self.selector(), path, self.inputs[self.input.get()], self.outputs[self.output.get()],
                              self.rate.get() == DEVICE_RATE, bool(self.exclusive.get()), delay, alpha,
                              round(self.chunk_size.get()))

    def start(self):
        if self.worker is not None:
            return
        try:
            settings = self.settings()
        except ValueError as error:
            self.set_status(f'Error · {error}')
            return
        self.guide.close()
        self.cancelled.clear()
        self.set_running(True)
        self.set_status('Starting…')
        self.worker = self.executor.submit(self.run, settings)

    def run(self, settings):
        try:
            stream_audio(self.args, settings, self.sounddevice, self.cancelled, self.updates)
        except Exception as error:
            logging.getLogger(__name__).exception('Desktop audio stream failed')
            self.updates.put(('status', f'Error · {str(error) or type(error).__name__}'))
        else:
            self.updates.put(('status', 'Stopped'))

    def set_running(self, running):
        self.start_button.configure(state='disabled' if running else 'normal')
        self.stop_button.configure(state='normal' if running else 'disabled')
        for control in self.controls:
            control.configure(state='disabled' if running else 'normal')
        if running:
            self.exclusive.configure(state='disabled')
        else:
            self._update_exclusive()

    def set_status(self, text):
        streaming = text.startswith('Streaming at ')
        self.status.configure(text='Streaming' if streaming else text)
        color = MUTED
        if streaming:
            color = ('#147551', '#2fa572')
        elif text.startswith('Error'):
            color = ('#b9403c', '#d35b58')
        elif text.endswith('…'):
            color = ('#8a620f', '#f0c040')
        self.status_dot.configure(text_color=color)
        rate = text.removeprefix('Streaming at ') if streaming else '—'
        self.sample_rate.configure(text=f'Sample rate  {rate}')

    def stop(self):
        if self.worker is not None and not self.cancelled.is_set():
            self.cancelled.set()
            self.stop_button.configure(state='disabled')
            self.set_status('Stopping…')

    def close(self):
        self.closing = True
        self.stop()

    def poll(self):
        try:
            while True:
                kind, text = self.updates.get_nowait()
                if kind == 'status':
                    self.set_status(text)
                elif not self.cancelled.is_set():
                    self.timing.configure(text=f'Inference  {text}')
        except Empty:
            pass
        if self.worker is not None and self.worker.done():
            self.worker = None
            self.set_running(False)
            self.timing.configure(text='Inference  —')
        if self.closing and self.worker is None:
            self.executor.shutdown()
            self.root.destroy()
        else:
            self.root.after(100, self.poll)


def main(argv=None):
    args = arguments(argv)
    try:
        import customtkinter as ctk
        import sounddevice
    except (ImportError, OSError) as error:
        raise SystemExit(f'Install the desktop GUI dependencies listed in the README. {error}') from error
    ctk.set_appearance_mode('dark')
    root = ctk.CTk()
    DesktopApp(root, args, ctk, sounddevice)
    root.mainloop()
