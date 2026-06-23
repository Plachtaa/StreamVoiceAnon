import os
import sys
from dotenv import load_dotenv
import shutil

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import warnings

warnings.simplefilter("ignore")

import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import torch

# ── Global state ─────────────────────────────────────────────────────────────
device = None
flag_vc = False
reference_wav_name = ""
decode_chunk_frames = 0
fp16 = False


@torch.no_grad()
def custom_infer(model_set, reference_wav, new_reference_wav_name, input_wav,
                 n_frame_delay=2, alpha=0.7):
    global reference_wav_name, decode_chunk_frames
    if reference_wav_name != new_reference_wav_name or decode_chunk_frames != input_wav.size(-1) // 2048:
        model_set.prefill_prompt(
            torch.from_numpy(reference_wav).to(device).unsqueeze(0),
            max_prompt_frames=64, delay=n_frame_delay, alpha=alpha,
        )
        model_set.setup_stream_caches(
            encode_window_frames=64, decode_window_frames=64,
            max_seq_frames=768, buffer_frames=32,
            decode_chunk_frames=input_wav.size(-1) // 2048,
        )
        reference_wav_name = new_reference_wav_name
        decode_chunk_frames = input_wav.size(-1) // 2048
    pred_wave = model_set.process_one_chunk(input_wav.to(device).unsqueeze(0))
    return pred_wave.squeeze().cpu()


def load_models(args):
    from evaluations.infer_arvc import InferenceWrapper
    return InferenceWrapper(
        args.config_path, args.checkpoint_path,
        compile_ar=True, compile_decoder=True, compile_encoder=True,
    )


class Config:
    def __init__(self):
        self.device = device


if __name__ == "__main__":
    import json
    import re
    import time
    import argparse
    from tkinter import filedialog

    import librosa
    import numpy as np
    import customtkinter as ctk
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat

    # ── Color palette ────────────────────────────────────────────────────
    COLORS = {
        "accent": "#1f6aa5",
        "green": "#2fa572",
        "green_hover": "#1a7a50",
        "red": "#d35b58",
        "red_hover": "#a83432",
        "yellow": "#d4a017",
        "gray": "#6b7280",
        "sidebar_bg": "#1a1a2e",
        "status_bg": "#16213e",
    }

    # ── Tooltip helper ───────────────────────────────────────────────────
    class Tooltip:
        def __init__(self, widget, text, delay=450, wraplength=280):
            self.widget = widget
            self.text = text
            self.delay = delay
            self.wraplength = wraplength
            self._tip = None
            self._after_id = None
            widget.bind("<Enter>", self._schedule, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<ButtonPress>", self._hide, add="+")

        def _schedule(self, _event=None):
            self._cancel()
            self._after_id = self.widget.after(self.delay, self._show)

        def _cancel(self):
            if self._after_id is not None:
                try:
                    self.widget.after_cancel(self._after_id)
                except Exception:
                    pass
                self._after_id = None

        def _show(self):
            if self._tip is not None:
                return
            try:
                x = self.widget.winfo_rootx() + 16
                y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            except Exception:
                return
            tip = ctk.CTkToplevel(self.widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            try:
                tip.attributes("-topmost", True)
            except Exception:
                pass
            frame = ctk.CTkFrame(
                tip, corner_radius=6, fg_color="#2b2b3e",
                border_width=1, border_color="#4a4a5e")
            frame.pack()
            ctk.CTkLabel(
                frame, text=self.text, font=("", 11),
                wraplength=self.wraplength, justify="left",
                text_color="#e5e5ef").pack(padx=10, pady=6)
            self._tip = tip

        def _hide(self, _event=None):
            self._cancel()
            if self._tip is not None:
                try:
                    self._tip.destroy()
                except Exception:
                    pass
                self._tip = None

    # ── Tutorial step definitions ────────────────────────────────────────
    TUTORIAL_CORE_STEPS = [
        {
            "widget": "ref_entry",
            "title": "Select Reference Audio",
            "description": (
                "Click Browse to pick the voice you want to sound like. "
                "This is the target speaker identity for voice conversion.\n\n"
                "Supports WAV, MP3, FLAC, M4A, OGG, Opus."
            ),
            "tab": "Voice Conversion",
        },
        {
            "widget": "preset_menu",
            "title": "Choose a Preset",
            "description": (
                "Quick-switch between anonymization profiles. "
                "Try 'Balanced' for a good starting point.\n\n"
                "Each preset adjusts alpha, delay, and block size automatically."
            ),
            "tab": None,
        },
        {
            "widget": "hostapi_menu",
            "title": "Configure Audio Devices",
            "description": (
                "Select your microphone as Input and speakers/headphones as Output.\n\n"
                "Use WASAPI for lowest latency on Windows."
            ),
            "tab": "Audio Devices",
        },
        {
            "widget": "start_btn",
            "title": "Start Conversion",
            "description": (
                "Click Start to begin real-time voice conversion. "
                "First run compiles model kernels (~1-2 minutes).\n\n"
                "Subsequent starts are instant."
            ),
            "tab": None,
        },
    ]

    TUTORIAL_ADVANCED_STEPS = [
        {
            "widget": "alpha_slider",
            "title": "Adjust Alpha",
            "description": (
                "Controls the privacy-utility trade-off.\n\n"
                "0.0 = Maximum privacy\n"
                "1.0 = Less anonymization (pure VC)\n\n"
                "Recommended: 0.7 (best out-of-box)."
            ),
            "tab": "Voice Conversion",
        },
        {
            "widget": "bf_slider",
            "title": "Block Frame Size",
            "description": (
                "Audio chunk size in 2048-sample units.\n\n"
                "Lower = less latency, more CPU load.\n"
                "Higher = more stable, more delay.\n\n"
                "Recommended: 1 (best out-of-box)."
            ),
            "tab": "Voice Conversion",
        },
        {
            "widget": "df_slider",
            "title": "Delay Frames",
            "description": (
                "Look-ahead frames for the decoder.\n\n"
                "0 = Minimum latency, lower quality\n"
                "2 = Recommended (best out-of-box)\n"
                "8 = Maximum quality, highest latency\n\n"
                "Each frame adds ~46ms of delay."
            ),
            "tab": "Voice Conversion",
        },
        {
            "widget": "status_bar",
            "title": "Status Bar",
            "description": (
                "Monitor your conversion in real-time.\n\n"
                "Delay: End-to-end latency\n"
                "Inference: Model processing time per chunk\n"
                "SR: Current sample rate\n\n"
                "Green dot = running, gray = stopped."
            ),
            "tab": None,
        },
    ]

    # ── Tutorial Manager ─────────────────────────────────────────────────
    class TutorialManager:
        """Step-by-step interactive tutorial with widget highlighting."""

        GLOW = "#f0c040"
        GLOW_DIM = "#c49a20"
        CARD_BG = "#1e1e32"
        CARD_BORDER = "#f0c040"
        DOT_ACTIVE = "#f0c040"
        DOT_INACTIVE = "#555555"

        def __init__(self, gui):
            self.gui = gui
            self.root = gui.root
            self.current_step = 0
            self.steps = []
            self.card = None
            self.is_active = False
            self._pulse_id = None
            self._pulse_on = True
            self._current_widget = None
            self._glow_frames = []
            self._dot_labels = []
            self._completion_toast = None

        def start(self, advanced=False):
            if self.is_active:
                return
            if self._completion_toast:
                try:
                    if self._completion_toast.winfo_exists():
                        self._completion_toast.destroy()
                except Exception:
                    pass
                self._completion_toast = None
            if advanced:
                self.steps = list(TUTORIAL_ADVANCED_STEPS)
            else:
                self.steps = list(TUTORIAL_CORE_STEPS)
            self.current_step = 0
            self.is_active = True
            self.root.bind("<Escape>", self._on_escape)
            self.root.bind("<Right>", self._on_right)
            self.root.bind("<Left>", self._on_left)
            self._show_step()

        def _on_escape(self, event):
            self.skip()

        def _on_right(self, event):
            self.next_step()

        def _on_left(self, event):
            if self.current_step > 0:
                self.goto_step(self.current_step - 1)

        def _get_widget(self, name):
            return getattr(self.gui, name, None)

        def _show_step(self):
            step = self.steps[self.current_step]
            widget = self._get_widget(step["widget"])
            if not widget:
                self.current_step += 1
                if self.current_step < len(self.steps):
                    self._show_step()
                else:
                    self.finish()
                return

            if step.get("tab"):
                self.gui.tabview.set(step["tab"])

            self.root.update_idletasks()
            self._highlight(widget)

            if self.card is None:
                self._create_card(widget, step)
            else:
                self._update_card(widget, step)

            self._start_pulse()

        def _highlight(self, widget):
            self._restore_highlight()
            self._current_widget = widget
            self.root.update_idletasks()

            s = self.gui.sidebar._get_widget_scaling()
            wx = (widget.winfo_rootx() - self.root.winfo_rootx()) / s
            wy = (widget.winfo_rooty() - self.root.winfo_rooty()) / s
            ww = widget.winfo_width() / s
            wh = widget.winfo_height() / s
            pad = 4
            bw = 3
            self._glow_frames = []
            for x, y, w, h in [
                (wx - pad, wy - pad, ww + 2 * pad, bw),
                (wx - pad, wy + wh + pad - bw, ww + 2 * pad, bw),
                (wx - pad, wy - pad, bw, wh + 2 * pad),
                (wx + ww + pad - bw, wy - pad, bw, wh + 2 * pad),
            ]:
                f = ctk.CTkFrame(
                    self.root, width=w, height=h,
                    fg_color=self.GLOW, corner_radius=0,
                )
                f.place(x=x, y=y)
                f.lift()
                self._glow_frames.append(f)

        def _restore_highlight(self):
            for f in self._glow_frames:
                try:
                    f.destroy()
                except Exception:
                    pass
            self._glow_frames = []
            self._current_widget = None

        def _start_pulse(self):
            self._stop_pulse()
            self._pulse_on = True
            self._pulse()

        def _stop_pulse(self):
            if self._pulse_id is not None:
                try:
                    self.root.after_cancel(self._pulse_id)
                except Exception:
                    pass
                self._pulse_id = None

        def _pulse(self):
            if not self.is_active or not self._glow_frames:
                return
            self._pulse_on = not self._pulse_on
            color = self.GLOW if self._pulse_on else self.GLOW_DIM
            for f in self._glow_frames:
                f.configure(fg_color=color)
            self._pulse_id = self.root.after(600, self._pulse)

        def _calc_card_pos(self, widget):
            self.root.update_idletasks()
            s = self.gui.sidebar._get_widget_scaling()
            wx = (widget.winfo_rootx() - self.root.winfo_rootx()) / s
            wy = (widget.winfo_rooty() - self.root.winfo_rooty()) / s
            ww = widget.winfo_width() / s
            wh = widget.winfo_height() / s
            rw = self.root.winfo_width() / s
            rh = self.root.winfo_height() / s
            card_w = 300
            card_h = 220
            pad = 15

            cx = wx + ww + pad
            if cx + card_w < rw:
                return cx, max(10, min(wy, rh - card_h - 10))
            cx = wx - card_w - pad
            if cx > 0:
                return cx, max(10, min(wy, rh - card_h - 10))
            cx = max(10, min(wx, rw - card_w - 10))
            cy = wy + wh + pad
            if cy + card_h < rh:
                return cx, cy
            return cx, max(10, wy - card_h - pad)

        def _create_card(self, widget, step):
            cx, cy = self._calc_card_pos(widget)
            self.card = ctk.CTkFrame(
                self.root, width=300, corner_radius=12,
                fg_color=self.CARD_BG,
                border_color=self.CARD_BORDER, border_width=2,
            )
            self.card.place(x=cx, y=cy)
            self.card.lift()

            n = len(self.steps)
            i = self.current_step

            self._step_label = ctk.CTkLabel(
                self.card, text=f"Step {i + 1} of {n}",
                font=("", 11), text_color=self.DOT_ACTIVE,
            )
            self._step_label.pack(padx=15, pady=(12, 2), anchor="w")

            self._title_label = ctk.CTkLabel(
                self.card, text=step["title"],
                font=("", 15, "bold"), anchor="w",
            )
            self._title_label.pack(padx=15, pady=(2, 6), anchor="w")

            self._desc_label = ctk.CTkLabel(
                self.card, text=step["description"],
                font=("", 12), anchor="w", justify="left",
                wraplength=270,
            )
            self._desc_label.pack(padx=15, pady=(0, 10), anchor="w")

            dots_frame = ctk.CTkFrame(self.card, fg_color="transparent")
            dots_frame.pack(pady=(0, 8))
            self._dot_labels = []
            for j in range(n):
                char = "\u25cf" if j == i else "\u25cb"
                color = self.DOT_ACTIVE if j == i else self.DOT_INACTIVE
                dot = ctk.CTkLabel(
                    dots_frame, text=char, font=("", 14),
                    text_color=color, cursor="hand2",
                )
                dot.pack(side="left", padx=3)
                dot.bind("<Button-1>",
                         lambda e, idx=j: self.goto_step(idx))
                self._dot_labels.append(dot)

            btn_frame = ctk.CTkFrame(self.card, fg_color="transparent")
            btn_frame.pack(padx=15, pady=(0, 12), fill="x")

            self._skip_btn = ctk.CTkButton(
                btn_frame, text="Skip", width=70, height=30,
                fg_color="transparent",
                hover_color=("gray75", "gray30"),
                text_color=COLORS["gray"], command=self.skip,
            )
            self._skip_btn.pack(side="left")

            is_last = (i == n - 1)
            self._next_btn = ctk.CTkButton(
                btn_frame,
                text="Got it!" if is_last else "Next \u2192",
                width=90, height=30,
                fg_color=self.GLOW, hover_color=self.GLOW_DIM,
                text_color="#000000", font=("", 12, "bold"),
                command=self.next_step,
            )
            self._next_btn.pack(side="right")

        def _update_card(self, widget, step):
            cx, cy = self._calc_card_pos(widget)
            self.card.place(x=cx, y=cy)
            self.card.lift()
            for f in self._glow_frames:
                f.lift()

            n = len(self.steps)
            i = self.current_step

            self._step_label.configure(text=f"Step {i + 1} of {n}")
            self._title_label.configure(text=step["title"])
            self._desc_label.configure(text=step["description"])

            for j, dot in enumerate(self._dot_labels):
                char = "\u25cf" if j == i else "\u25cb"
                color = self.DOT_ACTIVE if j == i else self.DOT_INACTIVE
                dot.configure(text=char, text_color=color)

            is_last = (i == n - 1)
            self._next_btn.configure(
                text="Got it!" if is_last else "Next \u2192")

        def next_step(self):
            self._stop_pulse()
            self._restore_highlight()
            self.current_step += 1
            if self.current_step >= len(self.steps):
                self.finish()
            else:
                self._show_step()

        def goto_step(self, idx):
            if 0 <= idx < len(self.steps) and idx != self.current_step:
                self._stop_pulse()
                self._restore_highlight()
                self.current_step = idx
                self._show_step()

        def skip(self):
            self._stop_pulse()
            self._restore_highlight()
            self.finish()

        def finish(self):
            self.is_active = False
            self._stop_pulse()
            self._restore_highlight()
            if self.card:
                try:
                    self.card.destroy()
                except Exception:
                    pass
                self.card = None
            self._dot_labels = []
            self.root.unbind("<Escape>")
            self.root.unbind("<Right>")
            self.root.unbind("<Left>")
            self.gui.mark_tutorial_completed()
            self._show_completion()

        def _show_completion(self):
            toast = ctk.CTkFrame(
                self.root, corner_radius=12,
                fg_color=self.CARD_BG,
                border_color=COLORS["green"], border_width=2,
            )
            toast.place(relx=0.5, rely=0.5, anchor="center")
            toast.lift()
            self._completion_toast = toast

            ctk.CTkLabel(
                toast, text="Tutorial Complete!",
                font=("", 16, "bold"),
            ).pack(padx=25, pady=(15, 5))

            ctk.CTkLabel(
                toast, text="You're ready to start using StreamVoiceAnon.",
                font=("", 12), text_color=COLORS["gray"],
            ).pack(padx=25, pady=(0, 10))

            btn_frame = ctk.CTkFrame(toast, fg_color="transparent")
            btn_frame.pack(padx=25, pady=(0, 15), fill="x")

            ctk.CTkButton(
                btn_frame, text="Close", width=80, height=30,
                fg_color=COLORS["gray"],
                hover_color=("gray75", "gray30"),
                command=toast.destroy,
            ).pack(side="left")

            ran_advanced = any(
                s.get("widget") == "alpha_slider" for s in self.steps
            )
            if not ran_advanced:
                def start_advanced():
                    toast.destroy()
                    self.start(advanced=True)

                ctk.CTkButton(
                    btn_frame, text="Advanced Tutorial",
                    width=140, height=30,
                    fg_color=self.GLOW, hover_color=self.GLOW_DIM,
                    text_color="#000000", font=("", 12, "bold"),
                    command=start_advanced,
                ).pack(side="right")

            def auto_close():
                try:
                    if toast.winfo_exists():
                        toast.destroy()
                except Exception:
                    pass

            self.root.after(8000, auto_close)

    # ── GUI Config ───────────────────────────────────────────────────────
    class GUIConfig:
        def __init__(self):
            self.reference_audio_path: str = ""
            self.sr_type: str = "sr_model"
            self.block_frame: int = 1
            self.n_frame_delay: int = 2
            self.alpha: float = 0.7
            self.sg_hostapi: str = ""
            self.wasapi_exclusive: bool = False
            self.sg_input_device: str = ""
            self.sg_output_device: str = ""
            self.samplerate: int = 44100
            self.channels: int = 2

    # ── Main GUI ─────────────────────────────────────────────────────────
    class GUI:
        def __init__(self, args, model_set=None, vad_model=None):
            self.gui_config = GUIConfig()
            self.config = Config()
            self.function = "vc"
            self.delay_time = 0
            self.stream = None
            self.model_set = model_set if model_set else load_models(args)
            if vad_model is not None:
                self.vad_model = vad_model
            else:
                from funasr import AutoModel
                self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
            self.hostapis = []
            self.input_devices = []
            self.output_devices = []
            self.input_devices_indices = []
            self.output_devices_indices = []
            self.presets = self.load_presets()
            self._applying_preset = False
            self.update_devices()
            self.build_ui()

        def load_presets(self):
            try:
                with open("configs/presets.json", "r") as f:
                    return json.load(f)
            except Exception:
                return {}

        def apply_preset(self, name):
            if name == "Custom" or name not in self.presets:
                self.preset_desc_label.configure(text="")
                return
            p = self.presets[name]
            self._applying_preset = True
            try:
                if "alpha" in p:
                    self.alpha_slider.set(p["alpha"])
                    self.alpha_label.configure(text=f"{p['alpha']:.2f}")
                    self.gui_config.alpha = p["alpha"]
                if "block_frame" in p:
                    self.bf_slider.set(p["block_frame"])
                    self.bf_label.configure(text=str(int(p["block_frame"])))
                    self.gui_config.block_frame = int(p["block_frame"])
                if "n_frame_delay" in p:
                    self.df_slider.set(p["n_frame_delay"])
                    self.df_label.configure(text=str(int(p["n_frame_delay"])))
                    self.gui_config.n_frame_delay = int(p["n_frame_delay"])
            finally:
                self._applying_preset = False
            self.preset_desc_label.configure(text=p.get("description", ""))

        # ── Config persistence ───────────────────────────────────────────
        def load_config(self):
            try:
                os.makedirs("configs/inuse", exist_ok=True)
                if not os.path.exists("configs/inuse/config.json"):
                    shutil.copy("configs/config.json", "configs/inuse/config.json")
                with open("configs/inuse/config.json", "r") as j:
                    data = json.load(j)
                    if data.get("sg_hostapi") in self.hostapis:
                        self.update_devices(hostapi_name=data["sg_hostapi"])
                        if (data.get("sg_input_device") not in self.input_devices
                                or data.get("sg_output_device") not in self.output_devices):
                            self.update_devices()
                            data["sg_hostapi"] = self.hostapis[0]
                            data["sg_input_device"] = self.input_devices[
                                self.input_devices_indices.index(sd.default.device[0])
                            ] if self.input_devices else ""
                            data["sg_output_device"] = self.output_devices[
                                self.output_devices_indices.index(sd.default.device[1])
                            ] if self.output_devices else ""
                    else:
                        data["sg_hostapi"] = self.hostapis[0] if self.hostapis else ""
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ] if self.input_devices else ""
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ] if self.output_devices else ""
            except Exception:
                data = {
                    "sg_hostapi": self.hostapis[0] if self.hostapis else "",
                    "sg_wasapi_exclusive": False,
                    "sg_input_device": self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ] if self.input_devices else "",
                    "sg_output_device": self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ] if self.output_devices else "",
                    "sr_type": "sr_model",
                    "alpha": 0.7,
                    "block_frame": 1,
                    "n_frame_delay": 2,
                }
            return data

        def save_config(self):
            settings = {
                "reference_audio_path": self.gui_config.reference_audio_path,
                "sg_hostapi": self.gui_config.sg_hostapi,
                "sg_wasapi_exclusive": self.gui_config.wasapi_exclusive,
                "sg_input_device": self.gui_config.sg_input_device,
                "sg_output_device": self.gui_config.sg_output_device,
                "sr_type": self.gui_config.sr_type,
                "alpha": self.gui_config.alpha,
                "block_frame": self.gui_config.block_frame,
                "n_frame_delay": self.gui_config.n_frame_delay,
                "preset": self.preset_var.get(),
                "tutorial_seen": getattr(self, "tutorial_seen", False),
            }
            os.makedirs("configs/inuse", exist_ok=True)
            with open("configs/inuse/config.json", "w") as j:
                json.dump(settings, j)

        # ── UI Construction ──────────────────────────────────────────────
        def build_ui(self):
            data = self.load_config()

            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")

            self.root = ctk.CTk()
            self.root.title("Stream-Voice-Anon")
            self.root.geometry("900x620")
            self.root.minsize(800, 580)
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)

            self.root.grid_columnconfigure(1, weight=1)
            self.root.grid_rowconfigure(0, weight=1)

            # ══════════════════════════════════════════════════════════════
            # SIDEBAR
            # ══════════════════════════════════════════════════════════════
            self.sidebar = ctk.CTkFrame(self.root, width=200, corner_radius=0,
                                        fg_color=COLORS["sidebar_bg"])
            self.sidebar.grid(row=0, column=0, sticky="nsew")
            self.sidebar.grid_rowconfigure(7, weight=1)

            ctk.CTkLabel(
                self.sidebar, text="Stream\nVoice Anon",
                font=("", 20, "bold"),
            ).grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

            ctk.CTkLabel(
                self.sidebar, text="Real-time speaker anonymization",
                font=("", 11), text_color=COLORS["gray"],
            ).grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

            # Start / Stop buttons
            self.start_btn = ctk.CTkButton(
                self.sidebar, text="Start", height=40,
                fg_color=COLORS["green"], hover_color=COLORS["green_hover"],
                font=("", 14, "bold"), command=self.on_start,
            )
            self.start_btn.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="ew")

            self.stop_btn = ctk.CTkButton(
                self.sidebar, text="Stop", height=32,
                fg_color=COLORS["red"], hover_color=COLORS["red_hover"],
                command=self.on_stop,
            )
            self.stop_btn.grid(row=3, column=0, padx=20, pady=(0, 5), sticky="new")

            # Preset selector
            ctk.CTkLabel(
                self.sidebar, text="PRESET", font=("", 11, "bold"),
                text_color=COLORS["gray"],
            ).grid(row=4, column=0, padx=20, pady=(18, 4), sticky="sw")

            preset_values = ["Custom"] + list(self.presets.keys())
            self.preset_var = ctk.StringVar(value=data.get("preset", "Custom"))
            self.preset_menu = ctk.CTkOptionMenu(
                self.sidebar, values=preset_values, variable=self.preset_var,
                command=self.on_preset_change, width=170)
            self.preset_menu.grid(row=5, column=0, padx=20, pady=2, sticky="ew")

            self.preset_desc_label = ctk.CTkLabel(
                self.sidebar, text="", font=("", 10),
                text_color=COLORS["gray"], anchor="w", justify="left",
                wraplength=160)
            self.preset_desc_label.grid(
                row=6, column=0, padx=20, pady=(4, 0), sticky="nw")

            # Mode selector at bottom
            self.mode_var = ctk.StringVar(value="vc")
            ctk.CTkLabel(
                self.sidebar, text="MODE", font=("", 11, "bold"),
                text_color=COLORS["gray"],
            ).grid(row=8, column=0, padx=20, pady=(10, 4), sticky="sw")

            self.mode_vc_radio = ctk.CTkRadioButton(
                self.sidebar, text="Voice Conversion", variable=self.mode_var,
                value="vc", command=self.on_mode_change)
            self.mode_vc_radio.grid(row=9, column=0, padx=20, pady=2, sticky="sw")

            self.mode_im_radio = ctk.CTkRadioButton(
                self.sidebar, text="Input Listening", variable=self.mode_var,
                value="im", command=self.on_mode_change)
            self.mode_im_radio.grid(row=10, column=0, padx=20, pady=(2, 10), sticky="sw")

            # Appearance mode
            ctk.CTkLabel(
                self.sidebar, text="THEME", font=("", 11, "bold"),
                text_color=COLORS["gray"],
            ).grid(row=11, column=0, padx=20, pady=(5, 4), sticky="sw")

            self.appearance_menu = ctk.CTkOptionMenu(
                self.sidebar, values=["Dark", "Light", "System"],
                command=lambda v: ctk.set_appearance_mode(v.lower()), width=170,
            )
            self.appearance_menu.set("Dark")
            self.appearance_menu.grid(row=12, column=0, padx=20, pady=(0, 8), sticky="sew")

            self.tutorial_btn = ctk.CTkButton(
                self.sidebar, text="Show Tutorial", height=28,
                fg_color="transparent", border_width=1,
                border_color=COLORS["gray"], text_color=COLORS["gray"],
                hover_color="#2a2a3e",
                command=self.run_tutorial, width=170)
            self.tutorial_btn.grid(row=13, column=0, padx=20, pady=(0, 15), sticky="sew")

            # ══════════════════════════════════════════════════════════════
            # MAIN CONTENT AREA — Tabview
            # ══════════════════════════════════════════════════════════════
            self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
            self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
            self.main_frame.grid_columnconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(0, weight=1)
            self.main_frame.grid_rowconfigure(1, weight=0)

            self.tabview = ctk.CTkTabview(self.main_frame)
            self.tabview.grid(row=0, column=0, sticky="nsew")

            # ── Tab 1: Voice Conversion ──────────────────────────────────
            tab_vc = self.tabview.add("Voice Conversion")
            tab_vc.grid_columnconfigure(0, weight=0)
            tab_vc.grid_columnconfigure(1, weight=1)
            tab_vc.grid_columnconfigure(2, weight=0)

            row = 0
            ctk.CTkLabel(tab_vc, text="Reference Audio",
                         font=("", 14, "bold")).grid(
                row=row, column=0, columnspan=3, padx=15, pady=(15, 8), sticky="w")
            row += 1

            self.ref_path_var = ctk.StringVar(value=data.get("reference_audio_path", ""))
            self.ref_entry = ctk.CTkEntry(tab_vc, textvariable=self.ref_path_var)
            self.ref_entry.grid(row=row, column=0, columnspan=2, padx=(15, 5), pady=4, sticky="ew")

            self.browse_btn = ctk.CTkButton(
                tab_vc, text="Browse", width=80, command=self.browse_audio)
            self.browse_btn.grid(row=row, column=2, padx=(0, 15), pady=4, sticky="e")
            row += 1

            sep1 = ctk.CTkFrame(tab_vc, height=2, fg_color=("gray80", "gray30"))
            sep1.grid(row=row, column=0, columnspan=3, padx=15, pady=10, sticky="ew")
            row += 1

            ctk.CTkLabel(tab_vc, text="Anonymization",
                         font=("", 14, "bold")).grid(
                row=row, column=0, columnspan=3, padx=15, pady=(5, 8), sticky="w")
            row += 1

            ctk.CTkLabel(tab_vc, text="Alpha:", width=90, anchor="w").grid(
                row=row, column=0, padx=(15, 5), pady=4, sticky="w")
            self.alpha_slider = ctk.CTkSlider(
                tab_vc, from_=0.0, to=1.0, number_of_steps=20,
                command=self.on_alpha_change)
            self.alpha_slider.set(data.get("alpha", 0.7))
            self.alpha_slider.grid(row=row, column=1, padx=5, pady=4, sticky="ew")
            self.alpha_label = ctk.CTkLabel(
                tab_vc, text=f"{data.get('alpha', 0.7):.2f}", width=40, anchor="e")
            self.alpha_label.grid(row=row, column=2, padx=(0, 15), pady=4, sticky="e")
            row += 1

            hint_frame = ctk.CTkFrame(tab_vc, fg_color="transparent", height=16)
            hint_frame.grid(row=row, column=0, columnspan=3, padx=15, pady=(0, 4), sticky="ew")
            ctk.CTkLabel(hint_frame, text="Max Privacy",
                         font=("", 10), text_color=COLORS["green"]).pack(side="left")
            ctk.CTkLabel(hint_frame, text="Less Anonymization",
                         font=("", 10), text_color=COLORS["gray"]).pack(side="right")
            row += 1

            sep2 = ctk.CTkFrame(tab_vc, height=2, fg_color=("gray80", "gray30"))
            sep2.grid(row=row, column=0, columnspan=3, padx=15, pady=10, sticky="ew")
            row += 1

            ctk.CTkLabel(tab_vc, text="Performance",
                         font=("", 14, "bold")).grid(
                row=row, column=0, columnspan=3, padx=15, pady=(5, 8), sticky="w")
            row += 1

            ctk.CTkLabel(tab_vc, text="Block frame:", width=90, anchor="w").grid(
                row=row, column=0, padx=(15, 5), pady=4, sticky="w")
            self.bf_slider = ctk.CTkSlider(
                tab_vc, from_=1, to=10, number_of_steps=9,
                command=self.on_block_frame_change)
            self.bf_slider.set(data.get("block_frame", 1))
            self.bf_slider.grid(row=row, column=1, padx=5, pady=4, sticky="ew")
            self.bf_label = ctk.CTkLabel(
                tab_vc, text=str(int(data.get("block_frame", 1))), width=40, anchor="e")
            self.bf_label.grid(row=row, column=2, padx=(0, 15), pady=4, sticky="e")
            row += 1

            ctk.CTkLabel(tab_vc, text="Delay frame:", width=90, anchor="w").grid(
                row=row, column=0, padx=(15, 5), pady=4, sticky="w")
            self.df_slider = ctk.CTkSlider(
                tab_vc, from_=0, to=8, number_of_steps=8,
                command=self.on_delay_frame_change)
            self.df_slider.set(data.get("n_frame_delay", 2))
            self.df_slider.grid(row=row, column=1, padx=5, pady=4, sticky="ew")
            self.df_label = ctk.CTkLabel(
                tab_vc, text=str(int(data.get("n_frame_delay", 2))), width=40, anchor="e")
            self.df_label.grid(row=row, column=2, padx=(0, 15), pady=4, sticky="e")
            row += 1

            # ── Tab 2: Audio Devices ─────────────────────────────────────
            tab_dev = self.tabview.add("Audio Devices")
            tab_dev.grid_columnconfigure(0, weight=0)
            tab_dev.grid_columnconfigure(1, weight=1)

            row = 0
            ctk.CTkLabel(tab_dev, text="Device Configuration",
                         font=("", 14, "bold")).grid(
                row=row, column=0, columnspan=2, padx=15, pady=(15, 8), sticky="w")
            row += 1

            ctk.CTkLabel(tab_dev, text="Audio API:", width=90, anchor="w").grid(
                row=row, column=0, padx=(15, 5), pady=6, sticky="w")
            api_ctrl = ctk.CTkFrame(tab_dev, fg_color="transparent")
            api_ctrl.grid(row=row, column=1, padx=(0, 15), pady=6, sticky="ew")
            api_ctrl.grid_columnconfigure(0, weight=0)

            self.hostapi_var = ctk.StringVar(
                value=data.get("sg_hostapi", self.hostapis[0] if self.hostapis else ""))
            self.hostapi_menu = ctk.CTkOptionMenu(
                api_ctrl, values=self.hostapis, variable=self.hostapi_var,
                command=self.on_hostapi_change, width=200)
            self.hostapi_menu.grid(row=0, column=0, padx=(0, 10), sticky="w")

            self.wasapi_var = ctk.BooleanVar(value=data.get("sg_wasapi_exclusive", False))
            self.wasapi_checkbox = ctk.CTkCheckBox(
                api_ctrl, text="Exclusive", variable=self.wasapi_var)
            self.wasapi_checkbox.grid(row=0, column=1, padx=(0, 10), sticky="w")
            self.reload_btn = ctk.CTkButton(
                api_ctrl, text="Reload", width=70, command=self.on_reload_devices)
            self.reload_btn.grid(row=0, column=2, sticky="e")
            row += 1

            ctk.CTkLabel(tab_dev, text="Input:", width=90, anchor="w").grid(
                row=row, column=0, padx=(15, 5), pady=6, sticky="w")
            self.input_dev_var = ctk.StringVar(value=data.get("sg_input_device", ""))
            self.input_dev_menu = ctk.CTkOptionMenu(
                tab_dev, values=self.input_devices if self.input_devices else [""],
                variable=self.input_dev_var, width=400)
            self.input_dev_menu.grid(row=row, column=1, padx=(0, 15), pady=6, sticky="ew")
            row += 1

            ctk.CTkLabel(tab_dev, text="Output:", width=90, anchor="w").grid(
                row=row, column=0, padx=(15, 5), pady=6, sticky="w")
            self.output_dev_var = ctk.StringVar(value=data.get("sg_output_device", ""))
            self.output_dev_menu = ctk.CTkOptionMenu(
                tab_dev, values=self.output_devices if self.output_devices else [""],
                variable=self.output_dev_var, width=400)
            self.output_dev_menu.grid(row=row, column=1, padx=(0, 15), pady=6, sticky="ew")
            row += 1

            sep3 = ctk.CTkFrame(tab_dev, height=2, fg_color=("gray80", "gray30"))
            sep3.grid(row=row, column=0, columnspan=2, padx=15, pady=10, sticky="ew")
            row += 1

            ctk.CTkLabel(tab_dev, text="Sample Rate:", width=90, anchor="w").grid(
                row=row, column=0, padx=(15, 5), pady=6, sticky="w")
            sr_ctrl = ctk.CTkFrame(tab_dev, fg_color="transparent")
            sr_ctrl.grid(row=row, column=1, padx=(0, 15), pady=6, sticky="ew")

            self.sr_type_var = ctk.StringVar(value=data.get("sr_type", "sr_model"))
            self.sr_model_radio = ctk.CTkRadioButton(
                sr_ctrl, text="Model (44100 Hz)", variable=self.sr_type_var,
                value="sr_model")
            self.sr_model_radio.pack(side="left", padx=(0, 15))
            self.sr_device_radio = ctk.CTkRadioButton(
                sr_ctrl, text="Device native", variable=self.sr_type_var,
                value="sr_device")
            self.sr_device_radio.pack(side="left", padx=(0, 15))
            row += 1

            # ══════════════════════════════════════════════════════════════
            # STATUS BAR
            # ══════════════════════════════════════════════════════════════
            self.status_bar = ctk.CTkFrame(self.main_frame, height=36,
                                           fg_color=COLORS["status_bg"],
                                           corner_radius=6)
            self.status_bar.grid(row=1, column=0, sticky="ew", pady=(5, 0))
            self.status_bar.grid_columnconfigure(3, weight=1)

            self.status_dot = ctk.CTkLabel(
                self.status_bar, text="\u25cf", font=("", 14),
                text_color=COLORS["gray"], width=20)
            self.status_dot.grid(row=0, column=0, padx=(10, 2), pady=6)

            self.status_label = ctk.CTkLabel(
                self.status_bar, text="Ready",
                font=("", 12), text_color=COLORS["gray"])
            self.status_label.grid(row=0, column=1, padx=(0, 20), pady=6)

            ctk.CTkLabel(self.status_bar, text="Delay:",
                         font=("", 11), text_color=COLORS["gray"]).grid(
                row=0, column=2, padx=(0, 2), pady=6)
            self.delay_display = ctk.CTkLabel(
                self.status_bar, text="-- ms", font=("", 11, "bold"))
            self.delay_display.grid(row=0, column=3, padx=(0, 15), pady=6, sticky="w")

            ctk.CTkLabel(self.status_bar, text="Inference:",
                         font=("", 11), text_color=COLORS["gray"]).grid(
                row=0, column=4, padx=(0, 2), pady=6)
            self.infer_display = ctk.CTkLabel(
                self.status_bar, text="-- ms", font=("", 11, "bold"))
            self.infer_display.grid(row=0, column=5, padx=(0, 15), pady=6)

            ctk.CTkLabel(self.status_bar, text="SR:",
                         font=("", 11), text_color=COLORS["gray"]).grid(
                row=0, column=6, padx=(0, 2), pady=6)
            self.sr_status = ctk.CTkLabel(
                self.status_bar, text="--", font=("", 11, "bold"))
            self.sr_status.grid(row=0, column=7, padx=(0, 15), pady=6)

            saved_preset = data.get("preset", "Custom")
            if saved_preset in self.presets:
                self.preset_desc_label.configure(
                    text=self.presets[saved_preset].get("description", ""))

            self._attach_tooltips()

            self.tutorial = TutorialManager(self)
            self.tutorial_seen = bool(data.get("tutorial_seen", False))
            if not self.tutorial_seen:
                self.root.after(800, self.tutorial.start)

            self.root.mainloop()

        def run_tutorial(self):
            self.tutorial.start()

        def mark_tutorial_completed(self):
            self.tutorial_seen = True
            try:
                self.save_config()
            except Exception:
                pass

        def _attach_tooltips(self):
            tips = {
                self.start_btn: "Start real-time voice conversion using the selected reference and devices.",
                self.stop_btn: "Stop the active stream and release audio devices.",
                self.mode_vc_radio: "Convert incoming microphone audio to the reference voice.",
                self.mode_im_radio: "Passthrough mode: play the input back without conversion (useful for checking the signal chain).",
                self.appearance_menu: "Switch between Dark, Light, and System appearance.",
                self.ref_entry: "Path to the reference audio file that defines the target voice.",
                self.browse_btn: "Pick a reference audio file (.wav / .mp3 / .flac / .m4a / .ogg / .opus).",
                self.preset_menu: "Quick presets for common privacy / latency tradeoffs. Moving any slider switches to Custom.",
                self.alpha_slider: "Noise mixing coefficient applied to the speaker embedding.\n0.00 = full noise (strongest anonymization).\n1.00 = untouched reference voice.",
                self.bf_slider: "Audio chunk size in frames. Larger blocks lower CPU load but add latency.",
                self.df_slider: "Decoder look-ahead in frames. Higher values give better quality but add latency.",
                self.hostapi_menu: "Audio backend. On Windows, WASAPI is recommended for the lowest latency.",
                self.wasapi_checkbox: "Enable WASAPI exclusive mode. Gives the lowest latency but blocks other apps from the device.",
                self.reload_btn: "Rescan available audio devices for the current backend.",
                self.input_dev_menu: "Input device (microphone) used as the source signal.",
                self.output_dev_menu: "Output device used for the converted voice. Pick a virtual cable to route into other apps.",
                self.sr_model_radio: "Use the model's native sample rate (44100 Hz). Requires resampling the device streams.",
                self.sr_device_radio: "Use the device's native sample rate. Avoids resampling on the output side.",
                self.tutorial_btn: "Replay the onboarding tutorial.",
            }
            self._tooltips = [Tooltip(w, t) for w, t in tips.items()]

        # ── Event handlers ───────────────────────────────────────────────
        def browse_audio(self):
            path = filedialog.askopenfilename(
                initialdir=os.path.join(os.getcwd(), "examples", "reference"),
                filetypes=[
                    ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg *.opus"),
                    ("All Files", "*.*"),
                ],
            )
            if path:
                self.ref_path_var.set(path)

        def on_alpha_change(self, value):
            self.alpha_label.configure(text=f"{value:.2f}")
            self.gui_config.alpha = value
            self._mark_custom_preset()
            global reference_wav_name
            reference_wav_name = ""

        def on_block_frame_change(self, value):
            val = int(round(value))
            self.bf_label.configure(text=str(val))
            self.gui_config.block_frame = val
            self._mark_custom_preset()

        def on_delay_frame_change(self, value):
            val = int(round(value))
            self.df_label.configure(text=str(val))
            self.gui_config.n_frame_delay = val
            self._mark_custom_preset()

        def on_preset_change(self, name):
            self.apply_preset(name)

        def _mark_custom_preset(self):
            if self._applying_preset:
                return
            if self.preset_var.get() != "Custom":
                self.preset_var.set("Custom")
                self.preset_desc_label.configure(text="")

        def on_hostapi_change(self, value):
            self.update_devices(hostapi_name=value)
            self.input_dev_menu.configure(
                values=self.input_devices if self.input_devices else [""])
            self.output_dev_menu.configure(
                values=self.output_devices if self.output_devices else [""])
            if self.input_devices:
                self.input_dev_var.set(self.input_devices[0])
            if self.output_devices:
                self.output_dev_var.set(self.output_devices[0])

        def on_reload_devices(self):
            self.on_hostapi_change(self.hostapi_var.get())

        def on_mode_change(self):
            self.function = self.mode_var.get()

        def set_status(self, text, color):
            self.status_label.configure(text=text, text_color=color)
            self.status_dot.configure(text_color=color)

        def on_start(self):
            global flag_vc
            if flag_vc:
                return
            ref_path = self.ref_path_var.get().strip()
            if not ref_path:
                self.show_error("Please choose a reference audio file.")
                return
            if not os.path.exists(ref_path):
                self.show_error(f"File not found:\n{ref_path}")
                return
            pattern = re.compile("[^\x00-\x7F]+")
            if pattern.findall(ref_path):
                self.show_error("Audio path contains non-ASCII characters.\n"
                                "Please move the file to a path with ASCII characters only.")
                return

            self.gui_config.reference_audio_path = ref_path
            self.gui_config.sg_hostapi = self.hostapi_var.get()
            self.gui_config.wasapi_exclusive = self.wasapi_var.get()
            self.gui_config.sg_input_device = self.input_dev_var.get()
            self.gui_config.sg_output_device = self.output_dev_var.get()
            self.gui_config.sr_type = self.sr_type_var.get()
            self.set_devices(self.gui_config.sg_input_device,
                             self.gui_config.sg_output_device)
            self.save_config()

            self.set_status("Starting...", COLORS["yellow"])
            self.root.update()
            self.start_vc()

        def on_stop(self):
            self.stop_stream()
            self.set_status("Stopped", COLORS["gray"])
            self.delay_display.configure(text="-- ms")
            self.infer_display.configure(text="-- ms")
            self.sr_status.configure(text="--")

        def on_close(self):
            self.stop_stream()
            self.root.destroy()

        def show_error(self, message):
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Error")
            dialog.geometry("400x150")
            dialog.attributes("-topmost", True)
            dialog.grab_set()
            ctk.CTkLabel(dialog, text=message, wraplength=360,
                         justify="center").pack(pady=(25, 15), padx=20)
            ctk.CTkButton(dialog, text="OK", width=100,
                          command=dialog.destroy).pack(pady=(0, 15))

        # ── Audio pipeline ───────────────────────────────────────────────
        def start_vc(self):
            if device.type == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

            self.reference_wav, _ = librosa.load(
                self.gui_config.reference_audio_path, sr=self.model_set.sr)
            self.gui_config.samplerate = (
                self.model_set.sr
                if self.gui_config.sr_type == "sr_model"
                else self.get_device_samplerate()
            )
            self.gui_config.channels = self.get_device_channels()
            self.block_frame = int(self.gui_config.block_frame * 2048)
            self.input_wav = torch.zeros(
                self.block_frame, device=self.config.device, dtype=torch.float32)
            self.output_buffer = self.input_wav.clone()
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate, new_freq=16000,
                dtype=torch.float32).to(self.config.device)
            if self.model_set.sr != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.model_set.sr, new_freq=self.gui_config.samplerate,
                    dtype=torch.float32).to(self.config.device)
            else:
                self.resampler2 = None
            self.n_frame_delay = self.gui_config.n_frame_delay
            self.alpha = self.gui_config.alpha

            # Warmup with progress
            n_warmup_chunks = int(self.gui_config.n_frame_delay) + 3
            self.set_status("Warming up...", COLORS["yellow"])
            self.root.update()

            warmup_win = ctk.CTkToplevel(self.root)
            warmup_win.withdraw()
            warmup_win.title("Warming Up")
            warmup_win.geometry("420x110")
            warmup_win.resizable(False, False)
            warmup_win.transient(self.root)
            warmup_label = ctk.CTkLabel(
                warmup_win, text="Compiling & warming up...", font=("", 12))
            warmup_label.pack(pady=(20, 8))
            warmup_bar = ctk.CTkProgressBar(warmup_win, width=370)
            warmup_bar.pack(pady=(0, 5))
            warmup_bar.set(0)
            ctk.CTkLabel(
                warmup_win, text="First run may take 1-2 minutes",
                font=("", 10), text_color=COLORS["gray"]).pack()
            warmup_win.update_idletasks()
            warmup_win.deiconify()
            warmup_win.lift()
            warmup_win.attributes("-topmost", True)
            warmup_win.grab_set()
            warmup_win.update()

            dummy_wav = torch.zeros(
                self.block_frame, device=self.config.device, dtype=torch.float32)
            for i in range(n_warmup_chunks):
                warmup_label.configure(
                    text=f"Compiling kernel {i + 1}/{n_warmup_chunks}...")
                warmup_bar.set((i + 1) / n_warmup_chunks)
                warmup_win.update()
                custom_infer(
                    self.model_set, self.reference_wav,
                    self.gui_config.reference_audio_path, dummy_wav,
                    self.n_frame_delay, self.alpha,
                )

            warmup_win.destroy()

            global reference_wav_name
            reference_wav_name = ""
            self.vad_cache = {}
            self.vad_chunk_size = min(
                500, int(1000 * self.gui_config.block_frame * 2048 / 44100))
            self.vad_speech_detected = False
            self.start_stream()
            self.sr_status.configure(text=f"{self.gui_config.samplerate} Hz")
            self.set_status("Streaming", COLORS["green"])
            if self.stream is not None:
                self.delay_display.configure(
                    text=f"{int(np.round(self.stream.latency[-1] * 1000))} ms")

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                extra_settings = None
                if ("WASAPI" in self.gui_config.sg_hostapi
                        and self.gui_config.wasapi_exclusive):
                    extra_settings = sd.WasapiSettings(exclusive=True)
                self.stream = sd.Stream(
                    callback=self.audio_callback,
                    blocksize=self.block_frame,
                    samplerate=self.gui_config.samplerate,
                    channels=self.gui_config.channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                )
                self.stream.start()

        def stop_stream(self):
            global flag_vc
            if flag_vc:
                flag_vc = False
                if self.stream is not None:
                    self.stream.abort()
                    self.stream.close()
                    self.stream = None

        def audio_callback(self, indata, outdata, frames, times, status):
            global flag_vc
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)

            self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
            self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(
                self.config.device)

            if self.function == "vc":
                if device.type == "mps":
                    start_event = torch.mps.event.Event(enable_timing=True)
                    end_event = torch.mps.event.Event(enable_timing=True)
                    torch.mps.synchronize()
                else:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                start_event.record()
                infer_wav = custom_infer(
                    self.model_set, self.reference_wav,
                    self.gui_config.reference_audio_path, self.input_wav,
                    self.n_frame_delay, self.alpha,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
                end_event.record()
                if device.type == "mps":
                    torch.mps.synchronize()
                else:
                    torch.cuda.synchronize()
            else:
                infer_wav = self.input_wav.clone()

            outdata[:] = (
                infer_wav[:self.block_frame][None]
                .repeat(self.gui_config.channels, 1)
                .t().cpu().numpy()
            )

            total_time = time.perf_counter() - start_time
            if flag_vc:
                self.root.after(0, lambda t=total_time: self.infer_display.configure(
                    text=f"{int(t * 1000)} ms"))

        # ── Device management ────────────────────────────────────────────
        def update_devices(self, hostapi_name=None):
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [h["name"] for h in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0] if self.hostapis else ""
            self.input_devices = [
                d["name"] for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices = [
                d["name"] for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]

        def set_devices(self, input_device, output_device):
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)]

        def get_device_samplerate(self):
            return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

        def get_device_channels(self):
            max_in = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
            max_out = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
            return min(max_in, max_out, 2)


    # ── Entry point ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="configs/config_firefly_arvcasr_8192_delay0_8.yaml")
    parser.add_argument("--checkpoint_path", type=str,
                        default="pretrained_checkpoints/dual_ar_delay_0_8.pth")
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()

    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"
    if torch.cuda.is_available():
        device = torch.device(cuda_target)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Loading screen ───────────────────────────────────────────────────
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    load_win = ctk.CTk()
    load_win.title("Stream-Voice-Anon")
    load_win.geometry("420x140")
    load_win.resizable(False, False)

    ctk.CTkLabel(load_win, text="Stream-Voice-Anon",
                 font=("", 18, "bold")).pack(pady=(20, 5))
    load_label = ctk.CTkLabel(load_win, text="Loading models...",
                              font=("", 12), text_color=COLORS["gray"])
    load_label.pack(pady=(0, 8))
    load_bar = ctk.CTkProgressBar(load_win, width=370)
    load_bar.pack(pady=(0, 10))
    load_bar.set(0)
    load_win.update()

    load_label.configure(text="Step 1/2: Loading main model...")
    load_bar.set(0.1)
    load_win.update()
    model_set = load_models(args)

    load_label.configure(text="Step 2/2: Loading VAD model...")
    load_bar.set(0.6)
    load_win.update()
    from funasr import AutoModel
    vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

    load_label.configure(text="Ready!")
    load_bar.set(1.0)
    load_win.update()
    load_win.destroy()

    gui = GUI(args, model_set=model_set, vad_model=vad_model)
