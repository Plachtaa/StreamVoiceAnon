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
import multiprocessing
import warnings
import yaml

warnings.simplefilter("ignore")

from tqdm import tqdm
import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import os
import sys
import torch
# Load model and configuration
device = None

flag_vc = False

reference_wav_name = ""
decode_chunk_frames = 0

fp16 = False
@torch.no_grad()
def custom_infer(model_set,
                 reference_wav,
                 new_reference_wav_name,
                 input_wav,
                 n_frame_delay=4,
                 ):
    global reference_wav_name, decode_chunk_frames
    if reference_wav_name != new_reference_wav_name or decode_chunk_frames != input_wav.size(-1) // 2048:
        model_set.prefill_prompt(torch.from_numpy(reference_wav).to(device).unsqueeze(0), max_prompt_frames=64, delay=n_frame_delay,)
        model_set.setup_stream_caches(
            encode_window_frames=64,
            decode_window_frames=64,
            max_seq_frames=768,
            buffer_frames=32,
            decode_chunk_frames=input_wav.size(-1) // 2048,
        )
        reference_wav_name = new_reference_wav_name
        decode_chunk_frames = input_wav.size(-1) // 2048
    print(f"input wave has shape {input_wav.shape}")
    pred_wave = model_set.process_one_chunk(input_wav.to(device).unsqueeze(0))
    return pred_wave.squeeze().cpu()

def load_models(args):
    from evaluations.infer_arvc import InferenceWrapper
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    infer_wrapper = InferenceWrapper(config_path, checkpoint_path, compile_ar=True,
                                     compile_decoder=True, compile_encoder=True
                                     )
    return infer_wrapper

def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)

class Config:
    def __init__(self):
        self.device = device


if __name__ == "__main__":
    import json
    import multiprocessing
    import re
    import threading
    import time
    import traceback
    from multiprocessing import Queue, cpu_count
    import argparse

    import librosa
    import numpy as np
    import FreeSimpleGUI as sg
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat


    current_dir = os.getcwd()
    n_cpu = cpu_count()
    class GUIConfig:
        def __init__(self) -> None:
            self.reference_audio_path: str = ""
            # self.index_path: str = ""
            self.sr_type: str = "sr_model"
            self.block_frame: int = 3  # 2048 wave length per frame
            self.n_frame_delay: int = 4,
            self.threhold: int = -60
            self.I_noise_reduce: bool = False
            self.O_noise_reduce: bool = False
            self.sg_hostapi: str = ""
            self.wasapi_exclusive: bool = False
            self.sg_input_device: str = ""
            self.sg_output_device: str = ""


    class GUI:
        def __init__(self, args) -> None:
            self.gui_config = GUIConfig()
            self.config = Config()
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.model_set = load_models(args)
            from funasr import AutoModel
            self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
            self.update_devices()
            self.launcher()

        def load(self):
            try:
                os.makedirs("configs/inuse", exist_ok=True)
                if not os.path.exists("configs/inuse/config.json"):
                    shutil.copy("configs/config.json", "configs/inuse/config.json")
                with open("configs/inuse/config.json", "r") as j:
                    data = json.load(j)
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
                    if data["sg_hostapi"] in self.hostapis:
                        self.update_devices(hostapi_name=data["sg_hostapi"])
                        if (
                            data["sg_input_device"] not in self.input_devices
                            or data["sg_output_device"] not in self.output_devices
                        ):
                            self.update_devices()
                            data["sg_hostapi"] = self.hostapis[0]
                            data["sg_input_device"] = self.input_devices[
                                self.input_devices_indices.index(sd.default.device[0])
                            ]
                            data["sg_output_device"] = self.output_devices[
                                self.output_devices_indices.index(sd.default.device[1])
                            ]
                    else:
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
            except:
                with open("configs/inuse/config.json", "w") as j:
                    data = {
                        "sg_hostapi": self.hostapis[0],
                        "sg_wasapi_exclusive": False,
                        "sg_input_device": self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ],
                        "sg_output_device": self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ],
                        "sr_type": "sr_model",
                        "block_frame": 3,
                        "n_frame_delay": 4,
                    }
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
            return data

        def launcher(self):
            self.config = Config()
            data = self.load()
            sg.theme("LightBlue3")
            layout = [
                [
                    sg.Frame(
                        title="Load reference audio",
                        layout=[
                            [
                                sg.Input(
                                    default_text=data.get("reference_audio_path", ""),
                                    key="reference_audio_path",
                                ),
                                sg.FileBrowse(
                                    "choose an audio file",
                                    initial_folder=os.path.join(
                                        os.getcwd(), "examples/reference"
                                    ),
                                    file_types=[
                                        ("WAV Files", "*.wav"),
                                        ("MP3 Files", "*.mp3"),
                                        ("FLAC Files", "*.flac"),
                                        ("M4A Files", "*.m4a"),
                                        ("OGG Files", "*.ogg"),
                                        ("Opus Files", "*.opus"),
                                    ],
                                ),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text("Device type"),
                                sg.Combo(
                                    self.hostapis,
                                    key="sg_hostapi",
                                    default_value=data.get("sg_hostapi", ""),
                                    enable_events=True,
                                    size=(20, 1),
                                ),
                                sg.Checkbox(
                                    "WASAPI Exclusive Device",
                                    key="sg_wasapi_exclusive",
                                    default=data.get("sg_wasapi_exclusive", False),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Input Device"),
                                sg.Combo(
                                    self.input_devices,
                                    key="sg_input_device",
                                    default_value=data.get("sg_input_device", ""),
                                    enable_events=True,
                                    size=(45, 1),
                                ),
                            ],
                            [
                                sg.Text("Output Device"),
                                sg.Combo(
                                    self.output_devices,
                                    key="sg_output_device",
                                    default_value=data.get("sg_output_device", ""),
                                    enable_events=True,
                                    size=(45, 1),
                                ),
                            ],
                            [
                                sg.Button("Reload devices", key="reload_devices"),
                                sg.Radio(
                                    "Use model sampling rate",
                                    "sr_type",
                                    key="sr_model",
                                    default=data.get("sr_model", True),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "Use device sampling rate",
                                    "sr_type",
                                    key="sr_device",
                                    default=data.get("sr_device", False),
                                    enable_events=True,
                                ),
                                sg.Text("Sampling rate:"),
                                sg.Text("", key="sr_stream"),
                            ],
                        ],
                        title="Sound Device",
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text("Block frame"),
                                sg.Slider(
                                    range=(1, 10),
                                    key="block_frame",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("block_frame", 3),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Delay frame"),
                                sg.Slider(
                                    range=(0, 8),
                                    key="n_frame_delay",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("n_frame_delay", 4),
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title="Performance settings",
                    ),
                ],
                [
                    sg.Button("Start Voice Conversion", key="start_vc"),
                    sg.Button("Stop Voice Conversion", key="stop_vc"),
                    sg.Radio(
                        "Input listening",
                        "function",
                        key="im",
                        default=False,
                        enable_events=True,
                    ),
                    sg.Radio(
                        "Voice Conversion",
                        "function",
                        key="vc",
                        default=True,
                        enable_events=True,
                    ),
                    sg.Text("Algorithm delay (ms):"),
                    sg.Text("0", key="delay_time"),
                    sg.Text("Inference time (ms):"),
                    sg.Text("0", key="infer_time"),
                ],
            ]
            self.window = sg.Window("Stream-VC - GUI", layout=layout, finalize=True)
            self.event_handler()

        def event_handler(self):
            global flag_vc
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    self.stop_stream()
                    exit()
                if event == "reload_devices" or event == "sg_hostapi":
                    self.gui_config.sg_hostapi = values["sg_hostapi"]
                    self.update_devices(hostapi_name=values["sg_hostapi"])
                    if self.gui_config.sg_hostapi not in self.hostapis:
                        self.gui_config.sg_hostapi = self.hostapis[0]
                    self.window["sg_hostapi"].Update(values=self.hostapis)
                    self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                    if (
                        self.gui_config.sg_input_device not in self.input_devices
                        and len(self.input_devices) > 0
                    ):
                        self.gui_config.sg_input_device = self.input_devices[0]
                    self.window["sg_input_device"].Update(values=self.input_devices)
                    self.window["sg_input_device"].Update(
                        value=self.gui_config.sg_input_device
                    )
                    if self.gui_config.sg_output_device not in self.output_devices:
                        self.gui_config.sg_output_device = self.output_devices[0]
                    self.window["sg_output_device"].Update(values=self.output_devices)
                    self.window["sg_output_device"].Update(
                        value=self.gui_config.sg_output_device
                    )
                if event == "start_vc" and not flag_vc:
                    if self.set_values(values) == True:
                        printt("cuda_is_available: %s", torch.cuda.is_available())
                        self.start_vc()
                        settings = {
                            "reference_audio_path": values["reference_audio_path"],
                            # "index_path": values["index_path"],
                            "sg_hostapi": values["sg_hostapi"],
                            "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "sr_type": ["sr_model", "sr_device"][
                                [
                                    values["sr_model"],
                                    values["sr_device"],
                                ].index(True)
                            ],
                            # "threhold": values["threhold"],
                            "block_frame": values["block_frame"],
                            "n_frame_delay": values["n_frame_delay"],
                        }
                        with open("configs/inuse/config.json", "w") as j:
                            json.dump(settings, j)
                        if self.stream is not None:
                            self.delay_time = (
                                self.stream.latency[-1]
                            )
                        self.window["sr_stream"].update(self.gui_config.samplerate)
                        self.window["delay_time"].update(
                            int(np.round(self.delay_time * 1000))
                        )
                # Parameter hot update
                # if event == "threhold":
                #     self.gui_config.threhold = values["threhold"]
                elif event == "block_frame":
                    self.gui_config.block_frame = values["block_frame"]
                elif event == "n_frame_delay":
                    self.gui_config.n_frame_delay = values["n_frame_delay"]
                elif event in ["vc", "im"]:
                    self.function = event
                elif event == "stop_vc" or event != "start_vc":
                    # Other parameters do not support hot update
                    self.stop_stream()

        def set_values(self, values):
            if len(values["reference_audio_path"].strip()) == 0:
                sg.popup("Choose an audio file")
                return False
            pattern = re.compile("[^\x00-\x7F]+")
            if pattern.findall(values["reference_audio_path"]):
                sg.popup("audio file path contains non-ascii characters")
                return False
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
            self.gui_config.sg_input_device = values["sg_input_device"]
            self.gui_config.sg_output_device = values["sg_output_device"]
            self.gui_config.reference_audio_path = values["reference_audio_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][
                [
                    values["sr_model"],
                    values["sr_device"],
                ].index(True)
            ]
            # self.gui_config.threhold = values["threhold"]
            self.gui_config.block_frame = values["block_frame"]
            self.gui_config.n_frame_delay = values["n_frame_delay"]
            return True

        def start_vc(self):
            if device.type == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
            self.reference_wav, _ = librosa.load(
                self.gui_config.reference_audio_path, sr=self.model_set.sr
            )
            self.gui_config.samplerate = (
                self.model_set.sr
                if self.gui_config.sr_type == "sr_model"
                else self.get_device_samplerate()
            )
            self.gui_config.channels = self.get_device_channels()
            self.block_frame = int(self.gui_config.block_frame * 2048)
            self.input_wav: torch.Tensor = torch.zeros(
                self.block_frame,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.output_buffer: torch.Tensor = self.input_wav.clone()
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.config.device)
            if self.model_set.sr != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.model_set.sr,
                    new_freq=self.gui_config.samplerate,
                    dtype=torch.float32,
                ).to(self.config.device)
            else:
                self.resampler2 = None
            self.n_frame_delay = self.gui_config.n_frame_delay
            self.vad_cache = {}
            self.vad_chunk_size = min(500, int(1000 * self.gui_config.block_frame * 2048 / 44100))
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False
            self.start_stream()

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                if (
                    "WASAPI" in self.gui_config.sg_hostapi
                    and self.gui_config.sg_wasapi_exclusive
                ):
                    extra_settings = sd.WasapiSettings(exclusive=True)
                else:
                    extra_settings = None
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

        def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):
            """
            Audio block callback function
            """
            global flag_vc
            print(indata.shape)
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)

            # # VAD first
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # torch.cuda.synchronize()
            # start_event.record()
            # indata_16k = librosa.resample(indata, orig_sr=self.gui_config.samplerate, target_sr=16000)
            # indata_16k = indata_16k[..., -(indata_16k.shape[-1] // 160 * 160):]
            # print(indata_16k.shape)
            # res = self.vad_model.generate(input=indata_16k, cache=self.vad_cache, is_final=False,
            #                               chunk_size=self.vad_chunk_size)
            # res_value = res[0]["value"]
            # print(res_value)
            # if len(res_value) % 2 == 1 and not self.vad_speech_detected:
            #     self.vad_speech_detected = True
            # elif len(res_value) % 2 == 1 and self.vad_speech_detected:
            #     self.set_speech_detected_false_at_end_flag = True
            # end_event.record()
            # torch.cuda.synchronize()  # Wait for the events to be recorded!
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f"Time taken for VAD: {elapsed_time_ms}ms")

            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
                self.config.device
            )
            print(f"preprocess time: {time.perf_counter() - start_time:.2f}")
            # infer
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
                    self.model_set,
                    self.reference_wav,
                    self.gui_config.reference_audio_path,
                    self.input_wav,
                    self.n_frame_delay,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
                end_event.record()
                if device.type == "mps":
                    torch.mps.synchronize()  # MPS - Wait for the events to be recorded!
                else:
                    torch.cuda.synchronize()  # Wait for the events to be recorded!
                elapsed_time_ms = start_event.elapsed_time(end_event)
                print(f"Time taken for VC: {elapsed_time_ms}ms")
                # if not self.vad_speech_detected:
                #     infer_wav = torch.zeros_like(infer_wav)
            else:
                infer_wav = self.input_wav.clone()

            outdata[:] = (
                infer_wav[: self.block_frame][None]
                .repeat(self.gui_config.channels, 1)
                .t()
                .cpu()
                .numpy()
            )

            total_time = time.perf_counter() - start_time
            if flag_vc:
                self.window["infer_time"].update(int(total_time * 1000))

            # if self.set_speech_detected_false_at_end_flag:
            #     self.vad_speech_detected = False
            #     self.set_speech_detected_false_at_end_flag = False

            print(f"Infer time: {total_time:.2f}")

        def update_devices(self, hostapi_name=None):
            """Get input and output devices."""
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0]
            self.input_devices = [
                d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices = [
                d["name"]
                for d in devices
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
            """set input and output devices."""
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)
            ]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)
            ]
            printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
            printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

        def get_device_samplerate(self):
            return int(
                sd.query_devices(device=sd.default.device[0])["default_samplerate"]
            )

        def get_device_channels(self):
            max_input_channels = sd.query_devices(device=sd.default.device[0])[
                "max_input_channels"
            ]
            max_output_channels = sd.query_devices(device=sd.default.device[1])[
                "max_output_channels"
            ]
            return min(max_input_channels, max_output_channels, 2)


    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config_firefly_arvcasr_8192_delay0_8.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="pretrained_checkpoints/dual_ar_delay_0_8.pth")

    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda" 

    if torch.cuda.is_available():
        device = torch.device(cuda_target)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    gui = GUI(args)