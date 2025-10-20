import sys
sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
import hydra
import yaml
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from tqdm import tqdm

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True

class InferenceWrapper:
    def __init__(self, config_path, checkpoint_path, compile_encoder=False, compile_decoder=False, compile_ar=False, fp16=False):
        self.config = yaml.safe_load(open(config_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models(compile_ar=compile_ar, compile_encoder=compile_encoder, compile_decoder=compile_decoder, fp16=fp16)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.sr = self.config['preprocess_params']['sr']


    def _init_models(self, **kwargs):
        """Initialize models and optimizers"""

        # Initialize main model
        self._init_main_model(**kwargs)

        # Initialize helper models
        self._init_helper_models(**kwargs)

    def _init_main_model(self, compile_ar=False, fp16=False, **kwargs):
        """Initialize the main model"""
        cfg = DictConfig(yaml.safe_load(open(self.config['model_params']['config_path'])))
        self.model = hydra.utils.instantiate(cfg)
        self.model.setup_caches(
            max_batch_size=1,
            max_seq_len=2048,
            dtype=torch.float16,
        )
        self.model.to(self.device)
        self.model.eval()
        if fp16:
            self.model.half()
        if compile_ar:
            self.model.compile_ar_decode_fn()

    def _init_helper_models(self, compile_decoder=False, compile_encoder=False, **kwargs):
        speechtokenizer_cfg = DictConfig(yaml.safe_load(open(self.config['speech_tokenizer']['config_path'])))
        self.speech_tokenizer = hydra.utils.instantiate(speechtokenizer_cfg).to(self.device)
        speech_tokenizer_sd = torch.load(
            self.config["speech_tokenizer"]["checkpoint_path"], map_location="cpu"
        )
        if 'net' in speech_tokenizer_sd:
            speech_tokenizer_sd = speech_tokenizer_sd['net']
        # strip 'module.' prefix if exists
        speech_tokenizer_sd = {
            k[7:] if k.startswith('module.') else k: v for k, v in speech_tokenizer_sd.items()
        }
        missing_keys, unexpected_keys = self.speech_tokenizer.load_state_dict(
            speech_tokenizer_sd, strict=False
        )
        print(f"Missing keys in speech tokenizer: {missing_keys}")
        print(f"Unexpected keys in speech tokenizer: {unexpected_keys}")
        for param in self.speech_tokenizer.parameters():
            param.requires_grad = False


        firefly_cfg = DictConfig(yaml.safe_load(open(self.config['firefly']['config_path'])))
        self.firefly = hydra.utils.instantiate(firefly_cfg).to(self.device)
        self.firefly.load_state_dict(
            torch.load(self.config["firefly"]["checkpoint_path"], map_location="cpu"),
            strict=False,
        )
        self.firefly.remove_parametrizations()
        for param in self.firefly.parameters():
            param.requires_grad = False

        self.style_encoder = hydra.utils.instantiate(
            OmegaConf.load(self.config["style_encoder"]["config_path"])
        ).to(self.device).eval()
        self.style_encoder.load_state_dict(
            torch.load(
                self.config["style_encoder"]["checkpoint_path"], map_location="cpu"
            ),
            strict=False,
        )
        for param in self.style_encoder.parameters():
            param.requires_grad = False

        # sparktts timbre encoder
        self.timbre_encoder = hydra.utils.instantiate(
            OmegaConf.load(self.config["timbre_encoder"]["config_path"])
        ).to(self.device)
        self.timbre_encoder.load_state_dict(
            torch.load(
                self.config["timbre_encoder"]["checkpoint_path"], map_location="cpu"
            ),
            strict=False,
        )
        for param in self.timbre_encoder.parameters():
            param.requires_grad = False

        self.speech_tokenizer.eval()
        self.firefly.eval()
        self.style_encoder.eval()
        self.timbre_encoder.eval()

        if compile_decoder:
            self.firefly.head = torch.compile(
                self.firefly.head,
                fullgraph=True,
                backend="inductor" if torch.cuda.is_available() else "aot_eager",
                mode="reduce-overhead" if torch.cuda.is_available() else None,
            )

        if compile_encoder:
            self.compiled_speech_tokenizer_encode = torch.compile(
                self.speech_tokenizer.encode,
                fullgraph=True,
                backend="inductor" if torch.cuda.is_available() else "aot_eager",
                mode="reduce-overhead" if torch.cuda.is_available() else None,
            )
        else:
            self.compiled_speech_tokenizer_encode = self.speech_tokenizer.encode

    def filter_state_dict_shapes(self, params, model):
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in params.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        skipped_keys = set(params.keys()) - set(filtered_state_dict.keys())
        if skipped_keys:
            print(
                f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}"
            )
        return filtered_state_dict, skipped_keys

    def _load_checkpoint(self, checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu")
        missing_keys, unexpected_keys = self.model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(missing_keys)
        print(unexpected_keys)

    @torch.no_grad()
    def wav2target_fn(self, waves, wave_lengths):
        (target, quantized), target_lengths = self.firefly.encode(waves, wave_lengths)
        target_size = target.size(2)
        return target, quantized, target_lengths, target_size

    @torch.no_grad()
    def code2wav_fn(self, code):
        wav = self.firefly.head(self.firefly.quantizer.decode(code))
        return wav

    @torch.no_grad()
    def calculate_style_vec(
            self,
            audio_16k_tensor: torch.Tensor,
            wave_lens: torch.Tensor,
    ):
        feat_list = []
        for bib in range(audio_16k_tensor.size(0)):
            feat = kaldi.fbank(
                audio_16k_tensor[bib: bib + 1, : wave_lens[bib]],
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        max_feat_len = max([feat.size(0) for feat in feat_list])
        feat_lens = (
                torch.tensor([feat.size(0) for feat in feat_list], dtype=torch.int32).to(
                    audio_16k_tensor.device
                )
                // 2
        )
        feat_list = [
            F.pad(
                feat,
                (0, 0, 0, max_feat_len - feat.size(0)),
                value=float(feat.min().item()),
            )
            for feat in feat_list
        ]
        feat = torch.stack(feat_list, dim=0)
        style_vectors = self.style_encoder(feat, feat_lens)
        return style_vectors

    @torch.no_grad()
    def calculate_timbre_latent(
            self,
            audio_16k_tensor: torch.Tensor,
            wave_lens: torch.Tensor,
    ):
        resample_timbre, resample_timbre_tokens = self.timbre_encoder.tokenize_wav(
            audio_16k_tensor, wave_lens
        )
        resample_timbre = resample_timbre.mT
        return resample_timbre

    def infer(self, src_path, ref_path, out_dir=None, output_path=None, delay=None, **sampling_kwargs):
        src_wav, _ = librosa.load(src_path, sr=self.sr)
        ref_wav, _ = librosa.load(ref_path, sr=self.sr)
        src_wav_tensor = torch.from_numpy(src_wav).unsqueeze(0).to(self.device)
        ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(self.device)
        src_wav_16k_tensor = torchaudio.functional.resample(
            src_wav_tensor, orig_freq=self.sr, new_freq=16000
        )
        ref_wav_16k_tensor = torchaudio.functional.resample(
            ref_wav_tensor, orig_freq=self.sr, new_freq=16000
        )
        src_wave_lens = torch.LongTensor([src_wav_tensor.size(1)]).to(self.device)
        ref_wave_lens = torch.LongTensor([ref_wav_tensor.size(1)]).to(self.device)
        src_wave_lens_16k = torch.LongTensor([src_wav_16k_tensor.size(1)]).to(self.device)
        ref_wave_lens_16k = torch.LongTensor([ref_wav_16k_tensor.size(1)]).to(self.device)

        # audio features
        ref_audio_codes, ref_quantized_features, ref_target_lengths, ref_target_size = self.wav2target_fn(
            ref_wav_tensor,
            ref_wave_lens,
        )

        style_vectors = self.calculate_style_vec(ref_wav_16k_tensor, ref_wave_lens_16k)
        timbre_latents = self.calculate_timbre_latent(
            ref_wav_16k_tensor, ref_wave_lens_16k
        )

        src_content_codes, src_code_lengths = self.speech_tokenizer.encode(
            src_wav_tensor, src_wave_lens
        )
        ref_content_codes, _ = self.speech_tokenizer.encode(
            ref_wav_tensor, ref_wave_lens
        )
        src_content_codes = src_content_codes.squeeze(0)
        ref_content_codes = ref_content_codes.squeeze(0)

        if delay is not None:
            self.model.set_delay(delay=delay)
        # vc_codes = self.model.infer(src_content_codes, style_vectors, timbre_latents)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            vc_codes = self.model.generate(
                ref_content_codes=ref_content_codes,
                ref_audio_codes=ref_audio_codes,
                src_content_codes=src_content_codes,
                style_vectors=style_vectors,
                timbre_latents=timbre_latents,
                **sampling_kwargs
            )
        pred_wave = self.code2wav_fn(
            vc_codes,
        )
        pred_wave = pred_wave.squeeze().cpu().numpy()

        # determine output file name
        src_name = Path(src_path).stem
        ref_name = Path(ref_path).stem
        out_name = f"{src_name}_{ref_name}.wav"
        if output_path:
            out_path = Path(output_path)
        elif out_dir:
            out_path = Path(out_dir) / out_name
        else:
            out_path = Path(src_path).parent / out_name
        # save output
        torchaudio.save(out_path, torch.from_numpy(pred_wave).unsqueeze(0), self.sr)
        print(f"Output saved to {out_path}")
        return pred_wave

    def calculate_prompt(self, ref_wav_tensor):
        ref_wav_16k_tensor = torchaudio.functional.resample(
            ref_wav_tensor, orig_freq=self.sr, new_freq=16000
        )
        ref_wave_lens = torch.LongTensor([ref_wav_tensor.size(1)]).to(self.device)
        ref_wave_lens_16k = torch.LongTensor([ref_wav_16k_tensor.size(1)]).to(self.device)
        # audio features
        ref_audio_codes, ref_quantized_features, ref_target_lengths, ref_target_size = self.wav2target_fn(
            ref_wav_tensor,
            ref_wave_lens,
        )

        style_vectors = self.calculate_style_vec(ref_wav_16k_tensor, ref_wave_lens_16k)
        timbre_latents = self.calculate_timbre_latent(
            ref_wav_16k_tensor, ref_wave_lens_16k
        )

        ref_content_codes, _ = self.speech_tokenizer.encode(
            ref_wav_tensor, ref_wave_lens
        )
        ref_content_codes = ref_content_codes.squeeze(0)

        return ref_audio_codes, ref_content_codes, style_vectors, timbre_latents

    def setup_stream_caches(self,
                            encode_window_frames=96,
                            decode_window_frames=64,
                            max_seq_frames=768,
                            buffer_frames=32,
                            decode_chunk_frames=1,
                            delay=None,
                            ):
        self.src_wav_tensor = torch.zeros(1, encode_window_frames * 2048).to(self.device)
        self.encode_window_wave_lens = encode_window_frames * 2048
        self.encode_window_frames = encode_window_frames
        self.decode_window_frames = decode_window_frames
        self.max_seq_frames = max_seq_frames
        self.buffer_frames = buffer_frames
        self.decode_chunk_frames = decode_chunk_frames
        self.src_content_codes = torch.zeros([1, 0]).to(self.device).long()
        self.pred_codes = torch.zeros([1, 8, 0]).to(self.device).long()
        self.src_condition4delay_prefilled = False


    def prefill_prompt(self, ref_wav_tensor, max_prompt_frames=256, delay=4,):
        ref_audio_codes, ref_content_codes, style_vectors, timbre_latents = self.calculate_prompt(ref_wav_tensor)
        # restrict prompt len
        self.ref_audio_codes = ref_audio_codes[:, :, :max_prompt_frames]
        self.ref_content_codes = ref_content_codes[:, :max_prompt_frames]
        self.style_vectors = style_vectors
        self.timbre_latents = timbre_latents
        self.ref_wav_tensor = ref_wav_tensor[:, :max_prompt_frames * 2048]
        self.ref_wav_tensor_len = self.ref_wav_tensor.size(-1)

        if isinstance(self.model.decoder.original_delay, int):
            self.delay = self.model.decoder.original_delay
        else:
            self.delay = int(delay)
        self.model.set_delay(delay=delay)

        # prefill prompt to kv cache
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.model.prefill_prompt(
                ref_content_codes,
                ref_audio_codes,
                style_vectors,
                timbre_latents,
            )


    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def process_one_chunk(self, src_wav_chunk, pitch_shift=0.0):
        self.src_wav_tensor[:, :-src_wav_chunk.size(-1)] = self.src_wav_tensor[:, src_wav_chunk.size(-1):].clone()
        self.src_wav_tensor[:, -src_wav_chunk.size(-1):] = src_wav_chunk.clone()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize before starting
        torch.cuda.synchronize()
        start_event.record()

        chunk_lens = torch.LongTensor([self.src_wav_tensor.size(1)]).to(self.device)
        src_chunk_content_codes = self.compiled_speech_tokenizer_encode(
            self.src_wav_tensor.reshape(1, self.encode_window_wave_lens), chunk_lens
        )[0].squeeze(0).clone()

        end_event.record()
        # Synchronize after recording
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Time taken for content encoder: {elapsed_time_ms}ms")

        # print(src_chunk_content_codes[..., -self.decode_chunk_frames:])
        self.src_content_codes = torch.cat([self.src_content_codes, src_chunk_content_codes[..., -self.decode_chunk_frames:]], dim=-1)
        if self.src_content_codes.size(-1) < self.delay:
            return torch.zeros_like(src_wav_chunk)
        elif self.src_content_codes.size(-1) >= self.delay and not self.src_condition4delay_prefilled and not self.delay == 0:
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.model.prefill_src_condition4delay(self.src_content_codes[:, -self.delay:])
            self.src_condition4delay_prefilled = True
            return torch.zeros_like(src_wav_chunk)
        else:
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Synchronize before starting
            torch.cuda.synchronize()
            start_event.record()
            for i in range(self.decode_chunk_frames):
                vc_code, current_pos = self.model.decode_one(
                    src_chunk_content_codes[..., -(self.decode_chunk_frames-i)].unsqueeze(0),
                )
                self.pred_codes = torch.cat([self.pred_codes, vc_code.clone()[None]], dim=-1)

            end_event.record()
            # Synchronize after recording
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Time taken for AR: {elapsed_time_ms}ms")

            if current_pos // 2 >= self.max_seq_frames:
                # refill prompt with ref codes and buffer
                extended_ref_audio_codes = torch.cat(
                    [self.ref_audio_codes, self.pred_codes[..., -self.buffer_frames:]], dim=-1
                )
                extended_ref_content_codes = torch.cat(
                    [self.ref_content_codes, self.src_content_codes[..., -self.buffer_frames - self.delay:-self.delay]],
                    dim=-1
                )
                self.model.prefill_prompt(
                    extended_ref_content_codes,
                    extended_ref_audio_codes,
                    self.style_vectors,
                    self.timbre_latents,
                )
                self.model.prefill_src_condition4delay(self.src_content_codes[..., -self.delay:])
                print(
                    f"Refill prompt with {len(self.pred_codes)} frames, current pos: {current_pos // 2}")

            # decode with vocoder
            vc_codes_chunk = self.pred_codes[..., -self.decode_window_frames:]
            pad_len = self.decode_window_frames - vc_codes_chunk.size(-1)
            vc_codes_chunk = F.pad(vc_codes_chunk, (pad_len, 0), value=0)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Synchronize before starting
            torch.cuda.synchronize()
            start_event.record()

            pred_wave = self.code2wav_fn(
                vc_codes_chunk.reshape(1, 8, self.decode_window_frames),
            ).clone()

            end_event.record()
            # Synchronize after recording
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Time taken for vocoder: {elapsed_time_ms}ms")

            # restric the maximum length of self kept pred_codes and src_content_codes
            self.pred_codes = self.pred_codes[..., -2048:]
            self.src_content_codes = self.src_content_codes[..., -2048:]

            return pred_wave[..., -2048 * self.decode_chunk_frames:].squeeze(1)
    def stream_infer(
            self,
            src_path,
            ref_path,
            out_dir=None,
            encode_window_frames=128,  # context size assigned to encoder
            decode_window_frames=64,  # context size assigned to vocoder
            max_prompt_frames=256,  # maximum length of the prompt
            max_seq_frames=768,   # maximum length of the sequence
            buffer_frames=32,  # when dealing with a long audio, how many frames to keep in memory when refilling prompt
            decode_chunk_frames=1,  # how many frames for vocoder to decode at once
            delay=None,  # delay for the decoder in frames, 0 means no delay
    ):
        src_wav, _ = librosa.load(src_path, sr=self.sr)
        ref_wav, _ = librosa.load(ref_path, sr=self.sr)
        src_wav_tensor = torch.from_numpy(src_wav).unsqueeze(0).to(self.device)
        ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(self.device)

        self.prefill_prompt(ref_wav_tensor, max_prompt_frames=max_prompt_frames, delay=delay,)
        self.setup_stream_caches(
            encode_window_frames=encode_window_frames,
            decode_window_frames=decode_window_frames,
            max_seq_frames=max_seq_frames,
            buffer_frames=buffer_frames,
            decode_chunk_frames=decode_chunk_frames,
        )

        # left pad src_wav_tensor to multiple of 2048
        pad_len = (2048 * decode_chunk_frames) - (src_wav_tensor.size(1) % (2048 * decode_chunk_frames))
        src_wav_tensor = F.pad(src_wav_tensor, (pad_len, 0), value=0)

        # divide src_wav_tensor into chunks of 2048 * decode_chunk_frames
        src_wav_chunks = src_wav_tensor.unfold(1, 2048 * decode_chunk_frames, 2048 * decode_chunk_frames).squeeze()
        max_window_len = encode_window_frames * 2048
        src_content_codes = []
        pred_codes = []
        pred_wave_chunks = []

        for src_wav_chunk in tqdm(src_wav_chunks):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Synchronize before starting
            torch.cuda.synchronize()
            start_event.record()

            pred_wave = self.process_one_chunk(src_wav_chunk[None])
            pred_wave_chunks.append(pred_wave)

            end_event.record()
            # Synchronize after recording
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Time taken: {elapsed_time_ms}ms")


        pred_wave = torch.cat(pred_wave_chunks, dim=-1)
        pred_wave = pred_wave.squeeze().cpu().numpy()

        # determine output file name
        src_name = Path(src_path).stem
        ref_name = Path(ref_path).stem
        out_name = f"{src_name}_{ref_name}.wav"
        if out_dir:
            out_path = Path(out_dir) / out_name
        else:
            out_path = Path(src_path).parent / out_name
        # save output
        torchaudio.save(out_path, torch.from_numpy(pred_wave).float().unsqueeze(0), self.sr)
        print(f"Output saved to {out_path}")
        return pred_wave

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference Wrapper")
    parser.add_argument("--config_path", type=str, default="configs/config_firefly_arvcasr_8192_delay0_8.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="pretrained_checkpoints/dual_ar_delay_0_8.pth")
    parser.add_argument("--src_path", type=str, default="./test_waves/azuma_0.wav")
    parser.add_argument("--ref_path", type=str, default="./test_waves/trump_0.wav")
    parser.add_argument("--out_dir", type=str, default="./audio_outputs/")
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--delay", type=int, default=2, help="Delay for the decoder (in frames), 0 means no delay")  # only used for dynamic-delay model

    # streaming args
    parser.add_argument("--simulate_streaming", action="store_true", help="Simulate streaming inference")
    parser.add_argument("--encode_window_frames", type=int, default=128, help="Encoder context window size in frames")  # only used for streaming
    parser.add_argument("--decode_window_frames", type=int, default=64, help="Vocoder context window size in frames")  # only used for streaming
    parser.add_argument("--max_prompt_frames", type=int, default=256, help="Maximum prompt length in frames")  # only used for streaming
    parser.add_argument("--max_seq_frames", type=int, default=768, help="Maximum sequence length in frames")  # only used for streaming
    parser.add_argument("--buffer_frames", type=int, default=32, help="Buffer frames when refilling prompt")  # only used for streaming
    parser.add_argument("--decode_chunk_frames", type=int, default=1, help="Decode chunk size in frames")  # only used for streaming
    args = parser.parse_args()
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    infer_wrapper = InferenceWrapper(
        config_path,
        checkpoint_path,
        compile_ar=args.compile,
        compile_decoder=args.compile if args.simulate_streaming else False,
        compile_encoder=args.compile if args.simulate_streaming else False,
    )
    src_path = args.src_path
    ref_path = args.ref_path
    out_dir = args.out_dir
    # Create output directory if it doesn't exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # vc_wav = infer_wrapper.infer(src_path, ref_path, out_dir, delay=args.delay)
    if args.simulate_streaming:
        vc_wav = infer_wrapper.stream_infer(
            src_path,
            ref_path,
            out_dir,
            encode_window_frames=args.encode_window_frames,
            decode_window_frames=args.decode_window_frames,
            max_prompt_frames=args.max_prompt_frames,
            max_seq_frames=args.max_seq_frames,
            buffer_frames=args.buffer_frames,
            decode_chunk_frames=args.decode_chunk_frames,
            delay=args.delay)
    else:
        vc_wav = infer_wrapper.infer(src_path, ref_path, out_dir, delay=args.delay)

