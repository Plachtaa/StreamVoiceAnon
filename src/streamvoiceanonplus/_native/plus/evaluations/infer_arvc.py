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
import torch._inductor.config
import sys
import numpy as np
import os




torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True

class InferenceWrapper:
    SAMPLES_PER_FRAME = 2048
    def __init__(self, config_path, checkpoint_path, compile_encoder=False, compile_decoder=False, compile_ar=False, fp16=False):
        self.config = yaml.safe_load(open(config_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.post_training_type = self.config.get('post_training_type', None)
        self.use_spk_embedding = self.config.get('use_spk_embedding', True)
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
        cfg['post_training_type'] = self.post_training_type
        cfg['decoder']['model']['config']['post_training_type'] = self.post_training_type
        cfg['use_spk_embedding'] = self.use_spk_embedding
        cfg['decoder']['model']['config']['use_spk_embedding'] = self.use_spk_embedding
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
        self.speech_tokenizer.load_state_dict(
            speech_tokenizer_sd, strict=False
        )
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

        # optional f0 extractor if model requires
        if self.config['preprocess_params']['f0_enabled']:
            from streamvoiceanonplus._native.plus.modules.rmvpe import RMVPE
            model_path = self.config["rmvpe"]["checkpoint_path"]
            self.rmvpe = RMVPE(model_path, is_half=False, device=self.device)
        else:
            self.rmvpe = None


    def _load_checkpoint(self, checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu")["net"]
        # strip module. prefix
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = v
        self.model.load_state_dict(new_sd, strict=False)

    @torch.no_grad()
    def wav2target_fn(self, waves, wave_lengths):
        (target, quantized), target_lengths = self.firefly.encode(waves, wave_lengths)
        target_size = target.size(2)
        return target, quantized, target_lengths, target_size

    @torch.no_grad()
    def code2wav_fn(self, code):
        wav = self.firefly.head(self.firefly.quantizer.decode(code)) #gaus_noise_params_output_audio_embs=gaus_noise_params_output_audio_embs)
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
                audio_16k_tensor[bib: bib + 1, : wave_lens[bib]].clone(),
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
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
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

    @torch.no_grad()
    def calculate_f0(
        self,
        audio_16k_tensor: torch.Tensor,
        max_feature_len: int,
    ):
        f0s = self.rmvpe.infer_from_audio(audio_16k_tensor)
        # interplate f0s to match the length of the features
        f0s = F.interpolate(
            f0s.unsqueeze(1),
            size=max_feature_len,
            mode="nearest",
        ).squeeze(1)
        # f0s, _ = torchcrepe.predict(
        #     audio=audio_16k_tensor,
        #     sample_rate=16000,
        #     hop_length=160,
        #     fmin=50,
        #     fmax=1100,
        #     model='tiny',
        #     batch_size=1,
        #     device=self.device,
        #     return_periodicity=True,
        # )
        # f0s = F.interpolate(f0s.unsqueeze(1), size=max_feature_len, mode='nearest').squeeze(1)
        return f0s

    @staticmethod
    def adjust_f0_semitones(f0_sequence, n_semitones):
        """Adjust F0 values by a number of semitones."""
        factor = 2 ** (n_semitones / 12)
        return f0_sequence * factor

    def infer(self, src_path, ref_path, out_dir=None, output_path=None, auto_f0_adjust=True, save_result=True, 
              anonymization_only=False, delay=None, alpha=1.0, **sampling_kwargs):
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
        style_vectors = self.apply_noise_mixing(style_vectors, alpha)
        timbre_latents = self.apply_noise_mixing(timbre_latents, alpha)

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
        # optional f0
        if self.rmvpe is not None:
            src_f0s = self.calculate_f0(
                src_wav_16k_tensor, src_content_codes.size(-1)
            )
            ref_f0s = self.calculate_f0(
                ref_wav_16k_tensor, ref_content_codes.size(-1)
            )
            voiced_src_f0s = src_f0s[src_f0s > 1]
            voiced_ref_f0s = ref_f0s[ref_f0s > 1]

            log_src_f0s = torch.log(src_f0s + 1e-5)
            voiced_log_src_f0s = torch.log(voiced_src_f0s + 1e-5)
            voiced_log_ref_f0s = torch.log(voiced_ref_f0s + 1e-5)
            median_log_src_f0s = torch.median(voiced_log_src_f0s)
            median_log_ref_f0s = torch.median(voiced_log_ref_f0s)

            # Shift alt log f0 level to ori log f0 level
            shifted_log_src_f0s = log_src_f0s.clone()
            if auto_f0_adjust:
                shifted_log_src_f0s[src_f0s > 1] = log_src_f0s[src_f0s > 1] - median_log_src_f0s + median_log_ref_f0s
            shifted_src_f0s = torch.exp(shifted_log_src_f0s)
            # if pitch_shift != 0:
            #     shifted_f0_alt[F0_alt > 1] = self.adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
        else:
            shifted_src_f0s = None
            ref_f0s = None

        if anonymization_only:
            if ref_f0s is not None:
                ref_f0s = torch.zeros_like(ref_f0s)
            ref_content_codes = torch.zeros_like(ref_content_codes)
            ref_audio_codes = torch.zeros_like(ref_audio_codes)
            style_vectors = torch.zeros_like(style_vectors)
            timbre_latents = torch.zeros_like(timbre_latents)
        # vc_codes = self.model.infer(src_content_codes, style_vectors, timbre_latents)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            vc_codes = self.model.generate(
                ref_content_codes=ref_content_codes,
                ref_audio_codes=ref_audio_codes,
                src_content_codes=src_content_codes,
                style_vectors=style_vectors,
                timbre_latents=timbre_latents,
                ref_f0=ref_f0s,
                src_f0=shifted_src_f0s,
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
        if save_result:
            torchaudio.save(out_path, torch.from_numpy(pred_wave).unsqueeze(0), self.sr)
            print(f"Output saved to {out_path}")
        return pred_wave

    def infer_multiple(self, src_path, ref_paths, out_dir=None, output_path=None, auto_f0_adjust=True, save_result=True, anonymization_only=False, delay=None, gaus_noise_params=None,
                       gaus_noise_params_emb_seq=None, gaus_noise_params_output_audio_embs=None, **sampling_kwargs):
        src_wav, _ = librosa.load(src_path, sr=self.sr)
        if len(ref_paths) == 1:
            ref_path = ref_paths[0]
            ref_wav, _ = librosa.load(ref_paths[0], sr=self.sr)
        else:
            ref_wavs = []
            for ref_path in ref_paths:
                ref_wav, _ = librosa.load(ref_path, sr=self.sr)
                ref_wavs.append(ref_wav[:4*self.sr])  # limit each wav to 4 seconds if we have multiple references
            ref_wav = np.concatenate(ref_wavs)
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

        # optional f0
        if self.rmvpe is not None:
            src_f0s = self.calculate_f0(
                src_wav_16k_tensor, src_content_codes.size(-1)
            )
            ref_f0s = self.calculate_f0(
                ref_wav_16k_tensor, ref_content_codes.size(-1)
            )
            voiced_src_f0s = src_f0s[src_f0s > 1]
            voiced_ref_f0s = ref_f0s[ref_f0s > 1]

            log_src_f0s = torch.log(src_f0s + 1e-5)
            voiced_log_src_f0s = torch.log(voiced_src_f0s + 1e-5)
            voiced_log_ref_f0s = torch.log(voiced_ref_f0s + 1e-5)
            median_log_src_f0s = torch.median(voiced_log_src_f0s)
            median_log_ref_f0s = torch.median(voiced_log_ref_f0s)

            # Shift alt log f0 level to ori log f0 level
            shifted_log_src_f0s = log_src_f0s.clone()
            if auto_f0_adjust:
                shifted_log_src_f0s[src_f0s > 1] = log_src_f0s[src_f0s > 1] - median_log_src_f0s + median_log_ref_f0s
            shifted_src_f0s = torch.exp(shifted_log_src_f0s)
            # if pitch_shift != 0:
            #     shifted_f0_alt[F0_alt > 1] = self.adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
        else:
            shifted_src_f0s = None
            ref_f0s = None

        if gaus_noise_params is not None:
            gaus_noise_mean_scale = float(gaus_noise_params["mean_scale"])
            gaus_noise_std_scale = float(gaus_noise_params["std_scale"])
            if ref_f0s is not None:
                ref_f0s_noise = torch.normal(ref_f0s.mean()*gaus_noise_mean_scale, ref_f0s.std()*gaus_noise_std_scale, ref_f0s.shape)
            style_vectors_noise = torch.normal(style_vectors.mean()*gaus_noise_mean_scale, style_vectors.std()*gaus_noise_std_scale, 
                            style_vectors.shape)
            timbre_latents_noise = torch.normal(timbre_latents.mean()*gaus_noise_mean_scale, timbre_latents.std()*gaus_noise_std_scale, 
                            timbre_latents.shape)


        if anonymization_only:
            if ref_f0s is not None:
                ref_f0s = torch.zeros_like(ref_f0s)
            ref_content_codes = torch.zeros_like(ref_content_codes)
            ref_audio_codes = torch.zeros_like(ref_audio_codes)
            style_vectors = torch.zeros_like(style_vectors)
            timbre_latents = torch.zeros_like(timbre_latents)

        if gaus_noise_params is not None:
            if ref_f0s is not None:
                ref_f0s += ref_f0s_noise
            # idx = torch.randperm(ref_content_codes.nelement())
            # ref_content_codes = ref_content_codes.view(-1)[idx].view(ref_content_codes.size())
            # idx = torch.randperm(ref_audio_codes.nelement())
            # ref_audio_codes = ref_audio_codes.view(-1)[idx].view(ref_audio_codes.size())
            style_vectors += style_vectors_noise.to(style_vectors.device)
            timbre_latents += timbre_latents_noise.to(timbre_latents.device)

        # vc_codes = self.model.infer(src_content_codes, style_vectors, timbre_latents)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            vc_codes = self.model.generate(
                ref_content_codes=ref_content_codes,
                ref_audio_codes=ref_audio_codes,
                src_content_codes=src_content_codes,
                style_vectors=style_vectors,
                timbre_latents=timbre_latents,
                ref_f0=ref_f0s,
                src_f0=shifted_src_f0s,
                gaus_noise_params_emb_seq=gaus_noise_params_emb_seq,
                **sampling_kwargs
            )
        pred_wave = self.code2wav_fn(
            vc_codes, gaus_noise_params_output_audio_embs=gaus_noise_params_output_audio_embs
        )
        pred_wave = pred_wave.squeeze().cpu().numpy()

        # determine output file name
        if save_result:
            src_name = Path(src_path).stem
            ref_name = Path(ref_path).stem
            out_name = f"{src_name}_{ref_name}.wav"
            if output_path:
                out_path = Path(output_path)
            elif out_dir:
                out_path = Path(out_dir) / out_name
            else:
                out_path = Path(src_path).parent / out_name

            torchaudio.save(out_path, torch.from_numpy(pred_wave).unsqueeze(0), self.sr)
            print(f"Output saved to {out_path}")
        return pred_wave

    def apply_noise_mixing(self, tensor, alpha):
        """Matched baseline mixing; called once per conditioning embedding."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        mean, std = tensor.mean(), tensor.std()
        noise = torch.randn_like(tensor) * std + mean
        return alpha * tensor + (1 - alpha) * noise

    def calculate_prompt(self, ref_wav_tensor, alpha=1.0):
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

        style_vectors = self.apply_noise_mixing(style_vectors, alpha)
        timbre_latents = self.apply_noise_mixing(timbre_latents, alpha)
        return ref_audio_codes, ref_content_codes, style_vectors, timbre_latents

    def setup_stream_caches(self,
                            encode_window_frames=96,
                            decode_window_frames=64,
                            max_seq_frames=768,
                            buffer_frames=32,
                            decode_chunk_frames=3,
                            pitch_shift=0,
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
        if self.rmvpe is not None:
            self.src_f0s = torch.zeros([1, 0]).to(self.device)
            self.pitch_shift = pitch_shift
        self.src_condition4delay_prefilled = False


    def prefill_prompt(self, ref_wav_tensor, max_prompt_frames=256, delay=None, alpha=1.0):
        ref_audio_codes, ref_content_codes, style_vectors, timbre_latents = self.calculate_prompt(ref_wav_tensor, alpha=alpha)
        # restrict prompt len
        self.ref_audio_codes = ref_audio_codes[:, :, :max_prompt_frames]
        self.ref_content_codes = ref_content_codes[:, :max_prompt_frames]
        self.style_vectors = style_vectors
        self.timbre_latents = timbre_latents
        self.ref_wav_tensor = ref_wav_tensor[:, :max_prompt_frames * 2048]
        self.ref_wav_tensor_len = self.ref_wav_tensor.size(-1)

        self.delay = delay if delay is not None else self.model.decoder.delay
        self.model.set_delay(delay=delay)

        if self.rmvpe is not None:
            ref_wav_16k_tensor = torchaudio.functional.resample(
                ref_wav_tensor, orig_freq=self.sr, new_freq=16000
            )
            ref_f0s = self.calculate_f0(
                ref_wav_16k_tensor, ref_content_codes.size(-1)
            )
            self.ref_f0s = ref_f0s[..., :max_prompt_frames]

            voiced_ref_f0s = ref_f0s[ref_f0s > 1]
            voiced_log_ref_f0s = torch.log(voiced_ref_f0s + 1e-5)
            self.median_log_ref_f0s = torch.median(voiced_log_ref_f0s)
        else:
            self.median_log_ref_f0s = None
            ref_f0s = None

        # prefill prompt to kv cache
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.model.prefill_prompt(
                ref_content_codes,
                ref_audio_codes,
                style_vectors,
                timbre_latents,
                ref_f0=ref_f0s,
                first_prefill=True
            )

    @staticmethod
    def adjust_f0_semitones(f0_sequence, n_semitones):
        """Adjust F0 values by a number of semitones."""
        factor = 2 ** (n_semitones / 12)
        return f0_sequence * factor

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def process_one_chunk(self, src_wav_chunk, pitch_shift=0.0, speech_tokenizer_include_prompt=False, chunk_idx=None):
        self.src_wav_tensor[:, :-src_wav_chunk.size(-1)] = self.src_wav_tensor[:, src_wav_chunk.size(-1):].clone()
        self.src_wav_tensor[:, -src_wav_chunk.size(-1):] = src_wav_chunk.clone()

        if speech_tokenizer_include_prompt:
            speech_tokenizer_input_wave_tensor = torch.cat([
                self.ref_wav_tensor,
                self.src_wav_tensor
            ], dim=-1).reshape(1, self.encode_window_wave_lens + self.ref_wav_tensor_len)
            chunk_lens = torch.LongTensor([self.src_wav_tensor.size(1) + self.ref_wav_tensor_len]).to(self.device)
            src_chunk_content_codes = self.compiled_speech_tokenizer_encode(
                speech_tokenizer_input_wave_tensor, chunk_lens
            )[0].squeeze(0).clone()
        else:
            chunk_lens = torch.LongTensor([self.src_wav_tensor.size(1)]).to(self.device)
            src_chunk_content_codes = self.compiled_speech_tokenizer_encode(
                self.src_wav_tensor.reshape(1, self.encode_window_wave_lens), chunk_lens
            )[0].squeeze(0).clone()
        # src_chunk_content_codes = self.compiled_speech_tokenizer_encode(
        #     torch.cat([self.src_wav_tensor.reshape(1, self.encode_window_wave_lens), torch.zeros(1, 2048).to(self.device)], dim=-1),
        #     chunk_lens + 2048
        # )[0].squeeze(0).clone()[..., :-1]
        if self.rmvpe is not None:
            src_wav_tensor_16k = torchaudio.functional.resample(
                self.src_wav_tensor, orig_freq=self.sr, new_freq=16000
            )
            src_chunk_f0s = self.calculate_f0(
                src_wav_tensor_16k, src_chunk_content_codes.size(-1)
            )
            if pitch_shift != 0:
                src_chunk_f0s = self.adjust_f0_semitones(src_chunk_f0s, pitch_shift)

        self.src_content_codes = torch.cat([self.src_content_codes, src_chunk_content_codes[..., -self.decode_chunk_frames:]], dim=-1)
        self.src_f0s = torch.cat(
            [self.src_f0s, src_chunk_f0s[..., -self.decode_chunk_frames:]], dim=-1
        ) if self.rmvpe is not None else None
        if self.src_content_codes.size(-1) < self.delay:
            return torch.zeros_like(src_wav_chunk)
        elif self.src_content_codes.size(-1) >= self.delay and not self.src_condition4delay_prefilled and not self.delay == 0:
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.model.prefill_src_condition4delay(self.src_content_codes[:, -self.delay:], src_f0=self.src_f0s[:, -self.delay:] if self.rmvpe is not None else None)
            self.src_condition4delay_prefilled = True
            return torch.zeros_like(src_wav_chunk)
        else:
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            for i in range(self.decode_chunk_frames):
                vc_code, current_pos = self.model.decode_one(
                    src_chunk_content_codes[..., -(self.decode_chunk_frames-i)].unsqueeze(0),
                    src_f0= self.src_f0s[..., -(self.decode_chunk_frames-i)].unsqueeze(0) if self.rmvpe is not None else None
                )
                self.pred_codes = torch.cat([self.pred_codes, vc_code.clone()[None]], dim=-1)

            if current_pos // 2 >= self.max_seq_frames:
                # refill prompt with ref codes and buffer
                extended_ref_audio_codes = torch.cat(
                    [self.ref_audio_codes, self.pred_codes[..., -self.buffer_frames:]], dim=-1
                )
                extended_ref_content_codes = torch.cat(
                    [self.ref_content_codes, self.src_content_codes[..., -self.buffer_frames - self.delay:-self.delay]],
                    dim=-1
                )
                if self.rmvpe is not None:
                    extended_ref_f0s = torch.cat(
                        [self.ref_f0s, self.src_f0s[..., -self.buffer_frames - self.delay:-self.delay]],
                        dim=-1
                    )
                else:
                    extended_ref_f0s = None
                self.model.prefill_prompt(
                    extended_ref_content_codes,
                    extended_ref_audio_codes,
                    self.style_vectors,
                    self.timbre_latents,
                    ref_f0=extended_ref_f0s,
                    first_prefill=False,
                )
                self.model.prefill_src_condition4delay(self.src_content_codes[..., -self.delay:], src_f0=self.src_f0s[:, -self.delay:] if self.rmvpe is not None else None)

            # # decode with vocoder
            vc_codes_chunk = self.pred_codes[..., -self.decode_window_frames:]
            pad_len = self.decode_window_frames - vc_codes_chunk.size(-1)
            if chunk_idx in range(self.delay):
                if pad_len > 0:  # ad-hoc but helps to avoid spiking in the start of generated audio
                    ref_pad = self.ref_audio_codes[..., -pad_len:]
                    if len(ref_pad) < pad_len:
                        ref_pad = F.pad(ref_pad, (pad_len - ref_pad.size(-1), 0), value=0)
                    vc_codes_chunk = torch.cat([ref_pad, vc_codes_chunk], dim=-1)
                else:
                    vc_codes_chunk = F.pad(vc_codes_chunk, (pad_len, 0), value=0)

            else:
                vc_codes_chunk = F.pad(vc_codes_chunk, (pad_len, 0), value=0)   
            pred_wave = self.code2wav_fn(
                vc_codes_chunk.reshape(1, 8, self.decode_window_frames),
            ).clone()

            # restric the maximum length of self kept pred_codes and src_content_codes
            self.pred_codes = self.pred_codes[..., -2048:]
            self.src_content_codes = self.src_content_codes[..., -2048:]

            return pred_wave[..., -2048 * self.decode_chunk_frames:].squeeze(1)
    def stream_infer(
            self,
            src_path,
            ref_path,
            save_result=True,
            out_dir=None,
            encode_window_frames=128,  # context size assigned to encoder
            decode_window_frames=64,  # context size assigned to vocoder
            max_prompt_frames=256,  # maximum length of the prompt
            max_seq_frames=768,   # maximum length of the sequence
            buffer_frames=32,  # when dealing with a long audio, how many frames to keep in memory when refilling prompt
            decode_chunk_frames=3,  # how many frames for vocoder to decode at once
            pitch_shift=0,  # pitch shift in semitones
            delay=None,  # delay for the decoder in frames, 0 means no delay
            alpha=1.0,
    ):
        src_wav, _ = librosa.load(src_path, sr=self.sr)
        ref_wav, _ = librosa.load(ref_path, sr=self.sr)
        src_wav_tensor = torch.from_numpy(src_wav).unsqueeze(0).to(self.device)
        ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(self.device)

        self.prefill_prompt(ref_wav_tensor, max_prompt_frames=max_prompt_frames, delay=delay, alpha=alpha)
        self.setup_stream_caches(
            encode_window_frames=encode_window_frames,
            decode_window_frames=decode_window_frames,
            max_seq_frames=max_seq_frames,
            buffer_frames=buffer_frames,
            decode_chunk_frames=decode_chunk_frames,
            pitch_shift=pitch_shift,
        )

        # left pad src_wav_tensor to multiple of 2048
        pad_len = (2048 * decode_chunk_frames) - (src_wav_tensor.size(1) % (2048 * decode_chunk_frames))
        src_wav_tensor = F.pad(src_wav_tensor, (pad_len, 0), value=0)
        # A delayed stream needs a second chunk to decode the first source
        # chunk. Without this flush context, very short inputs return silence.
        minimum_stream_samples = 2 * self.SAMPLES_PER_FRAME * decode_chunk_frames
        if src_wav_tensor.size(1) < minimum_stream_samples:
            src_wav_tensor = F.pad(src_wav_tensor, (0, minimum_stream_samples - src_wav_tensor.size(1)), value=0)

        # divide src_wav_tensor into chunks of 2048 * decode_chunk_frames
        src_wav_chunks = src_wav_tensor.unfold(1, 2048 * decode_chunk_frames, 2048 * decode_chunk_frames).squeeze(0)
        # n_chunks = src_wav_tensor.size(-1) // 2048
        max_window_len = encode_window_frames * 2048
        src_content_codes = []
        pred_codes = []
        pred_wave_chunks = []

        for chunk_idx, src_wav_chunk in tqdm(enumerate(src_wav_chunks)):

            torch.cuda.synchronize()

            pred_wave = self.process_one_chunk(src_wav_chunk[None], pitch_shift=pitch_shift, chunk_idx=chunk_idx)
            pred_wave_chunks.append(pred_wave)

            torch.cuda.synchronize()

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
        if save_result:
            torchaudio.save(out_path, torch.from_numpy(pred_wave).unsqueeze(0), self.sr)
            print(f"Output saved to {out_path}")
        return pred_wave


