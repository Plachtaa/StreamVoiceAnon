"""Strict helper loading from the previously validated inference profile."""
import torch
import hydra
import yaml
from omegaconf import OmegaConf, DictConfig

class StrictHelpers:
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
        if missing_keys or unexpected_keys:
            raise RuntimeError(f'Incomplete speech tokenizer: {missing_keys}, {unexpected_keys}')
        for param in self.speech_tokenizer.parameters():
            param.requires_grad = False


        firefly_cfg = DictConfig(yaml.safe_load(open(self.config['firefly']['config_path'])))
        self.firefly = hydra.utils.instantiate(firefly_cfg).to(self.device)
        missing, unexpected = self.firefly.load_state_dict(
            torch.load(self.config["firefly"]["checkpoint_path"], map_location="cpu"),
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(f'Incomplete Firefly weights: {missing}, {unexpected}')
        self.firefly.remove_parametrizations()
        for param in self.firefly.parameters():
            param.requires_grad = False

        self.style_encoder = hydra.utils.instantiate(
            OmegaConf.load(self.config["style_encoder"]["config_path"])
        ).to(self.device).eval()
        missing, unexpected = self.style_encoder.load_state_dict(
            torch.load(
                self.config["style_encoder"]["checkpoint_path"], map_location="cpu"
            ),
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(f'Incomplete CAMPPlus weights: {missing}, {unexpected}')
        for param in self.style_encoder.parameters():
            param.requires_grad = False

        # sparktts timbre encoder
        self.timbre_encoder = hydra.utils.instantiate(
            OmegaConf.load(self.config["timbre_encoder"]["config_path"])
        ).to(self.device)
        missing, unexpected = self.timbre_encoder.load_state_dict(
            torch.load(
                self.config["timbre_encoder"]["checkpoint_path"], map_location="cpu"
            ),
            strict=False,
        )
        allowed = {'mel_transformer.spectrogram.window', 'mel_transformer.mel_scale.fb'}
        if unexpected or not set(missing) <= allowed:
            raise RuntimeError(f'Incomplete Spark weights: {missing}, {unexpected}')
        for key in missing:
            value = dict(self.timbre_encoder.named_buffers())[key]
            if not value.numel() or not torch.isfinite(value).all():
                raise RuntimeError(f'Invalid deterministic Spark buffer: {key}')
        self.weight_audit['spark_constructor_buffers'] = missing
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
        # This release fixes pitch_shift=0 and f0_enabled=False. Native Plus
        # consults this attribute in the streaming path.
        self.rmvpe = None
