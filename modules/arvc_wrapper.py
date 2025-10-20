import torch
import torch.nn as nn
import torch.nn.functional as F



class ARVCWrapper(nn.Module):
    def __init__(
            self,
            embedding: nn.Module,
            decoder: nn.Module,
            context_dim: int = 128,
            style_dim: int = 192,
            model_dim: int = 768,
            spk_condition: bool = True,
    ):
        super(ARVCWrapper, self).__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.context_in = nn.Linear(context_dim, model_dim)
        self.style_in = nn.Linear(style_dim, model_dim)
        self.compiled_fn = None
        self.spk_condition = spk_condition

    def compile_ar_decode_fn(self):
        print("Compiling function...")
        from modules.dual_ar_stream import decode_one_token_ar

        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True

        self.compiled_fn = torch.compile(
            decode_one_token_ar,
            fullgraph=True,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )

    def setup_caches(self, **kwargs):
        self.decoder.setup_caches(**kwargs)

    def set_delay(self, **kwargs):
        self.decoder.set_delay(**kwargs)

    def forward(self,
                x_lens: torch.Tensor,
                condition: torch.Tensor,
                base_target: torch.Tensor,
                target: torch.Tensor,
                style_vectors: torch.Tensor = None,
                timbre_latents: torch.Tensor = None,
                ):
        condition = self.embedding(condition)

        if self.spk_condition:
            spk_condition = torch.cat([self.context_in(timbre_latents), self.style_in(style_vectors).unsqueeze(1)], dim=1)
        else:
            spk_condition = torch.zeros((condition.size(0), 0, condition.size(2)), device=condition.device)
        codebook_loss, semantic_loss, codebook_per_token_logps, semantic_per_token_logps\
            = self.decoder(condition, spk_condition, base_target.squeeze(0), target, x_lens)

        return codebook_loss, semantic_loss, codebook_per_token_logps, semantic_per_token_logps

    def infer(self,
              src_semanitcs: torch.Tensor,
              style_vectors: torch.Tensor,
              timbre_latents: torch.Tensor,
        ):
        condition = self.embedding(src_semanitcs)
        print(f"condition shape: {condition.shape}")
        if self.spk_condition:
            spk_condition = torch.cat([self.context_in(timbre_latents), self.style_in(style_vectors).unsqueeze(1)], dim=1)
        else:
            spk_condition = torch.zeros((condition.size(0), 0, condition.size(2)), device=condition.device)
        pred_audio_codes = self.decoder.infer(condition, spk_condition)
        return pred_audio_codes

    def generate(
            self,
            ref_content_codes,
            ref_audio_codes,
            src_content_codes,
            style_vectors,
            timbre_latents,
            **sampling_kwargs,
    ):
        src_condition = self.embedding(src_content_codes)
        ref_condition = self.embedding(ref_content_codes)
        if self.spk_condition:
            spk_condition = torch.cat([self.context_in(timbre_latents), self.style_in(style_vectors).unsqueeze(1)], dim=1)
        else:
            spk_condition = torch.zeros((src_condition.size(0), 0, src_condition.size(2)), device=src_condition.device)
        pred_audio_codes = self.decoder.generate(ref_condition, ref_audio_codes, src_condition, spk_condition, compiled_fn=self.compiled_fn, **sampling_kwargs)
        return pred_audio_codes.transpose(0, 1)

    def prefill_prompt(
            self,
            ref_content_codes,
            ref_audio_codes,
            style_vectors,
            timbre_latents,
    ):
        ref_condition = self.embedding(ref_content_codes)
        if self.spk_condition:
            spk_condition = torch.cat([self.context_in(timbre_latents), self.style_in(style_vectors).unsqueeze(1)], dim=1)
        else:
            spk_condition = torch.zeros((ref_condition.size(0), 0, ref_condition.size(2)), device=ref_condition.device)
        self.decoder.prefill_prompt(ref_condition, ref_audio_codes, spk_condition)

    def prefill_src_condition4delay(
            self,
            src_content_codes,
    ):
        src_condition = self.embedding(src_content_codes)
        self.decoder.prefill_src_condition4delay(src_condition)

    def decode_one(
            self,
            src_content_codes,
    ):
        src_condition = self.embedding(src_content_codes)
        return self.decoder.decode_one(src_condition, compiled_fn=self.compiled_fn)
