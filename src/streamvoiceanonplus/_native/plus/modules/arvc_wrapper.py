import torch
import torch.nn as nn
import torch.nn.functional as F



class ARVCWrapper(nn.Module):
    def __init__(
            self,
            embedding: nn.Module,
            decoder: nn.Module,
            f0_embedding: nn.Module = None,
            context_dim: int = 128,
            style_dim: int = 192,
            model_dim: int = 768,
            post_training_type: str = None,
            emotion_loss_params: dict = None,
            use_spk_embedding: bool = True
    ):
        super(ARVCWrapper, self).__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.context_in = nn.Linear(context_dim, model_dim)
        self.style_in = nn.Linear(style_dim, model_dim)
        self.f0_embedding = f0_embedding
        self.compiled_fn = None
        self.post_training_type = post_training_type
        self.decoder.post_training_type = post_training_type
        self.emotion_loss_params = emotion_loss_params
        self.decoder.emotion_loss_params = emotion_loss_params
        self.use_spk_embedding = use_spk_embedding
        self.decoder.use_spk_embedding = use_spk_embedding

    def compile_ar_decode_fn(self):
        print("Compiling function...")
        from streamvoiceanonplus._native.plus.modules.dual_ar_stream_v3 import decode_one_token_ar

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
                base_target: torch.Tensor,
                target: torch.Tensor,
                style_vectors: torch.Tensor,
                timbre_latents: torch.Tensor,
                f0: torch.Tensor = None,
                x1_lens: torch.Tensor = None,
                emo1_labels: torch.Tensor = None,
                emo2_labels: torch.Tensor = None,
                emo1_feats_orig: list = None,
                emo2_feats_orig: list = None
                ):
        condition = self.embedding(base_target.squeeze(0))
        if self.f0_embedding is not None:
            f0_embed = self.f0_embedding(f0)
            condition = condition + f0_embed
        spk_condition = torch.cat([self.context_in(timbre_latents), self.style_in(style_vectors).unsqueeze(1)], dim=1)
        if self.emotion_loss_params is not None:
            codebook_loss, semantic_loss, emo_loss, _ = self.decoder(condition, spk_condition, base_target.squeeze(0), target, x_lens,
                                                       x1_lens=x1_lens, emo1_labels=emo1_labels, emo2_labels=emo2_labels, 
                                                       emo1_feats_orig=emo1_feats_orig, emo2_feats_orig=emo2_feats_orig)
            return codebook_loss, semantic_loss, emo_loss
        else:
            codebook_loss, semantic_loss, _ = self.decoder(condition, spk_condition, base_target.squeeze(0), target, x_lens,
                                                        x1_lens=x1_lens, emo1_labels=emo1_labels, emo2_labels=emo2_labels,
                                                        emo1_feats_orig=emo2_feats_orig, emo2_feats_orig=emo2_feats_orig)

            return codebook_loss, semantic_loss

    def infer(self,
              src_semanitcs: torch.Tensor,
              style_vectors: torch.Tensor,
              timbre_latents: torch.Tensor,
              f0: torch.Tensor = None,
              feature1_lens: torch.Tensor = None,
        ):
        condition = self.embedding(src_semanitcs)
        if self.f0_embedding is not None:
            f0_embed = self.f0_embedding(f0)
            condition = condition + f0_embed
        spk_condition = torch.cat([self.context_in(timbre_latents), self.style_in(style_vectors).unsqueeze(1)], dim=1)
        pred_audio_codes = self.decoder.infer(condition, spk_condition, feature1_lens=feature1_lens)
        return pred_audio_codes

    def generate(
            self,
            ref_content_codes,
            ref_audio_codes,
            src_content_codes,
            style_vectors,
            timbre_latents,
            ref_f0=None,
            src_f0=None,
            **sampling_kwargs,
    ):
        src_condition = self.embedding(src_content_codes)
        ref_condition = self.embedding(ref_content_codes)
        if self.f0_embedding is not None:
            ref_f0_embed = self.f0_embedding(ref_f0)
            src_f0_embed = self.f0_embedding(src_f0)
            ref_condition = ref_condition + ref_f0_embed
            src_condition = src_condition + src_f0_embed
        spk_condition = torch.cat([self.context_in(timbre_latents), self.style_in(style_vectors).unsqueeze(1)], dim=1)
        pred_audio_codes = self.decoder.generate(ref_condition, ref_audio_codes, src_condition, spk_condition, compiled_fn=self.compiled_fn, **sampling_kwargs)
        return pred_audio_codes.transpose(0, 1)

    def prefill_prompt(
            self,
            ref_content_codes,
            ref_audio_codes,
            style_vectors,
            timbre_latents,
            ref_f0=None,
            first_prefill: bool = True,
    ):
        ref_condition = self.embedding(ref_content_codes)
        if self.f0_embedding is not None:
            ref_f0_embed = self.f0_embedding(ref_f0)
            ref_condition = ref_condition + ref_f0_embed
        spk_condition = torch.cat([self.context_in(timbre_latents), self.style_in(style_vectors).unsqueeze(1)], dim=1)
        self.decoder.prefill_prompt(ref_condition, ref_audio_codes, spk_condition, first_prefill=first_prefill)

    def prefill_src_condition4delay(
            self,
            src_content_codes,
            src_f0=None,
    ):
        src_condition = self.embedding(src_content_codes)
        if self.f0_embedding is not None:
            src_f0_embed = self.f0_embedding(src_f0)
            src_condition = src_condition + src_f0_embed
        self.decoder.prefill_src_condition4delay(src_condition)

    def decode_one(
            self,
            src_content_codes,
            src_f0=None,
    ):
        src_condition = self.embedding(src_content_codes)
        if self.f0_embedding is not None:
            src_f0_embed = self.f0_embedding(src_f0)
            src_condition = src_condition + src_f0_embed
        return self.decoder.decode_one(src_condition, compiled_fn=self.compiled_fn)
