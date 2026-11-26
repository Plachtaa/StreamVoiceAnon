import dataclasses
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from tqdm import tqdm

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class BaseModelArgs:
    model_type: str = "base"

    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False

    # Codebook configs
    codebook_size: int = 160
    num_codebooks: int = 4

    # Gradient checkpointing
    use_gradient_checkpointing: bool = False

    # Initialize the model
    initializer_range: float = 0.02

    # Dummy vars
    is_reward_model: bool = False
    share_codebook_embeddings: bool = True
    scale_codebook_embeddings: bool = False

    # Optional cross attention
    has_cross_attention: bool = False
    context_dim: int = 128

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        match data["model_type"]:
            case "naive":
                cls = NaiveModelArgs
            case "dual_ar":
                cls = DualARModelArgs
            case _:
                raise ValueError(f"Unknown model type: {data['model_type']}")

        return cls(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, ensure_ascii=False)


@dataclass
class NaiveModelArgs(BaseModelArgs):
    model_type: str = "naive"


@dataclass
class DualARModelArgs(BaseModelArgs):
    model_type: str = "dual_ar"
    n_fast_layer: int = 4
    fast_dim: int | None = None
    fast_n_head: int | None = None
    fast_n_local_heads: int | None = None
    fast_head_dim: int | None = None
    fast_intermediate_size: int | None = None
    fast_attention_qkv_bias: bool | None = None

    cond_input_dim: int = 512
    style_input_dim: int = 192
    delay: int | List[int] = 0

    def __post_init__(self):
        super().__post_init__()

        self.fast_dim = self.fast_dim or self.dim
        self.fast_n_head = self.fast_n_head or self.n_head
        self.fast_n_local_heads = self.fast_n_local_heads or self.n_local_heads
        self.fast_head_dim = self.fast_head_dim or self.head_dim
        self.fast_intermediate_size = (
            self.fast_intermediate_size or self.intermediate_size
        )
        self.fast_attention_qkv_bias = (
            self.fast_attention_qkv_bias
            if self.fast_attention_qkv_bias is not None
            else self.attention_qkv_bias
        )


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    token_targets: Tensor
    codebook_logits: Tensor
    codebook_targets: Tensor


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    def __init__(
        self,
        config: BaseModelArgs,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        # Slow transformer
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.output = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.dim // config.n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        # For kv cache
        self.max_batch_size = -1
        self.max_seq_len = -1

        if init_weights:
            self.apply(self._init_weights)

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        head_dim = self.config.dim // self.config.n_head
        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads if self.config.n_local_heads > 0 else self.config.n_head,
                head_dim,
                dtype=dtype,
            )

    def embed(self, x: Tensor) -> Tensor:
        # vocab_embeds = [self.embeddings(x[:, 0])]
        vocab_embeds = []
        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(x[:, i] + i * self.config.codebook_size)
            vocab_embeds.append(emb)

        x = torch.stack(vocab_embeds, dim=3)
        x = x.sum(dim=3)

        return x

    def embed_base(self, x: Tensor) -> Tensor:
        x_emb = self.embeddings(x)
        return x_emb

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
    ) -> BaseTransformerForwardResult:
        seq_len = inp.size(1)

        # Here we want to merge the embeddings of the codebooks
        # x = self.embed(inp)
        x = inp.clone()

        if input_pos is None:
            freqs_cis = self.freqs_cis[:seq_len].repeat(inp.size(0), 1, 1, 1)
        else:
            freqs_cis = self.freqs_cis[input_pos]

        if context is not None:
            context_input_pos = torch.arange(context.size(1), device=context.device).unsqueeze(0).repeat(inp.size(0), 1)
            context_freqs_cis = self.freqs_cis[context_input_pos]
        else:
            context_freqs_cis = None

        # Not that the causal mask here follows the definition of scaled_dot_product_attention
        # That is, FALSE means masked out
        # To maintain consistency, key_padding_mask use TRUE to mask out
        mask = None
        if key_padding_mask is not None:
            mask = self.causal_mask[None, None, :seq_len, :seq_len]  # (B, N, Q, K)
            mask = mask & key_padding_mask[:, None, None, :].logical_not()

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, mask, context, context_freqs_cis)

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def forward_generate(
        self,
        inp: Tensor,
        input_pos: Optional[Tensor] = None,
        kv_pos: Optional[Tensor] = None,
        vq_masks: Optional[Tensor] = None,  # this is not used in fact
        return_all: bool = False,
    ) -> BaseTransformerForwardResult:
        # This is used for generation, optimized for torch compile
        # assert (
        #     self.max_seq_len != -1 and self.max_batch_size != -1
        # ), "Please call setup_caches before forward_generate"

        x = inp

        if input_pos is None:
            input_pos = torch.arange(inp.shape[-1], device=x.device)
            max_seq_len = inp.shape[-1]
        else:
            max_seq_len = self.max_seq_len

        mask = self.causal_mask[None, None, kv_pos, :max_seq_len]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=kv_pos)

        # If prefill, we only calculate the logits of last token
        if x.size(1) > 1 and not return_all:
            x = x[:, -1:]

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.is_reward_model:
            token_logits = self.score_output(slow_out)
        elif self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def from_pretrained(
        path: str,
        load_weights: bool = False,
        max_length: int | None = None,
        rope_base: int | None = None,
        is_agent: bool = False,
    ) -> "BaseTransformer":
        config = BaseModelArgs.from_pretrained(str(path))
        if max_length is not None:
            config.max_seq_len = max_length

        if rope_base is not None:
            config.rope_base = rope_base

        match config.model_type:
            case "dual_ar":
                model_cls = DualARTransformer
            case _:
                raise ValueError(f"Unknown model type: {config.model_type}")

        model = model_cls(config)

        return model

    def save_pretrained(self, path: str, drop_lora: bool = False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save(path / "config.json")
        state_dict = self.state_dict()

        if drop_lora:
            for key in list(state_dict.keys()):
                if "lora" not in key:
                    continue

                state_dict.pop(key)

        torch.save(state_dict, path / "model.pth")
        self.tokenizer.save_pretrained(path)

class DualARTransformer(BaseTransformer):
    def __init__(self, config: DualARModelArgs) -> None:
        super().__init__(config, init_weights=False)

        # Project to fast dim if needed
        if config.fast_dim is not None and config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        else:
            self.fast_project_in = nn.Identity()


        # Fast transformer
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)

        # The equivalent bs is so large that sdpa doesn't work
        override_config = dataclasses.replace(
            config,
            dim=config.fast_dim,
            n_head=config.fast_n_head,
            n_local_heads=config.fast_n_local_heads,
            head_dim=config.fast_head_dim,
            intermediate_size=config.fast_intermediate_size,
            attention_qkv_bias=config.fast_attention_qkv_bias,
            has_cross_attention=False,
        )

        self.fast_layers = nn.ModuleList(
            TransformerBlock(override_config, use_sdpa=False)
            for _ in range(config.n_fast_layer)
        )
        self.fast_norm = RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(
            config.fast_dim,
            config.codebook_size,
            bias=False,
        )

        self.register_buffer(
            "fast_freqs_cis",
            precompute_freqs_cis(
                config.num_codebooks,
                config.fast_dim // config.fast_n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.apply(self._init_weights)

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        super().setup_caches(max_batch_size, max_seq_len, dtype)

        head_dim = self.config.fast_dim // self.config.fast_n_head

        # Fast transformer
        # The max seq len here is the number of codebooks
        for b in self.fast_layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.fast_n_local_heads if self.config.fast_n_local_heads > 0 else self.config.fast_n_head,
                head_dim,
                dtype=dtype,
            )

    def forward(
        self,
        inp: Tensor,
        base_target: Tensor,
        target: Tensor,
        target_lens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        delay: int = 0,
        spk_condition_len: int = 33,
    ) -> TransformerForwardResult:
        parent_result = super().forward(inp, key_padding_mask, input_pos, context)
        token_logits = parent_result.logits

        # construct targets for token_logits
        token_targets = torch.zeros(token_logits.size(0), token_logits.size(1), dtype=torch.long, device=target.device) - 100
        for bib in range(token_targets.size(0)):
            token_targets[bib, delay * 2 + spk_condition_len:delay * 2 + spk_condition_len + target_lens[bib] * 2:2]\
                = base_target[bib, :target_lens[bib]]

        x = parent_result.hidden_states
        x = self.fast_project_in(x)

        # Fast transformer
        fast_seq_len = self.config.num_codebooks
        fast_mask = self.causal_mask[
            None, None, :fast_seq_len, :fast_seq_len
        ]  # (B, N, Q, K)

        # Drop the last token and rotate left
        x_latents = torch.cat([
            x[i, delay * 2 + spk_condition_len:delay * 2 + spk_condition_len + target_lens[i] * 2:2] for i in range(x.size(0))
        ], dim=0)
        codebooks = torch.cat([
            target[i, :, :target_lens[i]] for i in range(target.size(0))
        ], dim=-1).transpose(0, 1)

        codebook_embeddings = self.fast_embeddings(codebooks[:, :-1])
        x = torch.cat([x_latents[:, None], codebook_embeddings], dim=1)
        b, s = x.size(0), x.size(1)

        x_bs, x_len = x.size(0), x.size(1)
        fast_freqs_cis = self.fast_freqs_cis[None].repeat(b, 1, 1, 1)
        for layer in self.fast_layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(
                    layer, x, fast_freqs_cis, fast_mask, use_reentrant=True
                )
            else:
                x = layer(x, fast_freqs_cis, fast_mask)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)
        codebook_logits = self.fast_output(fast_out)

        return TransformerForwardResult(
            token_logits=token_logits,
            token_targets=token_targets,
            codebook_logits=codebook_logits,
            codebook_targets=codebooks,
        )

    def forward_generate_fast(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        # Fast transformer
        x = x.view(1, 1, -1)

        fast_mask = self.causal_mask[
            None, None, input_pos, : self.config.num_codebooks
        ]  # (B, N, Q, K)
        fast_freqs_cis = self.fast_freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis, fast_mask, input_pos=input_pos)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)  # only take the last token
        codebook_logits = self.fast_output(fast_out)

        return codebook_logits

    def forward_generate(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        kv_pos: Optional[Tensor] = None,
        vq_masks: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        x = super().forward_generate(x, input_pos, kv_pos, vq_masks)
        x.hidden_states = self.fast_project_in(x.hidden_states)
        return x

    def infer_fast(self, x: Tensor,):
        # Fast transformer
        seq_len = x.size(1)
        fast_mask = self.causal_mask[
            None, None, : seq_len, : seq_len
        ]
        freq_cis = self.fast_freqs_cis[:seq_len][None].repeat(x.size(0), 1, 1, 1)
        for layer in self.fast_layers:
            x = layer(x, freq_cis, fast_mask)

        fast_out = self.fast_norm(x)
        codebook_logits = self.fast_output(fast_out)[:, -1]
        sampled, log_prob = topk_sampling(codebook_logits, top_k=-1, top_p=0.9)
        return sampled


    def infer_slow(self, inp: Tensor, input_pos: Optional[Tensor] = None, context: Optional[Tensor] = None,):
        # no kv cache used
        parent_result = super().forward(inp, input_pos=input_pos, context=context)
        latent = parent_result.hidden_states[:, -1]
        base_logits = parent_result.logits[:, -1]
        base_sampled, _ = topk_sampling(base_logits, top_k=-1, top_p=1.0)
        latent = self.fast_project_in(latent)
        x = latent.unsqueeze(1)

        pred_codes = []
        for i in range(self.config.num_codebooks):
            sampled_code = self.infer_fast(x)
            new_emb = self.fast_embeddings(sampled_code)
            x = torch.cat([x, new_emb], dim=1)
            pred_codes.append(sampled_code)
        return torch.cat(pred_codes, dim=1), base_sampled


class DualARWrapper(nn.Module):
    def __init__(self, model: DualARTransformer) -> None:
        super().__init__()
        self.model = model

        if not isinstance(self.model.config.delay, int):
            max_delay = max(self.model.config.delay)
            self.wait4start_embedding = nn.Embedding(max_delay, model.config.dim)
            self.wait4end_embedding = nn.Embedding(max_delay, model.config.dim)
            self.delay = self.model.config.delay
            self.original_delay = self.model.config.delay
        elif self.model.config.delay > 0:
            self.wait4start_embedding = nn.Embedding(self.model.config.delay, model.config.dim)
            self.wait4end_embedding = nn.Embedding(self.model.config.delay, model.config.dim)
            self.delay = self.model.config.delay
            self.original_delay = self.model.config.delay
        else:
            self.wait4start_embedding = None
            self.wait4end_embedding = None
            self.delay = 0
            self.original_delay = 0

    def setup_caches(self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16):
        self.model.setup_caches(max_batch_size, max_seq_len, dtype)

    def set_delay(self, delay: int):
        delay = int(delay)  # ensure delay is an integer
        if isinstance(self.original_delay, int):
            print("Setting delay is not supported to a model with a single delay value, ignoring operation...")
            return
        self.original_delay = self.model.config.delay
        self.delay = delay
        print(f"Setting delay to {self.delay} frames")

    def forward(self,
                condition: Tensor,
                spk_condition: Tensor,
                base_target: Tensor,
                x: Tensor,
                x_lens: Tensor,
                ) -> torch.Tensor:
        B, T, D = condition.size(0), condition.size(1), condition.size(2)
        x_emb = self.model.embed(x)  # [B, D, T]
        emb_seq_list = []
        if not isinstance(self.delay, int):
            # randomly select a delay
            delay = self.delay[torch.randint(0, len(self.delay), (1,)).item()]
            wait4end_emb = self.wait4end_embedding.weight[:delay]
            wait4start_emb = self.wait4start_embedding.weight[:delay]
        else:
            wait4end_emb = self.wait4end_embedding.weight if self.wait4end_embedding is not None else torch.zeros([0, condition.size(-1)]).to(condition.device)
            wait4start_emb = self.wait4start_embedding.weight if self.wait4start_embedding is not None else torch.zeros([0, condition.size(-1)]).to(condition.device)
            delay = self.delay
        for i in range(x.size(0)):
            cond_wait4end = torch.cat([condition[i, :x_lens[i]], wait4end_emb], dim=0)
            wait4start_x_emb = torch.cat([wait4start_emb, x_emb[i, :x_lens[i]]], dim=0)
            # alternating_emb = cond_wait4end + wait4start_x_emb
            alternating_emb = torch.stack([cond_wait4end, wait4start_x_emb], dim=0).transpose(0, 1).reshape(-1, self.model.config.dim)
            spk_alternating_emb = torch.cat([spk_condition[i], alternating_emb], dim=0)
            emb_seq_list.append(spk_alternating_emb)

        spk_condition_len = spk_condition.size(1)

        emb_seq = torch.nn.utils.rnn.pad_sequence(emb_seq_list, batch_first=True, padding_value=0.0)
        input_pos = torch.arange(emb_seq.size(1), device=emb_seq.device).unsqueeze(0).repeat(emb_seq.size(0), 1)
        out = self.model(emb_seq, base_target, x, x_lens, input_pos=input_pos, delay=delay, spk_condition_len=spk_condition_len)
        codebook_loss = F.cross_entropy(out.codebook_logits.transpose(1, 2), out.codebook_targets.long())
        token_loss = F.cross_entropy(out.token_logits.transpose(1, 2), out.token_targets.long(), ignore_index=-100)
        codebook_per_token_logps = selective_log_softmax(out.codebook_logits, out.codebook_targets.long())
        token_per_token_logps = selective_log_softmax(
            out.token_logits[out.token_targets.long()!=-100], out.token_targets.long()[out.token_targets.long()!=-100])
        return codebook_loss, token_loss, codebook_per_token_logps, token_per_token_logps

    @torch.no_grad()
    def infer(self, cond: Tensor, spk_condition: Tensor, context: Tensor = None,) -> torch.Tensor:
        delay = max(self.delay) if not isinstance(self.delay, int) else self.delay
        B, T, D = cond.size(0), cond.size(1), cond.size(2)
        wait4end_emb = self.wait4end_embedding.weight if self.wait4end_embedding is not None else torch.zeros([0, cond.size(-1)]).to(cond.device)
        wait4start_emb = self.wait4start_embedding.weight if self.wait4start_embedding is not None else torch.zeros([0, cond.size(-1)]).to(cond.device)
        cond = torch.cat([cond, wait4end_emb.unsqueeze(0).repeat(B, 1, 1)], dim=1)
        emb_seq = torch.stack([cond[:, :delay], wait4start_emb.unsqueeze(0).repeat(B, 1, 1)], dim=1).transpose(1, 2).reshape(B, -1, D)
        emb_seq = torch.cat([spk_condition, emb_seq], dim=1)
        pred_codes = []
        for i in tqdm(range(cond.size(1))):
            new_cond = cond[:, delay + i:delay + i + 1]
            emb_seq = torch.cat([emb_seq, new_cond], dim=1)
            input_pos = torch.arange(emb_seq.size(1), device=emb_seq.device)[None].repeat(B, 1)
            x, base = self.model.infer_slow(emb_seq, input_pos, context)
            new_emb = self.model.embed(x.unsqueeze(-1))
            emb_seq = torch.cat([emb_seq, new_emb], dim=1)
            pred_codes.append(x)
        return torch.stack(pred_codes, dim=-1)

    @torch.no_grad()
    def generate(self, ref_cond: Tensor,
                 ref_audio_codes: Tensor,
                 src_cond: Tensor,
                 spk_condition: Tensor,
                 compiled_fn: Callable = None,
                 **sampling_kwargs
                 ) -> torch.Tensor:
        B, T, D = ref_cond.size(0), ref_cond.size(1), ref_cond.size(2)
        # wait4end_emb = self.wait4end_embedding.weight if self.wait4end_embedding is not None else torch.zeros([0, D]).to(src_cond.device)
        # wait4start_emb = self.wait4start_embedding.weight if self.wait4start_embedding is not None else torch.zeros([0, D]).to(src_cond.device)
        wait4end_emb = self.wait4end_embedding.weight[:self.delay] if self.wait4end_embedding is not None else torch.zeros([0, D]).to(src_cond.device)
        wait4start_emb = self.wait4start_embedding.weight[:self.delay] if self.wait4start_embedding is not None else torch.zeros([0, D]).to(src_cond.device)
        ref_emb = self.model.embed(ref_audio_codes)
        ref_emb = torch.cat([wait4start_emb.unsqueeze(0), ref_emb], dim=1)
        prefill_cond = torch.cat([ref_cond, src_cond[:, :self.delay]], dim=1)
        emb_seq = torch.stack([prefill_cond, ref_emb], dim=1).transpose(1, 2).reshape(B, -1, D)
        emb_seq = torch.cat([spk_condition, emb_seq], dim=1)
        remaining_cond = torch.cat([src_cond[:, self.delay:], wait4end_emb.unsqueeze(0)], dim=1)
        pred_codes = []

        # prefill kv cache
        emb_seq = torch.cat([emb_seq, remaining_cond[:, :1]], dim=1)
        input_pos = torch.arange(emb_seq.size(1), device=emb_seq.device)[None]
        kv_pos = torch.arange(emb_seq.size(1), device=emb_seq.device)
        pred = decode_one_token_ar(self.model, emb_seq, input_pos, kv_pos)
        pred_x = pred[1:]
        pred_codes.append(pred_x)

        decode_fn = compiled_fn or decode_one_token_ar

        for i in tqdm(range(remaining_cond.size(1) - 1)):

            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            #
            # # Synchronize before starting
            # torch.cuda.synchronize()
            # start_event.record()

            new_audio_emb = self.model.embed(pred_codes[-1].unsqueeze(0))
            new_cond = remaining_cond[:, i + 1:i + 2]

            emb_seq = torch.cat([new_audio_emb, new_cond], dim=1)

            input_pos = input_pos[:, -2:] + 2
            kv_pos = kv_pos[-2:] + 2

            next_tokens = decode_fn(self.model,
                                  emb_seq.reshape(B, 2, D),
                                  input_pos.reshape(B, 2),
                                  kv_pos.reshape(2),
                                  **sampling_kwargs,
                                  )
            _, pred_x = next_tokens[0], next_tokens[1:]
            pred_codes.append(pred_x.clone())

            # end_event.record()
            # # Synchronize after recording
            # torch.cuda.synchronize()
            #
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f"Time taken: {elapsed_time_ms}ms")

        return torch.stack(pred_codes, dim=-1)

    @torch.no_grad()
    def prefill_prompt(self,
                       ref_cond: Tensor,
                       ref_audio_codes: Tensor,
                       spk_condition: Tensor,
                       compiled_fn: Callable = None,
                       ):
        B, T, D = ref_cond.size(0), ref_cond.size(1), ref_cond.size(2)
        wait4end_emb = self.wait4end_embedding.weight[:self.delay] if self.wait4end_embedding is not None else torch.zeros([0, D]).to(ref_cond.device)
        wait4start_emb = self.wait4start_embedding.weight[:self.delay] if self.wait4start_embedding is not None else torch.zeros([0, D]).to(ref_cond.device)
        ref_emb = self.model.embed(ref_audio_codes)
        self.cached_ref_emb = ref_emb[:, -self.delay:].clone() if self.delay != 0 else ref_emb
        if self.delay != 0:
            ref_emb = torch.cat([wait4start_emb.unsqueeze(0), ref_emb[:, :-self.delay]], dim=1)
        else:
            self.cached_new_audio_emb = ref_emb[:, -1:].clone()
        # prefill_cond = torch.cat([ref_cond, src_cond[:, :self.delay]], dim=1)
        prefill_cond = ref_cond
        emb_seq = torch.stack([prefill_cond, ref_emb], dim=1).transpose(1, 2).reshape(B, -1, D)
        emb_seq = torch.cat([spk_condition, emb_seq], dim=1)
        if self.delay == 0:
            emb_seq = emb_seq[:, :-1]

        # prefill kv cache
        input_pos = torch.arange(emb_seq.size(1), device=emb_seq.device)[None]
        kv_pos = torch.arange(emb_seq.size(1), device=emb_seq.device)
        _ = decode_one_token_ar(self.model, emb_seq, input_pos, kv_pos)

        # record current pos
        self.cached_input_pos = input_pos
        self.cached_kv_pos = kv_pos
        self.prompt_cached_input_pos = input_pos
        self.prompt_cached_kv_pos = kv_pos

    @torch.no_grad()
    def prefill_src_condition4delay(self, src_cond: Tensor, compiled_fn: Callable = None,):
        """
        prefill src condition for delay
        :param src_cond:
        :return:
        """
        assert src_cond.size(1) == self.delay
        B, T, D = src_cond.size(0), src_cond.size(1), src_cond.size(2)
        emb_seq = torch.stack([src_cond, self.cached_ref_emb], dim=1).transpose(1, 2).reshape(B, -1, D)
        self.cached_new_audio_emb = emb_seq[:, -1:].clone()
        emb_seq = emb_seq[:, :-1]
        input_pos = torch.arange(emb_seq.size(1), device=emb_seq.device)[None] + self.cached_input_pos[:, -1:] + 1
        kv_pos = torch.arange(emb_seq.size(1), device=emb_seq.device) + self.cached_kv_pos[-1:] + 1
        decode_fn = compiled_fn or decode_one_token_ar
        _ = decode_fn(self.model, emb_seq, input_pos, kv_pos)
        self.cached_input_pos = input_pos
        self.cached_kv_pos = kv_pos

    @torch.no_grad()
    def decode_one(self,
                   src_cond: Tensor,
                   compiled_fn: Callable = None,
                   ):
        B, T, D = src_cond.size(0), src_cond.size(1), src_cond.size(2)
        emb_seq = torch.cat([self.cached_new_audio_emb, src_cond], dim=1)
        input_pos = torch.arange(emb_seq.size(1), device=emb_seq.device)[None] + self.cached_input_pos[:, -1:] + 1
        kv_pos = torch.arange(emb_seq.size(1), device=emb_seq.device) + self.cached_kv_pos[-1:] + 1
        # pred = decode_one_token_ar(self.model, emb_seq, input_pos, kv_pos)
        decode_fn = compiled_fn or decode_one_token_ar
        next_tokens = decode_fn(self.model,
                                emb_seq.reshape(B, 2, D),
                                input_pos.reshape(B, 2),
                                kv_pos.reshape(2)
                                )
        _, pred_x = next_tokens[0], next_tokens[1:]
        self.cached_new_audio_emb = self.model.embed(pred_x.unsqueeze(0))
        self.cached_input_pos = input_pos
        self.cached_kv_pos = kv_pos
        return pred_x, kv_pos[-1]

class TransformerBlock(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attention = Attention(config, use_sdpa=use_sdpa)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

        if config.has_cross_attention:
            self.has_cross_attention = True
            self.cross_attention = Attention(config, is_cross_attention=True)
            self.cross_attention_norm = RMSNorm(config.dim, config.norm_eps)
        else:
            self.has_cross_attention = False

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None, context: Tensor = None, context_freqs_cis: Tensor = None
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, input_pos=input_pos, mask=mask)
        if self.has_cross_attention:
            h = h + self.cross_attention(self.cross_attention_norm(h), freqs_cis, context, context_freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True, is_cross_attention: bool = False) -> None:
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        if is_cross_attention:
            self.wq = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
            self.wkv = nn.Linear(config.context_dim, 2 * config.n_local_heads * config.head_dim, bias=False)
        else:
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        # self._register_load_state_dict_pre_hook(self.load_hook)
        self.is_cross_attention = is_cross_attention

    # def load_hook(self, state_dict, prefix, *args):
    #     if prefix + "wq.weight" in state_dict:
    #         wq = state_dict.pop(prefix + "wq.weight")
    #         wk = state_dict.pop(prefix + "wk.weight")
    #         wv = state_dict.pop(prefix + "wv.weight")
    #         state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_freqs_cis: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        if context is None:
            q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
            context_seqlen = seqlen
        else:
            q = self.wq(x)
            k, v = self.wkv(context).split([kv_size, kv_size], dim=-1)
            context_seqlen = context.shape[1]

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, context_freqs_cis if context_freqs_cis is not None else freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if input_pos is not None and self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                           is_causal=False if mask is not None or self.is_cross_attention else True,
                                           attn_mask=mask,)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)

        y = self.wo(y)
        return y

    def eq_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        # This is a standard scaled dot product attention
        # It's low efficient, but it doesn't raise cuda error

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight @ value


class FeedForward(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(x.size(0), xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    current_logprobs = logprobs[torch.arange(logprobs.shape[0]), token.squeeze(1)]
    return token, current_logprobs

def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[List[int]] = None,
    temperature: torch.Tensor = 0.7,
    top_p: torch.Tensor = 0.7,
    repetition_penalty: torch.Tensor = 1.5,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)
    if suppress_tokens is not None:
        for token in suppress_tokens:
            logits[token] = -float("Inf")

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

@torch.no_grad()
def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    kv_pos: torch.Tensor,
    # mask: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    suppress_tokens: Optional[List[int]] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos, kv_pos)

    sampling_kwargs_main = sampling_kwargs.copy()

    codebooks = [
        sample(
            x.logits,
            previous_tokens=(
                previous_tokens[0] if previous_tokens is not None else None
            ),  # Disable repetition penalty for the token codebook
            suppress_tokens=suppress_tokens,
            **sampling_kwargs_main,
        )[0]
    ]

    hidden_states = x.hidden_states

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample(
            logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a.clone())

    codebooks = torch.stack(codebooks, dim=0)
    return codebooks