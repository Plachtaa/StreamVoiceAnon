# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import time

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    dim: int = 512
    intermediate_size: int = 1536
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True  # to be compatible with conv1d input/output
    dw_conv: bool = False  # whether to use depthwise conv in feed-forward
    conv_kernel_size: int = 5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out[:, :, :input_pos.max() + 1, :], v_out[:, :, :input_pos.max() + 1, :]

    def clear_cache(self, prompt_len):
        self.k_cache[:, :, prompt_len:, :].fill_(0)
        self.v_cache[:, :, prompt_len:, :].fill_(0)


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.head_dim,
                                              self.config.rope_base)
        causal_mask = torch.tril(torch.ones(self.config.block_size, self.config.block_size, dtype=torch.bool))
        self.register_buffer('freqs_cis', freqs_cis)
        self.register_buffer('causal_mask', causal_mask)

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    def setup_caches(self, max_batch_size, max_seq_length):
        """
        This method will only be called during inference when using KV cache.
        """
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.weight.dtype
        device = self.norm.weight.device

        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype).to(device)

        self.use_kv_cache = True

    def forward(self,
                x: Tensor,
                input_pos: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if mask is None: # in case of non-causal model
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., :input_pos.max() + 1]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_layer_scale = LayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = LayerScale(config.dim, inplace=True)
        self.lconv1d = LConv1d(config) if config.dw_conv else nn.Identity()

    def forward(self,
                x: Tensor,
                input_pos: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                ) -> Tensor:
        h = x + self.attention_layer_scale(self.attention(self.attention_norm(x), freqs_cis, mask, input_pos))
        h = self.lconv1d(h)  # apply depthwise conv if specified
        out = h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate

    def forward(self,
                x: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                input_pos: Optional[Tensor] = None,
                ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        context_seqlen = seqlen

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_dropout_rate if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)

        y = self.wo(y)
        return y


class LConv1d(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.pre_layer_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.conv_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.layer_scale = LayerScale(config.dim, inplace=True)
        self.linear_start = nn.Linear(config.dim, config.dim * 2, bias=False)
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=config.dim,
            out_channels=config.dim,
            kernel_size=config.conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=config.dim,  # Depthwise
            bias=False,
        )
        self.linear_end = nn.Linear(config.dim, config.dim, bias=False)
        self.causal_padding = config.conv_kernel_size - 1

    def forward(self, x: Tensor) -> Tensor:
        audio_encodings = x  # rename
        audio_encodings_residual = audio_encodings  # Save for residual connection

        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.linear_start(audio_encodings)
        audio_encodings = torch.nn.functional.glu(audio_encodings, dim=-1)
        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)
        # Apply manual causal padding
        audio_encodings_permuted_padded = F.pad(audio_encodings_permuted, (self.causal_padding, 0))
        audio_encodings = self.depthwise_conv1d(audio_encodings_permuted_padded)
        # Permute back: [B, D, T_out] -> [B, T_out, D]
        audio_encodings = audio_encodings.permute(0, 2, 1)
        audio_encodings = self.conv_norm(audio_encodings)
        audio_encodings = nn.functional.silu(audio_encodings)
        audio_encodings = self.linear_end(audio_encodings)
        output = self.layer_scale(audio_encodings) + audio_encodings_residual
        return output

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

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
    
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-2,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class WindowLimitedTransformer(Transformer):
    """
    Transformer with window limited attention, causal or non-causal.
    TODO: test whether non-causal window works as expected.
    """
    def __init__(
            self,
            config: ModelArgs,
            window_size: Optional[int] = None,
            causal: bool = True,
            ):
        super().__init__(config)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
    
    def make_window_limited_mask(self,
                                 max_length: int,
                                 x_lens: Optional[Tensor] = None,
                                 ) -> Tensor:
        """
        Make mask to form window limited attention.
        """
        if self.causal:
            mask = torch.tril(torch.ones(max_length, max_length))
            row_indices = torch.arange(max_length).view(-1, 1)
            valid_range = (row_indices - self.window_size + 1).clamp(min=0)
            column_indices = torch.arange(max_length)
            mask = (column_indices >= valid_range) & mask.bool()
        else:
            mask = torch.ones(max_length, max_length)
            half_context = self.window_size // 2
            row_indices = torch.arange(max_length).view(-1, 1)
            column_indices = torch.arange(max_length)
            left_limit = (row_indices - half_context).clamp(min=0)
            right_limit = (row_indices + half_context + 1).clamp(
                max=max_length
            )
            mask = ((column_indices >= left_limit) & (column_indices < right_limit)).to(
                torch.float32
            )
        mask = mask.bool()[None, None]
        return mask

    def make_mask(self,
                  max_length: int,
                  x_lens: Optional[Tensor] = None,
                  ) -> Tensor:
        """
        Make ordinary mask if window size is not specified.
        """
        if self.causal:
            mask = torch.tril(torch.ones(max_length, max_length))
            mask = mask.bool()[None, None]
        else:
            mask = torch.ones(max_length, max_length)
            mask = mask.bool()[None, None].repeat(len(x_lens) if x_lens is not None else 1, 1, 1, 1)
            for i, x_len in enumerate(x_lens):
                mask[i, :, :, x_len:] = 0

        return mask

    def forward(self,
                x: Tensor,
                x_lens: Optional[Tensor] = None,
                ) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)
        input_pos = torch.arange(x.shape[1], device=x.device)
        # construct mask to form window limited attention
        max_length = x.shape[1]
        if self.window_size is not None:
            mask = self.make_window_limited_mask(max_length, x_lens)
        else:
            mask = self.make_mask(max_length, x_lens)
        mask = mask.to(x.device)
        x = super().forward(x, input_pos, mask)
        if self.channels_first:
            x = x.transpose(1, 2)
        return x

def precompute_freqs_cis(
        seq_len: int, n_elem: int, base: int = 10000,
        dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

if __name__ == "__main__":
    config = ModelArgs()
    transformer = WindowLimitedTransformer(config, window_size=128, causal=True)
    x = torch.randn(1, 1024, 512)
    x_lens = torch.LongTensor([1024])
    input_pos = torch.arange(1024)
    output = transformer(x, x_lens, input_pos)
    print(output.shape) 

