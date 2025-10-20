import torch
from torch import nn
import math

from modules.asr.transformer import ModelArgs, Transformer

from torch.nn.utils import weight_norm

import torch.nn.functional as F

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class ASRDecoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        block_size: int = 4096,
        in_channels: int = 512,
        loss_type: str = "s2s",  # ctc or s2s
        n_vocab: int = 51866,
        bos_id: int = 50528,
        eos_id: int = 50527,
        dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
    ):
        super(ASRDecoder, self).__init__()
        self.loss_type = loss_type
        model_args = ModelArgs(
            block_size=block_size,
            n_layer=depth,
            n_head=num_heads,
            dim=hidden_dim,
            head_dim=hidden_dim // num_heads,
            vocab_size=1024,
            has_cross_attention=False,
            context_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.transformer = Transformer(model_args)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.prediction_head = nn.Linear(hidden_dim, n_vocab)
        self.bos_id = bos_id
        self.eos_id = eos_id

        input_pos = torch.arange(block_size)
        self.register_buffer("input_pos", input_pos)

        self.text_embedding = nn.Embedding(n_vocab, hidden_dim)
        self.audio_feat_projection = nn.Linear(in_channels, hidden_dim) if in_channels != hidden_dim else nn.Identity()

    def forward(self, x, x_lens, text, text_lens):
        B = x.size(0)
        # add bos for text
        text = torch.cat([torch.zeros([text.size(0), 1]).long().to(text.device) + self.bos_id, text], dim=-1)
        # add eos for text
        text = torch.cat([text, torch.zeros([text.size(0), 1]).long().to(text.device)], dim=-1)
        text_lens = text_lens + 2
        for bib in range(text.size(0)):
            text[bib, text_lens[bib] - 1:] = self.eos_id

        text_embed = self.text_embedding(text)

        # concat audio feature as prefix
        x = self.audio_feat_projection(x)

        x_text = torch.zeros([x.size(0), text.size(1) + x.size(1), x.size(2)]).to(x.device)
        for bib in range(B):
            x_len = x_lens[bib]
            text_len = text_lens[bib]
            x_text[bib, :x_len] = x[bib, :x_len]
            x_text[bib, x_len:x_len + text_len] = text_embed[bib, :text_len]

        input_pos = self.input_pos[:x_text.size(1)]  # (T,)
        output = self.transformer(x_text, None, input_pos)
        logits = self.prediction_head(output[:, :-1])
        targets = text[:, 1:]
        s2s_loss = 0
        for bib in range(B):
            x_len = x_lens[bib]
            # x_len = 0
            text_len = text_lens[bib]
            s2s_loss = s2s_loss + torch.nn.functional.cross_entropy(
                logits[bib, x_len:x_len + text_len - 1],
                targets[bib, :text_len - 1],
                reduction='mean',
            )
        s2s_loss = s2s_loss / text.size(0)

        return s2s_loss
    @torch.no_grad()
    def decode(self, x):
        text = torch.zeros([x.size(0), 1]).long().to(x.device) + self.bos_id
        pred_phone_ids = []
        x = self.audio_feat_projection(x)
        while True:
            text_embed = self.text_embedding(text)
            x_text = torch.cat([x, text_embed], dim=1)
            input_pos = self.input_pos[:x_text.size(1)]  # (T,)
            output = self.transformer(x_text, None, input_pos)
            logits = self.prediction_head(output[:, -1:])

            pred_phone_id = topk_sampling(logits[0], top_k=-1, top_p=0.9, temperature=1.0)[0]
            pred_phone_ids.append(pred_phone_id.item())
            if pred_phone_id == self.eos_id or len(pred_phone_ids) > 500:
                break
            text = torch.cat([text, pred_phone_id], dim=-1)
        return pred_phone_ids

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
