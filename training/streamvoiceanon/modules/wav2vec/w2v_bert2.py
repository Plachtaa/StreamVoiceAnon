import torch
from torch import nn
import torchaudio

from typing import List, Optional

from transformers import AutoFeatureExtractor, Wav2Vec2BertModel


class Wav2Vec2BertEncoder(nn.Module):
    def __init__(
            self,
            input_sample_rate: int = 44100,
            output_layer: int | List[int] = 16,
            dtype: torch.dtype = torch.float16,
            output_hidden_states: bool = False,
    ):
        super().__init__()
        self.model_name = "facebook/w2v-bert-2.0"
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.w2v_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

        max_output_layer = max(output_layer) if not isinstance(output_layer, int) else output_layer
        self.w2v_model.encoder.layers = self.w2v_model.encoder.layers[:max_output_layer]
        self.w2v_model.eval()
        self.w2v_model.to(dtype=dtype)
        # set all parameters to untrainable
        for param in self.w2v_model.parameters():
            param.requires_grad = False
        self.sample_rate = input_sample_rate
        self.dtype = dtype
        self.mel_filters = torch.from_numpy(self.processor.mel_filters)
        self.window = torch.from_numpy(self.processor.window)
        self.output_hidden_states = output_hidden_states
        self.output_layer = output_layer if not isinstance(output_layer, int) else output_layer

    def train(self, mode: bool = True):
        super(Wav2Vec2BertEncoder, self).train(mode)
        # pretrained model is frozen
        self.w2v_model.eval()

    @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
    @torch.no_grad()
    def forward(
            self,
            x: torch.Tensor,
            x_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(self.dtype)
        if x_lens is None:
            x_lens = torch.LongTensor([x.shape[-1]] * x.shape[0]).to(x.device)
        # resample x to 16k
        if self.sample_rate != 16000:
            x = torchaudio.functional.resample(x, self.sample_rate, 16000)
            x_lens = (x_lens * 16000 / self.sample_rate).long()
        # print(f"x_lens: {x_lens}")
        # pad x left and right by 160 each
        # print(x.shape)
        x = torch.nn.functional.pad(x, (160, 160), mode='constant', value=0)
        x_lens = x_lens + 320  # pad left and right by 160 each
        np_wave_16k_list = [
            x[i, : x_lens[i]].cpu().numpy() for i in range(len(x))
        ]
        # for w in np_wave_16k_list:
        #     print(w.shape)
        inputs = self.processor(
            np_wave_16k_list,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = inputs.to(self.w2v_model.device).to(self.dtype)
        # print(inputs.input_features.shape)
        outputs = self.w2v_model(
            input_features=inputs.input_features.to(dtype=self.dtype),
            attention_mask=inputs.attention_mask.to(dtype=self.dtype),
            output_hidden_states=True,
        )
        feat = sum(outputs.hidden_states[i] for i in self.output_layer) / len(self.output_layer)
        if self.output_hidden_states:
            # print(outputs.last_hidden_state.shape)
            return feat.mT.to(input_dtype)
        else:
            return outputs