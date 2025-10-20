from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .bsq import GroupedResidualBSQ

from .firefly import ConvNeXtBlock, FishConvNet, FishTransConvNet


@dataclass
class BSQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor
    aux_loss: torch.Tensor


class DownsampleBinarySphericalQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_groups: int = 1,
        codebook_size: int = 10,  # 2 ** 10 = 1024
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
        pre_module: nn.Module | None = None,
        post_module: nn.Module | None = None,
        do_upsample: bool = False,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.residual_bsq = GroupedResidualBSQ(
            dim=all_dims[-1],
            codebook_size=codebook_size,
            groups=n_groups,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    FishConvNet(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    FishTransConvNet(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        ) if do_upsample else nn.Identity()

        self.apply(self._init_weights)

        self.pre_module = pre_module if pre_module is not None else nn.Identity()
        self.post_module = post_module if post_module is not None else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, z, z_lens) -> BSQResult:
        z = self.downsample(z)
        z = self.pre_module(z)  # B, T, D
        quantized, indices, aux_loss = self.residual_bsq(z.mT)
        result = BSQResult(
            z=quantized.mT,
            codes=indices.mT,
            latents=quantized.mT,  # Latents are the same as quantized in this case
            aux_loss=sum(aux_loss),
        )
        result.z = self.upsample(result.z)
        result.z = self.post_module(result.z, z_lens)

        return result

    def encode(self, z):
        z = self.downsample(z)
        z = self.pre_module(z)  # B, T, D
        _, indices, _ = self.residual_bsq(z.mT)
        return indices

    def decode(self, indices: torch.Tensor):
        indices = rearrange(indices, "b (g r) l -> g b l r", g=self.residual_fsq.groups)
        z_q = self.residual_fsq.get_output_from_indices(indices).mT
        return z_q
