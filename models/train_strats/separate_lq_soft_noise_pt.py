try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

import einops
import torch
from torch import Tensor, nn

from models.train_strats.noise_pt import (
    NoisePT,
    ChangeGenerator,
    ChangeStatSource,
    NoiseType,
)

import torch.nn.functional as F


class Q_DIM(StrEnum):
    C = "c"
    B = "b"
    BC = "bc"


class SeparateLrnNoisePT(NoisePT):
    _Q_PARAM_KEYS = ("change_noise_q_param", "base_noise_q_param")

    def __init__(
        self,
        q_dim: Q_DIM,
        change_noise_q: float,
        base_noise_q: float,
        *args,
        **kwargs,
    ):
        if not hasattr(self, "changen_class"):
            self.changen_class = LearnableSeparateChangeGenerator

        if q_dim == Q_DIM.C:
            self.reshape_rule = "b c h w -> (b h w) c"
        elif q_dim == Q_DIM.B:
            self.reshape_rule = "b c h w -> (c h w) b"
        elif q_dim == Q_DIM.BC:
            self.reshape_rule = "b c h w -> (h w) b c"
        else:
            raise ValueError(f"Unrecognized q_dims: {q_dim}")

        self.q_dim = q_dim
        self.other_changen_kwargs = {
            **getattr(self, "other_changen_kwargs", {}),
            "reshape_rule": self.reshape_rule,
            "q_dim": self.q_dim,
        }
        super().__init__(
            change_noise_q=change_noise_q, base_noise_q=base_noise_q, *args, **kwargs
        )

        def q_to_logit(q: float, eps: float = 1e-4) -> Tensor:
            q = torch.tensor(float(q)).clamp(eps, 1.0 - eps)
            return torch.log(q / (1.0 - q))

        self.change_noise_q_param = nn.Parameter(q_to_logit(change_noise_q))
        self.base_noise_q_param = nn.Parameter(q_to_logit(base_noise_q))
        self.register_state_dict_post_hook(lambda *args: self._drop_q_params(*args))
        self.register_load_state_dict_post_hook(self._allow_missing_q_params)

    def _drop_q_params(self, _, state_dict, prefix, __):
        for key in self._Q_PARAM_KEYS:
            state_dict.pop(prefix + key, None)

    def _allow_missing_q_params(self, _, incompatible_keys):
        incompatible_keys.missing_keys[:] = [
            k
            for k in incompatible_keys.missing_keys
            if not k.endswith(self._Q_PARAM_KEYS)
        ]

    def forward(self, batch, lbl=None):
        cq = torch.sigmoid(self.change_noise_q_param)
        bq = torch.sigmoid(self.base_noise_q_param)

        i1, i2 = batch["imageA"], batch["imageB"]
        x = torch.cat([i1, i2], dim=0)

        x = self.enc(x)

        if self.training:
            x = {k: torch.cat([v, v.detach()], dim=0) for k, v in x.items()}
            lbl = torch.cat([lbl, lbl], dim=0)
            x, lbl = self.noise_gen(x, lbl, cq=cq, bq=bq)

        x = self.diff(x)
        x = self.dec(x)

        return x, lbl

    def get_parameters(self, pretraining):
        return [{"params": self.change_noise_q_param, "lr": 1e-7}] + [
            {"params": self.base_noise_q_param, "lr": 1e-7}
        ]


def soft_quantile(
    values: Tensor,
    q: Tensor,
) -> Tensor:
    values = values.sort(dim=0).values
    n = values.shape[0]

    if n == 1:
        return values[0]

    position = q.clamp(0, 1) * (n - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp(max=n - 1)
    fraction = position - position.floor()
    return values[lower] + fraction * (values[upper] - values[lower])


class LearnableSeparateChangeGenerator(ChangeGenerator):
    def __init__(self, reshape_rule, q_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reshape_rule = reshape_rule
        self.q_dim = q_dim

    def gen_normal(self, stat_feat, q, shape, laplace=False):
        sb = stat_feat.shape[0]
        stat_feat = einops.rearrange(stat_feat, self.reshape_rule)

        if stat_feat.dtype == torch.float16:
            stat_feat = stat_feat.to(torch.float32)

        q = q.to(stat_feat.device, dtype=stat_feat.dtype) if torch.is_tensor(q) else q
        offset = soft_quantile(stat_feat, q).abs()

        b, c, h, w = shape

        if self.q_dim == Q_DIM.C:
            view_vals = (1, c, 1, 1)
            repeat_vals = (b, 1, h, w)
        elif self.q_dim == Q_DIM.B:
            view_vals = (sb, 1, 1, 1)
            repeat_vals = (1, c, h, w)
        elif self.q_dim == Q_DIM.BC:
            view_vals = (sb, c, 1, 1)
            repeat_vals = (1, 1, h, w)
        else:
            raise ValueError(f"Unknown q_dim: {self.q_dim}")

        offset = offset.view(*view_vals).repeat(*repeat_vals).to(stat_feat.device)

        if sb != b and self.q_dim != Q_DIM.C:
            offset = torch.cat((offset, offset), dim=0)

        offset = offset.clamp_min(1e-6)

        if laplace:
            u = torch.rand_like(offset) - 0.5
            noise = (
                -offset * torch.sign(u) * torch.log1p(-2.0 * u.abs()).clamp_min(-20.0)
            )
        else:
            noise = torch.randn_like(offset) * offset

        return noise

    def forward(
        self, feat_dict: dict[Tensor], mask: Tensor, cq, bq
    ) -> tuple[dict, Tensor]:
        feats = list(feat_dict.values())
        b, _, h, w = mask.shape

        binary_mask = self.generate_perlin(b, h, w).to(feats[0].device)
        noise_mask = torch.ones(b, 1, h, w, device=feats[0].device, requires_grad=False) * binary_mask
        mask = torch.where((mask + noise_mask) > 0, 1, 0)

        noised_feat = {}
        for feat_key, feat_val in feat_dict.items():
            noise_mask_scaled = F.interpolate(
                noise_mask,
                size=feat_val.shape[2:],
                mode="bilinear",
            )

            f1, f2 = feat_val.chunk(2, dim=0)
            f1_real, f2_real = f1.chunk(2, dim=0)
            diff = f1_real - f2_real

            # Base noise
            if self.base_noise_type != NoiseType.NONE:
                base_noise = self.gen_noise(self.base_noise_type, diff, f2.shape, bq)
                if not self.overlap_base:
                    base_noise = base_noise * (1 - noise_mask_scaled)
            else:
                base_noise = None

            if self.change_stat_source == ChangeStatSource.FEAT:
                ch_stat_feat = f2
            elif self.change_stat_source == ChangeStatSource.DIFF:
                ch_stat_feat = diff
            else:
                raise ValueError(
                    f"Unknown change_stat_source {self.change_stat_source}"
                )

            # Change noise
            change_noise = self.gen_noise(
                self.change_noise_type, ch_stat_feat, f2.shape, cq
            )
            f2 = f2 + change_noise * noise_mask_scaled

            if base_noise is not None:
                f2 = f2 + base_noise

            perturbed = torch.cat((f1, f2), dim=0)
            noised_feat[feat_key] = perturbed

        return noised_feat, mask
