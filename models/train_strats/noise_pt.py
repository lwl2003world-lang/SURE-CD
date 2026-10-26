try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass
from functools import partial

import torch
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss

from models.common.perlin import rand_perlin_2d
from models.loss.dice import dice_loss_smooth, dice_loss_eps
from models.train_strats.base import BaseStrategy

import torch.nn.functional as F


class NoiseType(StrEnum):
    NORMAL = "normal"
    LAPLACE = "laplace"
    UNIFORM = "uniform"
    NONE = "none"


class MaskType(StrEnum):
    PERLIN = "perlin"


class ChangeStatSource(StrEnum):
    FEAT = "feat"
    DIFF = "diff"


class NoisePT(BaseStrategy):
    """Noise-based latent perturbation strategy."""

    def __init__(
        self,
        change_noise_q: float | tuple[float, float],
        change_noise_type: NoiseType,
        change_stat_source: ChangeStatSource,
        base_noise_q: float | tuple[float, float],
        base_noise_type: NoiseType,
        overlap_base: bool = True,
        mask_type: MaskType = MaskType.PERLIN,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "changen_class"):
            self.changen_class = ChangeGenerator
        if not hasattr(self, "other_changen_kwargs"):
            self.other_changen_kwargs = {}
        self.transfer = False
        self.loss = self.build_loss()

        self.noise_gen = self.changen_class(
            change_noise_q=change_noise_q,
            change_noise_type=change_noise_type,
            change_stat_source=change_stat_source,
            base_noise_q=base_noise_q,
            base_noise_type=base_noise_type,
            overlap_base=overlap_base,
            mask_type=mask_type,
            **self.other_changen_kwargs,
        )

    def build_loss(self):
        loss_list = self.config.train.loss
        loss_fn_dict = {}

        for l in loss_list:
            if l == "focal":
                loss_fn_dict[l] = partial(
                    sigmoid_focal_loss, reduction="mean", alpha=-1
                )
            elif l == "ce":
                loss_fn_dict[l] = torch.nn.BCEWithLogitsLoss()
            elif l == "2chce":
                f = torch.nn.CrossEntropyLoss()

                def two_ch_ce(pred, t):
                    return f(pred, t.squeeze().long())

                loss_fn_dict[l] = two_ch_ce
            elif l == "dice":
                loss_fn_dict[l] = dice_loss_smooth
            elif l == "diceE":
                loss_fn_dict[l] = dice_loss_eps
            else:
                raise ValueError(f"Unknown loss function {l}")

        def loss_callable(preds, target):
            total_loss = 0
            loss_terms = {}
            for name, curr_loss_f in loss_fn_dict.items():
                curr_loss_val = curr_loss_f(preds, target.float())
                loss_terms[name] = curr_loss_val.item()
                total_loss += curr_loss_val
            return total_loss, loss_terms

        return loss_callable

    def forward(self, batch, lbl=None):
        i1, i2 = batch["imageA"], batch["imageB"]
        x = torch.cat([i1, i2], dim=0)

        x = self.enc(x)
        if self.training:
            x = {k: torch.cat([v, v.detach()], dim=0) for k, v in x.items()}
            lbl = torch.cat([lbl, lbl], dim=0)
            x, lbl = self.noise_gen(x, lbl)

        x = self.diff(x)
        x = self.dec(x)

        return x, lbl

    def __call__(self, batch, loss_only=True):
        b, _, h, w = batch["imageA"].shape
        lbl = torch.zeros(b, 1, h, w, dtype=torch.float, device=batch["imageA"].device)

        pred, lbl = self.forward(batch, lbl)

        if loss_only:
            return self.loss(pred, lbl)
        else:
            return pred, self.loss(pred, lbl)

    def get_parameters(self, pretraining):
        return []


class ChangeGenerator(nn.Module):
    def __init__(
        self,
        change_noise_q: float | tuple[float, float],
        change_noise_type: NoiseType,
        change_stat_source: ChangeStatSource,
        base_noise_q: float | tuple[float, float],
        base_noise_type: NoiseType,
        overlap_base: bool,
        mask_type: MaskType,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.change_noise_q = change_noise_q
        self.change_noise_type = change_noise_type
        self.change_stat_source = change_stat_source

        self.base_noise_q = base_noise_q
        self.base_noise_type = base_noise_type
        self.overlap_base = overlap_base

        self.mask_type = mask_type

        if isinstance(self.change_noise_q, (tuple, list)):
            self.change_noise_q = torch.tensor(self.change_noise_q, dtype=torch.float)
            assert self.change_noise_type != NoiseType.NORMAL
        if isinstance(self.base_noise_q, (tuple, list)):
            self.base_noise_q = torch.tensor(self.base_noise_q, dtype=torch.float)
            assert self.base_noise_type != NoiseType.NORMAL

        self.min_perlin_scale = 0
        self.max_perlin_scale = 6

    def generate_perlin(self, batches, h, w) -> Tensor:
        """Generate 2D Perlin noise masks."""
        perlin = []
        for _ in range(batches):
            perlin_scalex = (
                2
                ** (
                    torch.randint(
                        self.min_perlin_scale, self.max_perlin_scale, (1,)
                    ).numpy()[0]
                )
            )
            perlin_scaley = (
                2
                ** (
                    torch.randint(
                        self.min_perlin_scale, self.max_perlin_scale, (1,)
                    ).numpy()[0]
                )
            )

            perlin_noise = rand_perlin_2d(
                (h, w), (perlin_scalex, perlin_scaley)
            ).reshape(1, 1, h, w)

            perlin_thr = torch.where(perlin_noise > 0.5, 1, 0)

            chance_change = torch.rand(1).numpy()[0]
            if chance_change > 0.5:
                perlin_thr = torch.zeros_like(perlin_thr)

            perlin.append(perlin_thr)
        return torch.cat(perlin)

    def gen_uniform(self, stat_feat, q, shape) -> Tensor:
        if isinstance(q, torch.Tensor):
            low, up = torch.quantile(stat_feat, q.to(stat_feat.device)).detach()
        else:
            up = torch.quantile(stat_feat, q).item()
            low = -up

        uniform_noise = torch.rand(
            size=shape, device=stat_feat.device, requires_grad=False
        )
        return uniform_noise * (up - low) + low

    def gen_normal(self, stat_feat, q, shape, laplace=False) -> Tensor:
        offset = torch.quantile(stat_feat, q).abs().clamp_min(1e-6)
        dist = torch.distributions.Laplace if laplace else torch.distributions.Normal
        return dist(
            torch.zeros_like(offset),
            offset,
        ).sample(shape)

    def gen_noise(self, noise_type, stat_feat, shape, q):
        if noise_type == NoiseType.NORMAL:
            return self.gen_normal(stat_feat, q, shape)
        elif noise_type == NoiseType.LAPLACE:
            return self.gen_normal(stat_feat, q, shape, laplace=True)
        elif noise_type == NoiseType.UNIFORM:
            return self.gen_uniform(stat_feat, q, shape)
        else:
            raise ValueError(f"Unknown noise type {noise_type}")

    def forward(self, feat_dict: dict[Tensor], mask: Tensor) -> tuple[dict, Tensor]:
        feats = list(feat_dict.values())
        b, _, h, w = mask.shape

        if self.mask_type == MaskType.PERLIN:
            binary_mask = self.generate_perlin(b, h, w).to(feats[0].device)
        else:
            raise ValueError(f"Unknown mask type {self.mask_type}")

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
                base_noise = self.gen_noise(
                    self.base_noise_type, diff, f2.shape, self.base_noise_q
                )
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
                self.change_noise_type, ch_stat_feat, f2.shape, self.change_noise_q
            )
            f2 = f2 + change_noise * noise_mask_scaled

            if base_noise is not None:
                f2 = f2 + base_noise

            perturbed = torch.cat((f1, f2), dim=0)
            noised_feat[feat_key] = perturbed

        return noised_feat, mask
