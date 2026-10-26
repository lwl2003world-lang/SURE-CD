import math

import torch

from models.train_strats.cross_donor_pt import CrossDonorChangeGenerator
from models.train_strats.separate_lq_soft_noise_pt import SeparateLrnNoisePT


class StableResidualGenerator(CrossDonorChangeGenerator):
    """Stable-Residual (SR) generator."""

    def __init__(self, stable_mix=0.5, stable_fraction=0.5, *args, **kwargs):
        if not 0 <= stable_mix <= 1:
            raise ValueError('stable_mix must be in [0, 1]')
        if not 0 < stable_fraction <= 1:
            raise ValueError('stable_fraction must be in (0, 1]')
        super().__init__(*args, **kwargs)
        self.stable_mix = stable_mix
        self.stable_fraction = stable_fraction

    def stable_pool(self, diff):
        tokens = diff.detach().permute(0, 2, 3, 1).reshape(-1, diff.shape[1])
        scale = tokens.square().mean(0, keepdim=True).sqrt().clamp_min(1e-6)
        energy = (tokens / scale).square().mean(1)
        count = max(1, int(tokens.shape[0] * self.stable_fraction))
        pool = tokens[energy.argsort()[:count]]
        return pool - pool.mean(0, keepdim=True)

    def resample_stable(self, diff, shape):
        b, c, h, w = shape
        if diff.shape[1] != c:
            raise ValueError('Temporal residual and output channel counts must match')
        pool = self.stable_pool(diff)
        indices = torch.randint(pool.shape[0], (b * h * w,), device=diff.device)
        signs = torch.randint(2, (b * h * w, 1), device=diff.device).to(diff.dtype) * 2 - 1
        return (pool[indices] * signs).view(b, h, w, c).permute(0, 3, 1, 2)

    def gen_struct_base_noise(self, stat_feat, shape, bq):
        synthetic = super().gen_struct_base_noise(stat_feat, shape, bq)
        if self.stable_mix == 0:
            return synthetic
        empirical = self.resample_stable(stat_feat, shape)
        dims = (0, 2, 3)
        energy = empirical.square().mean(dims, keepdim=True).sqrt()
        target = synthetic.square().mean(dims, keepdim=True).sqrt()
        empirical = empirical * target / energy.clamp_min(1e-6)
        mixed = (math.sqrt(1 - self.stable_mix) * synthetic
                 + math.sqrt(self.stable_mix) * empirical)
        return torch.where(energy > 1e-6, mixed, synthetic)


class StableResidualPT(SeparateLrnNoisePT):
    """Stable-Residual training strategy."""

    def __init__(
        self, stable_mix=0.5, stable_fraction=0.5,
        donor_prob=0.8, alpha_min=0.5, alpha_max=0.85,
        box_mask_prob=0.3, spatial_blur_prob=0.8,
        spatial_blur_sigma=1.5, spatial_alpha=0.5, *args, **kwargs,
    ):
        self.changen_class = StableResidualGenerator
        self.other_changen_kwargs = dict(
            stable_mix=stable_mix, stable_fraction=stable_fraction,
            donor_prob=donor_prob, alpha_min=alpha_min, alpha_max=alpha_max,
            box_mask_prob=box_mask_prob, spatial_blur_prob=spatial_blur_prob,
            spatial_blur_sigma=spatial_blur_sigma, spatial_alpha=spatial_alpha,
        )
        super().__init__(*args, **kwargs)
