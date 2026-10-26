import math

import torch

from models.train_strats.cross_donor_pt import CrossDonorChangeGenerator
from models.train_strats.separate_lq_soft_noise_pt import SeparateLrnNoisePT


class EmpiricalNuisanceGenerator(CrossDonorChangeGenerator):
    """Empirical nuisance residual generator."""

    def __init__(self, empirical_mix=0.5, nuisance_clip_q=0.8, *args, **kwargs):
        if not 0 <= empirical_mix <= 1:
            raise ValueError("empirical_mix must be in [0, 1]")
        if not 0 < nuisance_clip_q < 1:
            raise ValueError("nuisance_clip_q must be in (0, 1)")
        super().__init__(*args, **kwargs)
        self.empirical_mix = empirical_mix
        self.nuisance_clip_q = nuisance_clip_q

    def empirical_residual(self, diff, shape):
        if shape[0] != 2 * diff.shape[0] or tuple(shape[1:]) != tuple(diff.shape[1:]):
            raise ValueError("Expected duplicated A/B feature batches")
        detached = diff.detach()
        values = detached.abs().permute(1, 0, 2, 3).flatten(1)
        cap = torch.quantile(values.float(), self.nuisance_clip_q, dim=1)
        cap = cap.to(detached).view(1, -1, 1, 1)
        clipped = torch.maximum(torch.minimum(detached, cap), -cap)
        return torch.cat((-clipped, clipped), dim=0)

    def gen_struct_base_noise(self, stat_feat, shape, bq):
        synthetic = super().gen_struct_base_noise(stat_feat, shape, bq)
        if self.empirical_mix == 0:
            return synthetic
        empirical = self.empirical_residual(stat_feat, shape)
        dims = (0, 2, 3)
        empirical_rms = empirical.square().mean(dims, keepdim=True).sqrt()
        target_rms = synthetic.square().mean(dims, keepdim=True).sqrt()
        empirical = empirical * target_rms / empirical_rms.clamp_min(1e-6)
        mixed = (math.sqrt(1 - self.empirical_mix) * synthetic
                 + math.sqrt(self.empirical_mix) * empirical)
        return torch.where(empirical_rms > 1e-6, mixed, synthetic)


class EmpiricalNuisancePT(SeparateLrnNoisePT):
    """Empirical nuisance training strategy."""

    def __init__(
        self, empirical_mix=0.5, nuisance_clip_q=0.8,
        donor_prob=0.8, alpha_min=0.5, alpha_max=0.85,
        box_mask_prob=0.3, spatial_blur_prob=0.8,
        spatial_blur_sigma=1.5, spatial_alpha=0.5, *args, **kwargs,
    ):
        self.changen_class = EmpiricalNuisanceGenerator
        self.other_changen_kwargs = dict(
            empirical_mix=empirical_mix, nuisance_clip_q=nuisance_clip_q,
            donor_prob=donor_prob, alpha_min=alpha_min, alpha_max=alpha_max,
            box_mask_prob=box_mask_prob, spatial_blur_prob=spatial_blur_prob,
            spatial_blur_sigma=spatial_blur_sigma, spatial_alpha=spatial_alpha,
        )
        super().__init__(*args, **kwargs)
