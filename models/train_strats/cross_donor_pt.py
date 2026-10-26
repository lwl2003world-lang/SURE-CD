import math
import einops
import torch
import torch.nn.functional as F
from torch import Tensor

from models.common.perlin import rand_perlin_2d
from models.train_strats.noise_pt import (
    ChangeStatSource,
    NoiseType,
)
from models.train_strats.separate_lq_soft_noise_pt import (
    Q_DIM,
    LearnableSeparateChangeGenerator,
    SeparateLrnNoisePT,
    soft_quantile,
)


def spatial_gaussian_blur(tensor: Tensor, kernel_size: int = 5, sigma: float = 1.5) -> Tensor:
    """2D separable Gaussian smoothing."""
    b, c, h, w = tensor.shape
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, dtype=tensor.dtype, device=tensor.device)
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()

    k1d_h = k1d.view(1, 1, kernel_size, 1).repeat(c, 1, 1, 1)
    k1d_w = k1d.view(1, 1, 1, kernel_size).repeat(c, 1, 1, 1)

    padded_h = F.pad(tensor, (0, 0, radius, radius), mode="reflect")
    blurred_h = F.conv2d(padded_h, k1d_h, groups=c)

    padded_w = F.pad(blurred_h, (radius, radius, 0, 0), mode="reflect")
    blurred = F.conv2d(padded_w, k1d_w, groups=c)

    std = blurred.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    mean = blurred.mean(dim=(-2, -1), keepdim=True)
    return (blurred - mean) / std


class CrossDonorChangeGenerator(LearnableSeparateChangeGenerator):
    """Cross-layer donor change generator."""

    def __init__(
        self,
        donor_prob: float = 0.8,
        alpha_min: float = 0.75,
        alpha_max: float = 1.0,
        box_mask_prob: float = 0.3,
        spatial_blur_prob: float = 0.8,
        spatial_blur_sigma: float = 1.5,
        spatial_alpha: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for name, value in (
            ("donor_prob", donor_prob),
            ("alpha_min", alpha_min),
            ("alpha_max", alpha_max),
            ("box_mask_prob", box_mask_prob),
            ("spatial_blur_prob", spatial_blur_prob),
            ("spatial_alpha", spatial_alpha),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if alpha_min > alpha_max:
            raise ValueError(f"alpha_min ({alpha_min}) cannot exceed alpha_max ({alpha_max})")

        self.donor_prob = donor_prob
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.box_mask_prob = box_mask_prob
        self.spatial_blur_prob = spatial_blur_prob
        self.spatial_blur_sigma = spatial_blur_sigma
        self.spatial_alpha = spatial_alpha

    def generate_masks(self, batches: int, h: int, w: int, device: torch.device) -> Tensor:
        masks = []
        for _ in range(batches):
            chance_change = torch.rand(1).item()
            if chance_change > 0.5:
                masks.append(torch.zeros(1, 1, h, w, device=device))
                continue

            if self.box_mask_prob > 0 and torch.rand(1).item() < self.box_mask_prob:
                m = torch.zeros(1, 1, h, w, device=device)
                num_boxes = torch.randint(1, 4, (1,)).item()
                for _ in range(num_boxes):
                    bh = torch.randint(max(4, int(h * 0.1)), max(5, int(h * 0.45)), (1,)).item()
                    bw = torch.randint(max(4, int(w * 0.1)), max(5, int(w * 0.45)), (1,)).item()
                    top = torch.randint(0, max(1, h - bh), (1,)).item()
                    left = torch.randint(0, max(1, w - bw), (1,)).item()
                    m[:, :, top:top + bh, left:left + bw] = 1.0
                masks.append(m)
            else:
                perlin_scalex = 2 ** torch.randint(self.min_perlin_scale, self.max_perlin_scale, (1,)).item()
                perlin_scaley = 2 ** torch.randint(self.min_perlin_scale, self.max_perlin_scale, (1,)).item()
                p_noise = rand_perlin_2d((h, w), (perlin_scalex, perlin_scaley)).reshape(1, 1, h, w).to(device)
                masks.append(torch.where(p_noise > 0.5, 1.0, 0.0))

        return torch.cat(masks, dim=0)

    def gen_struct_base_noise(
        self,
        stat_feat: Tensor,
        shape: tuple[int, int, int, int],
        bq: Tensor | float,
    ) -> Tensor:
        b, c, h, w = shape
        sb = stat_feat.shape[0]
        stat_rearranged = einops.rearrange(stat_feat, self.reshape_rule)

        if stat_rearranged.dtype == torch.float16:
            stat_rearranged = stat_rearranged.to(torch.float32)

        q = torch.as_tensor(bq, device=stat_rearranged.device, dtype=stat_rearranged.dtype)
        offset = soft_quantile(stat_rearranged, q).abs()

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

        white_noise = torch.randn_like(offset) * offset

        if (
            self.training
            and h >= 4
            and w >= 4
            and self.spatial_blur_prob > 0
            and self.spatial_alpha > 0
            and torch.rand(1).item() < self.spatial_blur_prob
        ):
            unit_noise = torch.randn(b, c, h, w, device=offset.device, dtype=offset.dtype)
            smooth_noise = spatial_gaussian_blur(
                unit_noise, kernel_size=5, sigma=self.spatial_blur_sigma
            ) * offset
            alpha = self.spatial_alpha
            return math.sqrt(alpha) * smooth_noise + math.sqrt(1.0 - alpha) * white_noise

        return white_noise

    def forward(
        self, feat_dict: dict[str, Tensor], mask: Tensor, cq: Tensor, bq: Tensor
    ) -> tuple[dict[str, Tensor], Tensor]:
        feats = list(feat_dict.values())
        b, _, h, w = mask.shape
        device = feats[0].device

        # Change mask
        noise_mask = self.generate_masks(b, h, w, device)
        target_mask = mask + noise_mask
        target_mask = torch.where(target_mask > 0, 1.0, 0.0)

        # Donor sampling
        if b > 1:
            shift = torch.randint(1, b, (1,), device=device).item()
            donor_idx = (torch.arange(b, device=device) + shift) % b
        else:
            donor_idx = torch.arange(b, device=device)

        alpha = torch.rand(b, 1, 1, 1, device=device) * (self.alpha_max - self.alpha_min) + self.alpha_min

        if self.donor_prob >= 1.0:
            use_donor = torch.ones(b, 1, 1, 1, dtype=torch.bool, device=device)
        elif self.donor_prob <= 0.0:
            use_donor = torch.zeros(b, 1, 1, 1, dtype=torch.bool, device=device)
        else:
            use_donor = torch.rand(b, 1, 1, 1, device=device) < self.donor_prob

        noised_feat = {}
        for feat_key, feat_val in feat_dict.items():
            fh, fw = feat_val.shape[2:]
            noise_mask_scaled = F.interpolate(
                noise_mask,
                size=(fh, fw),
                mode="nearest",
            )

            f1, f2 = feat_val.chunk(2, dim=0)
            f1_real, f2_real = f1.chunk(2, dim=0)
            diff = f1_real - f2_real

            # Nuisance perturbation
            if self.base_noise_type != NoiseType.NONE:
                if self.base_noise_type == NoiseType.NORMAL:
                    base_noise = self.gen_struct_base_noise(diff, f2.shape, bq)
                else:
                    base_noise = self.gen_noise(self.base_noise_type, diff, f2.shape, bq)
            else:
                base_noise = None

            # Donor perturbation
            f2_donor = f2[donor_idx]
            donor_diff = (f2_donor - f2) * alpha

            if self.change_stat_source == ChangeStatSource.FEAT:
                ch_stat_feat = f2
            elif self.change_stat_source == ChangeStatSource.DIFF:
                ch_stat_feat = diff
            else:
                raise ValueError(f"Unknown change_stat_source {self.change_stat_source}")

            calib_change_noise = self.gen_noise(
                self.change_noise_type, ch_stat_feat, f2.shape, cq
            )

            effective_change = torch.where(use_donor, donor_diff, calib_change_noise)

            # Combine change and base noise
            change_residual = effective_change * noise_mask_scaled

            if base_noise is not None:
                base_weight = 1.0 if self.overlap_base else (1.0 - noise_mask_scaled)
                residual = change_residual + base_noise * base_weight
            else:
                residual = change_residual

            f2_perturbed = f2 + residual
            perturbed = torch.cat((f1, f2_perturbed), dim=0)
            noised_feat[feat_key] = perturbed

        return noised_feat, target_mask


class CrossDonorPT(SeparateLrnNoisePT):
    """Cross-layer donor perturbation training strategy."""

    def __init__(
        self,
        q_dim: Q_DIM,
        change_noise_q: float,
        base_noise_q: float,
        donor_prob: float = 0.8,
        alpha_min: float = 0.75,
        alpha_max: float = 1.0,
        box_mask_prob: float = 0.3,
        spatial_blur_prob: float = 0.8,
        spatial_blur_sigma: float = 1.5,
        spatial_alpha: float = 0.5,
        *args,
        **kwargs,
    ):
        self.changen_class = CrossDonorChangeGenerator
        self.other_changen_kwargs = {
            "donor_prob": donor_prob,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "box_mask_prob": box_mask_prob,
            "spatial_blur_prob": spatial_blur_prob,
            "spatial_blur_sigma": spatial_blur_sigma,
            "spatial_alpha": spatial_alpha,
        }
        super().__init__(
            q_dim=q_dim,
            change_noise_q=change_noise_q,
            base_noise_q=base_noise_q,
            *args,
            **kwargs,
        )
