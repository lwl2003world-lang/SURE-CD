import math

import torch
from torch import nn
import torch.nn.functional as F

from models.modules.base import BaseCDDiff


class CompetingEvidenceDiff(BaseCDDiff):
    """Competing Evidence Gating (EG) module."""

    def __init__(self, dims, stable_fraction=0.5, edit_prior=0.25,
                 evidence_strength=0.25, evidence_lr=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < stable_fraction <= 1 or not 0 < edit_prior < 0.5:
            raise ValueError("Invalid stable fraction or sparse-edit prior")
        if not 0 <= evidence_strength <= 0.5:
            raise ValueError("Keep a bounded residual-preserving evidence gate")
        self.stable_fraction = stable_fraction
        self.log_prior_odds = math.log(edit_prior / (1 - edit_prior))
        self.evidence_strength = evidence_strength
        self.evidence_lr = evidence_lr

        # Learnable variance parameters per layer
        self.nuisance_raw = nn.Parameter(torch.full((len(dims),), math.log(0.5)))
        self.edit_raw = nn.Parameter(torch.full((len(dims),), math.log(2.75 / 13)))
        self.last_evidence = None
        self.last_energies = None

    def variances(self):
        v0 = 0.5 + 1.5 * self.nuisance_raw.sigmoid()
        ve = 0.25 + 15.75 * self.edit_raw.sigmoid()
        return v0, v0 + ve

    def normalized_energy(self, a, b):
        residual = a - b
        with torch.no_grad():
            tokens = residual.detach().flatten(2)
            rms = tokens.square().mean(2, keepdim=True).sqrt().clamp_min(1e-6)
            ranks = (tokens / rms).square().mean(1)
            count = max(1, int(tokens.shape[2] * self.stable_fraction))
            index = ranks.argsort(1)[:, :count]
            pool = tokens.gather(2, index[:, None].expand(-1, tokens.shape[1], -1))
            scale = pool.square().mean(2, keepdim=True).sqrt()
            feature_rms = ((a.detach().square() + b.detach().square()) / 2).flatten(2).mean(2, keepdim=True).sqrt()
            scale = torch.maximum(scale, 0.01 * feature_rms).clamp_min(1e-6)
        energy = (residual / scale.unsqueeze(-1)).square().mean(1, keepdim=True)
        return residual, energy.clamp_max(1e4)

    def score(self, energy, index):
        v0, v1 = self.variances()
        fit_gain = 0.5 * energy * (v0[index].reciprocal() - v1[index].reciprocal())
        complexity = 0.5 * torch.log(v1[index] / v0[index])
        return fit_gain - complexity + self.log_prior_odds

    def forward(self, feat_dict):
        out, evidence, energies = {}, [], []
        for index, (key, feature) in enumerate(feat_dict.items()):
            a, b = feature.chunk(2, dim=0)
            residual, energy = self.normalized_energy(a, b)
            score = self.score(energy, index)
            out[key] = residual * (1 + self.evidence_strength * score.tanh())
            evidence.append(score)
            energies.append(energy)
        self.last_evidence, self.last_energies = evidence, energies
        return out

    def evidence_loss(self, target, loss_fn):
        scores = [F.interpolate(x, target.shape[-2:], mode="bilinear", align_corners=False)
                  for x in self.last_evidence]
        score = torch.stack(scores).mean(0)
        evidence_loss, _ = loss_fn(score, target)
        v0, v1 = self.variances()
        calibration = score.new_zeros(())
        for index, energy in enumerate(self.last_energies):
            y = F.interpolate(target.float(), energy.shape[-2:], mode="area")
            for support, variance in ((1-y, v0[index]), (y, v1[index])):
                nll = 0.5 * (energy / variance + variance.log())
                calibration = calibration + (nll * support).sum() / support.sum().clamp_min(1)
        calibration = calibration / (2 * len(self.last_energies))
        diagnostics = {
            "evidence_bce_dice": evidence_loss.detach(),
            "hypothesis_nll": calibration.detach(),
            "h1_usage": (score > 0).float().mean().detach(),
            "synthetic_area": target.float().mean().detach(),
            "nuisance_variance": v0.mean().detach(),
            "edit_variance": (v1-v0).mean().detach(),
            "evidence_on_edit": ((score * target).sum() / target.sum().clamp_min(1)).detach(),
            "evidence_on_nuisance": ((score * (1-target)).sum() / (1-target).sum().clamp_min(1)).detach(),
        }
        return evidence_loss, calibration, diagnostics

    def get_parameters(self, pretraining=False):
        return [{"params": [self.nuisance_raw, self.edit_raw], "lr": self.evidence_lr}]


CompetingEvidenceGate = CompetingEvidenceDiff
