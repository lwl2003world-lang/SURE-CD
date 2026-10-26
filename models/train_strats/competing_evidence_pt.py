from models.train_strats.stable_residual_pt import StableResidualPT


class CompetingEvidencePT(StableResidualPT):
    """Training strategy with competing evidence auxiliary loss."""

    def __init__(self, evidence_loss_weight=0.1, calibration_weight=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidence_loss_weight = evidence_loss_weight
        self.calibration_weight = calibration_weight

    def __call__(self, batch, loss_only=True):
        image = batch["imageA"]
        target = image.new_zeros((image.shape[0], 1, *image.shape[-2:]))
        prediction, target = self.forward(batch, target)
        segmentation, terms = self.loss(prediction, target)
        evidence, calibration, diagnostics = self.diff.evidence_loss(target, self.loss)
        total = segmentation + self.evidence_loss_weight * evidence + self.calibration_weight * calibration
        terms.update(diagnostics)
        terms["prediction_area"] = (prediction.detach().sigmoid() > 0.5).float().mean()
        return (total, terms) if loss_only else (prediction, (total, terms))


SURETrainingStrategy = CompetingEvidencePT
