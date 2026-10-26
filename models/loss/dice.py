"""Dice loss functions."""


def dice_loss_smooth(inputs, targets):
    """Compute smooth Dice loss."""
    smooth = 1
    inputs = inputs.sigmoid().flatten()
    targets = targets.flatten()
    numerator = 2 * (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()
    return 1 - (numerator + smooth) / (denominator + smooth)


def dice_loss_eps(inputs, targets, naive=True, eps=1e-3):
    """Compute Dice loss with epsilon smoothing."""
    inputs = inputs.sigmoid().flatten()
    targets = targets.flatten()
    numerator = 2 * (inputs * targets).sum()
    if naive:
        denominator = inputs.sum() + targets.sum()
    else:
        denominator = (inputs * inputs).sum() + (targets * targets).sum()
    return 1 - numerator / (denominator + eps)
