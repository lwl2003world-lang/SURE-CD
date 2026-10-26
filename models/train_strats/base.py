from typing import Any

from torch import nn

from models.modules import BaseCDDecoder


class BaseStrategy(nn.Module):
    """Base training strategy."""

    def __init__(
        self,
        config: dict[str, Any] = None,
        enc: nn.Module = None,
        diff: nn.Module = None,
        dec: BaseCDDecoder = None,
    ):
        super().__init__()
        self.config = config
        object.__setattr__(self, "enc", enc)
        object.__setattr__(self, "diff", diff)
        object.__setattr__(self, "dec", dec)

    @staticmethod
    def build_loss(loss_list):
        raise NotImplementedError

    def __call__(self, batch, loss_only=True):
        raise NotImplementedError
