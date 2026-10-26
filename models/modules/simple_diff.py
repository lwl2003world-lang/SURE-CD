from models.modules.base import BaseCDDiff


class SubDiff(BaseCDDiff):
    """Feature difference module by subtraction."""

    name = "sub"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feat_dict):
        if self.pretraining and not self.use_in_pretrain:
            return feat_dict

        out_dict = {}
        for i, feat in feat_dict.items():
            f1, f2 = feat.chunk(2, dim=0)
            out_dict[i] = f1 - f2

        return out_dict

    def get_parameters(self, pretraining=False):
        return []
