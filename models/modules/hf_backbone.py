import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoBackbone,
)

from models.modules.base import BaseCDEncoder
from models.modules.dinov3 import Dinov3Backbone


class HFBackbone(BaseCDEncoder):
    """HuggingFace and Vision Transformer backbone wrapper."""
    name = "HFBB"

    def __init__(
        self,
        model_name: str,
        return_layers: list[int],
        pretraining: bool,
        train_args: dict | None = None,
        lr_adjustments: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            return_layers=return_layers, pretraining=pretraining, *args, **kwargs
        )
        self.model_name = model_name
        self.train_args = train_args

        self.finetune: bool
        self.pretrained: bool
        self.backbone: nn.Module | None = None
        self.patch_size: int

        self.lr_adjustments = lr_adjustments
        self.drop_path_rate: bool

        self.prepare_for_phase(train_args)
        self.register_state_dict_post_hook(
            lambda *args: self._drop_pretrained_dinov3_weights(*args)
        )
        self.register_load_state_dict_post_hook(self._allow_missing_pretrained_dinov3)

        if self.pretrained:
            print(f"Using pretrained backbone from {model_name}")
            self.name += "PT"

    def prepare_for_phase(self, arg_dict) -> None:
        """Configure backbone fine-tuning or frozen state."""
        self.finetune = arg_dict["finetune"]
        self.pretrained = arg_dict["pretrained"]
        self.drop_path_rate = arg_dict.get("drop_path_rate", None)

        if self.backbone is None:
            self.init_backbone()

        if not self.finetune:
            self.eval()
            self.freeze()
        else:
            self.train()

    def init_backbone(self):
        self.n_channels = 3

        model_name = self.model_name
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        out_list = [f"stage{i}" for i in self.return_layers]

        if "ImageClassification" in hf_config.architectures[0]:
            hf_config.out_features = out_list
            if self.drop_path_rate is not None:
                hf_config.drop_path_rate = self.drop_path_rate

            if self.pretrained:
                self.backbone = AutoBackbone.from_pretrained(
                    model_name, config=hf_config
                )
            else:
                self.backbone = AutoBackbone.from_config(config=hf_config)

            if "patch_size" in hf_config:
                self.patch_size = hf_config.patch_size
            else:
                self.patch_size = None
        elif "Dinov2" in hf_config.architectures[0]:
            hf_config.out_features = out_list
            if self.drop_path_rate is not None:
                hf_config.drop_path_rate = self.drop_path_rate

            self.backbone = AutoBackbone.from_pretrained(model_name, config=hf_config)
            self.patch_size = hf_config.patch_size
        elif "DINOv3" in hf_config.architectures[0]:
            if self.drop_path_rate is not None:
                hf_config.drop_path_rate = self.drop_path_rate
            self.backbone = Dinov3Backbone(
                model_name, hf_config, self.return_layers, self.pretrained
            )
        else:
            raise ValueError(f"HFBackbone doesn't support {model_name} models")

        if not self.pretrained and hasattr(self.backbone, "init_weights"):
            self.backbone.init_weights()

    def freeze(self):
        print("Freezing backbone")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _skip_pretrained_dinov3_weights(self):
        return (
            isinstance(self.backbone, Dinov3Backbone)
            and self.pretrained
            and not self.finetune
        )

    def _drop_pretrained_dinov3_weights(self, _, state_dict, prefix, __):
        if self._skip_pretrained_dinov3_weights():
            for key in list(state_dict):
                if key.startswith(prefix + "backbone.model."):
                    state_dict.pop(key)

    def _allow_missing_pretrained_dinov3(self, _, incompatible_keys):
        if self._skip_pretrained_dinov3_weights():
            incompatible_keys.missing_keys[:] = [
                k for k in incompatible_keys.missing_keys if "backbone.model." not in k
            ]

    def forward(self, x, mask=None, hidden3=False):
        if not self.finetune:
            self.backbone.eval()

        if mask is None:
            feat = self.backbone(x, output_hidden_states=hidden3)
        else:
            if len(mask.size()) == 3:
                mask = mask.flatten(1)
            feat = self.backbone(x, mask, output_hidden_states=hidden3)

        rd = {k: v for k, v in zip(self.return_layers, feat.feature_maps)}

        if hidden3:
            rd["h3"] = feat.hidden_states[3]
        return rd

    def get_parameters(self, pretraining: bool) -> list[dict]:
        del pretraining
        if self.finetune:
            self.lr = self.train_args.get("lr", self.lr)
            self.weight_decay = self.train_args.get("weight_decay", self.weight_decay)
            base_lr_dict = self.get_lr_dict()

            if hasattr(self.backbone, "lr_tunable") and self.lr_adjustments:
                print("Adjusting LR for backbone specifics")
                return self.backbone.get_parameters(base_lr_dict)
            else:
                base_params = self.backbone.parameters()
                return [{"params": base_params, **base_lr_dict}]
        else:
            return []

    @torch.no_grad()
    def get_strides(self) -> list[int]:
        was_train = self.backbone.training
        self.backbone.eval()
        feat = self.backbone(torch.rand(1, self.n_channels, 224, 224))
        if was_train:
            self.backbone.train()
        return [256 / f.shape[2] for f in feat.feature_maps]

    def get_out_dims(self) -> list[int]:
        return self.backbone.channels

    def transfer_from_pretrained(self, state_dict: dict):
        if "backbone.embeddings.mask_token" in state_dict:
            state_dict.pop("backbone.embeddings.mask_token")
        super().transfer_from_pretrained(state_dict)
        self.prepare_for_phase(self.train_args)


def build_hfbackbone(config, pretraining):
    return HFBackbone(**config.encoder.init_args.as_dict(), pretraining=pretraining)
