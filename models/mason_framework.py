import copy
import importlib
from typing import Any

import torch
import yaml
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection

from models.framework import Framework
from models.train_strats.base import BaseStrategy


def get_strategy_cls(config) -> tuple[BaseStrategy, dict]:
    split = config.train.strategy.class_path.split(".")
    module_path, class_name = ".".join(split[:-1]), split[-1]

    cls: BaseStrategy = getattr(importlib.import_module(module_path), class_name)
    init_args = config.train.strategy.get("init_args", {})
    init_args = {k: v for k, v in init_args.items() if v is not None}
    return cls, init_args


class MasonFramework(Framework):
    """Lightning framework module for change detection."""
    pretraining = False

    def __init__(
        self, train: bool, config_namespace, metrics: MetricCollection, *args, **kwargs
    ):
        super().__init__(config_namespace=config_namespace, *args, **kwargs)

        if train:
            assert (
                config_namespace.train.strategy is not None
            ), "Strategy required"
            strategy_class, strat_args = get_strategy_cls(config=config_namespace)
            self.strategy_class: BaseStrategy = strategy_class

            self.strategy = self.strategy_class(
                config=self.config,
                enc=self.enc,
                diff=self.diff,
                dec=self.dec,
                **strat_args,
            )

        self.val_metrics = metrics
        self.test_metrics = copy.deepcopy(metrics)

        self.val_metrics.prefix = "val/"
        self.test_metrics.prefix = "test/"

    def dump_config(self):
        with (self.res_path / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

    def forward(self, batch):
        i1, i2 = batch["imageA"], batch["imageB"]
        x = torch.cat([i1, i2], dim=0)

        x = self.enc(x)
        x = self.diff(x)
        x = self.dec(x)

        return x

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs

        loss, loss_terms = self.strategy(batch)

        if loss != loss:
            loss = None
            print(f"Train - NaN encountered in batch {batch.get('img_idx')}")
        else:
            self.log("loss", loss, prog_bar=True, logger=True)
            self.log_dict(loss_terms, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs

        x = self.forward(batch)

        return torch.sigmoid(x)

    def test_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        preds = self.predict_step(batch, *args, **kwargs)
        if isinstance(preds, dict):
            batch = {**batch, **preds}
        else:
            batch["pred"] = preds

        self.test_metrics.update(
            batch["pred"] > self.threshold, torch.where(batch["label"] > 0.5, 1, 0)
        )
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=True)

        return batch

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        try:
            preds = self.predict_step(batch, *args, **kwargs)
            if isinstance(preds, dict):
                batch = {**batch, **preds}
            else:
                batch["pred"] = preds

            results = batch

            self.val_metrics.update(
                batch["pred"] > self.threshold, torch.where(batch["label"] > 0.5, 1, 0)
            )
            self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        except Exception as e:
            print(f"Validation step error: {e}")
            results = None

        return results


SUREFramework = MasonFramework
