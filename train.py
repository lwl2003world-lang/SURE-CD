from pathlib import Path

import pandas as pd
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from configs.config_parser import get_parser
from data.datamodule import CDDataModule
from models.callbacks.visualiser import Visualizer
from models.mason_framework import SUREFramework


def checkpoint_directory(config):
    """Return checkpoint directory for experiment and seed."""
    output_dir = getattr(config.train, "ckpt_dir", None)
    if output_dir:
        return Path(output_dir) / str(config.seed)
    return Path("checkpoints") / f"{config.config_tag}_{config.data.dataset}" / str(config.seed)


def load_weights(config, framework, res_path):
    ckpt_path = config.ckpt_path or config.train.ckpt
    if ckpt_path is None and getattr(config.train, "ckpt_dir", None):
        ckpt_dir = checkpoint_directory(config)
        ckpt_path = str(ckpt_dir / "weights.ckpt")
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path is None:
        raise ValueError("Set --ckpt_path or train.ckpt for evaluation.")
    ckpt = Path(ckpt_path)
    if ckpt.is_dir():
        print(f"Loading local HF checkpoint directory: {ckpt_path}")
        framework = framework.from_pretrained(
            str(ckpt),
            metrics=MetricCollection(
                {
                    "F1": BinaryF1Score(),
                    "Recall": BinaryRecall(),
                    "Precision": BinaryPrecision(),
                    "cIoU": BinaryJaccardIndex(),
                    "OA": BinaryAccuracy(),
                }
            ),
            config_namespace=config,
        )
        framework.res_path = res_path
    elif ckpt.is_file():
        print(f"Loading checkpoint file: {ckpt_path}")
        if ckpt.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(ckpt))
        else:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
        framework.load_state_dict(state.get("state_dict", state), strict=True)
    else:
        print(f"Loading HF checkpoint: {ckpt_path}")
        if config.data.dataset not in ckpt_path:
            raise ValueError(
                f"Checkpoint dataset mismatch: {config.data.dataset} not in {ckpt_path}"
            )
        framework = framework.from_pretrained(
            ckpt_path,
            metrics=MetricCollection(
                {
                    "F1": BinaryF1Score(),
                    "Recall": BinaryRecall(),
                    "Precision": BinaryPrecision(),
                    "cIoU": BinaryJaccardIndex(),
                    "OA": BinaryAccuracy(),
                }
            ),
            config_namespace=config,
        )
        framework.res_path = res_path
    return framework


def train(framework, datamodule, config, wandb_logger):
    has_cuda = torch.cuda.is_available()

    callbacks = [TQDMProgressBar(refresh_rate=1)]
    if config.save_ckpt:
        ckpt_dir = checkpoint_directory(config)
        ckpt_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="weights",
            save_weights_only=True,
        )
        callbacks.append(ckpt_callback)

    trainer = Trainer(
        max_steps=config.train.steps if config.train.steps > 0 else -1,
        max_epochs=config.train.epochs if config.train.epochs > 0 else None,
        val_check_interval=config.train.val_freq if config.train.steps > 0 else None,
        check_val_every_n_epoch=None
        if config.train.steps > 0
        else config.train.val_freq,
        logger=None if config.dev else wandb_logger,
        accelerator="auto",
        devices=config.devices if has_cuda else "auto",
        enable_checkpointing=config.save_ckpt,
        precision=None,
        fast_dev_run=config.dev,
        callbacks=callbacks,
        gradient_clip_val=config.train.grad_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(
        model=framework,
        datamodule=datamodule,
    )
    if config.save_ckpt and not config.dev and getattr(config.train, "ckpt_dir", None):
        trainer.save_checkpoint(
            checkpoint_directory(config) / "weights.ckpt", weights_only=True
        )


def eval(framework, datamodule, config, wandb_logger):
    has_cuda = torch.cuda.is_available()
    framework.dump_config()

    callbacks = [TQDMProgressBar(refresh_rate=1)]
    if config.vis_path is not None and config.vis_path:
        visualizer = Visualizer(
            res_path=framework.res_path,
            add_vis_keys=[],
            threshold=0.5,
            criterion=None,
            criterion_limit=2,
        )
        callbacks.append(visualizer)

    test_trainer = Trainer(
        logger=None if config.dev else wandb_logger,
        accelerator="auto",
        devices=[0] if has_cuda else "auto",
        fast_dev_run=config.dev,
        callbacks=callbacks,
    )
    results = test_trainer.test(
        model=framework,
        datamodule=datamodule,
    )
    pd.DataFrame(results).to_csv(framework.res_path / "res.csv", index=False)


def main(config, seed=None):
    if seed is not None:
        config.seed = seed

    seed_everything(config.seed)

    datamodule = CDDataModule(
        config,
        data_path=config.data.data_path,
        use_hf=config.data.use_hf,
        load_in_mem=config.data.load_in_mem,
    )

    framework = SUREFramework(
        train=not config.eval_only,
        config_namespace=config,
        config=config.as_dict(),
        metrics=MetricCollection(
            {
                "F1": BinaryF1Score(),
                "Recall": BinaryRecall(),
                "Precision": BinaryPrecision(),
                "cIoU": BinaryJaccardIndex(),
                "OA": BinaryAccuracy(),
            }
        ),
    )

    res_path = framework.generate_and_verify_res_path(
        Path(config.res_path) / str(config.seed)
    )
    framework.dump_config()

    run_name = framework.generate_exp_name()
    if config.wandb_proj != "None":
        wandb_logger = WandbLogger(
            project=config.wandb_proj, log_model=False, name=run_name
        )
    else:
        wandb_logger = None

    if config.eval_only:
        framework = load_weights(config, framework, res_path)
    else:
        print(f"Starting SURE-CD training for {run_name}")
        train(framework, datamodule, config, wandb_logger)

    print(f"Starting evaluation for {run_name}")
    eval(framework, datamodule, config, wandb_logger)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args()

    import faulthandler
    faulthandler.enable()
    main(config)
