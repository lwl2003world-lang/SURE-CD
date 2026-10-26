from pathlib import Path

import lightning as L
from huggingface_hub import PyTorchModelHubMixin
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
    LinearLR,
    PolynomialLR,
    ExponentialLR,
)
from torchmetrics import MetricCollection

from models.modules import (
    build_encoder,
    build_diff,
    build_decoder,
)


class Framework(
    L.LightningModule,
    PyTorchModelHubMixin,
    repo_url="https://github.com/blaz-r/mason_cd",
    paper_url="https://arxiv.org/abs/2602.19881",
    docs_url="https://github.com/blaz-r/mason_cd",
):
    pretraining = None

    def __init__(
        self,
        config_namespace,
        config,
    ):
        super().__init__()
        self.config = config_namespace
        self.metrics: MetricCollection
        self.threshold = self.config.threshold

        self.enc = build_encoder(config_namespace, pretraining=self.pretraining)
        dims = self.enc.get_out_dims()
        strides = self.enc.get_strides()
        resolutions = [int(self.config.data.img_size / s) for s in strides]

        self.diff = build_diff(
            config_namespace,
            dims=dims,
            resolutions=resolutions,
            pretraining=self.pretraining,
            enc=self.enc,
        )
        dims = self.diff.adjust_dims(dims)

        self.dec = build_decoder(
            config_namespace,
            input_sizes=dims,
            encoder_strides=strides,
            pretraining=self.pretraining,
            enc=self.enc,
        )

        self.res_path = None
        self.exp_name = None

        self.save_hyperparameters(ignore=["metrics", "module_list", "config_namespace"])

    def generate_and_verify_res_path(self, save_path):
        if self.res_path is None:
            config = self.config
            if "noise" in config.train.strategy.class_path:
                bt = config.train.strategy.init_args.base_noise_type
                bq = config.train.strategy.init_args.base_noise_q
                ct = config.train.strategy.init_args.change_noise_type
                cq = config.train.strategy.init_args.change_noise_q
                csrc = config.train.strategy.init_args.change_stat_source

                res_run_name = f"{bt}_{bq}_{ct}_{cq}_{csrc}"
            else:
                res_run_name = ""

            res_path = (
                Path(save_path) / config.config_tag / res_run_name / config.data.dataset
            )
            res_path.mkdir(parents=True, exist_ok=True)
            self.res_path = res_path

        return self.res_path

    def generate_exp_name(self):
        """Return experiment name."""
        config = self.config
        if "noise" in config.train.strategy.class_path:
            bt = config.train.strategy.init_args.base_noise_type
            bq = config.train.strategy.init_args.base_noise_q
            ct = config.train.strategy.init_args.change_noise_type
            cq = config.train.strategy.init_args.change_noise_q
            csrc = config.train.strategy.init_args.change_stat_source

            self.exp_name = (
                f"{bt}_{bq}_{ct}_{cq}_{csrc}_{config.data.dataset}_"
                + f"{config.config_tag}_{config.tag}"
            )
        else:
            self.exp_name = f"{config.data.dataset}_{config.config_tag}_{config.tag}"
        return self.exp_name

    def configure_optimizers(self):
        params = []
        missing = []
        for name in [
            "enc",
            "diff",
            "dec",
            "strategy",
        ]:
            module = getattr(self, name, None)
            if module:
                params += module.get_parameters(pretraining=self.pretraining)
            else:
                missing.append(name)

        print(f"Following modules not used: {missing}")

        config = self.config.train

        optimizer = AdamW(
            params,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )

        if config.steps > 0:
            steps = config.steps
            step_type = "step"
            assert (
                config.epochs == -1
            ), "Training steps provided but epochs is non negative number. Check correctness."
        else:
            steps = config.epochs
            step_type = "epoch"
            assert (
                config.steps == -1
            ), "Training epochs provided but steps is non negative number. Check correctness."

        sched_obj = self.get_scheduler(optimizer, config, steps)

        if sched_obj is None:
            return optimizer

        scheduler = {
            "scheduler": sched_obj,
            "interval": step_type,
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def get_scheduler(self, optimizer, config, steps):
        if config.lr_scheduler.type == "none":
            print("Lr scheduler not used")
            return None
        elif config.lr_scheduler.type == "multistep":
            assert config.lr_scheduler.gamma
            steps = [int(0.8 * steps), int(0.9 * steps)]
            print(f"Using multistep LR with steps{steps}")
            return MultiStepLR(
                optimizer,
                milestones=steps,
                gamma=config.lr_scheduler.gamma,
            )
        elif config.lr_scheduler.type == "cosine":
            assert config.lr_scheduler.ratio_t0
            t0 = int(config.lr_scheduler.ratio_t0 * steps)
            print(f"Using cosine scheduler with T0={t0}")
            factor = 1e-3
            min_lr = config.base_lr * factor
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t0,
                eta_min=min_lr,
            )
        elif config.lr_scheduler.type == "linear":
            factor = 1e-3
            return LinearLR(
                optimizer, start_factor=1, end_factor=factor, total_iters=steps
            )
        elif config.lr_scheduler.type == "poly":
            assert config.lr_scheduler.gamma
            return PolynomialLR(
                optimizer, power=config.lr_scheduler.gamma, total_iters=steps
            )
        elif config.lr_scheduler.type == "exp":
            assert config.lr_scheduler.gamma
            return ExponentialLR(optimizer, gamma=config.lr_scheduler.gamma)
        elif config.lr_scheduler.type == "cosine_warmup":
            assert config.lr_scheduler.ratio_t0
            assert config.lr_scheduler.warmup_epoch_ratio
            assert config.lr_scheduler.warmup_lr_ratio
            factor = config.lr_scheduler.warmup_lr_ratio
            min_lr = config.base_lr * factor
            warmup_epochs = int(config.lr_scheduler.warmup_epoch_ratio * steps)

            t0 = int(config.lr_scheduler.ratio_t0 * steps)
            print(
                f"Using cosine scheduler with T0={t0} and {warmup_epochs} epoch warmup"
            )
            t0 -= warmup_epochs

            warmup = LinearLR(
                optimizer, start_factor=factor, end_factor=1, total_iters=warmup_epochs
            )
            cosine_lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t0,
                eta_min=min_lr,
            )
            return SequentialLR(
                optimizer, [warmup, cosine_lr_scheduler], milestones=[warmup_epochs]
            )
        elif config.lr_scheduler.type == "multistep_warmup":
            assert config.lr_scheduler.gamma
            assert config.lr_scheduler.warmup_epoch_ratio
            assert config.lr_scheduler.warmup_lr_ratio
            steps = [int(0.8 * steps), int(0.9 * steps)]
            factor = config.lr_scheduler.warmup_lr_ratio
            warmup_epochs = int(config.lr_scheduler.warmup_epoch_ratio * steps)
            print(
                f"Using multistep LR with steps {steps} and {warmup_epochs} epoch warmup"
            )
            steps = [s - warmup_epochs for s in steps]

            warmup = LinearLR(
                optimizer, start_factor=factor, end_factor=1, total_iters=warmup_epochs
            )
            multistep = MultiStepLR(
                optimizer,
                milestones=steps,
                gamma=config.lr_scheduler.gamma,
            )
            return SequentialLR(
                optimizer, [warmup, multistep], milestones=[warmup_epochs]
            )
        else:
            raise ValueError(f"Unknown lr scheduler {config.lr_scheduler.type}")
