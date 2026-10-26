from attr import dataclass
from jsonargparse import ArgumentParser, ActionConfigFile

from models.modules.base import (
    BaseCDEncoder,
    BaseCDDiff,
    BaseCDDecoder,
)
from models.train_strats.base import BaseStrategy


def get_parser():
    parser = ArgumentParser("SURE-CD")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to a configuration file in json or yaml format.",
    )

    add_arguments(parser)
    add_framework_arguments(parser)

    return parser


@dataclass
class SchedulerArgs:
    type: str
    gamma: float = None
    ratio_t0: float = None
    warmup_epoch_ratio: float = None
    warmup_lr_ratio: float = None


@dataclass
class TrainArgs:
    epochs: int
    val_freq: int
    base_lr: float
    weight_decay: float
    lr_scheduler: SchedulerArgs
    loss: list[str]
    transforms: list[dict]
    steps: int = -1
    use: bool = True
    ckpt: str | None = None
    ckpt_dir: str | None = None
    grad_clip_val: float = None
    strategy: BaseStrategy = None


@dataclass
class DataArgs:
    dataset: str
    img_size: int
    num_workers: int
    batch_size: int
    pin_memory: bool
    bands: str = "rgb"
    data_path: str = "./datasets"
    use_hf: bool = True
    load_in_mem: str | None = None


def add_arguments(parser: ArgumentParser):
    """Add CLI arguments."""
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    parser.add_argument("--save_ckpt", action="store_true", help="Save checkpoints")
    parser.add_argument("--devices", default="auto", help="Devices")
    parser.add_argument("--tag", type=str, default="", help="Experiment tag")
    parser.add_argument("--config_tag", type=str, help="Config tag")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path or HuggingFace model ID",
    )
    parser.add_argument(
        "--res_path", type=str, default="./results", help="Results save directory"
    )
    parser.add_argument("--dev", help="Fast dev run mode", action="store_true")
    parser.add_argument(
        "--wandb_proj",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--vis_path",
        default=None,
        help="Path to save prediction mask visualizations",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )


def add_framework_arguments(parser: ArgumentParser):
    parser.add_subclass_arguments(
        BaseCDEncoder,
        "encoder",
        required=True,
        skip={"config", "logger", "pretraining"},
    )
    parser.add_subclass_arguments(
        BaseCDDiff, "diff", required=True, skip={"dims", "resolutions", "pretraining"}
    )
    parser.add_subclass_arguments(
        BaseCDDecoder,
        "decoder.train",
        required=True,
        skip={"out_size", "input_sizes", "encoder_strides", "pretraining", "enc"},
    )
    parser.add_argument("--data", type=DataArgs)
    parser.add_argument("--train", type=TrainArgs)
