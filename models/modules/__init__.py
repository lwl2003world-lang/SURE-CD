from models.modules.base import BaseCDDecoder, BaseCDDiff
from models.modules.hf_backbone import HFBackbone, build_hfbackbone

import importlib


def build_encoder(config, pretraining):
    model_type = config.encoder.class_path.split(".")[-1]
    if model_type == HFBackbone.__name__:
        return build_hfbackbone(config, pretraining=pretraining)
    raise ValueError(f"Model type {model_type} not supported")


def build_diff(config, dims, resolutions, pretraining, enc):
    split = config.diff.class_path.split(".")
    module_path, class_name = ".".join(split[:-1]), split[-1]

    diff: BaseCDDiff = getattr(importlib.import_module(module_path), class_name)
    diff_args = config.diff.get("init_args", {})
    return diff(
        config=config,
        enc=enc,
        dims=dims,
        resolutions=resolutions,
        pretraining=pretraining,
        **diff_args,
    )


def build_decoder(config, input_sizes, encoder_strides, pretraining, enc):
    subconf = config.decoder.train
    split = subconf.class_path.split(".")
    module_path, class_name = ".".join(split[:-1]), split[-1]

    decoder = getattr(importlib.import_module(module_path), class_name)

    return decoder(
        out_size=config.data.img_size,
        input_sizes=input_sizes,
        encoder_strides=encoder_strides,
        pretraining=pretraining,
        enc=enc,
        **subconf.init_args.as_dict(),
    )
