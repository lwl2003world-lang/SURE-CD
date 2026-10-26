import importlib
from enum import Enum

from albumentations.pytorch import ToTensorV2
import albumentations as A


def build_transforms(config, test: bool, has_mask: bool = True):
    return DualTransform(config, test, has_mask)


class Target(str, Enum):
    I1 = "i1"
    I2 = "i2"
    BOTH_SAME = "both_same"
    BOTH_DIFF = "both_diff"


class DualTransform:
    """Bi-temporal image transformation wrapper."""

    def __init__(self, config, test: bool, has_mask: bool):
        self.img_size = (config.data.img_size, config.data.img_size)
        self.test = test
        self.has_mask = has_mask

        transforms = [(Target.BOTH_SAME, A.Resize(*self.img_size))]
        transforms.extend(self.build_from_config(config))

        assert isinstance(
            transforms[-1][1], A.Normalize
        ), "Normalize must be included in the transforms and on last position"
        assert (
            transforms[-1][0] == Target.BOTH_SAME
        ), "Normalize must be applied to both images"
        transforms.append((Target.BOTH_SAME, ToTensorV2()))

        assert (
            not test or len(transforms) == 3
        ), "Testing transforms should only contain resize, normalize and ToTensor"

        proc_str = "testing" if test else "training"
        print(f"Transforms used for {proc_str}:")
        for tr in transforms:
            print(tr)

        if self.has_mask:
            add = {"imageB": "image", "label": "mask"}
        else:
            add = {"imageB": "image"}

        if all([target == Target.BOTH_SAME for target, _ in transforms]):
            self.transforms = [
                (
                    Target.BOTH_SAME,
                    A.Compose(
                        [t_obj for _, t_obj in transforms],
                        additional_targets=add,
                        is_check_shapes=False,
                    ),
                )
            ]
        else:
            add_dict = {
                Target.BOTH_SAME: add,
                Target.BOTH_DIFF: {},
                Target.I1: {},
                Target.I2: {},
            }

            composed = []
            for target, t_obj in transforms:
                pair = (
                    target,
                    A.Compose(
                        t_obj,
                        additional_targets=add_dict[target],
                        is_check_shapes=False,
                    ),
                )
                composed.append(pair)

            self.transforms = composed

    def build_from_config(self, config) -> list:
        transforms = []
        transform_config = config.train.transforms

        if self.test:
            tr_config_list = [transform_config[-1]]
        else:
            tr_config_list = transform_config

        for tr_conf in tr_config_list:
            cls_name, init_args = next(iter(tr_conf.items()))
            if "size" in init_args:
                init_args["size"] = self.img_size

            target = Target(init_args["target"]) if "target" in init_args else Target.BOTH_SAME
            tr_cls = getattr(importlib.import_module("albumentations"), cls_name)
            t_obj = tr_cls(**init_args)
            transforms.append((target, t_obj))

        return transforms

    def transform(self, image, imageB, label=None) -> dict:
        if label is not None:
            out_dict = {"image": image, "imageB": imageB, "label": label}
        else:
            out_dict = {"image": image, "imageB": imageB}

        for target, t_obj in self.transforms:
            if target == Target.BOTH_SAME:
                out_dict = t_obj(**out_dict)
            elif target == Target.BOTH_DIFF:
                out_dict["image"] = t_obj(image=out_dict["image"])["image"]
                out_dict["imageB"] = t_obj(image=out_dict["imageB"])["image"]
            elif target == Target.I1:
                out_dict["image"] = t_obj(image=out_dict["image"])["image"]
            elif target == Target.I2:
                out_dict["imageB"] = t_obj(image=out_dict["imageB"])["image"]

        return out_dict

    def __call__(self, data):
        if self.has_mask:
            transformed = self.transform(
                image=data["imageA"], imageB=data["imageB"], label=data["label"]
            )
        else:
            transformed = self.transform(image=data["imageA"], imageB=data["imageB"])

        transformed_batch = {
            "imageA": transformed["image"],
            "imageB": transformed["imageB"],
        }
        if self.has_mask:
            transformed_batch["label"] = (transformed["label"] / 255).unsqueeze(0)

        return transformed_batch
