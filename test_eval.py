import json
import os
from datetime import datetime
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import TQDMProgressBar
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
from train import checkpoint_directory, load_weights


class ThresholdSweepCallback(TQDMProgressBar):
    """Callback to evaluate metrics across multiple thresholds."""

    def __init__(self, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thresholds = thresholds
        self.metrics_by_thr = {
            t: MetricCollection({
                "Precision": BinaryPrecision(),
                "Recall": BinaryRecall(),
                "F1": BinaryF1Score(),
                "cIoU": BinaryJaccardIndex(),
                "OA": BinaryAccuracy(),
            }) for t in thresholds
        }

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        preds = outputs.get("pred", batch.get("pred"))
        labels = torch.where(batch["label"] > 0.5, 1, 0)
        for t, m in self.metrics_by_thr.items():
            m.to(preds.device)
            m.update(preds > t, labels)

    def get_results(self):
        res = {}
        for t, m in self.metrics_by_thr.items():
            res[t] = {k: round(v.compute().item() * 100.0, 2) for k, v in m.items()}
        return res


def run_evaluation(config):
    seed_everything(config.seed)
    has_cuda = torch.cuda.is_available()
    resolved_checkpoint = config.ckpt_path or config.train.ckpt
    if resolved_checkpoint is None and getattr(config.train, "ckpt_dir", None):
        resolved_checkpoint = checkpoint_directory(config) / "weights.ckpt"

    print("=" * 50)
    print(" SURE-CD Evaluation")
    print(f" Dataset: {config.data.dataset} | Seed: {config.seed}")
    print(f" Checkpoint: {resolved_checkpoint}")
    print(f" Device: {torch.cuda.get_device_name(0) if has_cuda else 'CPU'}")
    print("=" * 50)

    datamodule = CDDataModule(
        config,
        data_path=config.data.data_path,
        use_hf=config.data.use_hf,
        load_in_mem=config.data.load_in_mem,
    )

    metrics = MetricCollection(
        {
            "F1": BinaryF1Score(),
            "Recall": BinaryRecall(),
            "Precision": BinaryPrecision(),
            "cIoU": BinaryJaccardIndex(),
            "OA": BinaryAccuracy(),
        }
    )

    framework = SUREFramework(
        train=False,
        config_namespace=config,
        config=config.as_dict(),
        metrics=metrics,
    )

    res_path = framework.generate_and_verify_res_path(
        Path(config.res_path) / str(config.seed)
    )
    framework.dump_config()

    framework = load_weights(config, framework, res_path)

    sweep_cb = ThresholdSweepCallback(thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8), refresh_rate=1)
    callbacks = [sweep_cb]
    if getattr(config, "vis_path", False):
        visualizer = Visualizer(
            res_path=framework.res_path,
            add_vis_keys=[],
            threshold=0.5,
            criterion=None,
            criterion_limit=2,
        )
        callbacks.append(visualizer)

    test_trainer = Trainer(
        logger=False,
        accelerator="auto",
        devices=[0] if has_cuda else "auto",
        fast_dev_run=config.dev,
        callbacks=callbacks,
    )

    start_time = datetime.now()
    results = test_trainer.test(
        model=framework,
        datamodule=datamodule,
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    sweep_results = sweep_cb.get_results()

    if results and len(results) > 0:
        res_dict = results[0]
        f1 = res_dict.get("test/F1", 0.0) * 100.0
        prec = res_dict.get("test/Precision", 0.0) * 100.0
        rec = res_dict.get("test/Recall", 0.0) * 100.0
        iou = res_dict.get("test/cIoU", 0.0) * 100.0
        oa = res_dict.get("test/OA", 0.0) * 100.0

        output_summary = {
            "dataset": config.data.dataset,
            "checkpoint": str(resolved_checkpoint),
            "seed": config.seed,
            "metrics": {
                "Precision": round(prec, 2),
                "Recall": round(rec, 2),
                "F1": round(f1, 2),
                "Class1_IoU": round(iou, 2),
                "OA": round(oa, 2),
            },
            "threshold_sweep": sweep_results,
            "raw_results": res_dict,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }

        csv_file = framework.res_path / "res.csv"
        json_file = framework.res_path / "test_results.json"
        pd.DataFrame([res_dict]).to_csv(csv_file, index=False)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(output_summary, f, indent=2, ensure_ascii=False)

        sweep_df = pd.DataFrame.from_dict(sweep_results, orient="index")
        sweep_df.index.name = "Threshold"
        sweep_df.to_csv(framework.res_path / "threshold_sweep.csv")

        print("\n" + "=" * 60)
        print("  Evaluation Results (Change Class)")
        print("=" * 60)
        print(f"Dataset   : {config.data.dataset}")
        print(f"Checkpoint: {resolved_checkpoint}")
        print(f"Results saved to: {json_file}")
        print("-" * 60)
        print("| Precision (%) | Recall (%) | F1-Score (%) | Class 1 IoU (%) | OA (%) |")
        print("| :---: | :---: | :---: | :---: | :---: |")
        print(f"| {prec:.2f}% | {rec:.2f}% | {f1:.2f}% | {iou:.2f}% | {oa:.2f}% |")
        print("=" * 60 + "\n")

        return output_summary
    return None


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args()
    config.eval_only = True

    import faulthandler
    faulthandler.enable()

    run_evaluation(config)
