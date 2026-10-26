# SURE-CD: Stable-Residual Synthesis and Competing Evidence Gating for Unsupervised Remote Sensing Change Detection

PyTorch implementation of **SURE-CD** for bi-temporal remote sensing change detection. A frozen pretrained DINOv3 ViT-L/16 encoder supplies features; task modules learn from synthetic supervision without target change annotations in the training loss. Validation and test metrics use target annotations.

## Method

- **Stable-Residual Resampling (SR)** selects the lowest-energy 50% residual vectors per layer from the training-batch pool, centers them per channel, and resamples full channel vectors with independent spatial indices and random signs. This reduces spatial correspondence with original changes; it does not guarantee a contamination-free pool or exact covariance preservation after scaling.
- **Competing Evidence Gating (EG)** compares shared-nuisance and sparse-edit hypotheses for bounded residual modulation. The four-layer gate adds eight learnable variance scalars.

## Release status

Source code and experiment configurations are present. Public task-weight links, exact original backbone repository/revision, dataset preparation provenance, and a validated environment lock remain to be provided. Files under `weights/` and `checkpoints/` are excluded from Git and will not accompany a source-only clone.

These instructions were checked against the source. A clean installation, training, and evaluation were **not run** during this documentation update. See [the reproducibility guide](docs/REPRODUCIBILITY.md) for details and known limitations.

## Installation

Start with a separate Python **3.10 or 3.11** environment. This is a suggested setup target, not a verified record of the historical experiment environment. Install a matching PyTorch/torchvision pair for your platform and driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` is a direct-dependency manifest, not an exact lock file. [DINOv3 release notes](https://github.com/facebookresearch/dinov3) specify Transformers support starting at 4.56.0. The manifest restricts Transformers to 4.x pending compatibility validation.

## Data preparation

The shipped YAMLs use local data and a machine-specific root. Override that root with `--data.data_path ./datasets` or your absolute data root. Each dataset follows this structure:

```text
datasets/
├── LEVIR/
├── SYSU/
└── CLCD_256/
    ├── train/
    │   ├── A/       # Time 1 RGB PNGs
    │   ├── B/       # Time 2 RGB PNGs, identical filenames
    │   └── label/   # Single-channel masks: 0 unchanged, 255 changed
    ├── val/         # Same A/B/label structure
    └── test/        # Same A/B/label structure
```

All three datasets use that split layout. Preserve exact train/val/test splits. The loader needs label files even during training, although target labels are not used for the synthetic training loss. Do not substitute arbitrary masks. The code resizes supplied images but does not reproduce patch extraction from full-resolution datasets; prepared 256-pixel patch splits and their provenance must be supplied separately.

## Backbone and task checkpoints

Prepare the original experiment's **Hugging Face-format DINOv3 ViT-L/16** directory:

```text
weights/dinov3/
├── config.json
└── model.safetensors
```

For sharded weights, include every shard and the index. Obtain the model through its provider under its license. The local config describes 24 layers, hidden size 1024, patch size 16, and four register tokens. Architecture alone does not identify the original pretraining variant: the exact repository/revision still needs documenting. Do not substitute another variant for paper reproduction.

Task checkpoints omit the frozen backbone weights; both assets are necessary. Default task paths are `checkpoints/sure_cd/{LEVIR,SYSU,CLCD}/42/weights.ckpt`. These files exist locally but public download links have not been provided.

## Evaluation

After preparing data, backbone, and task weights:

```bash
python test_eval.py -c configs/sure_cd_levir.yaml --data.data_path ./datasets --threshold 0.5
python test_eval.py -c configs/sure_cd_sysu.yaml --data.data_path ./datasets --threshold 0.5
python test_eval.py -c configs/sure_cd_clcd.yaml --data.data_path ./datasets --threshold 0.5
```

Use `--ckpt_path /path/to/weights.ckpt` for a specific compatible task checkpoint. See the guide for selecting a physical GPU. Main-config outputs are saved to `results/42/<config_tag>/<dataset>/`: raw metrics in `res.csv`, percentage metrics and provenance in `test_results.json`, and diagnostic results in `threshold_sweep.csv`.

## Existing reference results

The following values match existing local test JSON artifacts dated 2026-10-22. They were **not recomputed** during this update. Precision, Recall, F1 and IoU measure **change class 1** on the test split at fixed probability threshold **0.5**. OA is overall pixel accuracy. All values below are percentages.

| Dataset | Precision | Recall | F1 | Class 1 IoU | OA |
| --- | ---: | ---: | ---: | ---: | ---: |
| LEVIR-CD | 28.75 | 65.29 | 39.92 | 24.94 | 89.99 |
| SYSU-CD | 66.93 | 68.02 | 67.47 | 50.91 | 84.53 |
| CLCD_256 | 55.41 | 61.46 | 58.28 | 41.12 | 93.45 |

Report the fixed-threshold result. Do not select a threshold by maximizing scores on the test labels.

## Training

Training starts from a pretrained frozen encoder, not a randomly initialized full network. It evaluates the final in-memory model on the test split after training. Use fresh checkpoint/result directories to preserve existing artifacts:

```bash
python train.py -c configs/sure_cd_levir.yaml --data.data_path ./datasets --train.ckpt_dir ./checkpoints/reproduction/LEVIR --res_path ./results/reproduction
python train.py -c configs/sure_cd_sysu.yaml --data.data_path ./datasets --train.ckpt_dir ./checkpoints/reproduction/SYSU --res_path ./results/reproduction
python train.py -c configs/sure_cd_clcd.yaml --data.data_path ./datasets --train.ckpt_dir ./checkpoints/reproduction/CLCD --res_path ./results/reproduction
```

These are independent experiments; run only the intended dataset. Seed remains 42. LEVIR/SYSU use 3,500 optimizer steps and CLCD uses 1,500, with batch size 4. The final checkpoint is `<train.ckpt_dir>/42/weights.ckpt`. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for protocol settings and component/mechanism comparisons.

## Repository structure

```text
configs/               Main, component-ablation, and mechanism configurations
data/                  Paired-image loaders and transforms
models/                Framework, backbone, synthesis, gating, decoder
train.py               Training followed by evaluation; evaluation-only option
test_eval.py           Standalone test evaluation and threshold diagnostics
requirements.txt       Direct dependencies (not a validated lock file)
docs/                  Reproduction and publication guides
weights/               External backbone (ignored by Git)
checkpoints/           Task weights (ignored by Git)
results/               Generated artifacts (ignored by Git)
```

## License, acknowledgment, and citation

This code builds on [MaSoN](https://github.com/blaz-r/mason_cd) and uses [DINOv3](https://github.com/facebookresearch/dinov3). Retain upstream notices and cite the original methods. The existing [MIT LICENSE](LICENSE) retains the upstream copyright notice. Backbone weights and datasets have separate terms; the code license does not grant rights to those assets.

Final SURE-CD author information, publication link, and BibTeX will be added when available. No publication metadata is fabricated here. For GitHub setup, see [docs/GITHUB_PUBLISH.md](docs/GITHUB_PUBLISH.md).
