# Reproducing SURE-CD

## Verification status

This guide follows the current source and YAMLs. No clean environment installation, training, or evaluation was run for this documentation update. The default local Python is 3.13.9 without training dependencies; it is not the historical experiment environment. No historical environment lock or `exp_log/summary.md` is present in this checkout. Hardware requirements and exact historical package versions remain unverified.

The main README metrics match existing local test JSON artifacts. Matching a saved result is not a fresh reproduction. A full public release still needs original backbone model ID/revision, data preparation and split provenance, downloadable task weights, and a validated environment export.

## 1. Install dependencies

Suggested initial environment, pending clean-run validation:

```bash
conda create -n sure_cd python=3.10
conda activate sure_cd
```

Install a matching PyTorch/torchvision pair from the [official selector](https://pytorch.org/get-started/locally/) for the driver and platform. Then, from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, torchvision, transformers, lightning; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'transformers:', transformers.__version__, 'lightning:', lightning.__version__); print('CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
python -c "from transformers import DINOv3ViTModel; from configs.config_parser import get_parser; get_parser().parse_args(['-c','configs/sure_cd_clcd.yaml']); print('Imports and config parsing passed')"
```

The final command parses a config without constructing a backbone or launching training; it does not validate weights/data. [DINOv3 upstream notes](https://github.com/facebookresearch/dinov3) specify Transformers support starting at 4.56.0. The manifest stays within 4.x pending upgrade validation. The original NumPy 1.26.4 constraint is retained; Python 3.10/3.11 are initial setup targets. Do not use that NumPy pin with Python 3.13.

The manifest now explicitly includes pandas, attrs, PyYAML, Pillow, tqdm, OpenCV, torchmetrics, Hugging Face Hub, and safetensors. `jsonargparse[signatures]` supports typed configuration parsing. Matplotlib and wandb are imported by current entrypoints even with visualization/logging disabled. `datasets` is only needed by the optional Hugging Face data backend. `timm` and `yacs` were removed because current project source does not import them. The CUDA-specific extra index was removed so installations can select their appropriate torch wheels separately.

After a successful real run, record the source revision, Python, OS, GPU, driver, CUDA build, and working package versions:

```bash
python -m pip freeze > environment-validated.txt
```

Review this generated export for private URLs and local paths before publishing. It must come from the validated environment, not the unrelated default interpreter.

## 2. Prepare the fixed data splits

`data.data_path` is the parent directory of `LEVIR`, `SYSU`, and `CLCD_256`. Each dataset must contain `train`, `val`, and `test` directories with `A`, `B`, and `label` subdirectories. For example:

```text
datasets/LEVIR/train/A/0001.png
datasets/LEVIR/train/B/0001.png
datasets/LEVIR/train/label/0001.png
```

- Default on-demand loading discovers `A/*.png`; B and label files must have identical names.
- Images are RGB. Masks must be single-channel, encoded 0 unchanged / 255 changed. The transform divides by 255: 0/1 input masks would produce incorrect metrics.
- Training uses resize to 256, paired flips/rotation, ImageNet normalization and tensor conversion. Test transforms use resize, normalization and tensor conversion only.
- The code does not generate the paper's cropped patch splits from full-resolution data. Preserve the original crop size, overlap, file lists, and split assignment; document them before release.
- The datamodule constructs train and test datasets even in evaluation-only mode. Supply the full split structure. If validation is missing, code falls back to training data for validation; provide the real validation split for reproduction.
- The loader reads labels during training, but synthesis strategies do not use target annotations for the training loss. Validation/test metrics do use labels. Do not replace real labels with arbitrary masks.
- Keep `data.use_hf=False` for the supplied YAMLs. The optional backend recognizes lowercase aliases and requires separate protocol verification; changing the flag alone is insufficient.

The local loader does not download datasets. Public download links, licenses, exact preprocessing instructions, and prepared split manifests remain release prerequisites.

## 3. Prepare both backbone and task checkpoints

The YAMLs use `encoder.init_args.model_name: weights/dinov3`. Provide a complete Hugging Face model directory with `config.json` and `model.safetensors` (or every shard and its index). The local config describes DINOv3 ViT-L/16: 24 layers, hidden size 1024, patch size 16, four register tokens. The wrapper extracts feature indices `[7, 11, 15, 23]`.

The exact original model ID/revision cannot be established from this config/model card alone. Different pretraining variants can share this architecture. Recover the original source before claiming exact reproduction, and download under the provider's terms.

`HFBackbone` omits frozen DINOv3 weights from task state dictionaries. A task checkpoint alone is insufficient.

| Config | Data folder | Default task file |
| --- | --- | --- |
| `configs/sure_cd_levir.yaml` | `LEVIR` | `checkpoints/sure_cd/LEVIR/42/weights.ckpt` |
| `configs/sure_cd_sysu.yaml` | `SYSU` | `checkpoints/sure_cd/SYSU/42/weights.ckpt` |
| `configs/sure_cd_clcd.yaml` | `CLCD_256` | `checkpoints/sure_cd/CLCD/42/weights.ckpt` |

All three files exist locally and are non-empty, approximately 138 MB each in decimal units. They are ignored by Git and do not yet have public download links. Use trusted task files: current `.ckpt` loading uses `torch.load(..., weights_only=False)`.

Resolution order is `--ckpt_path`, then `train.ckpt`, then `<train.ckpt_dir>/<seed>/weights.ckpt`. The current loader does **not** automatically prefer `best_weights.ckpt`, despite an older mechanism-YAML comment. Pass the intended file explicitly if needed.

## 4. Select the GPU

Standalone evaluation uses logical CUDA device 0. To use physical GPU 1 on a shared machine, expose only that GPU before invoking Python. Keep `devices: [0]` in main configs; that is the visible logical index.

Linux:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = "1"
```

Run the commands below in the same shell. CPU fallback exists but its execution and memory requirements have not been validated. No minimum VRAM figure is claimed. Do not silently alter batch size, precision, or the backbone for paper reproduction.

## 5. Evaluate

```bash
python test_eval.py -c configs/sure_cd_levir.yaml --data.data_path ./datasets --threshold 0.5
python test_eval.py -c configs/sure_cd_sysu.yaml --data.data_path ./datasets --threshold 0.5
python test_eval.py -c configs/sure_cd_clcd.yaml --data.data_path ./datasets --threshold 0.5
```

To evaluate a separately trained checkpoint:

```bash
python test_eval.py -c configs/sure_cd_clcd.yaml --data.data_path ./datasets --ckpt_path ./checkpoints/reproduction/CLCD/42/weights.ckpt --res_path ./results/reproduction_eval --threshold 0.5
```

Main-method outputs go to `<res_path>/42/<config_tag>/<dataset>/`. Use a new result root to preserve earlier files; repeated runs overwrite the named artifacts.

| Artifact | Meaning |
| --- | --- |
| `res.csv` | Raw test metrics in [0, 1] |
| `test_results.json` | Percentage metrics, checkpoint path, seed, raw results, timing, timestamp |
| `threshold_sweep.csv` | Diagnostic metrics at 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 |
| `config.yaml` | Framework's dumped configuration |

Main results use `pred > 0.5` with labels thresholded after normalization. Precision/Recall/F1/cIoU are global binary change-class metrics; OA is overall accuracy. Do not select thresholds using test labels or replace the fixed 0.5 result with the sweep maximum.

`test_eval.py` unconditionally sets `HF_ENDPOINT=https://hf-mirror.com` at import time. Local paths avoid remote model fetching. If fetching remote assets, review that behavior; setting a different endpoint outside the script alone will not override it.

## 6. Train task modules

| Setting | LEVIR | SYSU | CLCD_256 |
| --- | --- | --- | --- |
| Seed | 42 | 42 | 42 |
| Input size | 256 | 256 | 256 |
| Batch size | 4 | 4 | 4 |
| Optimizer steps | 3500 | 3500 | 1500 |
| Validation interval, steps | 200 | 200 | 100 |
| Data workers | 4 | 4 | 0 |
| Base learning rate | 1e-5 | 1e-5 | 1e-5 |
| Base weight decay | 1e-4 | 1e-4 | 1e-4 |

AdamW uses per-module parameter groups, so base rate/decay need not equal every group's settings. The YAMLs specify cosine scheduling, Dice plus cross-entropy, frozen backbone, and SR/EG synthesis parameters. Keep them unchanged for reproduction. The shipped main decoder has `use_auxfcn: False`; check manuscript claims against the executable configuration.

Use fresh output paths:

```bash
python train.py -c configs/sure_cd_levir.yaml --data.data_path ./datasets --train.ckpt_dir ./checkpoints/reproduction/LEVIR --res_path ./results/reproduction
python train.py -c configs/sure_cd_sysu.yaml --data.data_path ./datasets --train.ckpt_dir ./checkpoints/reproduction/SYSU --res_path ./results/reproduction
python train.py -c configs/sure_cd_clcd.yaml --data.data_path ./datasets --train.ckpt_dir ./checkpoints/reproduction/CLCD --res_path ./results/reproduction
```

Run only the intended dataset, retaining seed 42. Do not reuse occupied output paths. The final explicit save writes `<train.ckpt_dir>/42/weights.ckpt`. The automatic test afterward evaluates the final in-memory model, not an automatically selected best-validation model. Use section 5 for standalone checkpoint evaluation.

The supplied `wandb_proj: None` is a string that disables online logging under the current entrypoint comparison. `--dev` runs a short development check; it is not a paper result. Fixed seeds do not guarantee bitwise agreement across software and hardware.

## 7. Component and mechanism comparisons

`configs/ablation/` contains five CLCD_256 component variants. `configs/mechanism/` contains Gaussian nuisance, aligned empirical residual, and stable residual configurations for three datasets. Optional independent examples:

```bash
python test_eval.py -c configs/ablation/abl1_baseline.yaml --data.data_path ./datasets --threshold 0.5
python test_eval.py -c configs/mechanism/stable_residual_sysu.yaml --data.data_path ./datasets --threshold 0.5
```

Each needs its own task weights at the selected YAML's `train.ckpt_dir/42/weights.ckpt`, or an explicit compatible `--ckpt_path`. Full-method weights must not be substituted into a different variant. These examples do not launch any experiment automatically.

## 8. Remaining public-release evidence

Supply original dataset links/preprocessing/split manifests; backbone model ID/revision; public task-weight links and checksums; validated environment and hardware records; and the source release commit/tag associated with result artifacts. Verify independent-test evaluation at 0.5. Add finalized paper metadata. Retain upstream MIT notices and the Apache-2.0 notice in the adapted `models/modules/upernet.py`; review applicable license-text/notice distribution requirements before publication.
