# SURE-CD

PyTorch implementation of **Stable-Residual Synthesis and Competing Evidence Gating for Unsupervised Remote Sensing Change Detection**.

## Installation

```bash
conda create -n sure_cd python=3.10
conda activate sure_cd
pip install -r requirements.txt
```

## Data preparation

Organize LEVIR-CD, SYSU-CD, and CLCD into the following structure:

```text
datasets/
├── LEVIR/
├── SYSU/
└── CLCD_256/
    ├── train/
    │   ├── A/
    │   ├── B/
    │   └── label/
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/
```

Each dataset follows this layout. Use matching PNG filenames for paired RGB images and single-channel labels (0: unchanged, 255: changed).

Place the Hugging Face-format DINOv3 ViT-L/16 backbone (`config.json` and `model.safetensors`) in `weights/dinov3/`.

## Pretrained checkpoints

Download the SURE-CD task checkpoints for LEVIR-CD, SYSU-CD, and CLCD from [Google Drive](https://drive.google.com/file/d/1cvxnfzW9yy8whDv84If30hfW4BMLBTqm/view?usp=sharing). The archive contains `LEVIR/weights.ckpt`, `SYSU/weights.ckpt`, and `CLCD/weights.ckpt`.

After extracting the archive, place each checkpoint at the corresponding path below (create the `42` directories if needed):

```text
checkpoints/sure_cd/LEVIR/42/weights.ckpt
checkpoints/sure_cd/SYSU/42/weights.ckpt
checkpoints/sure_cd/CLCD/42/weights.ckpt
```

The archive contains the dataset-specific SURE-CD task checkpoints. The DINOv3 ViT-L/16 backbone is a separate requirement; see [Data preparation](#data-preparation).

## Training

```bash
python train.py -c configs/sure_cd_levir.yaml --data.data_path ./datasets
python train.py -c configs/sure_cd_sysu.yaml --data.data_path ./datasets
python train.py -c configs/sure_cd_clcd.yaml --data.data_path ./datasets
```

## Evaluation

```bash
python test_eval.py -c configs/sure_cd_levir.yaml --data.data_path ./datasets
python test_eval.py -c configs/sure_cd_sysu.yaml --data.data_path ./datasets
python test_eval.py -c configs/sure_cd_clcd.yaml --data.data_path ./datasets
```

Task checkpoints are loaded from `checkpoints/sure_cd/{LEVIR,SYSU,CLCD}/42/weights.ckpt`. Use `--ckpt_path` to specify another checkpoint. The DINOv3 backbone is required for both training and evaluation. Results are saved under `results/`.

## Acknowledgments

This project builds on [MaSoN](https://github.com/blaz-r/mason_cd) and [DINOv3](https://github.com/facebookresearch/dinov3).

## License

[MIT](LICENSE). Third-party code retains its original license notices.
