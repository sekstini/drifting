# Drifting (Minimal)

## 1) Setup
```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision rich
```

## 2) Quick test (FashionMNIST)
```bash
uv run python drifting.py --dataset fashionmnist --data-root ./data --max-steps 200
```

## 3) Download a small ImageNet-style dataset (Imagenette)
```bash
./download_imagenette.sh
```

## 4) Train on Imagenette
```bash
uv run python drifting.py --dataset imagenet --data-root ./data/imagenette2-160 --image-size 64 --batch-size 32 --num-workers 0 --feature-backbone resnet18 --max-steps 2000
```

Outputs are saved under `./outputs/`.
