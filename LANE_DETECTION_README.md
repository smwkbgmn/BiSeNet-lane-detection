# BiSeNet for Lane Detection

Complete training pipeline for lane detection using BiSeNetV2 semantic segmentation.

## Overview

This implementation provides:
- **Dataset Conversion**: Convert lane coordinate annotations to segmentation masks
- **Custom Dataset Loader**: PyTorch dataset with augmentations for lane segmentation
- **Training Script**: Full training pipeline with checkpointing and logging
- **Diagnostics**: Comprehensive model evaluation and visualization tools

## Setup

### Requirements

```bash
pip install torch torchvision opencv-python numpy albumentations tqdm matplotlib wandb
```

### Directory Structure

Your dataset should be organized as:
```
dataset/
├── images/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── annotations/
    ├── frame_0001.json
    ├── frame_0002.json
    └── ...
```

JSON annotation format:
```json
{
  "image": "frame_0001.jpg",
  "lanes": [
    [[x1, y1], [x2, y2], ...],  // Lane 1
    [[x1, y1], [x2, y2], ...],  // Lane 2
    ...
  ]
}
```

## Quick Start

### Step 1: Convert Dataset to Segmentation Masks

Convert lane coordinates to binary/multi-class segmentation masks:

```bash
# Binary segmentation (lane vs background)
python convert_lanes_to_masks.py \
    --input_dir ../lane-detection-dl/dataset_augmented \
    --output_dir ./datasets/lane_binary \
    --binary \
    --thickness 10

# Multi-class segmentation (4 lanes + background)
python convert_lanes_to_masks.py \
    --input_dir ../lane-detection-dl/dataset_augmented \
    --output_dir ./datasets/lane_multiclass \
    --thickness 10
```

**Parameters:**
- `--input_dir`: Directory containing images/ and annotations/
- `--output_dir`: Where to save the masks/
- `--binary`: Use binary segmentation (all lanes = class 1)
- `--thickness`: Thickness of lane lines in pixels

**Output:**
```
output_dir/
├── masks/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
└── visualizations/  # Sample overlays for inspection
    └── viz_frame_0001.png
```

### Step 2: Train the Model

```bash
# Binary segmentation training
python train_lane_detection.py \
    --train_dir ./datasets/lane_binary \
    --val_dir ./datasets/lane_val \
    --n_classes 2 \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.005 \
    --image_size 512 1024 \
    --output_dir ./outputs/lane_binary \
    --use_ohem \
    --use_wandb \
    --wandb_project lane-detection

# Multi-class training (4 lanes + background = 5 classes)
python train_lane_detection.py \
    --train_dir ./datasets/lane_multiclass \
    --val_dir ./datasets/lane_val \
    --n_classes 5 \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.005 \
    --image_size 512 1024 \
    --output_dir ./outputs/lane_multiclass \
    --use_ohem
```

**Key Parameters:**
- `--train_dir`: Training dataset directory
- `--val_dir`: Validation dataset directory (optional)
- `--n_classes`: 2 for binary, 5 for 4 lanes + background
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (reduce if OOM)
- `--lr`: Learning rate
- `--image_size`: Input image size (H W)
- `--use_ohem`: Use Online Hard Example Mining loss
- `--use_wandb`: Enable Weights & Biases logging
- `--resume`: Resume from checkpoint

### Step 3: Evaluate and Diagnose

Run comprehensive diagnostics on your trained model:

```bash
python diagnose_model.py \
    --checkpoint ./outputs/lane_binary/checkpoints/best_model.pth \
    --data_dir ./datasets/lane_val \
    --output_dir ./diagnostics/lane_binary \
    --n_classes 2 \
    --image_size 512 1024 \
    --visualize \
    --num_viz 50 \
    --error_analysis
```

**Outputs:**
- `metrics.json`: Overall accuracy, mIoU, per-class metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `visualizations/`: Side-by-side predictions vs ground truth
- `error_analysis.json`: Best/worst performing samples

## Training Options

### Using Pre-converted Masks

If you've already converted your dataset:

```bash
python train_lane_detection.py \
    --train_dir ./datasets/lane_binary \
    --n_classes 2 \
    --epochs 100
```

### On-the-Fly Mask Generation

Generate masks during training (slower but saves disk space):

```bash
python train_lane_detection.py \
    --train_dir ../lane-detection-dl/dataset_augmented \
    --use_json \
    --n_classes 2 \
    --epochs 100
```

## Advanced Usage

### Custom Augmentations

Edit `lib/data/lane_dataset.py` to customize augmentations:

```python
def _get_default_transforms(self):
    return A.Compose([
        A.Resize(height=self.image_size[0], width=self.image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        # Add your custom augmentations here
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

### Learning Rate Scheduling

The trainer uses CosineAnnealingLR by default. Modify in `train_lane_detection.py`:

```python
self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
)
```

### Checkpoint Management

Checkpoints are saved in `output_dir/checkpoints/`:
- `best_model.pth`: Best model based on validation mIoU
- `checkpoint_epoch_N.pth`: Periodic checkpoints
- `final_model.pth`: Final model after training

Resume training:
```bash
python train_lane_detection.py \
    --train_dir ./datasets/lane_binary \
    --resume ./outputs/lane_binary/checkpoints/checkpoint_epoch_50.pth
```

## Model Architecture

BiSeNetV2 architecture:
- **Detail Branch**: Captures low-level spatial details
- **Semantic Branch**: Captures high-level semantic context
- **Bilateral Guided Aggregation**: Fuses both branches
- **Auxiliary Heads**: Deep supervision during training

## Metrics

Evaluation metrics:
- **Accuracy**: Pixel-wise classification accuracy
- **mIoU**: Mean Intersection over Union across all classes
- **Per-Class IoU**: IoU for each lane class
- **Precision/Recall**: Per-class precision and recall

## Tips for Better Performance

1. **Data Quality**: Ensure lane annotations are accurate
2. **Thickness**: Adjust `--thickness` based on your image resolution
3. **Image Size**: Larger images (e.g., 768x1536) for better accuracy
4. **Batch Size**: Reduce if you encounter OOM errors
5. **Learning Rate**: Start with 0.005, adjust based on loss curves
6. **OHEM Loss**: Use `--use_ohem` for handling class imbalance
7. **Augmentations**: Strong augmentations help generalization

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size and/or image size
python train_lane_detection.py ... --batch_size 4 --image_size 384 768
```

### Poor Performance
1. Check data quality with visualizations from conversion script
2. Verify class distribution in metrics
3. Increase lane thickness if lanes are too thin
4. Add more augmentations for robustness

### No GPU
```bash
# Training will automatically use CPU if CUDA is unavailable
# Expect significantly slower training
```

## Example Workflow

Complete workflow for a new dataset:

```bash
# 1. Convert dataset
python convert_lanes_to_masks.py \
    --input_dir ../lane-detection-dl/dataset_augmented \
    --output_dir ./datasets/lanes \
    --binary \
    --thickness 12

# 2. Check visualizations
# Inspect: datasets/lanes/visualizations/

# 3. Split into train/val (manually or using script)
# Expected structure:
#   datasets/lanes_train/
#   datasets/lanes_val/

# 4. Train model
python train_lane_detection.py \
    --train_dir ./datasets/lanes_train \
    --val_dir ./datasets/lanes_val \
    --n_classes 2 \
    --epochs 100 \
    --batch_size 8 \
    --output_dir ./outputs/experiment1 \
    --use_ohem \
    --use_wandb

# 5. Monitor training
# Check W&B dashboard or outputs/experiment1/

# 6. Evaluate best model
python diagnose_model.py \
    --checkpoint ./outputs/experiment1/checkpoints/best_model.pth \
    --data_dir ./datasets/lanes_val \
    --output_dir ./diagnostics/experiment1 \
    --n_classes 2

# 7. Review diagnostics
# Check: diagnostics/experiment1/metrics.json
#        diagnostics/experiment1/visualizations/
```

## Files Overview

- `convert_lanes_to_masks.py`: Dataset conversion script
- `lib/data/lane_dataset.py`: Custom PyTorch dataset loaders
- `train_lane_detection.py`: Main training script
- `diagnose_model.py`: Evaluation and visualization tools
- `lib/models/bisenetv2.py`: BiSeNetV2 model architecture (existing)
- `lib/ohem_ce_loss.py`: OHEM loss implementation (existing)

## Citation

If using BiSeNet, please cite:
```
@inproceedings{yu2021bisenet,
  title={Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation},
  author={Yu, Changqian and Gao, Changxin and Wang, Jingbo and Yu, Gang and Shen, Chunhua and Sang, Nong},
  booktitle={IJCV},
  year={2021}
}
```
