"""
BiSeNet Diagnostic and Evaluation Script
Comprehensive model evaluation, visualization, and debugging
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from models.bisenetv2 import BiSeNetV2
from data.lane_dataset import get_lane_dataloader


class ModelDiagnostics:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)

        # Load model
        print(f"Loading model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=self.device)

        # Auto-detect number of classes from checkpoint
        state_dict = checkpoint['model_state_dict']
        if 'head.conv_out.1.weight' in state_dict:
            detected_classes = state_dict['head.conv_out.1.weight'].shape[0]
            print(f"Auto-detected {detected_classes} classes from checkpoint")
            self.n_classes = detected_classes
        else:
            print(f"Using {args.n_classes} classes from command line")
            self.n_classes = args.n_classes

        self.model = BiSeNetV2(n_classes=self.n_classes, aux_mode='eval')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully (trained for {checkpoint.get('epoch', 'unknown')} epochs)")

        # Load dataset
        print(f"Loading dataset from {args.data_dir}")
        # Try with JSON annotations first (on-the-fly mask generation)
        self.dataloader = get_lane_dataloader(
            root_dir=args.data_dir,
            batch_size=1,
            num_workers=2,
            mode='val',
            image_size=tuple(args.image_size),
            n_classes=self.n_classes,
            use_json=True,
            binary_mode=False,
            lane_thickness=10
        )

        print(f"Loaded {len(self.dataloader.dataset)} samples")

        if len(self.dataloader.dataset) == 0:
            raise ValueError(
                f"No samples found in {args.data_dir}. "
                f"Please ensure the directory contains:\n"
                f"  - images/ subdirectory with image files\n"
                f"  - annotations/ subdirectory with JSON files (or masks/ with PNG masks)"
            )

    def calculate_iou(self, pred, target, num_classes):
        """Calculate IoU for each class"""
        ious = []
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        for cls in range(num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)

            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)

        return ious

    def calculate_metrics(self):
        """Calculate comprehensive metrics on the dataset"""
        print("\n" + "="*50)
        print("CALCULATING METRICS")
        print("="*50 + "\n")

        total_correct = 0
        total_pixels = 0
        class_iou_sum = np.zeros(self.n_classes)
        class_count = np.zeros(self.n_classes)
        class_pixel_count = np.zeros(self.n_classes)
        confusion_matrix = np.zeros((self.n_classes, self.n_classes))

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                logits = outputs[0]
                preds = logits.argmax(dim=1)

                # Calculate accuracy
                total_correct += (preds == masks).sum().item()
                total_pixels += masks.numel()

                # Calculate per-class metrics
                for cls in range(self.n_classes):
                    class_pixel_count[cls] += (masks == cls).sum().item()

                # Calculate IoU
                ious = self.calculate_iou(preds[0], masks[0], self.n_classes)
                for cls, iou in enumerate(ious):
                    if not np.isnan(iou):
                        class_iou_sum[cls] += iou
                        class_count[cls] += 1

                # Update confusion matrix
                for true_cls in range(self.n_classes):
                    for pred_cls in range(self.n_classes):
                        confusion_matrix[true_cls, pred_cls] += (
                            (masks == true_cls) & (preds == pred_cls)
                        ).sum().item()

        # Calculate final metrics
        accuracy = total_correct / total_pixels
        class_iou = class_iou_sum / (class_count + 1e-8)
        miou = np.nanmean(class_iou)

        # Calculate precision and recall per class
        precision = np.zeros(self.n_classes)
        recall = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            tp = confusion_matrix[cls, cls]
            fp = confusion_matrix[:, cls].sum() - tp
            fn = confusion_matrix[cls, :].sum() - tp

            precision[cls] = tp / (tp + fp + 1e-8)
            recall[cls] = tp / (tp + fn + 1e-8)

        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print(f"Mean IoU: {miou:.4f}\n")

        print("Per-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<10} {'IoU':<10} {'Precision':<12} {'Recall':<10} {'Pixel Count':<15}")
        print("-" * 80)
        for cls in range(self.n_classes):
            class_name = f"Class {cls}" if cls > 0 else "Background"
            print(f"{class_name:<10} {class_iou[cls]:<10.4f} {precision[cls]:<12.4f} "
                  f"{recall[cls]:<10.4f} {int(class_pixel_count[cls]):<15}")

        # Save metrics to JSON
        metrics = {
            'accuracy': float(accuracy),
            'miou': float(miou),
            'per_class': {
                f'class_{cls}': {
                    'iou': float(class_iou[cls]),
                    'precision': float(precision[cls]),
                    'recall': float(recall[cls]),
                    'pixel_count': int(class_pixel_count[cls])
                }
                for cls in range(self.n_classes)
            },
            'confusion_matrix': confusion_matrix.tolist()
        }

        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

        # Plot confusion matrix
        self.plot_confusion_matrix(confusion_matrix)

        return metrics

    def plot_confusion_matrix(self, confusion_matrix):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(self.n_classes)
        class_names = [f'Class {i}' if i > 0 else 'Background' for i in range(self.n_classes)]
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Add text annotations
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(int(confusion_matrix[i, j]), 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        save_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def visualize_predictions(self, num_samples=20):
        """Visualize model predictions"""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50 + "\n")

        colors = [
            [0, 0, 0],       # Background - black
            [255, 0, 0],     # Lane 1 - red
            [0, 255, 0],     # Lane 2 - green
            [0, 0, 255],     # Lane 3 - blue
            [255, 255, 0],   # Lane 4 - yellow
        ]

        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc='Visualizing')):
                if count >= num_samples:
                    break

                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                logits = outputs[0]
                preds = logits.argmax(dim=1)

                # Get image, ground truth, and prediction
                image = images[0].cpu().numpy().transpose(1, 2, 0)
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image * std + mean) * 255
                image = np.clip(image, 0, 255).astype(np.uint8)

                mask_gt = masks[0].cpu().numpy()
                mask_pred = preds[0].cpu().numpy()

                # Create colored masks
                colored_gt = np.zeros((*mask_gt.shape, 3), dtype=np.uint8)
                colored_pred = np.zeros((*mask_pred.shape, 3), dtype=np.uint8)

                for cls in range(min(self.n_classes, len(colors))):
                    colored_gt[mask_gt == cls] = colors[cls]
                    colored_pred[mask_pred == cls] = colors[cls]

                # Create overlays
                overlay_gt = cv2.addWeighted(image, 0.6, colored_gt, 0.4, 0)
                overlay_pred = cv2.addWeighted(image, 0.6, colored_pred, 0.4, 0)

                # Calculate IoU for this sample
                ious = self.calculate_iou(
                    torch.from_numpy(mask_pred),
                    torch.from_numpy(mask_gt),
                    self.n_classes
                )
                sample_miou = np.nanmean(ious)

                # Create comparison figure
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(colored_gt)
                axes[0, 1].set_title('Ground Truth Mask')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(cv2.cvtColor(overlay_gt, cv2.COLOR_BGR2RGB))
                axes[0, 2].set_title('Ground Truth Overlay')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title('Original Image')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(colored_pred)
                axes[1, 1].set_title(f'Prediction (mIoU: {sample_miou:.3f})')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(cv2.cvtColor(overlay_pred, cv2.COLOR_BGR2RGB))
                axes[1, 2].set_title('Prediction Overlay')
                axes[1, 2].axis('off')

                plt.suptitle(f'Sample {count + 1} - mIoU: {sample_miou:.4f}', fontsize=16)
                plt.tight_layout()

                save_path = self.viz_dir / f'prediction_{count:03d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

                count += 1

        print(f"\nGenerated {count} visualizations in {self.viz_dir}")

    def analyze_errors(self, num_samples=10):
        """Analyze worst-performing samples"""
        print("\n" + "="*50)
        print("ANALYZING ERRORS")
        print("="*50 + "\n")

        sample_ious = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Analyzing'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                outputs = self.model(images)
                logits = outputs[0]
                preds = logits.argmax(dim=1)

                ious = self.calculate_iou(preds[0], masks[0], self.n_classes)
                miou = np.nanmean(ious)

                sample_ious.append({
                    'miou': miou,
                    'image_path': batch['image_path'][0],
                    'mask_path': batch.get('mask_path', batch.get('annotation_path', ['N/A']))[0]
                })

        # Sort by mIoU (worst first)
        sample_ious.sort(key=lambda x: x['miou'])

        print(f"\nWorst {num_samples} samples:")
        print("-" * 80)
        for i, sample in enumerate(sample_ious[:num_samples]):
            print(f"{i+1}. mIoU: {sample['miou']:.4f} - {sample['image_path']}")

        print(f"\nBest {num_samples} samples:")
        print("-" * 80)
        for i, sample in enumerate(sample_ious[-num_samples:]):
            print(f"{i+1}. mIoU: {sample['miou']:.4f} - {sample['image_path']}")

        # Save error analysis
        error_analysis = {
            'worst_samples': sample_ious[:num_samples],
            'best_samples': sample_ious[-num_samples:],
            'mean_miou': float(np.mean([s['miou'] for s in sample_ious])),
            'std_miou': float(np.std([s['miou'] for s in sample_ious]))
        }

        error_path = self.output_dir / 'error_analysis.json'
        with open(error_path, 'w') as f:
            json.dump(error_analysis, f, indent=2)
        print(f"\nError analysis saved to {error_path}")

    def run_diagnostics(self):
        """Run all diagnostic tests"""
        print("\n" + "="*50)
        print("RUNNING BISENET DIAGNOSTICS")
        print("="*50)

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Visualize predictions
        if self.args.visualize:
            self.visualize_predictions(num_samples=self.args.num_viz)

        # Analyze errors
        if self.args.error_analysis:
            self.analyze_errors(num_samples=10)

        print("\n" + "="*50)
        print("DIAGNOSTICS COMPLETE")
        print("="*50)
        print(f"\nResults saved to {self.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose and evaluate BiSeNet model')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to evaluation dataset')
    parser.add_argument('--output_dir', type=str, default='./diagnostics',
                       help='Output directory for results')

    # Model config
    parser.add_argument('--n_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024],
                       help='Image size (height width)')

    # Diagnostics options
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate prediction visualizations')
    parser.add_argument('--num_viz', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--error_analysis', action='store_true', default=True,
                       help='Perform error analysis')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    diagnostics = ModelDiagnostics(args)
    diagnostics.run_diagnostics()
