"""
BiSeNet Training Script for Lane Detection
Train BiSeNetV2 on lane segmentation dataset
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from datetime import datetime

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from models.bisenetv2 import BiSeNetV2
from data.lane_dataset import get_lane_dataloader
from ohem_ce_loss import OhemCELoss


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize model
        print(f"Initializing BiSeNetV2 with {args.n_classes} classes")
        self.model = BiSeNetV2(n_classes=args.n_classes, aux_mode='train')
        self.model.to(self.device)

        # Initialize datasets
        print(f"Loading training data from {args.train_dir}")
        self.train_loader = get_lane_dataloader(
            root_dir=args.train_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode='train',
            image_size=tuple(args.image_size),
            n_classes=args.n_classes,
            use_json=args.use_json
        )

        if args.val_dir:
            print(f"Loading validation data from {args.val_dir}")
            self.val_loader = get_lane_dataloader(
                root_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                mode='val',
                image_size=tuple(args.image_size),
                n_classes=args.n_classes,
                use_json=args.use_json
            )
        else:
            self.val_loader = None

        # Initialize loss function
        if args.use_ohem:
            print("Using OHEM Cross Entropy Loss")
            self.criterion = OhemCELoss(thresh=0.7, device=self.device)
        else:
            print("Using standard Cross Entropy Loss")
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Initialize optimizer
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = self.model.get_params()
        self.optimizer = optim.AdamW([
            {'params': wd_params, 'weight_decay': args.weight_decay},
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': args.lr * 10, 'weight_decay': args.weight_decay},
            {'params': lr_mul_nowd_params, 'lr': args.lr * 10, 'weight_decay': 0},
        ], lr=args.lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        # Training state
        self.current_epoch = 0
        self.best_miou = 0.0
        self.global_step = 0

        # Wandb logging
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args)
            )

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.args.epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # BiSeNetV2 returns (logits, aux2, aux3, aux4, aux5_4) during training
            outputs = self.model(images)
            logits = outputs[0]

            # Calculate loss
            loss_main = self.criterion(logits, masks)

            # Add auxiliary losses
            if self.args.use_aux_loss:
                loss_aux2 = self.criterion(outputs[1], masks)
                loss_aux3 = self.criterion(outputs[2], masks)
                loss_aux4 = self.criterion(outputs[3], masks)
                loss_aux5_4 = self.criterion(outputs[4], masks)

                loss = loss_main + 0.4 * (loss_aux2 + loss_aux3 + loss_aux4 + loss_aux5_4)
            else:
                loss = loss_main

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': epoch_loss / (batch_idx + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Log to wandb
            if self.args.use_wandb and batch_idx % self.args.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_main': loss_main.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

        return epoch_loss / len(self.train_loader)

    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        class_iou_sum = torch.zeros(self.args.n_classes).to(self.device)
        class_count = torch.zeros(self.args.n_classes).to(self.device)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                logits = outputs[0]

                # Calculate loss
                loss = self.criterion(logits, masks)
                total_loss += loss.item()

                # Calculate accuracy
                preds = logits.argmax(dim=1)
                total_correct += (preds == masks).sum().item()
                total_pixels += masks.numel()

                # Calculate IoU per class
                for cls in range(self.args.n_classes):
                    pred_mask = (preds == cls)
                    true_mask = (masks == cls)

                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()

                    if union > 0:
                        class_iou_sum[cls] += intersection / union
                        class_count[cls] += 1

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_pixels

        # Calculate mean IoU
        class_iou = class_iou_sum / (class_count + 1e-8)
        miou = class_iou.mean().item()

        print(f"\nValidation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  mIoU: {miou:.4f}")
        for cls in range(self.args.n_classes):
            if class_count[cls] > 0:
                print(f"  Class {cls} IoU: {class_iou[cls].item():.4f}")

        # Log to wandb
        if self.args.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/accuracy': accuracy,
                'val/miou': miou,
                'epoch': self.current_epoch
            })

        return miou, avg_loss

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'global_step': self.global_step,
            'args': vars(self.args)
        }

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_miou = checkpoint['best_miou']
        self.global_step = checkpoint.get('global_step', 0)

        print(f"Resumed from epoch {self.current_epoch}, best mIoU: {self.best_miou:.4f}")

    def train(self):
        """Main training loop"""
        print(f"\nStarting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Epochs: {self.args.epochs}\n")

        for epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}")

            # Validate
            if self.val_loader and (epoch + 1) % self.args.val_interval == 0:
                miou, val_loss = self.validate()

                # Save best model
                if miou > self.best_miou:
                    self.best_miou = miou
                    self.save_checkpoint(is_best=True)
                    print(f"New best mIoU: {miou:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch_{epoch + 1}.pth')

            # Update learning rate
            self.scheduler.step()

        # Save final model
        self.save_checkpoint(filename='final_model.pth')
        print(f"\nTraining completed! Best mIoU: {self.best_miou:.4f}")

        if self.args.use_wandb:
            wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description='Train BiSeNet for Lane Detection')

    # Dataset
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Training dataset directory')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Validation dataset directory')
    parser.add_argument('--use_json', action='store_true',
                       help='Use on-the-fly mask generation from JSON')

    # Model
    parser.add_argument('--n_classes', type=int, default=2,
                       help='Number of classes (2 for binary, 5 for 4 lanes + background)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024],
                       help='Image size (height width)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Loss
    parser.add_argument('--use_ohem', action='store_true',
                       help='Use OHEM loss instead of standard CE loss')
    parser.add_argument('--use_aux_loss', action='store_true', default=True,
                       help='Use auxiliary losses from intermediate layers')

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Run validation every N epochs')

    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='bisenet-lane-detection',
                       help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log metrics every N batches')

    args = parser.parse_args()

    # Set experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'bisenet_lane_{timestamp}'

    return args


if __name__ == '__main__':
    args = parse_args()

    # Create trainer and start training
    trainer = Trainer(args)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()
