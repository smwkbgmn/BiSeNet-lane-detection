"""
Lane Detection Dataset for BiSeNet
Custom dataset loader for lane segmentation
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LaneSegmentationDataset(Dataset):
    """
    Lane Segmentation Dataset for BiSeNet training

    Expected directory structure:
        dataset_root/
            images/
                frame_0001.jpg
                frame_0002.jpg
                ...
            masks/
                frame_0001.png
                frame_0002.png
                ...
            annotations/ (optional, for reference)
                frame_0001.json
                ...
    """

    def __init__(self, root_dir, mode='train', image_size=(512, 1024),
                 n_classes=2, transforms=None):
        """
        Args:
            root_dir: Root directory of the dataset
            mode: 'train', 'val', or 'test'
            image_size: (height, width) for resizing images
            n_classes: Number of classes (2 for binary: background + lane,
                      5 for multi-class: background + 4 lanes)
            transforms: albumentations transforms (if None, use default)
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.image_size = image_size
        self.n_classes = n_classes

        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'masks'

        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) +
                                 list(self.images_dir.glob('*.png')))

        # Filter to only include images with corresponding masks
        self.samples = []
        for img_path in self.image_files:
            mask_path = self.masks_dir / (img_path.stem + '.png')
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        print(f"Loaded {len(self.samples)} samples for {mode} mode")

        # Setup transforms
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self._get_default_transforms()

    def _get_default_transforms(self):
        """Get default augmentation transforms based on mode"""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                    val_shift_limit=10, p=0.3),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert mask to long tensor for CrossEntropyLoss
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'mask': mask,
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        }


class LaneDatasetFromJSON(Dataset):
    """
    Lane Dataset that creates masks on-the-fly from JSON annotations
    Useful if you don't want to pre-generate all masks
    """

    def __init__(self, root_dir, mode='train', image_size=(512, 1024),
                 n_classes=2, binary_mode=False, lane_thickness=10, transforms=None):
        """
        Args:
            root_dir: Root directory containing images/ and annotations/
            mode: 'train', 'val', or 'test'
            image_size: (height, width)
            n_classes: Number of classes
            binary_mode: If True, all lanes are class 1
            lane_thickness: Thickness of lane lines in mask
            transforms: albumentations transforms
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.image_size = image_size
        self.n_classes = n_classes
        self.binary_mode = binary_mode
        self.lane_thickness = lane_thickness

        self.images_dir = self.root_dir / 'images'
        self.annotations_dir = self.root_dir / 'annotations'

        # Get all annotation files
        self.annotation_files = sorted(list(self.annotations_dir.glob('*.json')))

        # Filter to include only annotations with corresponding images
        self.samples = []
        for ann_path in self.annotation_files:
            with open(ann_path, 'r') as f:
                annotation = json.load(f)

            image_name = annotation.get('image', ann_path.stem + '.png')

            # Find corresponding image
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = self.images_dir / image_name.replace(Path(image_name).suffix, ext)
                if potential_path.exists():
                    img_path = potential_path
                    break

            if img_path is not None:
                self.samples.append((img_path, ann_path))

        print(f"Loaded {len(self.samples)} samples for {mode} mode (on-the-fly mask generation)")

        # Setup transforms
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self._get_default_transforms()

    def _get_default_transforms(self):
        """Get default augmentation transforms"""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def _create_mask_from_annotation(self, annotation, image_shape):
        """Create segmentation mask from lane annotations"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        lanes = annotation.get('lanes', [])

        # Assign sequential class IDs to non-empty lanes only
        class_id = 1
        for lane_idx, lane_points in enumerate(lanes):
            if len(lane_points) < 2:
                continue

            lane_id = 1 if self.binary_mode else class_id
            lane_id = min(lane_id, 255)

            points = np.array(lane_points, dtype=np.int32)
            cv2.polylines(mask, [points], isClosed=False,
                         color=lane_id, thickness=self.lane_thickness)

            if not self.binary_mode:
                class_id += 1

        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read annotation and create mask
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        mask = self._create_mask_from_annotation(annotation, image.shape)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'mask': mask,
            'image_path': str(img_path),
            'annotation_path': str(ann_path)
        }


def get_lane_dataloader(root_dir, batch_size=8, num_workers=4,
                       mode='train', image_size=(512, 1024), n_classes=2,
                       use_json=False, **kwargs):
    """
    Convenience function to create dataloader

    Args:
        root_dir: Dataset root directory
        batch_size: Batch size
        num_workers: Number of worker processes
        mode: 'train', 'val', or 'test'
        image_size: (height, width)
        n_classes: Number of classes
        use_json: If True, use LaneDatasetFromJSON (on-the-fly mask generation)
        **kwargs: Additional arguments for dataset

    Returns:
        DataLoader
    """
    if use_json:
        dataset = LaneDatasetFromJSON(root_dir, mode=mode, image_size=image_size,
                                     n_classes=n_classes, **kwargs)
    else:
        dataset = LaneSegmentationDataset(root_dir, mode=mode, image_size=image_size,
                                         n_classes=n_classes, **kwargs)

    shuffle = (mode == 'train')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )

    return dataloader


if __name__ == '__main__':
    # Test the dataset
    dataset = LaneSegmentationDataset(
        root_dir='../datasets/lane_detection',
        mode='train',
        image_size=(512, 1024),
        n_classes=2
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Mask unique values: {torch.unique(sample['mask'])}")
