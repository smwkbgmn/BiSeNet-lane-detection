"""
Dataset Conversion Script for BiSeNet in fine-tune3
Converts JSON lane annotations to segmentation masks
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def draw_lane_mask(mask, lane_points, lane_id, thickness=10):
    """
    Draw a single lane on the mask

    Args:
        mask: numpy array (H, W) for the mask
        lane_points: list of [x, y] coordinates
        lane_id: integer representing the lane class
        thickness: thickness of the lane line
    """
    if len(lane_points) < 2:
        return mask

    points = np.array(lane_points, dtype=np.int32)
    cv2.polylines(mask, [points], isClosed=False, color=lane_id, thickness=thickness)

    return mask


def convert_annotation_to_mask(json_path, image_path, output_path,
                                binary_mode=False, thickness=10):
    """
    Convert a single JSON annotation to segmentation mask
    """
    # Read image to get dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return False

    h, w = image.shape[:2]

    # Create empty mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Read JSON annotation
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)

    lanes = annotation.get('lanes', [])

    # Draw each lane on the mask with sequential class IDs
    class_id = 1
    for lane_idx, lane_points in enumerate(lanes):
        if len(lane_points) == 0:
            continue

        lane_id = 1 if binary_mode else class_id
        lane_id = min(lane_id, 255)

        mask = draw_lane_mask(mask, lane_points, lane_id, thickness=thickness)

        if not binary_mode:
            class_id += 1

    # Save mask
    cv2.imwrite(str(output_path), mask)
    return True


def convert_dataset(input_dir, output_dir, binary_mode=False, thickness=10):
    """
    Convert entire dataset from JSON annotations to segmentation masks
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    annotations_dir = input_path / 'annotations'
    images_dir = input_path / 'images'
    masks_dir = output_path / 'masks'

    # Create output directory
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to output directory
    output_images_dir = output_path / 'images'
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = list(annotations_dir.glob('*.json'))

    print(f"Found {len(json_files)} annotations")
    print(f"Mode: {'Binary' if binary_mode else 'Multi-class'}")
    print(f"Lane thickness: {thickness}px")

    successful = 0
    failed = 0

    # Process each annotation
    for json_file in tqdm(json_files, desc="Converting annotations"):
        with open(json_file, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        image_name = annotation.get('image', json_file.stem + '.png')

        # Find image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = images_dir / image_name.replace(Path(image_name).suffix, ext)
            if potential_path.exists():
                image_path = potential_path
                break

        if image_path is None or not image_path.exists():
            print(f"Warning: Image not found for {json_file.name}")
            failed += 1
            continue

        # Copy image to output
        import shutil
        shutil.copy(image_path, output_images_dir / image_path.name)

        # Create mask
        mask_filename = json_file.stem + '.png'
        mask_path = masks_dir / mask_filename

        if convert_annotation_to_mask(json_file, image_path, mask_path,
                                     binary_mode=binary_mode, thickness=thickness):
            successful += 1
        else:
            failed += 1

    print(f"\nConversion complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Images saved to: {output_images_dir}")
    print(f"Masks saved to: {masks_dir}")

    # Create visualizations
    visualize_samples(output_images_dir, masks_dir, output_path, num_samples=5)


def visualize_samples(images_dir, masks_dir, output_dir, num_samples=5):
    """Create visualization of image + mask overlays"""
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    mask_files = list(Path(masks_dir).glob('*.png'))[:num_samples]

    colors = [
        [0, 0, 0],       # Background
        [255, 0, 0],     # Lane 1 - red
        [0, 255, 0],     # Lane 2 - green
        [0, 0, 255],     # Lane 3 - blue
        [255, 255, 0],   # Lane 4 - yellow
    ]

    for mask_file in mask_files:
        # Find corresponding image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = images_dir / mask_file.name.replace('.png', ext)
            if potential_path.exists():
                image_path = potential_path
                break

        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        # Create colored overlay
        colored_mask = np.zeros_like(image)

        for i in range(min(5, len(colors))):
            colored_mask[mask == i] = colors[i]

        # Blend
        overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)

        # Save
        viz_path = viz_dir / f'viz_{mask_file.name}'
        cv2.imwrite(str(viz_path), overlay)

    print(f"Sample visualizations saved to: {viz_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert lane annotations to segmentation masks for BiSeNet')
    parser.add_argument('--input_dir', type=str, default='./dataset/augmented',
                       help='Input directory containing images and annotations folders')
    parser.add_argument('--output_dir', type=str, default='./dataset/bisenet_data',
                       help='Output directory to save masks')
    parser.add_argument('--binary', action='store_true',
                       help='Use binary segmentation (lane vs background)')
    parser.add_argument('--thickness', type=int, default=10,
                       help='Thickness of lane lines in pixels')

    args = parser.parse_args()

    convert_dataset(args.input_dir, args.output_dir,
                   binary_mode=args.binary, thickness=args.thickness)
