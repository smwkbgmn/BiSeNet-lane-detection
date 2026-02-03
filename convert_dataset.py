"""
Dataset Conversion Script: Lane Coordinates to Segmentation Masks
Converts JSON lane annotations to binary/multi-class segmentation masks for BiSeNet
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
        lane_id: integer representing the lane class (1, 2, 3, 4 for different lanes)
        thickness: thickness of the lane line
    """
    if len(lane_points) < 2:
        return mask

    points = np.array(lane_points, dtype=np.int32)

    # Draw polylines on the mask
    cv2.polylines(mask, [points], isClosed=False, color=lane_id, thickness=thickness)

    return mask


def convert_annotation_to_mask(json_path, image_path, output_path,
                                binary_mode=False, thickness=10):
    """
    Convert a single JSON annotation to segmentation mask

    Args:
        json_path: path to JSON annotation file
        image_path: path to corresponding image
        output_path: path to save the mask
        binary_mode: if True, all lanes are class 1; if False, each lane gets different class
        thickness: thickness of lane lines in pixels
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
    with open(json_path, 'r') as f:
        annotation = json.load(f)

    lanes = annotation.get('lanes', [])

    # Draw each lane on the mask
    for lane_idx, lane_points in enumerate(lanes):
        if len(lane_points) == 0:
            continue

        # Assign lane ID (1-indexed for segmentation, 0 is background)
        lane_id = 1 if binary_mode else (lane_idx + 1)

        # Ensure lane_id doesn't exceed 255 (uint8 max)
        lane_id = min(lane_id, 255)

        mask = draw_lane_mask(mask, lane_points, lane_id, thickness=thickness)

    # Save mask
    cv2.imwrite(str(output_path), mask)
    return True


def convert_dataset(input_dir, output_dir, binary_mode=False, thickness=10):
    """
    Convert entire dataset from JSON annotations to segmentation masks

    Args:
        input_dir: directory containing 'images' and 'annotations' folders
        output_dir: directory to save masks
        binary_mode: binary segmentation (lane vs background) or multi-class
        thickness: thickness of lane lines
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    annotations_dir = input_path / 'annotations'
    images_dir = input_path / 'images'
    masks_dir = output_path / 'masks'

    # Create output directory
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = list(annotations_dir.glob('*.json'))

    print(f"Found {len(json_files)} annotations")
    print(f"Mode: {'Binary' if binary_mode else 'Multi-class'}")
    print(f"Lane thickness: {thickness}px")

    successful = 0
    failed = 0

    # Process each annotation
    for json_file in tqdm(json_files, desc="Converting annotations"):
        # Read JSON to get image filename
        with open(json_file, 'r') as f:
            annotation = json.load(f)

        image_name = annotation.get('image', json_file.stem + '.png')

        # Try different image extensions
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

        # Create mask filename
        mask_filename = json_file.stem + '.png'
        mask_path = masks_dir / mask_filename

        # Convert annotation to mask
        if convert_annotation_to_mask(json_file, image_path, mask_path,
                                     binary_mode=binary_mode, thickness=thickness):
            successful += 1
        else:
            failed += 1

    print(f"\nConversion complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Masks saved to: {masks_dir}")

    # Create a visualization of the first few masks
    visualize_samples(images_dir, masks_dir, output_path, num_samples=5)


def visualize_samples(images_dir, masks_dir, output_dir, num_samples=5):
    """Create visualization of image + mask overlays for inspection"""
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    mask_files = list(Path(masks_dir).glob('*.png'))[:num_samples]

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
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0]   # Yellow
        ]

        for i in range(1, 5):
            colored_mask[mask == i] = colors[i-1]

        # Blend
        overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)

        # Save
        viz_path = viz_dir / f'viz_{mask_file.name}'
        cv2.imwrite(str(viz_path), overlay)

    print(f"Sample visualizations saved to: {viz_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert lane annotations to segmentation masks')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing images and annotations folders')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory to save masks')
    parser.add_argument('--binary', action='store_true',
                       help='Use binary segmentation (lane vs background) instead of multi-class')
    parser.add_argument('--thickness', type=int, default=10,
                       help='Thickness of lane lines in pixels (default: 10)')

    args = parser.parse_args()

    convert_dataset(args.input_dir, args.output_dir,
                   binary_mode=args.binary, thickness=args.thickness)
