"""
BiSeNet Lane Detection Inference Script
Run inference on images or video using trained model
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
from models.bisenetv2 import BiSeNetV2


class LaneDetectionInference:
    def __init__(self, checkpoint_path, n_classes=2, image_size=(512, 1024), device='cuda'):
        """
        Initialize inference engine

        Args:
            checkpoint_path: Path to trained model checkpoint
            n_classes: Number of classes
            image_size: (height, width) for model input
            device: 'cuda' or 'cpu'
        """
        self.n_classes = n_classes
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load model
        print(f"Loading model from {checkpoint_path}")
        self.model = BiSeNetV2(n_classes=n_classes, aux_mode='eval')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

        # Setup transforms
        self.transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Define colors for visualization
        self.colors = np.array([
            [0, 0, 0],       # Background - black
            [255, 0, 0],     # Lane 1 - red
            [0, 255, 0],     # Lane 2 - green
            [0, 0, 255],     # Lane 3 - blue
            [255, 255, 0],   # Lane 4 - yellow
        ], dtype=np.uint8)

    def preprocess(self, image):
        """Preprocess image for model input"""
        original_size = image.shape[:2]
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        return image_tensor, original_size

    def postprocess(self, logits, original_size):
        """Convert model output to segmentation mask"""
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        # Resize to original size
        pred_resized = cv2.resize(
            pred.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        return pred_resized

    def visualize(self, image, mask, alpha=0.5):
        """
        Create visualization overlay

        Args:
            image: Original image (H, W, 3)
            mask: Predicted mask (H, W)
            alpha: Transparency for overlay

        Returns:
            Overlay image
        """
        # Create colored mask
        colored_mask = np.zeros_like(image)
        for cls in range(min(self.n_classes, len(self.colors))):
            colored_mask[mask == cls] = self.colors[cls]

        # Blend with original image
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        return overlay, colored_mask

    def predict_image(self, image_path, output_path=None, visualize=True):
        """
        Run inference on a single image

        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            visualize: Whether to create visualization

        Returns:
            mask, overlay (if visualize=True)
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        image_tensor, original_size = self.preprocess(image_rgb)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            logits = outputs[0]

        # Postprocess
        mask = self.postprocess(logits, original_size)

        # Visualize
        overlay = None
        colored_mask = None
        if visualize:
            overlay, colored_mask = self.visualize(image, mask)

        # Save output
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save mask
            cv2.imwrite(str(output_path).replace('.png', '_mask.png'), mask)

            # Save overlay
            if overlay is not None:
                cv2.imwrite(str(output_path), overlay)

            # Save colored mask
            if colored_mask is not None:
                cv2.imwrite(str(output_path).replace('.png', '_colored.png'), colored_mask)

        return mask, overlay

    def predict_directory(self, input_dir, output_dir, extensions=['.jpg', '.png', '.jpeg']):
        """
        Run inference on all images in a directory

        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for results
            extensions: List of image file extensions to process
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))

        print(f"Found {len(image_files)} images")

        # Process each image
        for img_path in tqdm(image_files, desc='Processing images'):
            output_file = output_path / f'{img_path.stem}_result.png'
            try:
                self.predict_image(img_path, output_file, visualize=True)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"\nResults saved to {output_path}")

    def predict_video(self, video_path, output_path, fps=None):
        """
        Run inference on a video

        Args:
            video_path: Input video path
            output_path: Output video path
            fps: Output video FPS (if None, use input FPS)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Processing video: {total_frames} frames at {fps} FPS")

        with tqdm(total=total_frames, desc='Processing frames') as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor, original_size = self.preprocess(frame_rgb)

                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    logits = outputs[0]

                mask = self.postprocess(logits, original_size)
                overlay, _ = self.visualize(frame, mask, alpha=0.5)

                # Write frame
                out.write(overlay)
                pbar.update(1)

        cap.release()
        out.release()
        print(f"\nVideo saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='BiSeNet Lane Detection Inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image, directory, or video')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path or directory')

    # Model config
    parser.add_argument('--n_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024],
                       help='Image size (height width)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')

    # Input type
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'image', 'directory', 'video'],
                       help='Input mode (auto-detect by default)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Initialize inference engine
    inference = LaneDetectionInference(
        checkpoint_path=args.checkpoint,
        n_classes=args.n_classes,
        image_size=tuple(args.image_size),
        device=args.device
    )

    input_path = Path(args.input)

    # Determine mode
    mode = args.mode
    if mode == 'auto':
        if input_path.is_file():
            if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                mode = 'video'
            else:
                mode = 'image'
        elif input_path.is_dir():
            mode = 'directory'
        else:
            raise ValueError(f"Invalid input path: {input_path}")

    # Run inference
    print(f"Running inference in {mode} mode")

    if mode == 'image':
        mask, overlay = inference.predict_image(input_path, args.output, visualize=True)
        print(f"Result saved to {args.output}")

    elif mode == 'directory':
        inference.predict_directory(input_path, args.output)

    elif mode == 'video':
        inference.predict_video(input_path, args.output)

    print("Inference complete!")
