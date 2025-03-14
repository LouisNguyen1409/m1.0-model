import os
import sys
import argparse
import torch
import yaml
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Import model and utilities
from model import YOLOD11
from utils.yolo_utils import process_predictions
from utils.visualization import draw_boxes, plot_detections


def load_model(checkpoint_path, device, num_classes=7):
    """
    Load YOLOD11 model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        num_classes: Number of classes

    Returns:
        model: Loaded model
        anchors: Anchor boxes used for prediction
        strides: Strides for each detection scale
    """
    # Initialize model
    model = YOLOD11(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract anchors and strides if available in checkpoint
    anchors = None
    strides = [8, 16, 32]  # Default strides

    # Try to get anchors from checkpoint
    if 'anchors' in checkpoint:
        anchors = checkpoint['anchors']
    else:
        # Use default anchors
        anchors = [
            torch.tensor([[10, 13], [16, 30], [33, 23]]),  # Small scale
            torch.tensor([[30, 61], [62, 45], [59, 119]]),  # Medium scale
            torch.tensor([[116, 90], [156, 198], [373, 326]])  # Large scale
        ]

    # Move model to device
    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation Loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"mAP: {checkpoint.get('mAP', 'unknown')}")

    return model, anchors, strides


def preprocess_image(image_path, img_size=640):
    """
    Preprocess image for inference

    Args:
        image_path: Path to image file
        img_size: Input image size

    Returns:
        image: Preprocessed image tensor
        orig_image: Original image for display
        ratio: Scaling ratio for box coordinates
    """
    # Read image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path

    # Store original image
    orig_image = img.copy()

    # Get original dimensions
    height, width = img.shape[:2]

    # Calculate scaling ratio
    ratio = min(img_size / width, img_size / height)

    # Resize image
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    img = cv2.resize(img, (new_width, new_height))

    # Create padded image (black padding)
    padded_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded_img[:new_height, :new_width] = img

    # Normalize and convert to tensor
    padded_img = padded_img.astype(np.float32) / 255.0
    padded_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0)

    return padded_img, orig_image, (new_width, new_height, width, height)


def postprocess_detections(detections, image_info, class_names):
    """
    Post-process detections to convert to original image coordinates

    Args:
        detections: Tensor of detections from model
        image_info: Tuple of (new_width, new_height, width, height)
        class_names: List of class names

    Returns:
        detections: List of processed detections
    """
    new_width, new_height, orig_width, orig_height = image_info

    # No detections
    if len(detections) == 0 or detections.size(0) == 0:
        return []

    # Scale detection coordinates from padded image to original image
    width_ratio = orig_width / new_width
    height_ratio = orig_height / new_height

    processed_dets = []

    for det in detections:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det.cpu().numpy()

        # Scale coordinates to original image size
        x1 = x1 * width_ratio
        y1 = y1 * height_ratio
        x2 = x2 * width_ratio
        y2 = y2 * height_ratio

        # Clip to image boundaries
        x1 = max(0, min(x1, orig_width))
        y1 = max(0, min(y1, orig_height))
        x2 = max(0, min(x2, orig_width))
        y2 = max(0, min(y2, orig_height))

        # Get class name
        cls_id = int(cls_id)
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"

        # Calculate confidence
        confidence = float(obj_conf * cls_conf)

        # Create detection dictionary
        det_dict = {
            'bbox': [x1, y1, x2, y2],
            'class_id': cls_id,
            'class_name': class_name,
            'confidence': confidence
        }

        processed_dets.append(det_dict)

    return processed_dets


def run_inference(model, image_path, anchors, strides, class_names, img_size=640,
                  conf_threshold=0.25, iou_threshold=0.45, device='cpu'):
    """
    Run inference on a single image

    Args:
        model: YOLOD11 model
        image_path: Path to image file
        anchors: Anchor boxes for each scale
        strides: Strides for each detection scale
        class_names: List of class names
        img_size: Input image size
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device: Device to run inference on

    Returns:
        detections: List of detections
        orig_image: Original image
    """
    # Preprocess image
    img, orig_image, image_info = preprocess_image(image_path, img_size)

    # Move to device
    img = img.to(device)

    # Run inference
    with torch.no_grad():
        # Forward pass
        predictions = model(img)

        # Process predictions
        processed_preds = process_predictions(
            predictions,
            anchors,
            strides,
            img_size,
            conf_threshold,
            iou_threshold
        )

        # Get detections for the first (and only) image in batch
        detections = processed_preds[0]

    # Post-process detections to original image coordinates
    detections = postprocess_detections(detections, image_info, class_names)

    return detections, orig_image


def visualize_detections(image, detections, class_names, output_path=None, show=True):
    """
    Visualize detections on image

    Args:
        image: Original image
        detections: List of detections
        class_names: List of class names
        output_path: Path to save output image
        show: Whether to display image

    Returns:
        vis_image: Image with detections visualized
    """
    # Convert detections list to tensor format for drawing function
    if len(detections) > 0:
        det_tensor = torch.zeros((len(detections), 7))

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            confidence = det['confidence']

            det_tensor[i, 0] = x1
            det_tensor[i, 1] = y1
            det_tensor[i, 2] = x2
            det_tensor[i, 3] = y2
            det_tensor[i, 4] = confidence  # obj_conf
            det_tensor[i, 5] = confidence  # cls_conf
            det_tensor[i, 6] = cls_id
    else:
        det_tensor = torch.zeros((0, 7))

    # Draw boxes on image
    vis_image = draw_boxes(image, det_tensor, class_names)

    # Save image if output path is provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    # Display image if show is True
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title(f"Detections: {len(detections)}")
        plt.show()

    return vis_image


def main():
    parser = argparse.ArgumentParser(description="YOLOD11 Inference Script")

    # Required arguments
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-yaml", required=True, help="Path to data YAML file")

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to single image")
    group.add_argument("--images-dir", help="Directory containing images")
    group.add_argument("--video", help="Path to video file")

    # Output options
    parser.add_argument("--output-dir", default="./inference_results", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--save", action="store_true", help="Save results")

    # Inference parameters
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", default="", help="Device to run on (cuda or cpu)")

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Create output directory
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load YAML configuration
    with open(args.data_yaml, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    # Get class names
    class_names = yaml_cfg.get('names', [])
    num_classes = yaml_cfg.get('nc', len(class_names))

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Load model
    model, anchors, strides = load_model(args.model, device, num_classes)

    # Run inference
    if args.image:
        # Single image inference
        print(f"Running inference on {args.image}")
        detections, orig_image = run_inference(
            model, args.image, anchors, strides, class_names,
            args.img_size, args.conf_thres, args.iou_thres, device
        )

        # Print detections
        print(f"Found {len(detections)} detections:")
        for i, det in enumerate(detections):
            print(f"  {i+1}: {det['class_name']} ({det['confidence']:.2f}): {det['bbox']}")

        # Visualize detections
        if args.show or args.save:
            output_path = os.path.join(args.output_dir, os.path.basename(args.image)) if args.save else None
            visualize_detections(orig_image, detections, class_names, output_path, args.show)

    elif args.images_dir:
        # Directory of images
        image_files = [str(p) for p in Path(args.images_dir).glob("**/*")
                       if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

        print(f"Found {len(image_files)} images in {args.images_dir}")

        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            detections, orig_image = run_inference(
                model, img_path, anchors, strides, class_names,
                args.img_size, args.conf_thres, args.iou_thres, device
            )

            # Save results
            if args.save:
                # Save detections as JSON
                json_path = os.path.join(args.output_dir, os.path.basename(img_path) + ".json")
                with open(json_path, 'w') as f:
                    json.dump(detections, f, indent=4)

                # Save visualized image
                output_path = os.path.join(args.output_dir, os.path.basename(img_path))
                visualize_detections(orig_image, detections, class_names, output_path, False)

            # Show results if requested (may be slow for many images)
            if args.show:
                visualize_detections(orig_image, detections, class_names, None, True)

    elif args.video:
        # Video inference
        print(f"Running inference on video: {args.video}")

        # Open video
        cap = cv2.VideoCapture(args.video)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video writer if saving
        if args.save:
            output_path = os.path.join(args.output_dir, os.path.basename(args.video))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for better quality
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process video frames
        frame_idx = 0
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run inference
                detections, _ = run_inference(
                    model, frame_rgb, anchors, strides, class_names,
                    args.img_size, args.conf_thres, args.iou_thres, device
                )

                # Visualize detections
                vis_frame = visualize_detections(frame_rgb, detections, class_names, None, False)

                # Convert back to BGR for OpenCV
                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

                # Write frame if saving
                if args.save:
                    out.write(vis_frame_bgr)

                # Show frame if requested
                if args.show:
                    cv2.imshow('YOLOD11 Detection', vis_frame_bgr)

                    # Break on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_idx += 1
                pbar.update(1)

        # Release resources
        cap.release()
        if args.save:
            out.release()
        cv2.destroyAllWindows()

        print(f"Processed {frame_idx} frames")
        if args.save:
            print(f"Output video saved to {output_path}")


if __name__ == "__main__":
    main()
