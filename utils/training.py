import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

from utils.visualization import plot_detections, evaluate_detections
from utils.yolo_utils import process_predictions


def generate_anchors(dataset, num_anchors=9, strides=[8, 16, 32], img_size=640):
    """
    Generate anchor boxes using k-means clustering on training data

    Args:
        dataset: Dataset object containing training images and annotations
        num_anchors: Total number of anchors to generate
        strides: List of strides for each detection scale
        img_size: Input image size

    Returns:
        anchors: List of anchor tensors for each stride
    """
    print("Generating anchors from dataset statistics...")

    # Collect all bounding box dimensions
    all_wh = []

    # Go through dataset to collect width and height of ground truth boxes
    for i in range(len(dataset)):
        _, target = dataset[i]
        boxes = target['boxes']

        if len(boxes) > 0:
            # Convert from [x1, y1, x2, y2] to [w, h]
            wh = boxes[:, 2:] - boxes[:, :2]
            all_wh.append(wh)

    # Concatenate all width-height pairs
    if all_wh:
        all_wh = torch.cat(all_wh, dim=0)
    else:
        print("No boxes found in dataset. Using default anchors.")
        # Default anchors if no boxes found
        return [
            torch.tensor([[10, 13], [16, 30], [33, 23]]),  # Small scale
            torch.tensor([[30, 61], [62, 45], [59, 119]]),  # Medium scale
            torch.tensor([[116, 90], [156, 198], [373, 326]])  # Large scale
        ]

    # Normalize by image size
    all_wh = all_wh / img_size

    # Use k-means clustering to find optimal anchor boxes
    anchors = kmeans_anchors(all_wh.numpy(), num_anchors, iterations=300)

    # Sort anchors by area (smallest to largest)
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

    # Split anchors by scale (small, medium, large)
    anchors_per_scale = num_anchors // len(strides)
    anchor_list = []

    for i in range(len(strides)):
        start_idx = i * anchors_per_scale
        end_idx = (i + 1) * anchors_per_scale

        # Make sure we don't go out of bounds
        if end_idx > len(anchors):
            end_idx = len(anchors)

        # Get anchors for this scale
        scale_anchors = anchors[start_idx:end_idx]

        # Convert back to absolute size
        scale_anchors = torch.tensor(scale_anchors) * img_size

        anchor_list.append(scale_anchors)

    # Print anchor statistics
    print("Generated anchors:")
    for i, (anchors, stride) in enumerate(zip(anchor_list, strides)):
        print(f"Scale {i+1} (stride {stride}):")
        for j, anchor in enumerate(anchors):
            print(f"  Anchor {j+1}: {anchor[0]:.1f} x {anchor[1]:.1f} pixels")

    return anchor_list


def kmeans_anchors(wh, k, iterations=300, distance='iou'):
    """
    K-means clustering to find optimal anchor boxes

    Args:
        wh: Width-height pairs, shape [n, 2]
        k: Number of clusters
        iterations: Maximum number of iterations
        distance: Distance metric ('iou' or 'euclidean')

    Returns:
        centroids: Anchor boxes, shape [k, 2]
    """
    # Number of boxes
    n = wh.shape[0]

    # Initialize centroids randomly
    idx = np.random.choice(n, k, replace=False)
    centroids = wh[idx]

    # Last cluster assignments
    last_clusters = np.zeros(n)

    # Main loop
    for _ in range(iterations):
        # Calculate distances between boxes and centroids
        if distance == 'iou':
            # IoU-based distance (1 - IoU)
            d = 1 - box_iou(wh, centroids)
        else:
            # Euclidean distance
            d = np.sum((wh[:, np.newaxis] - centroids[np.newaxis])**2, axis=2)

        # Assign boxes to nearest centroid
        clusters = np.argmin(d, axis=1)

        # Check for convergence
        if (clusters == last_clusters).all():
            break

        # Update centroids
        for i in range(k):
            if np.sum(clusters == i) > 0:
                centroids[i] = np.mean(wh[clusters == i], axis=0)

        # Remember assignments
        last_clusters = clusters

    return centroids


def box_iou(box1, box2):
    """
    Calculate IoU between boxes, assuming they are centered at origin

    Args:
        box1: Width-height pairs, shape [n, 2]
        box2: Width-height pairs, shape [k, 2]

    Returns:
        iou: IoU matrix, shape [n, k]
    """
    # Convert to numpy arrays if needed
    if isinstance(box1, torch.Tensor):
        box1 = box1.numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.numpy()

    # Make sure inputs are 2D
    if len(box1.shape) == 1:
        box1 = box1.reshape(1, 2)
    if len(box2.shape) == 1:
        box2 = box2.reshape(1, 2)

    # Number of boxes
    n = box1.shape[0]
    k = box2.shape[0]

    # Calculate areas
    area1 = box1[:, 0] * box1[:, 1]  # [n]
    area2 = box2[:, 0] * box2[:, 1]  # [k]

    # Calculate intersections
    # We assume boxes are centered at (0,0), so the intersection width is
    # the minimum of the two widths, and same for height
    inter_w = np.minimum(box1[:, 0][:, np.newaxis], box2[:, 0])  # [n, k]
    inter_h = np.minimum(box1[:, 1][:, np.newaxis], box2[:, 1])  # [n, k]

    # Intersection area
    inter_area = inter_w * inter_h  # [n, k]

    # Union area (broadcasting)
    union_area = area1[:, np.newaxis] + area2 - inter_area  # [n, k]

    # IoU
    iou = inter_area / union_area  # [n, k]

    return iou


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch, anchors, strides):
    """
    Train model for one epoch

    Args:
        model: YOLOD11 model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        epoch: Current epoch number
        anchors: List of anchor tensors for each stride
        strides: List of strides for each feature map

    Returns:
        metrics: Dictionary with training metrics
    """
    model.train()
    epoch_loss = 0
    epoch_box_loss = 0
    epoch_obj_loss = 0
    epoch_cls_loss = 0

    start_time = time.time()
    total_batches = len(dataloader)

    # Create progress bar
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}")

    for batch_idx, (images, targets) in pbar:
        # Move inputs to device
        images = images.to(device)
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in targets.items()}

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss, loss_components = loss_fn(predictions, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        epoch_loss += loss.item()
        epoch_box_loss += loss_components['box']
        epoch_obj_loss += loss_components['obj']
        epoch_cls_loss += loss_components['cls']

        # Update progress bar
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Box": f"{loss_components['box']:.4f}",
            "Obj": f"{loss_components['obj']:.4f}",
            "Cls": f"{loss_components['cls']:.4f}"
        })

        # Print detailed progress at specific intervals
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)

            print(f"\nBatch {batch_idx+1}/{total_batches} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Box: {loss_components['box']:.4f} | "
                  f"Obj: {loss_components['obj']:.4f} | "
                  f"Cls: {loss_components['cls']:.4f} | "
                  f"Time: {elapsed:.2f}s | "
                  f"ETA: {eta:.2f}s")

    # Calculate average losses
    avg_loss = epoch_loss / total_batches
    avg_box_loss = epoch_box_loss / total_batches
    avg_obj_loss = epoch_obj_loss / total_batches
    avg_cls_loss = epoch_cls_loss / total_batches

    return {
        'loss': avg_loss,
        'box_loss': avg_box_loss,
        'obj_loss': avg_obj_loss,
        'cls_loss': avg_cls_loss
    }


def validate(model, dataloader, loss_fn, device, anchors, strides, class_names, conf_threshold=0.25, iou_threshold=0.45):
    """
    Validate model on validation dataset

    Args:
        model: YOLOD11 model
        dataloader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to validate on
        anchors: List of anchor tensors for each stride
        strides: List of strides for each feature map
        class_names: List of class names
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        metrics: Dictionary with validation metrics
    """
    model.eval()
    val_loss = 0
    val_box_loss = 0
    val_obj_loss = 0
    val_cls_loss = 0

    total_batches = len(dataloader)

    # For mAP calculation
    all_predictions = []
    all_targets = []

    # For visualization
    sample_images = []
    sample_preds = []
    sample_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move inputs to device
            images = images.to(device)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}

            # Forward pass
            predictions = model(images)

            # Calculate loss
            loss, loss_components = loss_fn(predictions, targets)

            # Process predictions
            processed_preds = process_predictions(
                predictions,
                anchors,
                strides,
                images.shape[2],  # Assuming square images (height=width)
                conf_threshold,
                iou_threshold
            )

            # Store predictions and targets for mAP calculation
            all_predictions.extend(processed_preds)

            # Create a copy of targets with tensors on the same device as processed_preds
            batch_target = {}
            if 'boxes' in targets:
                batch_target['boxes'] = targets['boxes'].to(device)
            if 'labels' in targets:
                batch_target['labels'] = targets['labels'].to(device)

            all_targets.append(batch_target)

            # Save some samples for visualization
            if batch_idx == 0:
                sample_images = images.cpu()
                sample_preds = processed_preds
                sample_targets = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                  for k, v in targets.items()}

            # Update metrics
            val_loss += loss.item()
            val_box_loss += loss_components['box']
            val_obj_loss += loss_components['obj']
            val_cls_loss += loss_components['cls']

            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Box": f"{loss_components['box']:.4f}",
                "Obj": f"{loss_components['obj']:.4f}",
                "Cls": f"{loss_components['cls']:.4f}"
            })

    # Calculate average losses
    avg_loss = val_loss / total_batches
    avg_box_loss = val_box_loss / total_batches
    avg_obj_loss = val_obj_loss / total_batches
    avg_cls_loss = val_cls_loss / total_batches

    # Calculate mAP
    eval_metrics = evaluate_detections(all_predictions, all_targets, iou_threshold=0.5)

    return {
        'loss': avg_loss,
        'box_loss': avg_box_loss,
        'obj_loss': avg_obj_loss,
        'cls_loss': avg_cls_loss,
        'mAP': eval_metrics['mAP'],
        'precision': eval_metrics['precision'],
        'recall': eval_metrics['recall'],
        'sample_images': sample_images,
        'sample_preds': sample_preds,
        'sample_targets': sample_targets
    }


def save_checkpoint(model, optimizer, epoch, loss_dict, save_path, is_best=False):
    """
    Save model checkpoint with detailed loss information

    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch number
        loss_dict: Dictionary containing loss values
        save_path: Path to save the checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': loss_dict.get('val_loss', float('inf')),
        'mAP': loss_dict.get('mAP', 0.0),
        'losses': loss_dict,
    }

    # Save the model
    torch.save(checkpoint, save_path)

    # Also save a separate losses file in JSON format for easy access
    loss_path = save_path.replace('.pth', '_losses.json')
    with open(loss_path, 'w') as f:
        json.dump(loss_dict, f, indent=4)

    print(f"Model saved: {save_path}")
    print(f"Loss info saved: {loss_path}")
