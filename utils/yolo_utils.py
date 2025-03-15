import torch
import math
import numpy as np


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between box1 and box2

    Args:
        box1: First box coordinates
        box2: Second box coordinates
        x1y1x2y2: If True, boxes are in [x1, y1, x2, y2] format, else [x_center, y_center, width, height]
        GIoU/DIoU/CIoU: Use generalized/distance/complete IoU loss

    Returns:
        IoU or modified IoU value
    """
    # Ensure input tensors have the right shape
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)

    # Convert from center-width to corner if needed
    if not x1y1x2y2:
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2
        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Ensure coordinates are properly ordered (x1 < x2, y1 < y2)
    box1_x1, box1_x2 = torch.min(box1_x1, box1_x2), torch.max(box1_x1, box1_x2)
    box1_y1, box1_y2 = torch.min(box1_y1, box1_y2), torch.max(box1_y1, box1_y2)
    box2_x1, box2_x2 = torch.min(box2_x1, box2_x2), torch.max(box2_x1, box2_x2)
    box2_y1, box2_y2 = torch.min(box2_y1, box2_y2), torch.max(box2_y1, box2_y2)

    # Intersection area
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    # Intersection width and height
    w_inter = torch.clamp(inter_x2 - inter_x1, min=0)
    h_inter = torch.clamp(inter_y2 - inter_y1, min=0)

    # Intersection area
    inter_area = w_inter * h_inter

    # Union Area
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Add small epsilon to prevent division by zero
    area1 = torch.clamp(area1, min=eps)
    area2 = torch.clamp(area2, min=eps)

    union_area = area1 + area2 - inter_area + eps

    # IoU
    iou = inter_area / union_area

    if GIoU or DIoU or CIoU:
        # Enclosing box
        encl_x1 = torch.min(box1_x1, box2_x1)
        encl_y1 = torch.min(box1_y1, box2_y1)
        encl_x2 = torch.max(box1_x2, box2_x2)
        encl_y2 = torch.max(box1_y2, box2_y2)

        # Enclosing box dimensions
        encl_w = encl_x2 - encl_x1
        encl_h = encl_y2 - encl_y1

        # Enclosing box diagonal squared
        c2 = encl_w**2 + encl_h**2 + eps

        # GIoU
        if GIoU:
            encl_area = encl_w * encl_h + eps
            giou = iou - (encl_area - union_area) / encl_area
            return giou

        # DIoU/CIoU
        if DIoU or CIoU:
            # Center distance squared
            box1_cx = (box1_x1 + box1_x2) / 2
            box1_cy = (box1_y1 + box1_y2) / 2
            box2_cx = (box2_x1 + box2_x2) / 2
            box2_cy = (box2_y1 + box2_y2) / 2

            center_dist2 = (box1_cx - box2_cx)**2 + (box1_cy - box2_cy)**2

            # DIoU
            if DIoU:
                diou = iou - center_dist2 / c2
                return diou

            # CIoU
            if CIoU:
                # Width and height of boxes
                w1 = box1_x2 - box1_x1
                h1 = box1_y2 - box1_y1
                w2 = box2_x2 - box2_x1
                h2 = box2_y2 - box2_y1

                # Aspect ratio consistency term
                v = (4 / (math.pi**2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                alpha = v / (1 - iou + v + eps)

                ciou = iou - (center_dist2 / c2 + alpha * v)
                return ciou

    return iou


def process_predictions(predictions, anchors, strides, img_size, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=300):
    """
    Process raw YOLO predictions into boxes, objectness scores, and class probabilities
    Apply NMS to filter detections

    Args:
        predictions: List of tensors from model output for each scale
        anchors: List of anchor box shapes for each scale
        strides: List of strides for each feature map
        img_size: Original image size (int) - assuming square image
        conf_thres: Confidence threshold for filtering detections
        iou_thres: IoU threshold for NMS
        classes: List of classes to keep (filter by class)
        max_det: Maximum number of detections to return

    Returns:
        detections: List of tensors, one per image in batch
                   Each tensor: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
    """
    device = predictions[0].device
    batch_size = predictions[0].shape[0]
    num_classes = (predictions[0].shape[1] // len(anchors[0])) - 5

    # Output list, one entry per image in batch
    output = [[] for _ in range(batch_size)]

    # Process each detection scale
    for i, (pred, anchor_set, stride) in enumerate(zip(predictions, anchors, strides)):
        # Get grid size
        batch_size, _, grid_h, grid_w = pred.shape
        num_anchors = len(anchor_set)

        # Reshape prediction to [batch, anchors, grid_h, grid_w, box+obj+classes]
        pred = pred.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()

        # Generate grid coordinates
        grid_y, grid_x = torch.meshgrid([torch.arange(grid_h), torch.arange(grid_w)])
        grid = torch.stack([grid_x, grid_y], dim=2).to(device).float()
        grid = grid.view(1, 1, grid_h, grid_w, 2)

        # Get box coordinates and apply sigmoid to xy, exp to wh
        box_xy = (torch.sigmoid(pred[..., 0:2]) + grid) * stride  # center x, y

        # Fix the anchor reshaping to match the expected dimensions
        anchors_reshaped = anchor_set.view(1, num_anchors, 1, 1, 2).to(device)
        box_wh = torch.exp(pred[..., 2:4]) * anchors_reshaped * stride  # width, height

        # Convert to corner coordinates [x1, y1, x2, y2]
        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        boxes = torch.cat([box_x1y1, box_x2y2], dim=-1)

        # Get objectness and class probabilities
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_prob = torch.sigmoid(pred[..., 5:])

        # Reshape to [batch, -1, box_params]
        boxes = boxes.reshape(batch_size, -1, 4)
        obj_conf = obj_conf.reshape(batch_size, -1, 1)
        cls_prob = cls_prob.reshape(batch_size, -1, num_classes)

        # Get scores (obj_conf * cls_prob) for each class
        scores = obj_conf * cls_prob

        # Process each image in batch
        for batch_idx in range(batch_size):
            # Get detections for this scale and image
            # Filter by confidence threshold
            score, class_idx = scores[batch_idx].max(1)
            mask = score > conf_thres

            if mask.sum() == 0:
                continue  # No detections for this image at this scale

            # Get filtered detections
            filtered_boxes = boxes[batch_idx][mask]
            filtered_scores = score[mask]
            filtered_class_idx = class_idx[mask]

            # Create detection tensor [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
            detections = torch.cat([
                filtered_boxes,
                filtered_scores.unsqueeze(1),
                filtered_scores.unsqueeze(1),  # Use class score as obj_conf for simplicity
                filtered_class_idx.float().unsqueeze(1)
            ], dim=1)

            # Add to output for this image
            output[batch_idx].append(detections)

    # Process each image's detections with NMS
    for batch_idx in range(batch_size):
        # Combine detections from all scales
        if not output[batch_idx]:
            # No detections for this image
            output[batch_idx] = torch.zeros((0, 7), device=device)
            continue

        # Concatenate detections from all scales
        output[batch_idx] = torch.cat(output[batch_idx], dim=0)

        # Apply NMS
        # Filter by confidence again (in case threshold changed)
        confidence = output[batch_idx][:, 4] * output[batch_idx][:, 5]
        confidence_mask = confidence > conf_thres
        detections = output[batch_idx][confidence_mask]

        if len(detections) == 0:
            output[batch_idx] = torch.zeros((0, 7), device=device)
            continue

        # Sort by confidence (highest first)
        _, sort_idx = detections[:, 4].sort(descending=True)
        detections = detections[sort_idx]

        # Get boxes and scores
        boxes = detections[:, :4]
        scores = detections[:, 4] * detections[:, 5]
        class_ids = detections[:, 6]

        # Filter by class if specified
        if classes is not None:
            mask = torch.zeros_like(scores, dtype=torch.bool)
            for cls in classes:
                mask = mask | (class_ids == cls)
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            if len(boxes) == 0:
                output[batch_idx] = torch.zeros((0, 7), device=device)
                continue

        # Apply NMS - using an improved algorithm that guarantees progress
        keep = []
        remaining_indices = torch.arange(boxes.shape[0], device=device)

        while remaining_indices.shape[0] > 0:
            # Select the detection with highest confidence
            first_box_idx = remaining_indices[0]
            keep.append(first_box_idx.item())

            # If only one box remains, we're done
            if remaining_indices.shape[0] == 1:
                break

            # Calculate IoU of the first box with all other remaining boxes
            first_box = boxes[first_box_idx].unsqueeze(0)
            other_boxes = boxes[remaining_indices[1:]]
            ious = bbox_iou(first_box, other_boxes)

            # Find indices of boxes with IoU < threshold
            below_threshold_mask = ious < iou_thres
            below_threshold_mask = below_threshold_mask.squeeze()

            # Create a new tensor to store the indices to keep
            # Always keep the first element (1) and elements with IoU < threshold
            if below_threshold_mask.numel() > 0:  # Check if mask is not empty
                # Combine the first box index with other indices below threshold
                indices_to_keep = torch.cat([
                    remaining_indices[0:1],
                    remaining_indices[1:][below_threshold_mask]
                ])
            else:
                # If all remaining boxes have IoU >= threshold with the first box,
                # only keep the first box
                indices_to_keep = remaining_indices[0:1]

            # Remove the first index from the remaining indices list
            # This is crucial: even if no boxes pass the IoU threshold,
            # we still make progress by removing at least one box
            remaining_indices = remaining_indices[1:]

            # If there are any indices below threshold, keep them
            if below_threshold_mask.numel() > 0 and below_threshold_mask.any():
                remaining_indices = torch.cat([
                    remaining_indices,
                    indices_to_keep[1:]  # Skip the first element which we've already processed
                ])

        # Get kept detections
        keep = torch.tensor(keep, device=device)
        output[batch_idx] = detections[keep]

        # Limit to max_det
        if len(output[batch_idx]) > max_det:
            output[batch_idx] = output[batch_idx][:max_det]

    return output
