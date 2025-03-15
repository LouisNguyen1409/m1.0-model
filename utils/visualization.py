import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import torch

from utils.yolo_utils import bbox_iou


def plot_detections(image, detections, class_names, conf_threshold=0.25, figsize=(12, 12)):
    """
    Plot detection results on an image

    Args:
        image: PIL Image or numpy array
        detections: tensor of detections in format [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        class_names: list of class names
        conf_threshold: confidence threshold for displaying detections
        figsize: figure size for matplotlib

    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Convert PIL Image to numpy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)

    # Display image
    ax.imshow(image)

    # Generate color map for classes
    num_classes = len(class_names)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes))

    # Filter detections by confidence
    if detections.size(0) > 0:
        # Calculate final confidence (obj_conf * cls_conf)
        confidence = detections[:, 4] * detections[:, 5]
        mask = confidence > conf_threshold
        filtered_detections = detections[mask]

        # Draw each detection
        for det in filtered_detections:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det.cpu().numpy()
            cls_id = int(cls_id)

            # Calculate confidence
            conf = float(obj_conf * cls_conf)

            # Get class name and color
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            color = colors[cls_id]

            # Create rectangle patch
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )

            # Add rectangle to plot
            ax.add_patch(rect)

            # Add label
            label = f"{class_name} {conf:.2f}"
            ax.text(
                x1, y1 - 5, label,
                color='white', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.8, pad=2)
            )

    # Set axis off and tight layout
    ax.set_axis_off()
    plt.tight_layout()

    return fig, ax


def draw_boxes(image, detections, class_names, conf_threshold=0.25):
    """
    Draw detection boxes on an image using OpenCV

    Args:
        image: numpy array (H, W, C) in RGB format
        detections: tensor of detections in format [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        class_names: list of class names
        conf_threshold: confidence threshold for displaying detections

    Returns:
        image: numpy array with drawn boxes
    """
    # Make a copy of the image
    image_copy = image.copy()

    # Convert to BGR for OpenCV if needed (assuming input is RGB)
    if len(image_copy.shape) == 3 and image_copy.shape[2] == 3:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)

    # Generate color map for classes
    num_classes = len(class_names)
    colors = []
    for i in range(num_classes):
        # Generate a color for each class (using HSV color space for better distinction)
        hue = i / num_classes
        # Convert HSV to RGB
        rgb = plt.cm.hsv(hue)[:3]
        # Convert to BGR and scale to 0-255
        bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        colors.append(bgr)

    # Filter detections by confidence
    if detections.size(0) > 0:
        # Calculate final confidence (obj_conf * cls_conf)
        confidence = detections[:, 4] * detections[:, 5]
        mask = confidence > conf_threshold
        filtered_detections = detections[mask]

        # Draw each detection
        for det in filtered_detections:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det.cpu().numpy()

            # Convert to integers for drawing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)

            # Calculate confidence
            conf = float(obj_conf * cls_conf)

            # Get class name and color
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            color = colors[cls_id]

            # Draw rectangle
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image_copy,
                (x1, y1 - label_height - 5),
                (x1 + label_width, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                image_copy,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return image_copy


def evaluate_detections(predictions, targets, iou_threshold=0.5, conf_threshold=0.25):
    """
    Evaluate detection performance by calculating precision, recall, and mAP

    Args:
        predictions: list of tensors, each tensor contains detections for one image
                    format: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        targets: list of dictionaries, each dictionary contains ground truth for one image
                {'boxes': tensor of shape [num_boxes, 4], 'labels': tensor of shape [num_boxes]}
        iou_threshold: IoU threshold for a detection to be considered correct
        conf_threshold: confidence threshold for filtering detections

    Returns:
        metrics: dictionary with precision, recall, and mAP values
    """
    # Determine device from first prediction tensor
    device = predictions[0].device if predictions and len(
        predictions) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 0
    for target in targets:
        # Handle the case where target['labels'] could be either a tensor or a list
        if isinstance(target['labels'], torch.Tensor):
            # If it's a tensor, use max()
            if target['labels'].numel() > 0:  # Check if tensor is not empty
                num_classes = max(num_classes, target['labels'].max().item() + 1)
        elif isinstance(target['labels'], list):
            # If it's a list, use Python's max function
            if target['labels']:  # Check if list is not empty
                # Safely convert tensor elements to integers
                max_label = 0
                for l in target['labels']:
                    if isinstance(l, torch.Tensor):
                        if l.numel() == 1:  # Single element tensor
                            max_label = max(max_label, l.item())
                        else:  # Multi-element tensor
                            if l.numel() > 0:
                                max_label = max(max_label, l.max().item())
                    else:
                        try:
                            max_label = max(max_label, int(l))
                        except (TypeError, ValueError):
                            print(f"Warning: Could not convert {l} of type {type(l)} to int")

                num_classes = max(num_classes, max_label + 1)
        else:
            # If it's something else, try to get a reasonable value
            try:
                if hasattr(target['labels'], 'max'):
                    # If it has a max method, try to use it
                    num_classes = max(num_classes, target['labels'].max().item() + 1)
                else:
                    # Otherwise try Python's built-in max
                    num_classes = max(num_classes, max(target['labels']) + 1)
            except:
                print(f"Warning: Cannot determine max class from {type(target['labels'])}")

    # If we couldn't determine the number of classes, default to 1
    if num_classes == 0:
        print("Warning: Could not determine number of classes, defaulting to 1")
        num_classes = 1

    # Initialize statistics
    true_positives = [[] for _ in range(num_classes)]
    false_positives = [[] for _ in range(num_classes)]
    gt_count = [0] * num_classes

    # Process each image
    for pred, target in zip(predictions, targets):
        # Move target tensors to the same device as predictions
        if isinstance(target['boxes'], torch.Tensor):
            target['boxes'] = target['boxes'].to(device)

        if isinstance(target['labels'], torch.Tensor):
            target['labels'] = target['labels'].to(device)

        # Count ground truth objects for each class
        for label in target['labels']:
            # Convert label to integer if it's a tensor
            if isinstance(label, torch.Tensor):
                if label.numel() == 1:
                    label_idx = label.item()
                else:
                    # If tensor has multiple elements, we need to handle each one
                    for i in range(label.numel()):
                        label_idx = label[i].item()
                        gt_count[label_idx] += 1
                    continue  # Skip the final gt_count increment since we've already counted each
            else:
                label_idx = int(label)

            gt_count[label_idx] += 1

        # Filter predictions by confidence
        if pred.size(0) > 0:
            # Get confidence
            confidence = pred[:, 4] * pred[:, 5]

            # Sort by confidence (highest first)
            _, indices = confidence.sort(descending=True)
            pred = pred[indices]

            # Filter by confidence threshold
            mask = confidence[indices] > conf_threshold
            pred = pred[mask]

        # Create detection status array (0 = not matched, 1 = matched)
        gt_matched = torch.zeros(len(target['labels']), dtype=torch.bool, device=device)

        # Process detections
        for detection in pred:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = detection
            cls_id = int(cls_id)

            # Skip if no ground truth in this image for this class
            # Handle different types of target['labels']
            labels_match = False
            if isinstance(target['labels'], torch.Tensor):
                # For tensor labels
                if target['labels'].dim() == 1:  # 1D tensor
                    labels_match = cls_id in target['labels']
                else:  # Multi-dimensional tensor
                    for i in range(target['labels'].size(0)):
                        if cls_id == target['labels'][i].item():
                            labels_match = True
                            break
            else:
                # For list labels
                for l in target['labels']:
                    if isinstance(l, torch.Tensor):
                        if l.numel() == 1 and l.item() == cls_id:
                            labels_match = True
                            break
                        elif l.numel() > 1:
                            for i in range(l.numel()):
                                if l[i].item() == cls_id:
                                    labels_match = True
                                    break
                    elif int(l) == cls_id:
                        labels_match = True
                        break

            if not labels_match:
                false_positives[cls_id].append(1)
                true_positives[cls_id].append(0)
                continue

            # Get ground truth boxes for this class
            # Handle different types of target['labels']
            if isinstance(target['labels'], torch.Tensor):
                if target['labels'].dim() == 1:  # 1D tensor
                    # Create a boolean mask and use it directly for indexing
                    mask = target['labels'] == cls_id
                    if not mask.any():
                        # No ground truth boxes for this class
                        false_positives[cls_id].append(1)
                        true_positives[cls_id].append(0)
                        continue

                    # Get boxes directly using the mask
                    gt_boxes = target['boxes'][mask]

                    # Create a mapping from mask indices to original indices
                    gt_indices = mask.nonzero().flatten()
                else:  # Multi-dimensional tensor
                    indices_list = []
                    for i in range(target['labels'].size(0)):
                        if target['labels'][i].item() == cls_id:
                            indices_list.append(i)

                    if not indices_list:
                        # No ground truth boxes for this class
                        false_positives[cls_id].append(1)
                        true_positives[cls_id].append(0)
                        continue

                    gt_indices = torch.tensor(indices_list, device=device)
                    gt_boxes = torch.stack([target['boxes'][i] for i in indices_list])
            else:
                # For list labels
                indices_list = []
                for i, l in enumerate(target['labels']):
                    if isinstance(l, torch.Tensor):
                        if l.numel() == 1 and l.item() == cls_id:
                            indices_list.append(i)
                        elif l.numel() > 1:
                            for j in range(l.numel()):
                                if l[j].item() == cls_id:
                                    indices_list.append(i)
                                    break
                    elif int(l) == cls_id:
                        indices_list.append(i)

                if not indices_list:
                    # No ground truth boxes for this class
                    false_positives[cls_id].append(1)
                    true_positives[cls_id].append(0)
                    continue

                # Safely gather the boxes
                try:
                    # Check if we have tensors of different sizes
                    first_box = target['boxes'][indices_list[0]]
                    if all(target['boxes'][i].size() == first_box.size() for i in indices_list[1:]):
                        gt_boxes = torch.stack([target['boxes'][i] for i in indices_list])
                    else:
                        # Boxes have different sizes, can't stack directly
                        raise RuntimeError("Boxes have different sizes")
                except (IndexError, TypeError, RuntimeError):
                    # If we encounter an error, handle boxes individually
                    box_list = []
                    valid_indices = []

                    for idx, i in enumerate(indices_list):
                        try:
                            box = target['boxes'][i]

                            # Check if this is a properly shaped box (should be 2D with 4 columns for x1,y1,x2,y2)
                            if box.dim() == 2 and box.size(1) == 4:
                                # Each box might have multiple rows, we need to select one row
                                # For simplicity, we'll take the first box
                                box_single = box[0].unsqueeze(0)  # Take first box and ensure it's 2D
                                box_list.append(box_single)
                                valid_indices.append(i)
                            elif box.dim() == 1 and box.size(0) == 4:
                                # Single box with correct shape
                                box_list.append(box.unsqueeze(0))  # Ensure it's 2D
                                valid_indices.append(i)
                        except (IndexError, TypeError, RuntimeError):
                            continue

                    if not box_list:
                        # Couldn't get any valid boxes
                        false_positives[cls_id].append(1)
                        true_positives[cls_id].append(0)
                        continue

                    try:
                        gt_boxes = torch.cat(box_list, dim=0)  # Use cat instead of stack
                        # Update gt_indices to match the valid boxes
                        gt_indices = torch.tensor(valid_indices, device=device)
                    except RuntimeError:
                        # If cat still fails, just use the first box as a fallback
                        gt_boxes = box_list[0]
                        gt_indices = torch.tensor([valid_indices[0]], device=device)

            # If all already matched, this is a false positive
            if len(gt_indices) == 0 or gt_matched[gt_indices].all():
                false_positives[cls_id].append(1)
                true_positives[cls_id].append(0)
                continue

            # Get IoU with all ground truth boxes of this class
            detection_box = detection[:4].unsqueeze(0)  # [1, 4]

            # We already have gt_boxes properly set up above
            ious = bbox_iou(detection_box, gt_boxes)

            # Get best IoU and corresponding index
            best_iou, best_idx = ious.max(1)

            # Extract scalar values from tensors
            best_iou = best_iou.item()  # Convert tensor to scalar
            best_idx = best_idx.item()  # Convert tensor to scalar

            # Get the actual ground truth index
            gt_idx = gt_indices[best_idx].item() if isinstance(
                gt_indices[best_idx], torch.Tensor) else int(gt_indices[best_idx])

            # If IoU > threshold and not already matched, this is a true positive
            if best_iou > iou_threshold and not gt_matched[gt_idx]:
                true_positives[cls_id].append(1)
                false_positives[cls_id].append(0)
                gt_matched[gt_idx] = True
            else:
                true_positives[cls_id].append(0)
                false_positives[cls_id].append(1)

    # Calculate metrics for each class
    precision = []
    recall = []
    ap = []

    for cls_id in range(num_classes):
        # Skip if no ground truth for this class
        if gt_count[cls_id] == 0:
            precision.append(0)
            recall.append(0)
            ap.append(0)
            continue

        # Convert to numpy arrays
        tp = np.array(true_positives[cls_id])
        fp = np.array(false_positives[cls_id])

        # Skip if no detections for this class
        if len(tp) == 0:
            precision.append(0)
            recall.append(0)
            ap.append(0)
            continue

        # Accumulate true positives and false positives
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Calculate precision and recall
        prec = tp_cumsum / (tp_cumsum + fp_cumsum)
        rec = tp_cumsum / gt_count[cls_id]

        # Append 0 precision at 0 recall
        prec = np.concatenate(([0], prec))
        rec = np.concatenate(([0], rec))

        # Ensure precision decreases as recall increases
        for i in range(len(prec) - 1, 0, -1):
            prec[i-1] = max(prec[i-1], prec[i])

        # Find points where recall increases
        i = np.where(rec[1:] != rec[:-1])[0]

        # Calculate AP using all points with precision interpolation
        cls_ap = np.sum((rec[i+1] - rec[i]) * prec[i+1])

        # Append metrics
        precision.append(prec[-1])
        recall.append(rec[-1])
        ap.append(cls_ap)

    # Calculate mAP
    mAP = np.mean(ap)

    return {
        'precision': precision,
        'recall': recall,
        'ap': ap,
        'mAP': mAP
    }
