import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou
import math
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('YOLOD11')


def xywh2xyxy(x):
    """Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2] format"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    """Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h] format"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Non-Maximum Suppression to filter detections

    Args:
        predictions: tensor of shape (batch, num_boxes, 5+num_classes)
                    where 5+num_classes is [x1, y1, x2, y2, obj_conf, class1_conf, class2_conf, ...]
        conf_thres: confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: maximum number of detections per image

    Returns:
        list of detections, each item is a tensor of shape (num_det, 6)
                    where 6 is [x1, y1, x2, y2, obj_conf, class_id]
    """
    # Ensure predictions is a list
    if isinstance(predictions, (list, tuple)):
        # Process each scale separately and concatenate
        # Assuming each scale output has shape (batch, anchors*(5+num_classes), grid_h, grid_w)
        batch_size = predictions[0].shape[0]
        processed_preds = []

        for pred in predictions:
            # reshape to (batch, -1, 5+num_classes)
            pred = pred.view(batch_size, -1, pred.shape[1]//3)
            processed_preds.append(pred)

        # Concatenate along anchor/grid dimension
        predictions = torch.cat(processed_preds, dim=1)

    nc = predictions.shape[2] - 5  # number of classes

    # Initial filtering based on confidence threshold
    try:
        xc = predictions[..., 4] > conf_thres  # candidates with conf > threshold
    except:
        # Handle unexpected prediction format
        logger.error(f"Unexpected prediction shape: {predictions.shape}")
        return [None] * predictions.shape[0]

    # Settings
    min_wh, max_wh = 2, 4096  # min and max box width and height

    # For each image in batch
    output = [None] * predictions.shape[0]
    for xi, x in enumerate(predictions):
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain, process next image
        if not x.shape[0]:
            continue

        # Compute confidence scores
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        x = x[x[:, 4].argsort(descending=True)[:max_det]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        try:
            i = torch.ops.torchvision.nms(boxes, scores, iou_thres)  # NMS
        except Exception as e:
            logger.error(f"Error in NMS operation: {e}")
            continue

        output[xi] = x[i]

    return output


def compute_ap(recall, precision):
    """Compute the average precision for a single class"""
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Calculate area under PR curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where recall changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def calculate_metrics(detections, targets, iou_thres=0.5):
    """
    Calculate precision, recall, and mAP for the detections

    Args:
        detections: list of tensors, each tensor is (n, 6) where 6 is [x1, y1, x2, y2, conf, class]
        targets: tensor of shape (m, 6) where 6 is [batch_idx, class, x, y, w, h]
        iou_thres: IoU threshold for a true positive

    Returns:
        dict with metrics: precision, recall, mAP
    """
    # Add some debug logging
    logger.info(f"Calculating metrics with {len(detections)} detections and {len(targets)} targets")
    print(f"Calculating metrics with {len(detections)} detections and {len(targets)} targets")
    
    # Count total detections
    total_dets = sum(len(d) if d is not None else 0 for d in detections)
    logger.info(f"Total detection boxes: {total_dets}")
    print(f"Total detection boxes: {total_dets}")
    
    # Check targets shape and content
    print(f"Targets type: {type(targets)}")
    if isinstance(targets, torch.Tensor):
        print(f"Targets shape: {targets.shape}")
        print(f"Targets data type: {targets.dtype}")
        if len(targets) > 0:
            print(f"First few targets: {targets[:5]}")
            # Count number of targets by class
            classes, counts = targets[:, 1].unique(return_counts=True)
            print(f"Target classes: {classes.tolist()}")
            print(f"Class counts: {counts.tolist()}")
    
    # Handle empty cases
    if len(targets) == 0:
        logger.info("No targets - returning zero metrics")
        print("No targets - returning zero metrics")
        return {'precision': 0, 'recall': 0, 'mAP': 0}

    # Group targets by image (batch index)
    targets_by_image = {}
    for target in targets:
        batch_idx = int(target[0])
        if batch_idx not in targets_by_image:
            targets_by_image[batch_idx] = []
        # Convert target from [batch_idx, class, x, y, w, h] to [x1, y1, x2, y2, class]
        box = xywh2xyxy(target[2:6].unsqueeze(0)).squeeze(0)
        cls = target[1]
        targets_by_image[batch_idx].append(torch.cat((box, cls.view(1))))
    
    logger.info(f"Targets grouped into {len(targets_by_image)} images")
    print(f"Targets grouped into {len(targets_by_image)} images")
    print(f"Batch indices: {list(targets_by_image.keys())}")
    for batch_idx, targets_list in list(targets_by_image.items())[:3]:  # Show first 3 images
        print(f"Batch {batch_idx}: {len(targets_list)} targets")
        if len(targets_list) > 0:
            print(f"  First target: {targets_list[0]}")

    stats = []
    for i, det in enumerate(detections):
        # Skip if no detections for this image
        if det is None or len(det) == 0:
            continue

        # Get targets for this image
        targets_i = targets_by_image.get(i, [])
        if not targets_i:
            # No targets for this image
            if len(det):
                # All detections are false positives
                logger.debug(f"Image {i}: {len(det)} detections but no targets - all false positives")
                print(f"Image {i}: {len(det)} detections but no targets - all false positives")
                stats.append((torch.zeros(len(det)), torch.ones(len(det)),
                             torch.zeros(len(det)), det[:, 5].cpu(), det[:, 4].cpu()))
            continue

        # Convert list of targets to tensor
        try:
            targets_i = torch.stack(targets_i) if len(targets_i) > 0 else torch.zeros((0, 5))
            print(f"Image {i}: {len(targets_i)} targets after stacking, shape: {targets_i.shape}")
        except Exception as e:
            print(f"Error stacking targets for image {i}: {e}")
            print(f"Target types: {[type(t) for t in targets_i]}")
            print(f"Target shapes: {[t.shape for t in targets_i]}")
            targets_i = torch.zeros((0, 5))

        # Extract detections and targets
        pred_boxes = det[:, :4].cpu()
        pred_cls = det[:, 5].cpu()
        pred_conf = det[:, 4].cpu()

        target_boxes = targets_i[:, :4]
        target_cls = targets_i[:, 4]
        
        logger.debug(f"Image {i}: {len(det)} detections, {len(targets_i)} targets")
        print(f"Image {i}: {len(det)} detections, {len(targets_i)} targets")
        if i < 3:  # Only for first 3 images
            print(f"  Pred classes: {pred_cls.unique().tolist()}")
            print(f"  Target classes: {target_cls.unique().tolist()}")

        # Assign true positives, false positives, and false negatives
        correct = torch.zeros(len(det))
        detected = []

        # For each detection, find the best matching target
        for j, (*pred_box, _, _) in enumerate(det):
            # Extract class for this detection
            cls = pred_cls[j]

            # Find targets with matching class
            try:
                targets_cls = target_cls == cls
                if not targets_cls.any():
                    continue

                # Find targets with matching class
                target_boxes_cls = target_boxes[targets_cls]
                target_ids = torch.nonzero(targets_cls).squeeze(1)

                # Calculate IoUs
                ious = box_iou(pred_box.unsqueeze(0), target_boxes_cls)
                
                # Log the max IoU
                if len(ious) > 0:
                    logger.debug(f"Det {j} (class {int(cls)}): max IoU = {ious.max().item():.4f}")
                    if i < 3 and j < 5:  # Only for first 3 images, first 5 detections
                        print(f"  Det {j} (class {int(cls)}): max IoU = {ious.max().item():.4f}")

                # Find the best match
                if ious.max() >= iou_thres:
                    max_iou, max_i = ious.max(1)
                    max_i = max_i.item()
                    i_target = target_ids[max_i].item()

                    # Ensure this target hasn't been assigned already
                    if i_target not in detected:
                        correct[j] = 1
                        detected.append(i_target)
                        logger.debug(f"Det {j} (class {int(cls)}) is a true positive with IoU {max_iou.item():.4f}")
                        if i < 3 and j < 5:  # Only for first 3 images, first 5 detections
                            print(f"  Det {j} (class {int(cls)}) is a true positive with IoU {max_iou.item():.4f}")
            except Exception as e:
                print(f"Error processing detection {j} in image {i}: {e}")

        # Add batch results
        try:
            # Check if target_cls can be repeated
            if len(pred_cls) > 0:
                repeated_target_cls = target_cls.repeat(len(pred_cls))
                stats.append((correct, torch.ones_like(correct), pred_cls, repeated_target_cls, pred_conf))
                if i < 3:  # Only for first 3 images
                    print(f"  Added stats: TP={correct.sum().item()}/{len(correct)}, classes={pred_cls.unique().tolist()}")
        except Exception as e:
            print(f"Error adding stats for image {i}: {e}")
            # Try alternative approach if repeating fails
            if len(pred_cls) > 0 and len(target_cls) > 0:
                try:
                    # Create a tensor of target_cls with the same length as pred_cls
                    repeated_cls = torch.zeros_like(pred_cls)
                    if len(target_cls) > 0:
                        # Fill with the first target class or 0
                        repeated_cls.fill_(target_cls[0] if len(target_cls) > 0 else 0)
                    stats.append((correct, torch.ones_like(correct), pred_cls, repeated_cls, pred_conf))
                    print(f"  Used alternative approach for adding stats")
                except Exception as e2:
                    print(f"  Alternative approach also failed: {e2}")

    # Calculate metrics
    if not stats:
        logger.info("No valid statistics collected - returning zero metrics")
        print("No valid statistics collected - returning zero metrics")
        return {'precision': 0, 'recall': 0, 'mAP': 0}

    # Concatenate stats from all images
    try:
        print(f"Concatenating stats from {len(stats)} images")
        for i, stat in enumerate(stats):
            print(f"  Stats {i}: {[s.shape for s in stat]}")
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        print(f"Stats after concatenation: {[s.shape for s in stats]}")
    except Exception as e:
        print(f"Error concatenating stats: {e}")
        print(f"Stats shapes: {[[s.shape for s in stat] for stat in stats]}")
        return {'precision': 0, 'recall': 0, 'mAP': 0}

    # Handle empty stats case
    if len(stats[0]) == 0:
        logger.info("Empty statistics array - returning zero metrics")
        print("Empty statistics array - returning zero metrics")
        return {'precision': 0, 'recall': 0, 'mAP': 0}

    tp, fp, p, t, conf = stats
    
    logger.info(f"Evaluation stats: {len(tp)} total detections, {tp.sum()} true positives, {fp.sum()} false positives")
    print(f"Evaluation stats: {len(tp)} total detections, {tp.sum()} true positives, {fp.sum()} false positives")
    print(f"Target classes present: {np.unique(t)}")
    print(f"Predicted classes present: {np.unique(p)}")

    # Calculate precision and recall
    precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
    recall = tp.sum() / (t.shape[0] + 1e-10)
    
    logger.info(f"Raw precision: {precision}, recall: {recall}")
    print(f"Raw precision: {precision}, recall: {recall}")

    # Calculate mAP
    try:
        unique_classes = torch.unique(torch.cat((torch.tensor(p), torch.tensor(t))).int())
        print(f"Unique classes for mAP calculation: {unique_classes.tolist()}")
        ap = []
        for c in unique_classes:
            c_int = int(c)
            i = (p == c_int)
            n_gt = (t == c_int).sum()
            n_p = i.sum()
            
            logger.debug(f"Class {c_int}: {n_p} predictions, {n_gt} ground truth")
            print(f"Class {c_int}: {n_p} predictions, {n_gt} ground truth")

            if n_p == 0 or n_gt == 0:
                continue

            # Sort by confidence
            i = i[conf.argsort(descending=True)]
            tp_c, fp_c = tp[i], fp[i]

            # Calculate cumulative false positives and true positives
            fpc = fp_c.cumsum()
            tpc = tp_c.cumsum()

            # Calculate recall and precision
            recall_curve = tpc / (n_gt + 1e-10)
            precision_curve = tpc / (tpc + fpc + 1e-10)

            # Calculate average precision
            ap_value = compute_ap(recall_curve, precision_curve)
            ap.append(ap_value)
            logger.debug(f"Class {c_int}: AP = {ap_value:.4f}")
            print(f"Class {c_int}: AP = {ap_value:.4f}")

        mAP = np.mean(ap) if ap else 0
    except Exception as e:
        print(f"Error calculating mAP: {e}")
        mAP = 0
    
    logger.info(f"Final metrics - mAP: {mAP:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
    print(f"Final metrics - mAP: {mAP:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")

    return {'precision': precision, 'recall': recall, 'mAP': mAP}


def visualize_detections(img, detections, class_names=None, conf_thres=0.25):
    """
    Visualize detections on an image

    Args:
        img: tensor of shape (3, H, W)
        detections: tensor of shape (n, 6) where 6 is [x1, y1, x2, y2, conf, class]
        class_names: list of class names

    Returns:
        matplotlib figure
    """
    # Convert tensor to numpy
    img = img.permute(1, 2, 0).cpu().numpy()

    # Denormalize if needed
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Create figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw each detection
    if detections is not None and len(detections) > 0:
        for *box, conf, cls in detections:
            if conf < conf_thres:
                continue

            # Get coordinates
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # Create rectangle
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add label
            cls_id = int(cls)
            label = f"{class_names[cls_id] if class_names and cls_id < len(class_names) else cls_id}: {conf:.2f}"
            ax.text(x1, y1, label, bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('off')
    return fig


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, best_map, scheduler=None, ema=None, filename='checkpoint.pth'):
    """Save model checkpoint"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_map': best_map
        }

        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()

        if ema is not None:
            checkpoint['ema'] = ema.state_dict()

        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def load_checkpoint(model, optimizer=None, scheduler=None, ema=None, filename='checkpoint.pth'):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(filename, map_location='cpu')

        model.load_state_dict(checkpoint['model'])

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        if ema is not None and 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])

        logger.info(f"Checkpoint loaded from {filename}")
        return checkpoint.get('epoch', 0), checkpoint.get('best_map', 0)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0, 0


class ModelEMA:
    """
    Model Exponential Moving Average
    Keeps a moving average of model weights for better validation results
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = model.eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        self.momentum = 1.0  # momentum at start

        # Deep copy of model
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            setattr(self.ema, k, v)
            
    def state_dict(self):
        """Return the state dictionary of the EMA model"""
        return self.ema.state_dict()
        
    def load_state_dict(self, state_dict):
        """Load state dictionary into the EMA model"""
        self.ema.load_state_dict(state_dict)


def get_lr_scheduler(optimizer, lr_schedule='cosine', epochs=300, min_lr=1e-6):
    """Get learning rate scheduler"""
    if lr_schedule == 'cosine':
        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr)
    elif lr_schedule == 'linear':
        # Linear warmup + cosine decay
        def lr_lambda(x):
            if x < 10:  # First 10 epochs as warmup
                return x / 10
            else:
                # Cosine decay after warmup
                return 0.5 * (1 + math.cos(math.pi * (x - 10) / (epochs - 10)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        # Step decay
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.1)

    return scheduler


def debug_model_shapes(model, img_size=640, device='cpu'):
    """Debug model output shapes with a dummy input"""
    model.eval()
    dummy_input = torch.zeros(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        try:
            outputs = model(dummy_input)

            logger.info("Model output shapes:")
            if isinstance(outputs, (list, tuple)):
                for i, output in enumerate(outputs):
                    logger.info(f"  Output {i}: {output.shape}")
            else:
                logger.info(f"  Output: {outputs.shape}")

            return True
        except Exception as e:
            logger.error(f"Error running model forward pass: {e}")
            return False


def check_dataset(dataloader, class_names, num_batches=3):
    """Verify that the dataset is loading correctly"""
    for batch_i, (imgs, targets, paths) in enumerate(dataloader):
        if batch_i >= num_batches:
            break

        logger.info(f"Batch {batch_i}:")
        logger.info(f"  Images shape: {imgs.shape}")
        logger.info(f"  Targets shape: {targets.shape}")
        logger.info(f"  Number of targets: {len(targets)}")

        if len(targets) > 0:
            # Print a few target samples
            logger.info(f"  Target samples (batch_idx, class, x, y, w, h):")
            for i in range(min(5, len(targets))):
                target = targets[i]
                class_idx = int(target[1])
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"Unknown ({class_idx})"
                logger.info(
                    f"    {i}: {target[0]:.0f}, {class_name}, {target[2]:.4f}, {target[3]:.4f}, {target[4]:.4f}, {target[5]:.4f}")
        else:
            logger.info("  No targets in this batch")
