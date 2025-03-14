import torch
import torch.nn as nn
import math

from utils.yolo_utils import bbox_iou


class YOLOD11Loss(nn.Module):
    def __init__(self, num_classes, anchors=None):
        super().__init__()
        self.num_classes = num_classes

        # Default anchors if not provided
        if anchors is None:
            self.anchors = [
                # Small objects (for high resolution feature map)
                torch.tensor([[10, 13], [16, 30], [33, 23]]),
                # Medium objects (for medium resolution feature map)
                torch.tensor([[30, 61], [62, 45], [59, 119]]),
                # Large objects (for low resolution feature map)
                torch.tensor([[116, 90], [156, 198], [373, 326]])
            ]
        else:
            self.anchors = anchors

        # Corresponding strides for feature maps
        self.strides = [8, 16, 32]  # Strides for the different detection scales

        # Loss functions
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

        # Constants
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
        self.lambda_iou = 1.0

        # Use CIoU loss for better bounding box regression
        self.use_ciou = True

        # Balance weights for each detection scale
        self.balance = [4.0, 1.0, 0.4]

    def forward(self, predictions, targets):
        """
        Calculate YOLO loss

        Args:
            predictions: tuple of (small_pred, medium_pred, large_pred)
            targets: dictionary with ground truth boxes and labels

        Returns:
            loss: total loss value
            loss_components: loss components dictionary
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        # Initialize loss components
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)

        # Process each scale (small, medium, large objects)
        for scale_idx, (pred, anchor_set, stride) in enumerate(zip(predictions, self.anchors, self.strides)):
            # Get grid dimensions
            batch_size, num_channels, grid_h, grid_w = pred.shape
            num_anchors = len(anchor_set)
            num_attrib = 5 + self.num_classes  # x, y, w, h, obj, classes

            # Reshape prediction to [batch, anchors, grid_h, grid_w, num_classes + 5]
            pred = pred.view(batch_size, num_anchors, num_attrib, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [b, a, h, w, attrib]

            # Get prediction components
            pred_box = pred[..., :4]  # [x, y, w, h]
            pred_obj = pred[..., 4:5]  # objectness
            pred_cls = pred[..., 5:]  # class probabilities

            # Create grid for x, y coordinates
            grid_y, grid_x = torch.meshgrid([torch.arange(grid_h), torch.arange(grid_w)])
            grid = torch.stack([grid_x, grid_y], dim=2).to(device).float()
            grid = grid.view(1, 1, grid_h, grid_w, 2)

            # Apply sigmoid to x, y to get offset within cell, then add grid coordinates
            pred_xy = torch.sigmoid(pred_box[..., :2]) + grid
            # Apply exp to w, h and multiply by anchor dimensions
            pred_wh = torch.exp(pred_box[..., 2:4]) * anchor_set.view(1, num_anchors, 1, 1, 2).to(device)

            # Scale to absolute coordinates
            pred_xy = pred_xy * stride
            pred_wh = pred_wh * stride

            # Convert to [x_center, y_center, width, height] format
            pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1)

            # Initialize tensors for target assignments
            obj_mask = torch.zeros_like(pred_obj)
            noobj_mask = torch.ones_like(pred_obj)
            target_box = torch.zeros_like(pred_boxes)
            target_cls = torch.zeros_like(pred_cls)

            # Process targets for each batch
            for batch_idx in range(batch_size):
                # Get ground truth for this batch
                gt_boxes = targets['boxes'][batch_idx]  # [num_gt, 4] in [x1, y1, x2, y2] format
                gt_labels = targets['labels'][batch_idx]  # [num_gt]

                # Skip if no ground truth
                if gt_boxes.size(0) == 0:
                    continue

                # Convert ground truth to [x_center, y_center, width, height] format
                gt_xy = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # center points
                gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]  # width, height
                gt_cxcywh = torch.cat([gt_xy, gt_wh], dim=1)

                # Calculate grid cell for each ground truth
                gt_xy_grid = gt_xy / stride
                grid_i = torch.clamp(gt_xy_grid[:, 1].long(), min=0, max=grid_h-1)
                grid_j = torch.clamp(gt_xy_grid[:, 0].long(), min=0, max=grid_w-1)

                # For each ground truth box
                for gt_idx in range(gt_boxes.size(0)):
                    i, j = grid_i[gt_idx], grid_j[gt_idx]
                    gt_box = gt_cxcywh[gt_idx]
                    gt_label = gt_labels[gt_idx]

                    # Calculate best matching anchor by IoU
                    anchors_wh = anchor_set.to(device) * stride

                    # Create anchor boxes and ground truth boxes
                    anchor_boxes = torch.zeros(num_anchors, 4, device=device)
                    anchor_boxes[:, 2:4] = anchors_wh  # Only set w, h for anchors at origin

                    # Calculate IoU between each anchor and ground truth box
                    ious = []
                    for a_idx in range(num_anchors):
                        # Calculate ratio of widths and heights
                        w_ratio = gt_wh[gt_idx, 0] / anchors_wh[a_idx, 0]
                        h_ratio = gt_wh[gt_idx, 1] / anchors_wh[a_idx, 1]

                        # Use shape metrics (width/height ratios) as simplified IoU approximation
                        shape_metric = torch.max(w_ratio, 1/w_ratio) * torch.max(h_ratio, 1/h_ratio)
                        ious.append(1 / shape_metric)

                    # Find best anchor
                    best_anchor_idx = torch.argmax(torch.tensor(ious))

                    # Assign ground truth to best matching anchor
                    obj_mask[batch_idx, best_anchor_idx, i, j] = 1
                    noobj_mask[batch_idx, best_anchor_idx, i, j] = 0

                    # Target box: convert to grid units for x, y
                    tx = gt_xy_grid[gt_idx, 0] - j  # x offset within cell (0-1)
                    ty = gt_xy_grid[gt_idx, 1] - i  # y offset within cell (0-1)
                    tw = torch.log(gt_wh[gt_idx, 0] / (anchors_wh[best_anchor_idx, 0] + 1e-16))
                    th = torch.log(gt_wh[gt_idx, 1] / (anchors_wh[best_anchor_idx, 1] + 1e-16))

                    # Set target values
                    target_box[batch_idx, best_anchor_idx, i, j] = torch.tensor([tx, ty, tw, th], device=device)
                    target_cls[batch_idx, best_anchor_idx, i, j, gt_label] = 1

            # Calculate box loss using CIoU
            if self.use_ciou and obj_mask.sum() > 0:
                # Get predicted boxes where objectness=1
                pred_boxes_pos = pred_boxes[obj_mask.bool()]

                # Get target boxes
                target_boxes_pos = torch.zeros_like(pred_boxes_pos)

                # Get indices where obj_mask is 1
                indices = obj_mask.nonzero(as_tuple=True)

                # Convert target box from tx, ty, tw, th to actual x, y, w, h
                target_tx = target_box[..., 0][indices]
                target_ty = target_box[..., 1][indices]
                target_tw = target_box[..., 2][indices]
                target_th = target_box[..., 3][indices]

                # Calculate grid positions
                grid_x = indices[3].float()
                grid_y = indices[2].float()

                # Convert to absolute coordinates
                tx = (grid_x + target_tx) * stride
                ty = (grid_y + target_ty) * stride
                tw = torch.exp(target_tw) * anchor_set[indices[1], 0].to(device) * stride
                th = torch.exp(target_th) * anchor_set[indices[1], 1].to(device) * stride

                # Set target boxes as center format
                target_boxes_pos[:, 0] = tx
                target_boxes_pos[:, 1] = ty
                target_boxes_pos[:, 2] = tw
                target_boxes_pos[:, 3] = th

                # Calculate CIoU loss
                ciou_loss = 1 - bbox_iou(
                    pred_boxes_pos,
                    target_boxes_pos,
                    x1y1x2y2=False,
                    CIoU=True
                )
                box_loss += ciou_loss.mean() * self.lambda_iou
            else:
                # Use MSE loss for box coordinates as fallback
                box_loss += self.mse(
                    pred_box[obj_mask.bool()],
                    target_box[obj_mask.bool()]
                ).sum() * self.lambda_box

            # Objectness loss (binary cross entropy)
            obj_loss_pos = self.bce(
                pred_obj[obj_mask.bool()],
                obj_mask[obj_mask.bool()]
            ).sum()

            obj_loss_neg = self.bce(
                pred_obj[noobj_mask.bool()],
                obj_mask[noobj_mask.bool()]
            ).sum()

            # Apply balance factor for this scale
            obj_loss += (obj_loss_pos + 0.5 * obj_loss_neg) * self.lambda_obj * self.balance[scale_idx]

            # Classification loss
            if obj_mask.sum() > 0:
                cls_loss += self.bce(
                    pred_cls[obj_mask.bool()],
                    target_cls[obj_mask.bool()]
                ).sum() * self.lambda_cls

        # Normalize losses by batch size
        box_loss /= batch_size
        obj_loss /= batch_size
        cls_loss /= batch_size

        # Calculate total loss
        total_loss = box_loss + obj_loss + cls_loss

        # Loss components dictionary
        loss_components = {
            'total': total_loss.item(),
            'box': box_loss.item(),
            'obj': obj_loss.item(),
            'cls': cls_loss.item()
        }

        return total_loss, loss_components
