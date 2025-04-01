import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, anchors, strides, img_size=640):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors  # List of anchors for each scale [small, medium, large]
        self.strides = strides  # Strides for each scale [8, 16, 32] for example
        self.num_anchors = len(anchors[0]) // 2  # Number of anchors per grid
        self.img_size = img_size

        # Hyper parameters
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_cls = 0.5
        self.lambda_box = 1.0

        # BCE loss for objectness (confidence that object exists)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # BCE loss for classification
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')

        # MSE loss for box regression
        self.mse = nn.MSELoss(reduction='none')

        # Generate grid cells for each detection scale
        self.grid_cells = [self._make_grid(int(img_size / stride)) for stride in strides]

    def _make_grid(self, size):
        """Generate grid cells for a given size"""
        y, x = torch.meshgrid([torch.arange(size), torch.arange(size)], indexing='ij')
        grid = torch.stack((x, y), dim=2).float()
        return grid.view(1, size, size, 1, 2)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: list of tensors [small_pred, medium_pred, large_pred]
                         each with shape (batch_size, num_anchors*(5+num_classes), grid_h, grid_w)
            targets: tensor of shape (num_targets, 6) where each target is
                    [batch_idx, class_idx, x, y, w, h]
        """
        # Ensure predictions is a list
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        device = predictions[0].device
        total_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)

        # Handle empty targets case
        if targets.shape[0] == 0:
            # Just compute objectness loss on all predictions
            for i, pred in enumerate(predictions):
                # Reshape prediction to get objectness predictions
                batch_size = pred.shape[0]
                num_anchors = self.num_anchors
                grid_size = pred.shape[2]  # Assuming square grid

                pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, grid_size, grid_size)
                pred = pred.permute(0, 1, 3, 4, 2).contiguous()

                # Get objectness predictions
                pred_obj = pred[..., 4]

                # All targets are background, so objectness should be 0
                obj_loss += self.lambda_noobj * self.bce(pred_obj, torch.zeros_like(pred_obj)).sum()

            total_loss = obj_loss
            return total_loss, {
                'loss': total_loss.item(),
                'cls_loss': 0.0,
                'box_loss': 0.0,
                'obj_loss': obj_loss.item()
            }

        # Continue with normal case
        # Build target tensors
        target_boxes = targets[:, 2:6]  # x, y, w, h
        target_cls = targets[:, 1].long()  # class
        target_imgs = targets[:, 0].long()  # image index in batch

        # Process each scale
        for i, pred in enumerate(predictions):
            stride = self.strides[i]
            anchors = torch.tensor(self.anchors[i], device=device).view(-1, 2) / stride
            grid_size = pred.shape[2]  # Assuming square grid

            # Reshape prediction to (batch, num_anchors, grid_size, grid_size, 5+num_classes)
            # Where 5+num_classes is [x, y, w, h, obj, class1, class2, ...]
            batch_size = pred.shape[0]
            num_anchors = self.num_anchors
            pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()

            # Sigmoid activation for xy, objectness and class scores
            x = torch.sigmoid(pred[..., 0])  # Center x
            y = torch.sigmoid(pred[..., 1])  # Center y
            w = pred[..., 2]  # Width
            h = pred[..., 3]  # Height
            pred_obj = pred[..., 4]  # Objectness
            pred_cls = pred[..., 5:]  # Class predictions

            # Get grid cells
            grid = self.grid_cells[i].to(device)
            if grid.shape[1:3] != (grid_size, grid_size):
                grid = self._make_grid(grid_size).to(device)
                self.grid_cells[i] = grid

            # Add offset to xy predictions (x,y predictions are offsets from cell corner)
            pred_boxes = torch.zeros_like(pred[..., :4]).to(device)
            pred_boxes[..., 0] = x + grid[..., 0]  # x center
            pred_boxes[..., 1] = y + grid[..., 1]  # y center

            # Apply anchor dimensions to width and height predictions
            pred_boxes[..., 2] = torch.exp(w) * anchors[:, 0].view(1, -1, 1, 1, 1)  # width
            pred_boxes[..., 3] = torch.exp(h) * anchors[:, 1].view(1, -1, 1, 1, 1)  # height

            # Scale back to original image size
            pred_boxes = pred_boxes * stride

            # Initialize target masks
            obj_mask = torch.zeros_like(pred_obj)
            cls_mask = torch.zeros_like(pred_cls)
            target_boxes_grid = torch.zeros_like(pred_boxes)

            # Assign targets to anchors
            if len(targets) > 0:
                # Scale target boxes to grid size
                target_boxes_scaled = target_boxes.clone()
                target_boxes_scaled[:, :2] = target_boxes_scaled[:, :2] * self.img_size / stride  # center xy
                target_boxes_scaled[:, 2:4] = target_boxes_scaled[:, 2:4] * self.img_size / stride  # width, height

                # For each target find the best matching anchor
                target_wh = target_boxes_scaled[:, 2:4]
                anchor_wh = anchors.view(-1, 2)

                # Calculate IoU between targets and anchors
                intersect_wh = torch.min(target_wh[:, None, :], anchor_wh[None, :, :])
                intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]
                target_area = target_wh[:, 0] * target_wh[:, 1]
                anchor_area = anchor_wh[:, 0] * anchor_wh[:, 1]
                iou = intersect_area / (target_area[:, None] + anchor_area[None, :] - intersect_area + 1e-6)

                # Find the best anchor for each target
                best_anchors = torch.max(iou, dim=1)[1]

                # Assign targets to grid cells and anchors
                for j, target_idx in enumerate(range(len(targets))):
                    # Get the best anchor for this target
                    anchor_idx = best_anchors[j]

                    # Get target image index within the batch
                    img_idx = target_imgs[target_idx]

                    # Skip targets with invalid batch index
                    if img_idx >= batch_size:
                        continue

                    # Get grid cell coordinates for this target
                    cell_x = int(target_boxes_scaled[target_idx, 0])
                    cell_y = int(target_boxes_scaled[target_idx, 1])

                    # Skip targets outside grid
                    if cell_x >= grid_size or cell_y >= grid_size or cell_x < 0 or cell_y < 0:
                        continue

                    # Set objectness target
                    obj_mask[img_idx, anchor_idx, cell_y, cell_x] = 1

                    # Set classification target
                    cls_idx = target_cls[target_idx]
                    if cls_idx < cls_mask.shape[-1]:  # Safety check
                        cls_mask[img_idx, anchor_idx, cell_y, cell_x, cls_idx] = 1

                    # Set box targets
                    target_boxes_grid[img_idx, anchor_idx, cell_y, cell_x] = target_boxes[target_idx]

            # Calculate losses
            # Objectness loss (for positive examples)
            obj_loss += self.lambda_obj * self.bce(pred_obj, obj_mask).sum()

            # No object loss (for negative examples)
            noobj_mask = 1 - obj_mask
            obj_loss += self.lambda_noobj * self.bce(pred_obj, torch.zeros_like(pred_obj)) * noobj_mask

            # Classification loss (only where objects exist)
            cls_loss += self.lambda_cls * (
                self.bce_cls(pred_cls, cls_mask) * obj_mask.unsqueeze(-1)
            ).sum()

            # Box loss (only where objects exist)
            # Convert to x,y,w,h format for loss calculation
            target_xy = target_boxes_grid[..., :2] / stride - grid
            target_wh = torch.log(target_boxes_grid[..., 2:4] / (anchors.view(1, -1, 1, 1, 2) * stride) + 1e-6)

            # MSE loss for xy
            box_loss += self.lambda_box * (
                self.mse(x, target_xy[..., 0]) * obj_mask
            ).sum()
            box_loss += self.lambda_box * (
                self.mse(y, target_xy[..., 1]) * obj_mask
            ).sum()

            # MSE loss for wh
            box_loss += self.lambda_box * (
                self.mse(w, target_wh[..., 0]) * obj_mask
            ).sum()
            box_loss += self.lambda_box * (
                self.mse(h, target_wh[..., 1]) * obj_mask
            ).sum()

        total_loss = cls_loss + box_loss + obj_loss

        # Return total loss and components
        return total_loss, {
            'loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item()
        }
