import os
import cv2
import math
import csv
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import YOLOD11  # Ensure this module is available

# -------------------------
# 1. Load data.yaml configuration
# -------------------------


def load_data_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # Base path for the dataset
    base_path = data['path']
    train_images = os.path.join(base_path, data['train'])
    val_images = os.path.join(base_path, data['val'])
    test_images = os.path.join(base_path, data['test'])
    nc = data['nc']
    names = data['names']
    return train_images, val_images, test_images, nc, names

# -------------------------
# 2. Dataset & Collate Function
# -------------------------


class YOLODataset(Dataset):
    """
    Dataset that loads images and YOLO-format labels.
    Each label file should contain one or more lines:
         class x_center y_center width height
    where coordinates are normalized (0-1).
    Assumes that labels are in a directory parallel to images,
    with "images" replaced by "labels" in the path.
    """

    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
        # Derive labels directory from images directory (replace 'images' with 'labels')
        self.labels_dir = images_dir.replace("images", "labels")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_filename)[0] + '.txt')
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        # Load targets; each target: [class, x_center, y_center, width, height]
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        targets.append([cls, x, y, w, h])
        targets = torch.tensor(targets) if len(targets) > 0 else torch.zeros((0, 5))
        return image, targets


def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, 0)
    return images, list(targets)

# -------------------------
# 3. Helper Functions: IoU and Loss Components
# -------------------------


def compute_iou_wh(box1, box2):
    """
    Compute IoU for two boxes specified only by width and height.
    box1: tensor [w, h]
    box2: tensor [w, h]
    Assumes boxes are centered at the origin.
    """
    inter = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
    union = box1[0] * box1[1] + box2[0] * box2[1] - inter
    return inter / (union + 1e-16)


def yolo_loss_components(pred, targets, anchors, stride, num_classes, ignore_thresh=0.5):
    """
    Compute YOLO loss for one scale and return each component.
    Returns:
        loss_box, loss_obj, loss_noobj, loss_cls, total_loss
    """
    device = pred.device
    batch_size = pred.size(0)
    num_anchors = len(anchors)
    grid_size = pred.size(2)

    # Reshape predictions: (batch, num_anchors, grid, grid, 5+num_classes)
    pred = pred.view(batch_size, num_anchors, 5+num_classes, grid_size, grid_size)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()

    # Apply sigmoid activations
    pred[..., 0] = torch.sigmoid(pred[..., 0])  # x
    pred[..., 1] = torch.sigmoid(pred[..., 1])  # y
    pred[..., 4] = torch.sigmoid(pred[..., 4])  # objectness
    pred[..., 5:] = torch.sigmoid(pred[..., 5:])  # class scores

    # Build grid
    grid_x = torch.arange(grid_size, device=device).repeat(grid_size, 1).view(1, 1, grid_size, grid_size).float()
    grid_y = torch.arange(grid_size, device=device).repeat(grid_size, 1).t().view(1, 1, grid_size, grid_size).float()

    # Predicted bounding boxes in pixels (assume input image 640x640)
    pred_boxes = torch.zeros_like(pred[..., :4])
    pred_boxes[..., 0] = (grid_x + pred[..., 0]) * stride
    pred_boxes[..., 1] = (grid_y + pred[..., 1]) * stride
    anchor_w = torch.tensor([a[0] for a in anchors], device=device).view(1, num_anchors, 1, 1).float()
    anchor_h = torch.tensor([a[1] for a in anchors], device=device).view(1, num_anchors, 1, 1).float()
    pred_boxes[..., 2] = torch.exp(pred[..., 2]) * anchor_w * stride
    pred_boxes[..., 3] = torch.exp(pred[..., 3]) * anchor_h * stride

    # Prepare target tensors for each grid cell and anchor
    obj_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
    noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, device=device)
    tx = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
    ty = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
    tw = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
    th = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
    tconf = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
    tcls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, num_classes, device=device)

    # Assign targets to grid cells and anchors
    for b in range(batch_size):
        if targets[b].numel() == 0:
            continue
        for target in targets[b]:
            cls = int(target[0])
            gx = target[1] * grid_size  # scale normalized x_center to grid
            gy = target[2] * grid_size  # scale normalized y_center to grid
            gw = target[3] * 640         # scale normalized width to pixels
            gh = target[4] * 640         # scale normalized height to pixels
            gi = int(gx)
            gj = int(gy)
            best_iou = 0
            best_anchor = 0
            for i, anchor in enumerate(anchors):
                iou = compute_iou_wh(torch.tensor([gw, gh], device=device),
                                     torch.tensor(anchor, device=device))
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            obj_mask[b, best_anchor, gj, gi] = 1
            noobj_mask[b, best_anchor, gj, gi] = 0
            tx[b, best_anchor, gj, gi] = gx - gi
            ty[b, best_anchor, gj, gi] = gy - gj
            tw[b, best_anchor, gj, gi] = math.log(gw / (anchors[best_anchor][0] + 1e-16) + 1e-16)
            th[b, best_anchor, gj, gi] = math.log(gh / (anchors[best_anchor][1] + 1e-16) + 1e-16)
            tconf[b, best_anchor, gj, gi] = 1
            tcls[b, best_anchor, gj, gi, cls] = 1

            # Optionally ignore anchors with high IoU but not best
            for i in range(num_anchors):
                if i == best_anchor:
                    continue
                if compute_iou_wh(torch.tensor([gw, gh], device=device),
                                  torch.tensor(anchors[i], device=device)) > ignore_thresh:
                    noobj_mask[b, i, gj, gi] = 0

    mse_loss = nn.MSELoss(reduction='sum')
    bce_loss = nn.BCELoss(reduction='sum')

    loss_x = mse_loss(pred[..., 0] * obj_mask, tx * obj_mask)
    loss_y = mse_loss(pred[..., 1] * obj_mask, ty * obj_mask)
    loss_w = mse_loss(pred[..., 2] * obj_mask, tw * obj_mask)
    loss_h = mse_loss(pred[..., 3] * obj_mask, th * obj_mask)
    loss_box = loss_x + loss_y + loss_w + loss_h

    loss_obj = bce_loss(pred[..., 4] * obj_mask, tconf * obj_mask)
    loss_noobj = bce_loss(pred[..., 4] * noobj_mask, tconf * noobj_mask)
    loss_cls = bce_loss(pred[..., 5:] * obj_mask.unsqueeze(-1), tcls * obj_mask.unsqueeze(-1))

    total_loss = loss_box + loss_obj + loss_noobj + loss_cls
    return loss_box, loss_obj, loss_noobj, loss_cls, total_loss / batch_size

# -------------------------
# 4. Training and Validation Functions
# -------------------------


def train_one_epoch(model, dataloader, optimizer, device, anchors, num_classes):
    model.train()
    loss_history = {
        "loss_box": 0.0,
        "loss_obj": 0.0,
        "loss_noobj": 0.0,
        "loss_cls": 0.0,
        "total_loss": 0.0
    }
    anchors_small = anchors['small']
    anchors_medium = anchors['medium']
    anchors_large = anchors['large']

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        optimizer.zero_grad()

        # Forward pass – model outputs predictions at three scales.
        small_pred, medium_pred, large_pred = model(images)

        lb_box, lb_obj, lb_noobj, lb_cls, loss_small = yolo_loss_components(
            small_pred, targets, anchors_small, stride=8, num_classes=num_classes)
        lm_box, lm_obj, lm_noobj, lm_cls, loss_medium = yolo_loss_components(
            medium_pred, targets, anchors_medium, stride=16, num_classes=num_classes)
        ll_box, ll_obj, ll_noobj, ll_cls, loss_large = yolo_loss_components(
            large_pred, targets, anchors_large, stride=32, num_classes=num_classes)

        # Sum losses from all scales
        loss_box = lb_box + lm_box + ll_box
        loss_obj = lb_obj + lm_obj + ll_obj
        loss_noobj = lb_noobj + lm_noobj + ll_noobj
        loss_cls = lb_cls + lm_cls + ll_cls
        loss = loss_small + loss_medium + loss_large

        loss.backward()
        optimizer.step()

        loss_history["loss_box"] += loss_box.item()
        loss_history["loss_obj"] += loss_obj.item()
        loss_history["loss_noobj"] += loss_noobj.item()
        loss_history["loss_cls"] += loss_cls.item()
        loss_history["total_loss"] += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Average losses over batches
    num_batches = len(dataloader)
    for key in loss_history:
        loss_history[key] /= num_batches

    return loss_history


@torch.no_grad()
def validate(model, dataloader, device, anchors, num_classes):
    model.eval()
    loss_history = {
        "loss_box": 0.0,
        "loss_obj": 0.0,
        "loss_noobj": 0.0,
        "loss_cls": 0.0,
        "total_loss": 0.0
    }
    anchors_small = anchors['small']
    anchors_medium = anchors['medium']
    anchors_large = anchors['large']

    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        small_pred, medium_pred, large_pred = model(images)

        lb_box, lb_obj, lb_noobj, lb_cls, loss_small = yolo_loss_components(
            small_pred, targets, anchors_small, stride=8, num_classes=num_classes)
        lm_box, lm_obj, lm_noobj, lm_cls, loss_medium = yolo_loss_components(
            medium_pred, targets, anchors_medium, stride=16, num_classes=num_classes)
        ll_box, ll_obj, ll_noobj, ll_cls, loss_large = yolo_loss_components(
            large_pred, targets, anchors_large, stride=32, num_classes=num_classes)

        loss_box = lb_box + lm_box + ll_box
        loss_obj = lb_obj + lm_obj + ll_obj
        loss_noobj = lb_noobj + lm_noobj + ll_noobj
        loss_cls = lb_cls + lm_cls + ll_cls
        loss = loss_small + loss_medium + loss_large

        loss_history["loss_box"] += loss_box.item()
        loss_history["loss_obj"] += loss_obj.item()
        loss_history["loss_noobj"] += loss_noobj.item()
        loss_history["loss_cls"] += loss_cls.item()
        loss_history["total_loss"] += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    num_batches = len(dataloader)
    for key in loss_history:
        loss_history[key] /= num_batches

    return loss_history

# -------------------------
# 5. Main Training Script
# -------------------------


def main():
    # Load YAML configuration
    train_images_dir, val_images_dir, _, num_classes, names = load_data_yaml("dataset/data.yaml")
    print(f"Loaded dataset with {num_classes} classes: {names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-4

    # Define anchors for each scale (in pixels)
    anchors = {
        'small': [(10, 13), (16, 30), (33, 23)],      # For 80x80 feature map (stride 8)
        'medium': [(30, 61), (62, 45), (59, 119)],      # For 40x40 feature map (stride 16)
        'large': [(116, 90), (156, 198), (373, 326)]     # For 20x20 feature map (stride 32)
    }

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    train_dataset = YOLODataset(train_images_dir, transform=transform)
    val_dataset = YOLODataset(val_images_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    model = YOLOD11(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare a list to store losses for CSV saving
    csv_rows = []
    headers = [
        "epoch",
        "train_total", "train_loss_box", "train_loss_obj", "train_loss_noobj", "train_loss_cls",
        "val_total", "val_loss_box", "val_loss_obj", "val_loss_noobj", "val_loss_cls"
    ]
    csv_rows.append(headers)

    for epoch in range(num_epochs):
        train_loss_components = train_one_epoch(model, train_loader, optimizer, device, anchors, num_classes)
        val_loss_components = validate(model, val_loader, device, anchors, num_classes)

        # Print losses
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss_components['total_loss']:.4f} (Box: {train_loss_components['loss_box']:.4f}, Obj: {train_loss_components['loss_obj']:.4f}, NoObj: {train_loss_components['loss_noobj']:.4f}, Cls: {train_loss_components['loss_cls']:.4f})")
        print(f"  Val Loss:   {val_loss_components['total_loss']:.4f} (Box: {val_loss_components['loss_box']:.4f}, Obj: {val_loss_components['loss_obj']:.4f}, NoObj: {val_loss_components['loss_noobj']:.4f}, Cls: {val_loss_components['loss_cls']:.4f})")

        # Append current epoch's losses to CSV rows
        row = [
            epoch+1,
            train_loss_components["total_loss"], train_loss_components["loss_box"], train_loss_components["loss_obj"],
            train_loss_components["loss_noobj"], train_loss_components["loss_cls"],
            val_loss_components["total_loss"], val_loss_components["loss_box"], val_loss_components["loss_obj"],
            val_loss_components["loss_noobj"], val_loss_components["loss_cls"]
        ]
        csv_rows.append(row)

    # Save the model weights
    torch.save(model.state_dict(), "yolod11_final.pth")

    # Save losses to CSV file
    csv_file = "loss_history.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("Training complete, model and loss history saved to CSV.")


if __name__ == '__main__':
    main()
