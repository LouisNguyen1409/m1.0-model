import os
import sys
import math
import cv2
import torch
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms
import torchvision.ops as ops
from model import YOLOD11
# -------------------------
# 1. Load data.yaml configuration
# -------------------------


def load_data_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    base_path = data['path']
    test_images = os.path.join(base_path, data['test'])
    nc = data['nc']
    names = data['names']
    return test_images, nc, names

# -------------------------
# 2. Decode predictions and NMS functions
# -------------------------


def decode_predictions(pred, anchors, stride, conf_thresh, num_classes):
    """
    Decode predictions for one scale.
    """
    device = pred.device
    batch_size = pred.size(0)
    num_anchors = len(anchors)
    grid_size = pred.size(2)

    # Reshape: (batch, num_anchors, grid, grid, 5+num_classes)
    pred = pred.view(batch_size, num_anchors, 5+num_classes, grid_size, grid_size)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()

    # Apply sigmoid activations
    pred[..., 0] = torch.sigmoid(pred[..., 0])
    pred[..., 1] = torch.sigmoid(pred[..., 1])
    pred[..., 4] = torch.sigmoid(pred[..., 4])
    pred[..., 5:] = torch.sigmoid(pred[..., 5:])

    # Build grid offsets
    grid_x = torch.arange(grid_size, device=device).repeat(grid_size, 1).view(1, 1, grid_size, grid_size).float()
    grid_y = torch.arange(grid_size, device=device).repeat(grid_size, 1).t().view(1, 1, grid_size, grid_size).float()

    # Compute bounding box predictions in pixels (assuming input size 640x640)
    pred_boxes = torch.zeros_like(pred[..., :4])
    pred_boxes[..., 0] = (grid_x + pred[..., 0]) * stride  # center x
    pred_boxes[..., 1] = (grid_y + pred[..., 1]) * stride  # center y
    anchor_w = torch.tensor([a[0] for a in anchors], device=device).view(1, num_anchors, 1, 1).float()
    anchor_h = torch.tensor([a[1] for a in anchors], device=device).view(1, num_anchors, 1, 1).float()
    pred_boxes[..., 2] = torch.exp(pred[..., 2]) * anchor_w * stride  # width
    pred_boxes[..., 3] = torch.exp(pred[..., 3]) * anchor_h * stride  # height

    # Convert from center (x,y) and size to [x1, y1, x2, y2]
    boxes = pred_boxes.clone()
    boxes[..., 0] = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    boxes[..., 1] = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    boxes[..., 2] = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    boxes[..., 3] = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    # Get detection scores and class predictions
    obj_conf = pred[..., 4]
    class_conf, class_pred = torch.max(pred[..., 5:], dim=-1)
    scores = obj_conf * class_conf

    # Filter by confidence threshold
    mask = scores > conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]
    class_pred = class_pred[mask]

    return boxes, scores, class_pred

# -------------------------
# 3. Inference function for one image
# -------------------------


def inference_image(image_path, model, device, anchors_dict, num_classes, conf_thresh=0.5, iou_thresh=0.4):
    """
    Run inference on a single image.
    """
    # Load image and record original size
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Could not load image {image_path}")
        return None
    orig_h, orig_w = orig_img.shape[:2]

    # Preprocess: resize to 640x640 and convert to tensor
    pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    input_img = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Model outputs predictions at three scales: small, medium, large
        small_pred, medium_pred, large_pred = model(input_img)

    boxes_all, scores_all, classes_all = [], [], []
    for pred, stride, anchors in zip(
        [small_pred, medium_pred, large_pred],
        [8, 16, 32],
        [anchors_dict['small'], anchors_dict['medium'], anchors_dict['large']]
    ):
        boxes, scores, class_pred = decode_predictions(pred, anchors, stride, conf_thresh, num_classes)
        if boxes.numel() > 0:
            boxes_all.append(boxes)
            scores_all.append(scores)
            classes_all.append(class_pred)

    if len(boxes_all) == 0:
        return orig_img  # No detections, return original

    boxes_all = torch.cat(boxes_all, dim=0)
    scores_all = torch.cat(scores_all, dim=0)
    classes_all = torch.cat(classes_all, dim=0)

    # Apply NMS
    keep = ops.nms(boxes_all, scores_all, iou_thresh)
    boxes_all = boxes_all[keep].cpu().numpy().astype(np.int32)
    scores_all = scores_all[keep].cpu().numpy()
    classes_all = classes_all[keep].cpu().numpy()

    # Scale boxes back to original image size
    scale_w = orig_w / 640
    scale_h = orig_h / 640
    boxes_all[:, [0, 2]] *= scale_w
    boxes_all[:, [1, 3]] *= scale_h

    # Draw detections
    for box, score, cls in zip(boxes_all, scores_all, classes_all):
        x1, y1, x2, y2 = box
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{names[cls]}: {score:.2f}"
        cv2.putText(orig_img, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return orig_img


# -------------------------
# 4. Main function to run inference on test dataset
# -------------------------
if __name__ == '__main__':
    # Load configuration from data.yaml
    yaml_path = "dataset/data.yaml"
    test_images_dir, num_classes, names = load_data_yaml(yaml_path)
    print(f"Test images directory: {test_images_dir}")
    print(f"Number of classes: {num_classes}, Names: {names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define anchors (should match those used in training)
    anchors_dict = {
        'small': [(10, 13), (16, 30), (33, 23)],
        'medium': [(30, 61), (62, 45), (59, 119)],
        'large': [(116, 90), (156, 198), (373, 326)]
    }

    # Import your YOLOD11 model here. Adjust as needed.
    try:
        from yolod11 import YOLOD11
    except ImportError:
        print("YOLOD11 model not found. Please ensure it is defined in yolod11.py.")
        sys.exit(1)

    model = YOLOD11(num_classes=num_classes).to(device)
    weights_path = "yolod11_final.pth"
    if not os.path.exists(weights_path):
        print(f"Model weights not found at {weights_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Create output directory for results
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in the test directory
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.png'))]
    for img_file in image_files:
        img_path = os.path.join(test_images_dir, img_file)
        result_img = inference_image(img_path, model, device, anchors_dict,
                                     num_classes, conf_thresh=0.5, iou_thresh=0.4)
        if result_img is not None:
            out_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + "_det.jpg")
            cv2.imwrite(out_path, result_img)
            print(f"Saved result: {out_path}")

    print("Inference on test dataset complete.")
