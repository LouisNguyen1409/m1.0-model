import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=True, class_names=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.class_names = class_names

        # Get image paths
        self.img_files = [os.path.join(img_dir, img) for img in os.listdir(img_dir)
                          if img.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']]

        # Sort to ensure same order
        self.img_files.sort()

        # Get corresponding label paths
        self.label_files = []
        for img_path in self.img_files:
            img_name = os.path.basename(img_path).split('.')[0]
            label_path = os.path.join(label_dir, f"{img_name}.txt")
            if os.path.exists(label_path):
                self.label_files.append(label_path)
            else:
                # If no label exists, use empty label
                self.label_files.append('')

        # Transformations
        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        # Read image
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Read labels
        try:
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                labels = np.loadtxt(label_path).reshape(-1, 5)
                boxes = labels[:, 1:5]  # YOLO format: [class, x_center, y_center, width, height]
                class_labels = labels[:, 0].astype(int)
            else:
                # No labels for this image
                boxes = np.zeros((0, 4))
                class_labels = np.zeros(0)
        except Exception as e:
            print(f"Error loading labels {label_path}: {e}")
            boxes = np.zeros((0, 4))
            class_labels = np.zeros(0)

        # Apply augmentations
        try:
            if boxes.size > 0:
                transformed = self.transform(image=img, bboxes=boxes, class_labels=class_labels)
                img = transformed['image']
                boxes = torch.tensor(transformed['bboxes'],
                                     dtype=torch.float32) if transformed['bboxes'] else torch.zeros((0, 4))
                class_labels = torch.tensor(
                    transformed['class_labels'], dtype=torch.long) if transformed['class_labels'] else torch.zeros(0, dtype=torch.long)

                # Create targets tensor with format [batch_idx, class, x, y, w, h]
                targets = torch.zeros((len(boxes), 6))
                if len(boxes) > 0:
                    targets[:, 1] = class_labels
                    targets[:, 2:] = boxes
            else:
                img = self.transform(image=img)['image']
                targets = torch.zeros((0, 6))
        except Exception as e:
            print(f"Error applying transformations: {e}")
            # Return a transformed blank image and empty targets
            img = self.transform(image=np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))['image']
            targets = torch.zeros((0, 6))

        return img, targets, img_path


def create_dataloaders(yaml_path, batch_size=16, img_size=640, workers=4):
    """
    Create train and validation dataloaders from a YAML config file
    """
    try:
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading YAML file {yaml_path}: {e}")

    # Get paths
    train_path = data_config.get('train', '')
    val_path = data_config.get('val', '')
    test_path = data_config.get('test', '')
    nc = data_config.get('nc', 0)  # number of classes
    names = data_config.get('names', [f'class{i}' for i in range(nc)])

    # Ensure train path exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training path {train_path} does not exist")

    # Split paths into image and label directories
    # Assuming standard YOLO directory structure
    train_img_dir = os.path.join(train_path, 'images')
    train_label_dir = os.path.join(train_path, 'labels')

    # Check that directories exist
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"Training images directory {train_img_dir} does not exist")
    if not os.path.exists(train_label_dir):
        raise FileNotFoundError(f"Training labels directory {train_label_dir} does not exist")

    val_img_dir = os.path.join(val_path, 'images') if val_path else None
    val_label_dir = os.path.join(val_path, 'labels') if val_path else None

    # Check validation directories if specified
    if val_path:
        if not os.path.exists(val_img_dir):
            print(f"Warning: Validation images directory {val_img_dir} does not exist")
            val_img_dir = None
        if not os.path.exists(val_label_dir):
            print(f"Warning: Validation labels directory {val_label_dir} does not exist")
            val_label_dir = None

    # Create datasets
    try:
        train_dataset = YOLODataset(
            train_img_dir,
            train_label_dir,
            img_size=img_size,
            augment=True,
            class_names=names
        )
    except Exception as e:
        raise RuntimeError(f"Error creating training dataset: {e}")

    val_dataset = None
    if val_img_dir and val_label_dir:
        try:
            val_dataset = YOLODataset(
                val_img_dir,
                val_label_dir,
                img_size=img_size,
                augment=False,
                class_names=names
            )
        except Exception as e:
            print(f"Warning: Error creating validation dataset: {e}")

    # Create dataloaders
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    except Exception as e:
        raise RuntimeError(f"Error creating training dataloader: {e}")

    val_loader = None
    if val_dataset:
        try:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
        except Exception as e:
            print(f"Warning: Error creating validation dataloader: {e}")

    return train_loader, val_loader, nc, names


def collate_fn(batch):
    """
    Custom collate function to handle variable number of targets
    """
    # Skip invalid samples (where image is None)
    valid_batch = [sample for sample in batch if sample[0] is not None]

    # If entire batch is invalid, return empty tensors
    if len(valid_batch) == 0:
        return torch.zeros((0, 3, 640, 640)), torch.zeros((0, 6)), []

    imgs, targets, paths = zip(*valid_batch)

    # Remove empty targets and add batch index
    targets_with_batch_idx = []
    for i, target in enumerate(targets):
        if target.shape[0] > 0:
            # Make a copy and set batch index
            t = target.clone()
            t[:, 0] = i  # set batch index
            targets_with_batch_idx.append(t)

    # Stack images
    try:
        imgs = torch.stack([img for img in imgs])
    except Exception as e:
        print(f"Error stacking images: {e}")
        print(f"Image shapes: {[img.shape for img in imgs]}")
        # Try to resize all images to the same shape
        resized_imgs = []
        for img in imgs:
            if img.shape != imgs[0].shape:
                # Resize to match first image
                resized = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(imgs[0].shape[1], imgs[0].shape[2])
                ).squeeze(0)
                resized_imgs.append(resized)
            else:
                resized_imgs.append(img)
        imgs = torch.stack(resized_imgs)

    # Concatenate targets
    targets = torch.cat(targets_with_batch_idx, dim=0) if targets_with_batch_idx else torch.zeros((0, 6))

    return imgs, targets, paths
