import os
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SolarPanelDataset(Dataset):
    def __init__(self, data_root, yaml_file, split='train', transform=None, img_size=640):
        """
        Dataset for solar panel inspection using YAML configuration.

        Args:
            data_root (str): Root directory containing the dataset
            yaml_file (str): Path to YAML configuration file
            split (str): 'train', 'val', or 'test'
            transform: Image transformations
            img_size (int): Image size for resizing
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.img_size = img_size

        # Load YAML configuration
        with open(yaml_file, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Map split name to YAML config keys
        split_map = {'train': 'train', 'val': 'val', 'test': 'test'}
        yaml_split = split_map.get(split, split)

        # Set paths based on YAML config
        if yaml_split in self.cfg:
            self.img_dir = os.path.join(data_root, self.cfg[yaml_split])
        else:
            raise ValueError(f"Split '{split}' not found in YAML configuration")

        # Get image list
        self.img_files = sorted([os.path.join(self.img_dir, img) for img in os.listdir(self.img_dir)
                                if img.endswith(('.jpg', '.jpeg', '.png'))])

        # Get class names from YAML config
        self.class_names = self.cfg.get('names', [])
        self.num_classes = len(self.class_names)

        # Set label directory - assume it's in the same structure as images but with 'labels' instead of 'images'
        self.ann_dir = self.img_dir.replace('images', 'labels')
        if not os.path.exists(self.ann_dir):
            print(f"Warning: Label directory not found at {self.ann_dir}")

        print(f"Loaded {len(self.img_files)} images for {split}")
        print(f"Classes: {self.class_names}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Resize image
        image = image.resize((self.img_size, self.img_size))

        # Convert to tensor and normalize
        if self.transform is None:
            image = torch.from_numpy(np.array(image))
            image = image.permute(2, 0, 1).float() / 255.0

        # Load annotation (YOLO format: class x_center y_center width height)
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        ann_path = os.path.join(self.ann_dir, f"{base_name}.txt")

        boxes = []
        labels = []

        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:  # class, x, y, w, h
                        class_id = int(data[0])
                        x_center = float(data[1])
                        y_center = float(data[2])
                        width = float(data[3])
                        height = float(data[4])

                        # Convert from YOLO format to [x1, y1, x2, y2]
                        x1 = (x_center - width/2)
                        y1 = (y_center - height/2)
                        x2 = (x_center + width/2)
                        y2 = (y_center + height/2)

                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

        # Convert to tensor
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Empty tensors if no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'img_path': img_path
        }

        # Apply transformations if any
        if self.transform:
            image, target = self.transform(image, target)

        return image, target
