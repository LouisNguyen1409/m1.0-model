import torch
import numpy as np
import cv2
import random
import math
from PIL import Image, ImageEnhance, ImageFilter


class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """
    Convert PIL Image to tensor.
    """

    def __call__(self, image, target):
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert numpy array to tensor
        if isinstance(image, np.ndarray):
            # Handle RGB or grayscale
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)

            # Convert HWC to CHW
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float() / 255.0

        return image, target


class RandomHorizontalFlip:
    """
    Random horizontal flip with probability p.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            # Flip image
            if isinstance(image, torch.Tensor):
                image = image.flip(-1)
            elif isinstance(image, np.ndarray):
                image = np.fliplr(image)
            elif isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Flip boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                boxes = boxes.clone()

                # Flip x coordinates
                if isinstance(image, torch.Tensor):
                    width = image.shape[-1]
                elif isinstance(image, np.ndarray):
                    width = image.shape[1]
                elif isinstance(image, Image.Image):
                    width = image.width
                else:
                    width = 1.0

                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes

        return image, target


class RandomVerticalFlip:
    """
    Random vertical flip with probability p.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            # Flip image
            if isinstance(image, torch.Tensor):
                image = image.flip(-2)
            elif isinstance(image, np.ndarray):
                image = np.flipud(image)
            elif isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

            # Flip boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                boxes = boxes.clone()

                # Flip y coordinates
                if isinstance(image, torch.Tensor):
                    height = image.shape[-2]
                elif isinstance(image, np.ndarray):
                    height = image.shape[0]
                elif isinstance(image, Image.Image):
                    height = image.height
                else:
                    height = 1.0

                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target['boxes'] = boxes

        return image, target


class ColorJitter:
    """
    Randomly change brightness, contrast, saturation, and hue.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        # Only apply to PIL Images
        if isinstance(image, Image.Image):
            # Apply random brightness
            if self.brightness > 0:
                factor = random.uniform(1-self.brightness, 1+self.brightness)
                image = ImageEnhance.Brightness(image).enhance(factor)

            # Apply random contrast
            if self.contrast > 0:
                factor = random.uniform(1-self.contrast, 1+self.contrast)
                image = ImageEnhance.Contrast(image).enhance(factor)

            # Apply random saturation
            if self.saturation > 0:
                factor = random.uniform(1-self.saturation, 1+self.saturation)
                image = ImageEnhance.Color(image).enhance(factor)

            # Apply random hue
            if self.hue > 0:
                # Convert to HSV and adjust hue
                image = np.array(image)
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)

                # Adjust hue
                hue_shift = random.uniform(-self.hue, self.hue) * 180
                h = (h + hue_shift) % 180

                # Merge channels and convert back to RGB
                hsv = cv2.merge([h, s, v])
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                image = Image.fromarray(image)

        return image, target


class RandomScale:
    """
    Randomly scale image and boxes within range.
    """

    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, image, target):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        # Scale image
        if isinstance(image, Image.Image):
            width, height = image.size
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.BILINEAR)

            # Scale boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                boxes = boxes * scale
                target['boxes'] = boxes

        return image, target


class RandomCrop:
    """
    Randomly crop image and adjust boxes.
    """

    def __init__(self, crop_range=(0.6, 1.0)):
        self.crop_range = crop_range

    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            width, height = image.size

            # Determine crop size
            crop_ratio = random.uniform(self.crop_range[0], self.crop_range[1])
            crop_width = int(width * crop_ratio)
            crop_height = int(height * crop_ratio)

            # Determine crop position
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            right = left + crop_width
            bottom = top + crop_height

            # Crop image
            image = image.crop((left, top, right, bottom))

            # Adjust boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()

                # Clip boxes to crop region
                boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=left, max=right)
                boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=top, max=bottom)

                # Shift boxes to new origin
                boxes[:, [0, 2]] -= left
                boxes[:, [1, 3]] -= top

                # Filter out boxes with no area
                keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes = boxes[keep]

                if 'labels' in target:
                    target['labels'] = target['labels'][keep]

                target['boxes'] = boxes

        return image, target


class RandomRotation:
    """
    Randomly rotate image and boxes.
    """

    def __init__(self, angle_range=(-5, 5)):
        self.angle_range = angle_range

    def __call__(self, image, target):
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        if isinstance(image, Image.Image):
            # Get image center
            width, height = image.size
            center_x, center_y = width // 2, height // 2

            # Rotate image
            image = image.rotate(angle, Image.BILINEAR, expand=False)

            # Rotate boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()

                # Convert boxes to corners
                corners = torch.zeros((len(boxes), 4, 2))
                corners[:, 0, 0] = boxes[:, 0]  # x1
                corners[:, 0, 1] = boxes[:, 1]  # y1
                corners[:, 1, 0] = boxes[:, 2]  # x2
                corners[:, 1, 1] = boxes[:, 1]  # y1
                corners[:, 2, 0] = boxes[:, 2]  # x2
                corners[:, 2, 1] = boxes[:, 3]  # y2
                corners[:, 3, 0] = boxes[:, 0]  # x1
                corners[:, 3, 1] = boxes[:, 3]  # y2

                # Rotate corners
                angle_rad = -angle * math.pi / 180.0
                cos_angle = math.cos(angle_rad)
                sin_angle = math.sin(angle_rad)

                # Translate corners to origin
                corners[:, :, 0] -= center_x
                corners[:, :, 1] -= center_y

                # Rotate corners
                rot_corners = torch.zeros_like(corners)
                rot_corners[:, :, 0] = corners[:, :, 0] * cos_angle - corners[:, :, 1] * sin_angle
                rot_corners[:, :, 1] = corners[:, :, 0] * sin_angle + corners[:, :, 1] * cos_angle

                # Translate corners back
                rot_corners[:, :, 0] += center_x
                rot_corners[:, :, 1] += center_y

                # Convert back to boxes [x1, y1, x2, y2]
                # Use min/max to get axis-aligned bounding box
                new_boxes = torch.zeros_like(boxes)
                new_boxes[:, 0] = torch.min(rot_corners[:, :, 0], dim=1)[0]
                new_boxes[:, 1] = torch.min(rot_corners[:, :, 1], dim=1)[0]
                new_boxes[:, 2] = torch.max(rot_corners[:, :, 0], dim=1)[0]
                new_boxes[:, 3] = torch.max(rot_corners[:, :, 1], dim=1)[0]

                # Clip boxes to image boundaries
                new_boxes[:, [0, 2]] = torch.clamp(new_boxes[:, [0, 2]], min=0, max=width)
                new_boxes[:, [1, 3]] = torch.clamp(new_boxes[:, [1, 3]], min=0, max=height)

                # Filter out boxes with no area
                keep = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
                new_boxes = new_boxes[keep]

                if 'labels' in target:
                    target['labels'] = target['labels'][keep]

                target['boxes'] = new_boxes

        return image, target


class RandomMixUp:
    """
    Implement MixUp augmentation.

    MixUp creates a new image that is a weighted combination of two images,
    and their labels are similarly combined.
    """

    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, image, target, image2=None, target2=None):
        # Only apply with probability p
        if random.random() > self.p or image2 is None or target2 is None:
            return image, target

        # Generate mixup ratio from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Make sure images are tensors
        if not isinstance(image, torch.Tensor) or not isinstance(image2, torch.Tensor):
            return image, target

        # Mixup images
        mixed_image = lam * image + (1 - lam) * image2

        # Combine targets (we'll just concatenate boxes and labels)
        if 'boxes' in target and 'boxes' in target2:
            mixed_boxes = torch.cat([target['boxes'], target2['boxes']], dim=0)
            target['boxes'] = mixed_boxes

        if 'labels' in target and 'labels' in target2:
            mixed_labels = torch.cat([target['labels'], target2['labels']], dim=0)
            target['labels'] = mixed_labels

        return mixed_image, target


class RandomMosaic:
    """
    Implement Mosaic augmentation.

    Mosaic combines 4 images into one.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, targets):
        # Only apply with probability p and if we have enough images
        if random.random() > self.p or len(images) < 4:
            return images[0], targets[0]

        # Ensure we have 4 images
        indices = list(range(min(len(images), 4)))
        random.shuffle(indices)
        imgs = [images[i] for i in indices[:4]]
        tgts = [targets[i] for i in indices[:4]]

        # Make sure images are tensors with the same shape
        if not all(isinstance(img, torch.Tensor) for img in imgs):
            return images[0], targets[0]

        # Get image dimensions
        _, h, w = imgs[0].shape

        # Create a new image that's 2x the dimensions
        mosaic_img = torch.zeros((3, h*2, w*2), dtype=imgs[0].dtype)

        # Place images in mosaic
        positions = [
            (0, 0),     # top-left
            (w, 0),     # top-right
            (0, h),     # bottom-left
            (w, h)      # bottom-right
        ]

        # Combined target
        mosaic_boxes = []
        mosaic_labels = []

        # Place each image
        for i in range(4):
            img = imgs[i]
            tgt = tgts[i]
            x, y = positions[i]

            # Place image
            mosaic_img[:, y:y+h, x:x+w] = img

            # Adjust boxes
            if 'boxes' in tgt and len(tgt['boxes']) > 0:
                boxes = tgt['boxes'].clone()
                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y
                mosaic_boxes.append(boxes)

                if 'labels' in tgt:
                    mosaic_labels.append(tgt['labels'])

        # Combine targets
        combined_target = {}
        if mosaic_boxes:
            combined_target['boxes'] = torch.cat(mosaic_boxes, dim=0)
        if mosaic_labels:
            combined_target['labels'] = torch.cat(mosaic_labels, dim=0)

        return mosaic_img, combined_target


def get_train_transforms(img_size=640):
    """
    Get transforms for training.
    """
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.1),
        # Additional transforms can be enabled based on needs
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # RandomRotation(angle_range=(-5, 5)),
        # RandomScale(scale_range=(0.8, 1.2)),
    ])


def get_val_transforms(img_size=640):
    """
    Get transforms for validation.
    """
    return Compose([
        ToTensor(),
    ])


def apply_mosaic_augmentation(dataset, batch_size=4, p=0.5):
    """
    Apply mosaic augmentation to a batch of images.

    Args:
        dataset: Dataset to sample images from
        batch_size: Number of images to combine (4 for standard mosaic)
        p: Probability of applying mosaic

    Returns:
        mosaic_transform: Callable that applies mosaic to a batch
    """
    mosaic_transform = RandomMosaic(p=p)

    def get_mosaic_batch(indices):
        images = []
        targets = []

        for idx in indices:
            img, tgt = dataset[idx]
            images.append(img)
            targets.append(tgt)

        return mosaic_transform(images, targets)

    return get_mosaic_batch


def apply_mixup_augmentation(dataset, alpha=1.0, p=0.5):
    """
    Apply mixup augmentation to a pair of images.

    Args:
        dataset: Dataset to sample images from
        alpha: Alpha parameter for beta distribution
        p: Probability of applying mixup

    Returns:
        mixup_transform: Callable that applies mixup to a pair of images
    """
    mixup_transform = RandomMixUp(alpha=alpha, p=p)

    def get_mixup_pair(idx1, idx2):
        img1, tgt1 = dataset[idx1]
        img2, tgt2 = dataset[idx2]

        return mixup_transform(img1, tgt1, img2, tgt2)

    return get_mixup_pair
