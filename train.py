import os
import time
import json
import yaml
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import model
from model import YOLOD11
# Import dataset
from datasets import SolarPanelDataset
# Import utilities
from utils.loss import YOLOD11Loss
from utils.yolo_utils import process_predictions
from utils.visualization import plot_detections, evaluate_detections
from utils.augmentation import get_train_transforms, get_val_transforms
from utils.training import (
    generate_anchors, train_one_epoch, validate, save_checkpoint
)


def collate_fn(batch):
    """
    Custom collate function for batching data with variable-sized objects
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, 0)

    # Convert list of dicts to dict of lists for train_one_epoch
    if targets and isinstance(targets[0], dict):
        # Initialize an empty dict to store batched targets
        batched_targets = {}
        # Get all keys from the first target dict
        keys = targets[0].keys()
        
        # For each key, collect values from all targets
        for key in keys:
            if key == 'img_path':  # Handle non-tensor data differently
                batched_targets[key] = [target[key] for target in targets]
            else:
                # Stack tensors if possible, otherwise store as list
                try:
                    batched_targets[key] = torch.stack([target[key] for target in targets])
                except:
                    batched_targets[key] = [target[key] for target in targets]
        
        return images, batched_targets
    
    return images, targets


def main(args):
    """
    Main training function

    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create logs directory
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Create visualization directory
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    # Load YAML configuration
    with open(args.data_yaml, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    # Get number of classes and class names from YAML
    num_classes = yaml_cfg.get('nc', args.num_classes)
    class_names = yaml_cfg.get('names', [])

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Create CSV log file for losses
    csv_path = os.path.join(logs_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train_Loss', 'Train_Box', 'Train_Obj', 'Train_Cls',
                         'Val_Loss', 'Val_Box', 'Val_Obj', 'Val_Cls', 'mAP', 'LR'])

    # Create datasets
    print("Loading datasets...")
    train_dataset = SolarPanelDataset(
        data_root=args.data_root,
        yaml_file=args.data_yaml,
        split='train',
        img_size=args.img_size,
        transform=get_train_transforms(args.img_size)
    )

    val_dataset = SolarPanelDataset(
        data_root=args.data_root,
        yaml_file=args.data_yaml,
        split='val',
        img_size=args.img_size,
        transform=get_val_transforms(args.img_size)
    )

    # Define dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Generate anchors from dataset
    anchors = generate_anchors(
        train_dataset,
        num_anchors=9,
        strides=[8, 16, 32],
        img_size=args.img_size
    )

    # Define strides for each feature map scale
    strides = [8, 16, 32]

    # Create model
    print("Initializing YOLOD11 model...")
    model = YOLOD11(num_classes=num_classes)
    model.to(device)

    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")

    # Define loss function with generated anchors
    loss_fn = YOLOD11Loss(num_classes=num_classes, anchors=anchors)
    loss_fn.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Load pretrained weights if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_mAP = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
            if 'mAP' in checkpoint:
                best_mAP = checkpoint['mAP']

            print(f"Resuming from epoch {start_epoch}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Best mAP: {best_mAP:.6f}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch")

    # Training loop
    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            anchors=anchors,
            strides=strides
        )

        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            anchors=anchors,
            strides=strides,
            class_names=class_names,
            conf_threshold=0.25,
            iou_threshold=0.45
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])

        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('BoxLoss/train', train_metrics['box_loss'], epoch)
        writer.add_scalar('BoxLoss/val', val_metrics['box_loss'], epoch)
        writer.add_scalar('ObjLoss/train', train_metrics['obj_loss'], epoch)
        writer.add_scalar('ObjLoss/val', val_metrics['obj_loss'], epoch)
        writer.add_scalar('ClsLoss/train', train_metrics['cls_loss'], epoch)
        writer.add_scalar('ClsLoss/val', val_metrics['cls_loss'], epoch)
        writer.add_scalar('mAP', val_metrics['mAP'], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Log precision and recall for each class
        for i, (prec, rec) in enumerate(zip(val_metrics['precision'], val_metrics['recall'])):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            writer.add_scalar(f'Precision/{class_name}', prec, epoch)
            writer.add_scalar(f'Recall/{class_name}', rec, epoch)

        # Log to CSV
        csv_writer.writerow([
            epoch + 1,
            train_metrics['loss'], train_metrics['box_loss'], train_metrics['obj_loss'], train_metrics['cls_loss'],
            val_metrics['loss'], val_metrics['box_loss'], val_metrics['obj_loss'], val_metrics['cls_loss'],
            val_metrics['mAP'],
            current_lr
        ])
        csv_file.flush()  # Make sure data is written immediately

        # Save visualization of predictions
        if 'sample_images' in val_metrics and len(val_metrics['sample_images']) > 0:
            for i in range(min(3, len(val_metrics['sample_images']))):
                image = val_metrics['sample_images'][i].permute(1, 2, 0).numpy()

                # Plot detections
                fig, _ = plot_detections(
                    image,
                    val_metrics['sample_preds'][i],
                    class_names,
                    conf_threshold=0.25
                )

                # Save figure
                fig_path = os.path.join(viz_dir, f'epoch{epoch+1:03d}_sample{i+1}.png')
                fig.savefig(fig_path)
                plt.close(fig)

                # Log to tensorboard
                writer.add_figure(f'Predictions/Sample{i+1}', fig, epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"  Train Box: {train_metrics['box_loss']:.6f}")
        print(f"  Train Obj: {train_metrics['obj_loss']:.6f}")
        print(f"  Train Cls: {train_metrics['cls_loss']:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Val Box: {val_metrics['box_loss']:.6f}")
        print(f"  Val Obj: {val_metrics['obj_loss']:.6f}")
        print(f"  Val Cls: {val_metrics['cls_loss']:.6f}")
        print(f"  mAP@0.5: {val_metrics['mAP']:.6f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        print(f"  Time: {(time.time() - start_time)/60:.2f} minutes")

        # Combined train/val loss dictionary
        loss_dict = {
            'train_loss': train_metrics['loss'],
            'train_box_loss': train_metrics['box_loss'],
            'train_obj_loss': train_metrics['obj_loss'],
            'train_cls_loss': train_metrics['cls_loss'],
            'val_loss': val_metrics['loss'],
            'val_box_loss': val_metrics['box_loss'],
            'val_obj_loss': val_metrics['obj_loss'],
            'val_cls_loss': val_metrics['cls_loss'],
            'mAP': val_metrics['mAP'],
            'learning_rate': current_lr,
            'epoch': epoch + 1
        }

        # Save checkpoint every save_interval epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f"checkpoint_epoch{epoch+1}.pth"
            )
            save_checkpoint(model, optimizer, epoch, loss_dict, checkpoint_path)

        # Save best model by loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(args.output_dir, "best_loss_model.pth")
            save_checkpoint(model, optimizer, epoch, loss_dict, best_model_path, is_best=True)
            print(f"  New best model saved with validation loss: {best_val_loss:.6f}")

        # Save best model by mAP
        if val_metrics['mAP'] > best_mAP:
            best_mAP = val_metrics['mAP']
            best_map_model_path = os.path.join(args.output_dir, "best_map_model.pth")
            loss_dict['best_metric'] = 'mAP'
            save_checkpoint(model, optimizer, epoch, loss_dict, best_map_model_path, is_best=True)
            print(f"  New best model saved with mAP: {best_mAP:.6f}")

        # Save latest model (overwrite)
        latest_model_path = os.path.join(args.output_dir, "latest_model.pth")
        save_checkpoint(model, optimizer, epoch, loss_dict, latest_model_path)

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    save_checkpoint(model, optimizer, args.epochs-1, loss_dict, final_model_path)
    print(f"\nTraining completed. Final model saved: {final_model_path}")

    # Close CSV file and tensorboard writer
    csv_file.close()
    writer.close()

    # Print total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOD11 for solar panel inspection')

    # Dataset parameters
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing the dataset')
    parser.add_argument('--data-yaml', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='Number of classes (overridden by YAML if present)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay for optimizer')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')

    # Saving parameters
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save model every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()
    main(args)
