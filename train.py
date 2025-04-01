import os
import time
import yaml
import math
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from utils import xywh2xyxy


# Import local modules
from model import YOLOD11, TaskAlignedPredictor, DetectionHead
from dataset import create_dataloaders
from loss import YOLOLoss
from utils import (AverageMeter, non_max_suppression, calculate_metrics, visualize_detections,
                   save_checkpoint, load_checkpoint, ModelEMA, get_lr_scheduler,
                   debug_model_shapes, check_dataset, logger)
from config import YOLOConfig, parse_args


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(model, dataloader, device, config, class_names=None):
    """Validate model on validation dataset"""
    model.eval()

    all_predictions = []
    all_targets = []

    # Progress bar
    pbar = tqdm(dataloader, desc='Validating', leave=False)

    with torch.no_grad():
        for batch_i, (imgs, targets, paths) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)

            # Run model
            try:
                outputs = model(imgs)

                # Apply NMS to get final detections
                # Ensure outputs is suitable for NMS (batch, anchor*boxes, 5+num_classes)
                if isinstance(outputs, (list, tuple)):
                    # Process and combine outputs from different scales
                    batch_size = outputs[0].shape[0]
                    processed_outputs = []

                    for output in outputs:
                        # Reshape to (batch, -1, 5+num_classes)
                        nc = config.num_classes if hasattr(config, 'num_classes') else 80
                        total_items = 5 + nc  # x, y, w, h, obj, classes
                        output = output.view(batch_size, -1, total_items)
                        processed_outputs.append(output)

                    # Concatenate predictions from different scales
                    predictions = torch.cat(processed_outputs, dim=1)
                else:
                    # Single output case
                    predictions = outputs

                # Convert from [x, y, w, h] to [x1, y1, x2, y2] for NMS
                predictions[..., :4] = xywh2xyxy(predictions[..., :4])

                # Apply NMS
                detections = non_max_suppression(predictions, conf_thres=0.1, iou_thres=0.5)
                all_predictions.extend(detections)
            except Exception as e:
                logger.error(f"Error in validation forward pass: {e}")
                # Return dummy metrics
                return {'precision': 0, 'recall': 0, 'mAP': 0}

            # Save targets for metric calculation
            all_targets.append(targets)

            # For visualization, process only the first batch
            if batch_i == 0:
                # Visualize a few images
                for i in range(min(3, len(imgs))):
                    if i < len(detections) and detections[i] is not None and len(detections[i]) > 0:
                        fig = visualize_detections(imgs[i], detections[i], class_names)
                        plt.close(fig)

    # Concatenate all targets
    all_targets = torch.cat(all_targets, 0) if all_targets else torch.zeros((0, 6))

    # Compute validation metrics
    try:
        metrics = calculate_metrics(all_predictions, all_targets, iou_thres=0.5)
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        metrics = {'precision': 0, 'recall': 0, 'mAP': 0}

    return metrics


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config, scaler=None, ema=None):
    """Train the model for one epoch"""
    model.train()

    # Meters for loss tracking
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    box_loss_meter = AverageMeter()
    obj_loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')

    # Train loop
    for batch_i, (imgs, targets, paths) in enumerate(pbar):
        # Move data to device
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)

        # Skip empty batches
        if imgs.shape[0] == 0 or targets.shape[0] == 0:
            continue

        # Mixed precision training
        if config.use_amp:
            with amp.autocast():
                try:
                    # Forward pass
                    outputs = model(imgs)

                    # Calculate loss
                    loss, loss_items = loss_fn(outputs, targets)
                except Exception as e:
                    logger.error(f"Error in forward pass or loss calculation: {e}")
                    continue
        else:
            try:
                # Forward pass
                outputs = model(imgs)

                # Calculate loss
                if isinstance(outputs, (list, tuple)):
                    loss, loss_items = loss_fn(outputs, targets)
                else:
                    # Handle case where model outputs a single tensor
                    loss, loss_items = loss_fn([outputs], targets)
            except Exception as e:
                logger.error(f"Error in forward pass or loss calculation: {e}")
                continue

        # Backward pass
        optimizer.zero_grad()

        try:
            if config.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        except Exception as e:
            logger.error(f"Error in backward pass: {e}")
            continue

        # Update EMA model
        if ema is not None:
            ema.update(model)

        # Update loss meters
        loss_meter.update(loss_items['loss'])
        cls_loss_meter.update(loss_items['cls_loss'])
        box_loss_meter.update(loss_items['box_loss'])
        obj_loss_meter.update(loss_items['obj_loss'])

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'cls_loss': f"{cls_loss_meter.avg:.4f}",
            'box_loss': f"{box_loss_meter.avg:.4f}",
            'obj_loss': f"{obj_loss_meter.avg:.4f}"
        })

    return {
        'loss': loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'box_loss': box_loss_meter.avg,
        'obj_loss': obj_loss_meter.avg
    }


def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = YOLOConfig(args.config)
    config.update_from_args(args)

    # Set up logging
    if config.debug:
        logger.setLevel(logging.DEBUG)

    # Create output directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Save configuration
    config.save_to_file(os.path.join(config.save_dir, 'config.yaml'))

    # Set seed for reproducibility
    set_seed(42)

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info(f"Creating model: {config.model_name}")

    try:
        model = YOLOD11(num_classes=80)  # Default to COCO classes
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return

    # Debug model architecture
    if config.debug:
        debug_model_shapes(model, config.img_size, device)

    # Load pretrained weights if specified
    if config.pretrained and os.path.exists(config.pretrained):
        logger.info(f"Loading pretrained weights: {config.pretrained}")
        try:
            state_dict = torch.load(config.pretrained, map_location='cpu')
            # Handle different state_dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.error(f"Error loading pretrained weights: {e}")

    # Data parallel if multiple GPUs available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        if config.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # Create dataloaders
    logger.info(f"Creating dataloaders from {config.data_yaml}")
    try:
        train_loader, val_loader, num_classes, class_names = create_dataloaders(
            config.data_yaml,
            batch_size=config.batch_size,
            img_size=config.img_size,
            workers=config.workers
        )
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        return

    # Update model's number of classes if needed
    if hasattr(model, 'num_classes') and model.num_classes != num_classes:
        logger.info(f"Updating model to use {num_classes} classes instead of {model.num_classes}")

        try:
            # For DataParallel model
            if isinstance(model, nn.DataParallel):
                base_model = model.module
            else:
                base_model = model

            # Update task aligned predictors for different scales
            if hasattr(base_model, 'task_predictor_small'):
                base_model.task_predictor_small = TaskAlignedPredictor(256, num_classes)
            if hasattr(base_model, 'task_predictor_medium'):
                base_model.task_predictor_medium = TaskAlignedPredictor(512, num_classes)
            if hasattr(base_model, 'task_predictor_large'):
                base_model.task_predictor_large = TaskAlignedPredictor(1024, num_classes)

            # Update standard detection heads
            if hasattr(base_model, 'small_head'):
                base_model.small_head = DetectionHead(256, num_classes)
            if hasattr(base_model, 'medium_head'):
                base_model.medium_head = DetectionHead(512, num_classes)
            if hasattr(base_model, 'large_head'):
                base_model.large_head = DetectionHead(1024, num_classes)

            # Update model's num_classes attribute
            base_model.num_classes = num_classes
        except Exception as e:
            logger.error(f"Error updating model's class count: {e}")

    # Check dataset
    if config.debug:
        logger.info("Checking training dataset...")
        check_dataset(train_loader, class_names, num_batches=2)
        if val_loader:
            logger.info("\nChecking validation dataset...")
            check_dataset(val_loader, class_names, num_batches=2)

    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    # Create loss function
    loss_fn = YOLOLoss(
        num_classes=num_classes,
        anchors=config.anchors,
        strides=config.strides,
        img_size=config.img_size
    )

    # Create learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        lr_schedule=config.lr_schedule,
        epochs=config.epochs
    )

    # AMP scaler for mixed precision training
    scaler = amp.GradScaler() if config.use_amp else None

    # Create EMA model
    ema = ModelEMA(model) if config.use_ema else None

    # Initialize variables
    best_map = 0.0
    start_epoch = 0

    # Resume training if specified
    if config.resume:
        last_checkpoint = os.path.join(config.save_dir, 'last.pth')
        if os.path.exists(last_checkpoint):
            logger.info(f"Resuming from {last_checkpoint}")
            start_epoch, best_map = load_checkpoint(
                model, optimizer, scheduler, ema, filename=last_checkpoint
            )
            start_epoch += 1  # Start from the next epoch

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.save_dir, 'logs'))

    # Start training
    logger.info(f"Start training for {config.epochs} epochs...")
    try:
        for epoch in range(start_epoch, config.epochs):
            # Train for one epoch
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, loss_fn,
                device, epoch, config, scaler, ema
            )

            # Update learning rate
            scheduler.step()

            # Log training metrics
            for k, v in train_metrics.items():
                writer.add_scalar(f'train/{k}', v, epoch)

            # Save last checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_map=best_map,
                scheduler=scheduler,
                ema=ema,
                filename=os.path.join(config.save_dir, 'last.pth')
            )

            # Evaluate model periodically
            if val_loader and (epoch % config.eval_period == 0 or epoch == config.epochs - 1):
                # Use EMA model if available
                eval_model = ema.ema if ema is not None else model

                # Validate
                try:
                    val_metrics = validate(eval_model, val_loader, device, config, class_names)

                    # Log validation metrics
                    logger.info(f"Epoch {epoch} validation: mAP={val_metrics['mAP']:.4f}, "
                                f"precision={val_metrics['precision']:.4f}, "
                                f"recall={val_metrics['recall']:.4f}")

                    for k, v in val_metrics.items():
                        writer.add_scalar(f'val/{k}', v, epoch)

                    # Save best model
                    if val_metrics['mAP'] > best_map:
                        best_map = val_metrics['mAP']
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            best_map=best_map,
                            scheduler=scheduler,
                            ema=ema,
                            filename=os.path.join(config.save_dir, 'best.pth')
                        )
                except Exception as e:
                    logger.error(f"Error during validation: {e}")

            # Save checkpoint periodically
            if epoch % config.save_period == 0 and epoch > 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_map=best_map,
                    scheduler=scheduler,
                    ema=ema,
                    filename=os.path.join(config.save_dir, f'epoch_{epoch}.pth')
                )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
    finally:
        # Close TensorBoard writer
        writer.close()

        logger.info(f"Training completed. Best mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()
