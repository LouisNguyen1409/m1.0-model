import os
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import YOLODataset, collate_fn

def main():
    parser = argparse.ArgumentParser(description='Direct Training Script')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--device', default='cuda', help='Device')
    args = parser.parse_args()
    
    # Hardcoded paths
    train_img_dir = "./dataset/train/images"
    train_label_dir = "./dataset/train/labels"
    val_img_dir = "./dataset/valid/images"
    val_label_dir = "./dataset/valid/labels"
    
    # Class names
    class_names = ['bird_drop', 'bird_feather', 'cracked', 'dust_partical', 'healthy', 'leaf', 'snow']
    
    print(f"Using device: {args.device}")
    
    # Check paths
    for path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        if not os.path.exists(path):
            print(f"WARNING: Path does not exist: {path}")
        else:
            print(f"Path exists: {path}")
            # Count files in directory
            files = os.listdir(path)
            print(f"  - Contains {len(files)} files")
    
    # Create datasets
    try:
        print(f"Creating training dataset from: {train_img_dir} and {train_label_dir}")
        train_dataset = YOLODataset(
            train_img_dir,
            train_label_dir,
            img_size=args.img_size,
            augment=True,
            class_names=class_names
        )
        print(f"Training dataset created with {len(train_dataset)} samples")
        
        print(f"Creating validation dataset from: {val_img_dir} and {val_label_dir}")
        val_dataset = YOLODataset(
            val_img_dir,
            val_label_dir,
            img_size=args.img_size,
            augment=False,
            class_names=class_names
        )
        print(f"Validation dataset created with {len(val_dataset)} samples")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        print(f"Training dataloader created with {len(train_loader)} batches")
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        print(f"Validation dataloader created with {len(val_loader)} batches")
        
        # Try to iterate through the first batch
        print("Trying to load first batch...")
        for batch_idx, (images, targets, paths) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images shape {images.shape}, Targets shape {targets.shape if isinstance(targets, torch.Tensor) else [t.shape for t in targets]}")
            if batch_idx == 0:
                break
                
        print("Successfully loaded first batch!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main() 