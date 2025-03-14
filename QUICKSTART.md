# YOLOD11 Quick Start Guide

This guide provides step-by-step instructions to get started with YOLOD11 for solar panel inspection.

## Setting Up Your Environment

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/yolod11.git
   cd yolod11
   ```

2. **Install the dependencies**:

   ```bash
   pip install torch torchvision tqdm matplotlib opencv-python pillow tensorboard pyyaml
   ```

3. **Create the directory structure**:

   ```bash
   mkdir -p utils
   touch utils/__init__.py
   ```

4. **Copy the code files to their respective locations** as shown in the [project structure](#project-structure).

## Project Structure

Make sure your files are organized as follows:

```
yolod11/
├── model.py                 # YOLOD11 model architecture
├── datasets.py              # SolarPanelDataset class
├── train.py                 # Training script
├── inference.py             # Inference script
├── export.py                # Model export script
└── utils/
    ├── __init__.py          # Package initialization
    ├── yolo_utils.py        # IoU and prediction processing
    ├── loss.py              # YOLOD11Loss implementation
    ├── visualization.py     # Visualization utilities
    ├── augmentation.py      # Data augmentation tools
    └── training.py          # Training helper functions
```

## Preparing Your Dataset

1. **Organize your dataset** in YOLO format:

   ```
   dataset/
   ├── images/
   │   ├── train/
   │   │   ├── img1.jpg
   │   │   └── ...
   │   └── val/
   │       ├── img1.jpg
   │       └── ...
   └── labels/
       ├── train/
       │   ├── img1.txt
       │   └── ...
       └── val/
           ├── img1.txt
           └── ...
   ```

2. **Create a data.yaml file** with your dataset configuration:

   ```yaml
   train: images/train
   val: images/val
   test: images/test

   nc: 7 # Number of classes
   names: ['crack', 'delamination', 'disconnection', 'hotspot', 'soiling', 'corrosion', 'cell_defect']
   ```

## Training the Model

1. **Start training**:

   ```bash
   python train.py --data-root /path/to/dataset \
                   --data-yaml data.yaml \
                   --output-dir ./output \
                   --batch-size 16 \
                   --epochs 100 \
                   --cuda
   ```

2. **Monitor training progress**:

   ```bash
   tensorboard --logdir ./output/tensorboard
   ```

3. **Resume training** if needed:
   ```bash
   python train.py --data-root /path/to/dataset \
                   --data-yaml data.yaml \
                   --output-dir ./output \
                   --batch-size 16 \
                   --epochs 100 \
                   --cuda \
                   --resume ./output/latest_model.pth
   ```

## Running Inference

1. **Inference on a single image**:

   ```bash
   python inference.py --model ./output/best_map_model.pth \
                      --data-yaml data.yaml \
                      --image /path/to/image.jpg \
                      --show \
                      --save
   ```

2. **Inference on a directory of images**:

   ```bash
   python inference.py --model ./output/best_map_model.pth \
                      --data-yaml data.yaml \
                      --images-dir /path/to/images \
                      --save
   ```

3. **Inference on a video**:
   ```bash
   python inference.py --model ./output/best_map_model.pth \
                      --data-yaml data.yaml \
                      --video /path/to/video.mp4 \
                      --save
   ```

## Exporting the Model

1. **Export to ONNX format**:

   ```bash
   python export.py --model ./output/best_map_model.pth \
                   --data-yaml data.yaml \
                   --format onnx \
                   --output-dir ./exported_models
   ```

2. **Export to multiple formats**:
   ```bash
   python export.py --model ./output/best_map_model.pth \
                   --data-yaml data.yaml \
                   --format all \
                   --output-dir ./exported_models
   ```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA out of memory error**:

   - Reduce batch size
   - Reduce image size
   - Use mixed precision training

2. **Import errors**:

   - Make sure your utils directory has an **init**.py file
   - Check your Python path

3. **Dataset loading errors**:

   - Verify the dataset path in data.yaml
   - Check that image and label directories match

4. **Training not improving**:
   - Adjust learning rate
   - Check data quality and annotations
   - Try different augmentation techniques
