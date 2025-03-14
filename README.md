# YOLOD11: YOLO-based Object Detection for Solar Panel Inspection

YOLOD11 is a custom object detection model based on YOLO architecture with enhancements specifically designed for solar panel inspection tasks. The model includes several state-of-the-art techniques such as spatial and channel attention, complete IoU (CIoU) loss, and task-aligned predictors for improved detection performance.

## Project Structure

```
yolod11/
├── model.py                 # Contains the YOLOD11 model architecture
├── datasets.py              # Contains the SolarPanelDataset class
├── train.py                 # Main training script
├── inference.py             # Script for running inference
├── export.py                # Script for exporting models
└── utils/
    ├── __init__.py          # Makes utils a proper package
    ├── yolo_utils.py        # Contains IoU calculations and prediction processing
    ├── loss.py              # Contains the YOLOD11Loss implementation
    ├── visualization.py     # Contains visualization utilities
    ├── augmentation.py      # Contains data augmentation tools
    └── training.py          # Contains training helper functions
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/yolod11.git
cd yolod11
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset should follow the YOLO format with images and labels organized as follows:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── val/
        ├── img1.txt
        ├── img2.txt
        └── ...
```

Label files should follow the YOLO format: `class_id x_center y_center width height` (all normalized between 0 and 1).

## Data Configuration

Create a `data.yaml` file with the following format:

```yaml
train: images/train
val: images/val
test: images/test

nc: 7 # Number of classes
names: ['a', 'b', 'c', 'd', 'e', 'f', 'g'] # Class names
```

## Training

To train the model:

```bash
python train.py --data-root /path/to/dataset --data-yaml data.yaml --output-dir ./output --batch-size 16 --epochs 100 --cuda
```

Arguments:

- `--data-root`: Path to the dataset directory
- `--data-yaml`: Path to the data YAML configuration file
- `--output-dir`: Directory to save outputs
- `--batch-size`: Batch size for training
- `--epochs`: Number of epochs to train for
- `--lr`: Initial learning rate (default: 0.001)
- `--weight-decay`: Weight decay for optimizer (default: 0.0005)
- `--num-workers`: Number of worker threads for data loading (default: 4)
- `--cuda`: Use CUDA if available
- `--save-interval`: Save model every N epochs (default: 5)
- `--resume`: Path to checkpoint to resume training from

## Inference

To run inference on images or videos:

```bash
python inference.py --model ./output/best_map_model.pth --data-yaml data.yaml --image /path/to/image.jpg --show --save
```

Arguments:

- `--model`: Path to model checkpoint
- `--data-yaml`: Path to data YAML configuration file
- `--image`: Path to single image
- `--images-dir`: Directory containing images (alternative to `--image`)
- `--video`: Path to video file (alternative to `--image`)
- `--output-dir`: Output directory (default: ./inference_results)
- `--show`: Show results
- `--save`: Save results
- `--img-size`: Input image size (default: 640)
- `--conf-thres`: Confidence threshold (default: 0.25)
- `--iou-thres`: IoU threshold for NMS (default: 0.45)
- `--device`: Device to run inference on (default: cuda if available, otherwise cpu)

## Export

To export the model to different formats for deployment:

```bash
python export.py --model ./output/best_map_model.pth --data-yaml data.yaml --format onnx --output-dir ./exported_models
```

Arguments:

- `--model`: Path to model checkpoint
- `--data-yaml`: Path to data YAML configuration file
- `--format`: Export format (choices: torchscript, onnx, tensorrt, openvino, tflite, coreml, all)
- `--output-dir`: Output directory (default: ./exported_models)
- `--img-size`: Input image size (default: 640)
- `--opset`: ONNX opset version (default: 12)
- `--half`: Export in half precision (FP16)
- `--int8`: Export in INT8 precision (TensorRT only)
- `--workspace`: TensorRT maximum workspace size in GB (default: 8)
- `--quantize`: Quantize TFLite model

## Model Architecture

YOLOD11 incorporates several enhancements to the standard YOLO architecture:

1. **Backbone**: Enhanced CSPDarknet with ELAN (Efficient Layer Aggregation Network) blocks
2. **Neck**: Enhanced PANet with CRAM (Channel and Spatial Attention) blocks and SPP (Spatial Pyramid Pooling)
3. **Head**: Task-aligned predictors for improved classification and localization
4. **Loss**: Complete IoU (CIoU) loss for better bounding box regression

## Key Features

- **Attention Mechanisms**: Channel and spatial attention for enhanced feature extraction
- **Complete IoU Loss**: Improved bounding box regression with CIoU loss
- **Data Augmentation**: Comprehensive augmentation techniques including MixUp and Mosaic
- **Anchor Generation**: Optimal anchor box generation using k-means clustering
- **Task-Aligned Predictors**: Separate predictors for classification and localization tasks

## Evaluation Metrics

Performance is evaluated using:

- Mean Average Precision (mAP) at 0.5 IoU threshold
- Precision and recall for each class
- Box loss, objectness loss, and classification loss
