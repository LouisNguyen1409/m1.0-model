import os
import sys
import argparse
import torch
import yaml
import json
import onnx
import tempfile
import numpy as np
import warnings
from pathlib import Path
import datetime
import subprocess

# Import model
from model import YOLOD11


def load_model(checkpoint_path, device, num_classes=7):
    """
    Load YOLOD11 model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        num_classes: Number of classes

    Returns:
        model: Loaded model
        anchors: Anchor boxes used for prediction
        strides: Strides for each detection scale
    """
    # Initialize model
    model = YOLOD11(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract anchors and strides if available in checkpoint
    anchors = None
    strides = [8, 16, 32]  # Default strides

    # Try to get anchors from checkpoint
    if 'anchors' in checkpoint:
        anchors = checkpoint['anchors']
    else:
        # Use default anchors
        anchors = [
            torch.tensor([[10, 13], [16, 30], [33, 23]]),  # Small scale
            torch.tensor([[30, 61], [62, 45], [59, 119]]),  # Medium scale
            torch.tensor([[116, 90], [156, 198], [373, 326]])  # Large scale
        ]

    # Move model to device
    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation Loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"mAP: {checkpoint.get('mAP', 'unknown')}")

    return model, anchors, strides


def export_torchscript(model, img_size=640, save_path=None):
    """
    Export model to TorchScript format

    Args:
        model: PyTorch model
        img_size: Input image size
        save_path: Path to save exported model

    Returns:
        script_model: TorchScript model
    """
    print(f"Exporting model to TorchScript format...")

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.zeros(1, 3, img_size, img_size, device=next(model.parameters()).device)

    # Trace the model
    with torch.no_grad():
        script_model = torch.jit.trace(model, dummy_input)

    # Save model if path provided
    if save_path:
        script_model.save(save_path)
        print(f"TorchScript model saved to {save_path}")

    return script_model


def export_onnx(model, img_size=640, save_path=None, opset_version=12, simplify=True):
    """
    Export model to ONNX format

    Args:
        model: PyTorch model
        img_size: Input image size
        save_path: Path to save exported model
        opset_version: ONNX opset version
        simplify: Whether to simplify ONNX model

    Returns:
        onnx_path: Path to exported ONNX model
    """
    try:
        try:
            from onnxsim import simplify as onnxsim_simplify
        except ImportError:
            warnings.warn("ONNX simplification requires onnx-simplifier package. Please install it with: "
                          "pip install onnx-simplifier")
            simplify = False
    except ImportError:
        warnings.warn("ONNX export requires onnx package. Please install it with: "
                      "pip install onnx")
        return None

    print(f"Exporting model to ONNX format (opset {opset_version})...")

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.zeros(1, 3, img_size, img_size, device=next(model.parameters()).device)

    # Create temporary file if no save path is provided
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        save_path = tmp.name
        tmp.close()

    # Export model to ONNX
    input_names = ['input']
    output_names = ['output_small', 'output_medium', 'output_large']

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output_small': {0: 'batch_size'},
            'output_medium': {0: 'batch_size'},
            'output_large': {0: 'batch_size'}
        }
    )

    # Check ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    # Simplify model if requested
    if simplify:
        try:
            model_simple, check = onnxsim_simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simple, save_path)
            print(f"ONNX model simplified")
        except Exception as e:
            warnings.warn(f"ONNX simplification failed: {e}")

    print(f"ONNX model saved to {save_path}")
    return save_path


def export_tensorrt(onnx_path, img_size=640, save_path=None, fp16=False, int8=False, workspace=8):
    """
    Export ONNX model to TensorRT format

    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
        save_path: Path to save exported model
        fp16: Whether to use FP16 precision
        int8: Whether to use INT8 precision
        workspace: Maximum workspace size in GB

    Returns:
        engine_path: Path to exported TensorRT engine
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # This is needed to initialize CUDA
    except ImportError:
        warnings.warn("TensorRT export requires tensorrt and pycuda packages. Please install them.")
        return None

    print(f"Exporting model to TensorRT format...")

    # Create temporary file if no save path is provided
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.engine', delete=False)
        save_path = tmp.name
        tmp.close()

    # Create logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    # Create network
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # Create parser and parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Create config
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30  # Convert to bytes

    # Set precision
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision")

        # INT8 calibrator would be set here if available
        # This is a simplified example, a proper calibration dataset would be needed
        warnings.warn("INT8 calibration not implemented, using default calibration")

    # Create engine
    engine = builder.build_engine(network, config)

    # Save engine
    with open(save_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {save_path}")
    return save_path


def export_openvino(onnx_path, img_size=640, save_path=None):
    """
    Export ONNX model to OpenVINO format

    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
        save_path: Path to save exported model

    Returns:
        ir_path: Path to exported OpenVINO IR model
    """
    try:
        # Check if OpenVINO is installed
        subprocess.check_output(['mo', '--version'], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn("OpenVINO export requires OpenVINO toolkit. Please install it.")
        return None

    print(f"Exporting model to OpenVINO format...")

    # Create output directory
    if save_path is None:
        output_dir = tempfile.mkdtemp()
        save_path = os.path.join(output_dir, "model")
    else:
        output_dir = os.path.dirname(save_path)
        os.makedirs(output_dir, exist_ok=True)

    # Use model optimizer to convert ONNX to OpenVINO IR
    cmd = [
        "mo",
        "--input_model", onnx_path,
        "--output_dir", output_dir,
        "--model_name", os.path.basename(save_path),
        "--input", "input",
        "--input_shape", f"[1,3,{img_size},{img_size}]",
        "--data_type", "FP32"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"OpenVINO model saved to {save_path}.xml and {save_path}.bin")
        return save_path + ".xml"
    except subprocess.CalledProcessError as e:
        warnings.warn(f"OpenVINO conversion failed: {e}")
        return None


def export_tflite(onnx_path, img_size=640, save_path=None, quantize=False):
    """
    Export ONNX model to TensorFlow Lite format

    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
        save_path: Path to save exported model
        quantize: Whether to quantize model

    Returns:
        tflite_path: Path to exported TFLite model
    """
    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError:
        warnings.warn("TFLite export requires tensorflow and tf2onnx packages. Please install them.")
        return None

    print(f"Exporting model to TensorFlow Lite format...")

    # Create temporary file if no save path is provided
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.tflite', delete=False)
        save_path = tmp.name
        tmp.close()

    # Create TF saved model from ONNX
    tf_model_path = os.path.splitext(save_path)[0] + "_tf_saved_model"
    os.makedirs(tf_model_path, exist_ok=True)

    # Convert ONNX to TF SavedModel
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--input", onnx_path,
        "--output", tf_model_path,
        "--opset", "12"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        warnings.warn(f"TF conversion failed: {e}")
        return None

    # Load SavedModel
    saved_model = tf.saved_model.load(tf_model_path)

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

    # Set optimization flags
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Using quantization")

    # Convert to TFLite
    tflite_model = converter.convert()

    # Save model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TensorFlow Lite model saved to {save_path}")
    return save_path


def export_coreml(onnx_path, img_size=640, save_path=None):
    """
    Export ONNX model to Apple Core ML format

    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
        save_path: Path to save exported model

    Returns:
        coreml_path: Path to exported Core ML model
    """
    try:
        import coremltools as ct
    except ImportError:
        warnings.warn("Core ML export requires coremltools package. Please install it.")
        return None

    print(f"Exporting model to Apple Core ML format...")

    # Create temporary file if no save path is provided
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.mlmodel', delete=False)
        save_path = tmp.name
        tmp.close()

    # Load ONNX model
    onnx_model = ct.converters.onnx.load(onnx_path)

    # Set input and output descriptions
    input_desc = ct.ImageType(name="input", shape=(1, 3, img_size, img_size),
                              bias=[-0.485, -0.456, -0.406],
                              scale=[1/0.229, 1/0.224, 1/0.225])

    # Convert model
    coreml_model = ct.converters.onnx.convert(
        model=onnx_model,
        minimum_ios_deployment_target='13',
        inputs=[input_desc]
    )

    # Set model metadata
    coreml_model.author = "YOLOD11 Exporter"
    coreml_model.license = "MIT"
    coreml_model.short_description = "YOLOD11 Object Detection Model"

    # Save model
    coreml_model.save(save_path)

    print(f"Core ML model saved to {save_path}")
    return save_path


def create_config_file(model_format, exported_path, anchors, strides, class_names, img_size=640, save_path=None):
    """
    Create a configuration file for the exported model

    Args:
        model_format: Format of the exported model
        exported_path: Path to the exported model
        anchors: Anchor boxes used for prediction
        strides: Strides for each detection scale
        class_names: List of class names
        img_size: Input image size
        save_path: Path to save configuration file

    Returns:
        config_path: Path to configuration file
    """
    print(f"Creating configuration file...")

    # Create temporary file if no save path is provided
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        save_path = tmp.name
        tmp.close()

    # Convert anchors to list for JSON serialization
    anchors_list = []
    for anchor_set in anchors:
        if isinstance(anchor_set, torch.Tensor):
            anchors_list.append(anchor_set.tolist())
        else:
            anchors_list.append(anchor_set)

    # Create configuration dictionary
    config = {
        'model_format': model_format,
        'model_path': exported_path,
        'anchors': anchors_list,
        'strides': strides,
        'class_names': class_names,
        'input_size': img_size,
        'export_date': f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    }

    # Save configuration
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration file saved to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="YOLOD11 Model Export")

    # Required arguments
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-yaml", required=True, help="Path to data YAML file")

    # Export format
    parser.add_argument("--format", default="onnx",
                        choices=["torchscript", "onnx", "tensorrt", "openvino", "tflite", "coreml", "all"],
                        help="Export format")

    # Output directory
    parser.add_argument("--output-dir", default="./exported_models", help="Output directory")

    # Model parameters
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")

    # Format-specific options
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--half", action="store_true", help="Export in half precision (FP16)")
    parser.add_argument("--int8", action="store_true", help="Export in INT8 precision (TensorRT only)")
    parser.add_argument("--workspace", type=int, default=8, help="TensorRT maximum workspace size (GB)")
    parser.add_argument("--quantize", action="store_true", help="Quantize TFLite model")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load YAML configuration
    with open(args.data_yaml, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    # Get class names
    class_names = yaml_cfg.get('names', [])
    num_classes = yaml_cfg.get('nc', len(class_names))

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Load model
    model, anchors, strides = load_model(args.model, device, num_classes)

    # Base name for saved models
    base_name = os.path.splitext(os.path.basename(args.model))[0]

    # Export model based on specified format
    if args.format == "torchscript" or args.format == "all":
        ts_path = os.path.join(args.output_dir, f"{base_name}_torchscript.pt")
        script_model = export_torchscript(model, args.img_size, ts_path)

        if script_model is not None:
            config_path = os.path.join(args.output_dir, f"{base_name}_torchscript_config.json")
            create_config_file("torchscript", ts_path, anchors, strides, class_names, args.img_size, config_path)

    if args.format == "onnx" or args.format == "all" or args.format in ["tensorrt", "openvino", "tflite", "coreml"]:
        onnx_path = os.path.join(args.output_dir, f"{base_name}_onnx.onnx")
        onnx_path = export_onnx(model, args.img_size, onnx_path, args.opset)

        if onnx_path is not None:
            config_path = os.path.join(args.output_dir, f"{base_name}_onnx_config.json")
            create_config_file("onnx", onnx_path, anchors, strides, class_names, args.img_size, config_path)

    if args.format == "tensorrt" or args.format == "all":
        if onnx_path is not None:
            trt_path = os.path.join(args.output_dir, f"{base_name}_tensorrt.engine")
            trt_path = export_tensorrt(onnx_path, args.img_size, trt_path, args.half, args.int8, args.workspace)

            if trt_path is not None:
                config_path = os.path.join(args.output_dir, f"{base_name}_tensorrt_config.json")
                create_config_file("tensorrt", trt_path, anchors, strides, class_names, args.img_size, config_path)

    if args.format == "openvino" or args.format == "all":
        if onnx_path is not None:
            openvino_path = os.path.join(args.output_dir, f"{base_name}_openvino")
            openvino_path = export_openvino(onnx_path, args.img_size, openvino_path)

            if openvino_path is not None:
                config_path = os.path.join(args.output_dir, f"{base_name}_openvino_config.json")
                create_config_file("openvino", openvino_path, anchors, strides, class_names, args.img_size, config_path)

    if args.format == "tflite" or args.format == "all":
        if onnx_path is not None:
            tflite_path = os.path.join(args.output_dir, f"{base_name}_tflite.tflite")
            tflite_path = export_tflite(onnx_path, args.img_size, tflite_path, args.quantize)

            if tflite_path is not None:
                config_path = os.path.join(args.output_dir, f"{base_name}_tflite_config.json")
                create_config_file("tflite", tflite_path, anchors, strides, class_names, args.img_size, config_path)

    if args.format == "coreml" or args.format == "all":
        if onnx_path is not None:
            coreml_path = os.path.join(args.output_dir, f"{base_name}_coreml.mlmodel")
            coreml_path = export_coreml(onnx_path, args.img_size, coreml_path)

            if coreml_path is not None:
                config_path = os.path.join(args.output_dir, f"{base_name}_coreml_config.json")
                create_config_file("coreml", coreml_path, anchors, strides, class_names, args.img_size, config_path)

    print("Export completed!")


if __name__ == "__main__":
    main()
