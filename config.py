import yaml
import os
import argparse


class YOLOConfig:
    """
    Configuration class for YOLO model training
    """

    def __init__(self, config_path=None):
        # Default configuration
        self.model_name = 'YOLOD11'
        self.data_yaml = 'data/coco.yaml'
        self.img_size = 640
        self.batch_size = 16
        self.epochs = 300
        self.workers = 8
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.momentum = 0.937
        self.warmup_epochs = 3
        self.warmup_bias_lr = 0.1
        self.warmup_momentum = 0.8
        self.save_dir = 'runs/train'
        self.save_period = 10
        self.eval_period = 5
        self.device = 'cuda'
        self.sync_bn = False
        self.resume = False
        self.pretrained = ''
        self.freeze = []
        self.use_amp = False
        self.use_ema = True
        self.lr_schedule = 'cosine'
        self.anchors = [
            # Small objects (stride 8)
            [10, 13, 16, 30, 33, 23],
            # Medium objects (stride 16)
            [30, 61, 62, 45, 59, 119],
            # Large objects (stride 32)
            [116, 90, 156, 198, 373, 326]
        ]
        self.strides = [8, 16, 32]
        self.debug = False

        # Load from config file if provided
        if config_path is not None and os.path.exists(config_path):
            self.load_from_file(config_path)

    def load_from_file(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            # Update config with values from file
            for key, value in cfg.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")

    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def save_to_file(self, filename):
        """Save configuration to YAML file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Convert config to dict
            config_dict = {k: v for k, v in self.__dict__.items()}

            # Save to file
            with open(filename, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            print(f"Configuration saved to {filename}")
        except Exception as e:
            print(f"Error saving config to {filename}: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOD11 Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--data-yaml', type=str, default=None, help='Path to data YAML file')
    parser.add_argument('--img-size', type=int, default=None, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    return parser.parse_args()
