import yaml

# Create a complete data configuration
data_config = {
    "path": ".",
    "train": "dataset/train",
    "val": "dataset/valid",
    "test": "dataset/test",
    "nc": 7,
    "names": ['bird_drop', 'bird_feather', 'cracked', 'dust_partical', 'healthy', 'leaf', 'snow'],
    "augmentation": {
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0
    },
    "image_size": 640,  # Try multiple naming conventions
    "img_size": 640,
    "imgsz": 640,
    "size": 640,
    "transforms": {
        "size": 640,  # This is likely what it's looking for
        "scale": (0.8, 1.0),
        "hflip": 0.5,
        "vflip": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "interpolation": 0,
        "p": 1.0
    }
}

# Save to YAML file
with open("dataset/data_full.yaml", "w") as f:
    yaml.dump(data_config, f, sort_keys=False)

print("Created dataset/data_full.yaml with all possible fields") 