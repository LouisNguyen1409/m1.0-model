import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNSiLU(nn.Module):
    """
    Convolution + BatchNorm + SiLU activation
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """
    Channel attention module
    """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAMBlock(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) Block
    """

    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SPPModule(nn.Module):
    """
    Spatial Pyramid Pooling module
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.pool1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(13, stride=1, padding=6)
        self.conv2 = ConvBNSiLU(mid_channels * 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        x = torch.cat([x, pool1, pool2, pool3], dim=1)
        x = self.conv2(x)
        return x


class ELANBlock(nn.Module):
    """
    ELAN (Efficient Layer Aggregation Network) Block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNSiLU(mid_channels, mid_channels, 3, padding=1)
        self.conv3 = ConvBNSiLU(mid_channels, mid_channels, 3, padding=1)
        self.conv4 = ConvBNSiLU(mid_channels, mid_channels, 3, padding=1)
        self.conv5 = ConvBNSiLU(mid_channels * 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        x = torch.cat([x, c2, c3, c4], dim=1)
        x = self.conv5(x)
        return x


class CSPDarknet(nn.Module):
    """
    Enhanced CSPDarknet backbone with ELAN blocks
    """

    def __init__(self):
        super().__init__()
        # Initial conv
        self.conv1 = ConvBNSiLU(3, 32, 3, stride=1, padding=1)

        # Downsampling + stage 1
        self.down1 = ConvBNSiLU(32, 64, 3, stride=2, padding=1)
        self.stage1 = ELANBlock(64, 64)

        # Downsampling + stage 2
        self.down2 = ConvBNSiLU(64, 128, 3, stride=2, padding=1)
        self.stage2 = ELANBlock(128, 128)

        # Downsampling + stage 3
        self.down3 = ConvBNSiLU(128, 256, 3, stride=2, padding=1)
        self.stage3 = ELANBlock(256, 256)

        # Downsampling + stage 4
        self.down4 = ConvBNSiLU(256, 512, 3, stride=2, padding=1)
        self.stage4 = ELANBlock(512, 512)

        # Downsampling + stage 5
        self.down5 = ConvBNSiLU(512, 1024, 3, stride=2, padding=1)
        self.stage5 = ELANBlock(1024, 1024)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)

        # Stage 1
        x = self.down1(x)
        c1 = self.stage1(x)

        # Stage 2
        x = self.down2(c1)
        c2 = self.stage2(x)

        # Stage 3
        x = self.down3(c2)
        c3 = self.stage3(x)

        # Stage 4
        x = self.down4(c3)
        c4 = self.stage4(x)

        # Stage 5
        x = self.down5(c4)
        c5 = self.stage5(x)

        return c3, c4, c5  # Return features at different scales


class YOLOD11Neck(nn.Module):
    """
    YOLOD11 Neck: Enhanced PANet for information flow between scales
    """

    def __init__(self):
        super().__init__()
        # Small object branch (high resolution)
        self.cbam_small = CBAMBlock(256)

        # Medium object branch (medium resolution)
        self.cbam_medium = CBAMBlock(512)

        # Large object branch (low resolution)
        self.cbam_large = CBAMBlock(1024)
        self.spp = SPPModule(1024, 1024)

        # Top-down path (large to small)
        self.top_conv1 = ConvBNSiLU(1024, 512, 1)
        self.top_upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_elan1 = ELANBlock(1024, 512)  # 512 + 512 = 1024

        self.top_conv2 = ConvBNSiLU(512, 256, 1)
        self.top_upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_elan2 = ELANBlock(512, 256)  # 256 + 256 = 512

        # Bottom-up path (small to large)
        self.bottom_conv1 = ConvBNSiLU(256, 256, 3, stride=2, padding=1)
        self.bottom_elan1 = ELANBlock(768, 512)  # 256 + 512 = 768 channels

        self.bottom_conv2 = ConvBNSiLU(512, 512, 3, stride=2, padding=1)
        self.bottom_elan2 = ELANBlock(1536, 1024)  # 512 + 1024 = 1536 channels

        # Output layers
        self.out_small = ELANBlock(256, 256)
        self.out_medium = ELANBlock(512, 512)
        self.out_large = ELANBlock(1024, 1024)

    def forward(self, features):
        c3, c4, c5 = features

        # Apply CBAM blocks and SPP to each feature level
        p3 = self.cbam_small(c3)
        p4 = self.cbam_medium(c4)
        p5 = self.cbam_large(c5)
        p5 = self.spp(p5)

        # Top-down path
        p5_td = self.top_conv1(p5)
        p5_td_upsampled = self.top_upsample1(p5_td)
        p4 = torch.cat([p5_td_upsampled, p4], dim=1)
        p4 = self.top_elan1(p4)

        p4_td = self.top_conv2(p4)
        p4_td_upsampled = self.top_upsample2(p4_td)
        p3 = torch.cat([p4_td_upsampled, p3], dim=1)
        p3 = self.top_elan2(p3)

        # Bottom-up path
        p3_bu = self.bottom_conv1(p3)
        p4 = torch.cat([p3_bu, p4], dim=1)
        p4 = self.bottom_elan1(p4)

        p4_bu = self.bottom_conv2(p4)
        p5 = torch.cat([p4_bu, p5], dim=1)
        p5 = self.bottom_elan2(p5)

        # Output feature maps
        p3 = self.out_small(p3)
        p4 = self.out_medium(p4)
        p5 = self.out_large(p5)

        return p3, p4, p5


class DetectionHead(nn.Module):
    """
    Detection head for object classification and bounding box regression
    """

    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Conv layers before prediction
        self.conv1 = ConvBNSiLU(in_channels, in_channels, 3, padding=1)
        self.conv2 = ConvBNSiLU(in_channels, in_channels, 3, padding=1)

        # Prediction layers
        # For each anchor: [x, y, w, h, objectness, class_1, class_2, ..., class_n]
        self.pred = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pred(x)


class TaskAlignedPredictor(nn.Module):
    """
    Task-aligned predictor to improve prediction quality
    """

    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.classification = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, 3, padding=1),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )

        self.regression = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, 3, padding=1),
            nn.Conv2d(in_channels, num_anchors * 4, 1)  # x, y, w, h
        )

        self.objectness = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, 3, padding=1),
            nn.Conv2d(in_channels, num_anchors, 1)
        )

    def forward(self, x):
        cls = self.classification(x)
        reg = self.regression(x)
        obj = self.objectness(x)

        return torch.cat([reg, obj, cls], dim=1)


class YOLOD11(nn.Module):
    """
    YOLOD11 model with ELAN-CSPDarknet backbone
    """

    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = CSPDarknet()
        self.neck = YOLOD11Neck()

        # Detection heads for different scales
        self.small_head = DetectionHead(256, num_classes)
        self.medium_head = DetectionHead(512, num_classes)
        self.large_head = DetectionHead(1024, num_classes)

        # Task-aligned predictor for final predictions
        self.task_predictor_small = TaskAlignedPredictor(256, num_classes)
        self.task_predictor_medium = TaskAlignedPredictor(512, num_classes)
        self.task_predictor_large = TaskAlignedPredictor(1024, num_classes)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # Feature enhancement through neck
        p3, p4, p5 = self.neck(features)

        # Apply detection heads
        small_pred = self.task_predictor_small(p3)
        medium_pred = self.task_predictor_medium(p4)
        large_pred = self.task_predictor_large(p5)

        if self.training:
            return small_pred, medium_pred, large_pred

        # Apply post-processing for inference (e.g., NMS)
        # This would typically be done outside the model during inference
        return small_pred, medium_pred, large_pred
