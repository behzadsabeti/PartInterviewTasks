import torch.nn as nn
import torch.nn.functional as F


# --- Building blocks ---

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Two conv layers with an identity skip connection."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
    def forward(self, x):
        return F.relu(self.net(x) + x, inplace=True)


# --- Architectures ---

class SimpleCNN(nn.Module):
    """3-block baseline CNN (~0.5 M params)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(3, 32),   ConvBnRelu(32, 32),   nn.MaxPool2d(2), nn.Dropout2d(0.25),
            ConvBnRelu(32, 64),  ConvBnRelu(64, 64),   nn.MaxPool2d(2), nn.Dropout2d(0.25),
            ConvBnRelu(64, 128), ConvBnRelu(128, 128),  nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(128, num_classes))
    def forward(self, x):
        return self.head(self.features(x))


class DeepCNN(nn.Module):
    """4-block deeper CNN with narrower channels (~250K params)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(3, 16),   ConvBnRelu(16, 16),   nn.MaxPool2d(2), nn.Dropout2d(0.2),
            ConvBnRelu(16, 32),  ConvBnRelu(32, 32),   nn.MaxPool2d(2), nn.Dropout2d(0.2),
            ConvBnRelu(32, 64),  ConvBnRelu(64, 64),   nn.MaxPool2d(2), nn.Dropout2d(0.2),
            ConvBnRelu(64, 96),  ConvBnRelu(96, 96),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(96, num_classes),
        )
    def forward(self, x):
        return self.head(self.features(x))


class ResNetCIFAR(nn.Module):
    """Lightweight ResNet for 32x32 inputs (~316K params)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem   = ConvBnRelu(3, 32)
        self.stage1 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.down1  = ConvBnRelu(32, 64, stride=2)
        self.stage2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down2  = ConvBnRelu(64, 64, stride=2)
        self.stage3 = nn.Sequential(ResBlock(64))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down1(x))
        x = self.stage3(self.down2(x))
        return self.head(x)


# --- Factory ---

_REGISTRY = {"SimpleCNN": SimpleCNN, "DeepCNN": DeepCNN, "ResNetCIFAR": ResNetCIFAR}

def get_model(name, num_classes=10):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
