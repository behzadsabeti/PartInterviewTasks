"""
models.py

All CNN architecture definitions live here.
The notebook only calls get_model(name) — no architecture code leaks out.

Architectures to implement:

  SimpleCNN
    - 3 blocks of [Conv → BN → ReLU] × 2 + MaxPool
    - Global average pool → Linear classifier
    - Goal: a fast baseline (~few hundred K params)

  DeepCNN
    - Same block structure but more filters and more blocks
    - Helps study the effect of depth / width vs SimpleCNN

  ResNetCIFAR
    - Residual blocks (skip connections) adapted for 32×32 inputs
    - No pre-trained weights; trained from scratch
    - Helps study whether skip connections improve training

Helpers to implement:
  - ConvBnRelu  : reusable Conv → BatchNorm → ReLU building block
  - ResidualBlock : two conv layers with an identity shortcut
  - get_model(name) → nn.Module  : factory so the notebook is model-agnostic
  - count_parameters(model) → int : quick param count for comparison table
"""

# TODO: imports (torch, torch.nn, etc.)


# TODO: class ConvBnRelu(nn.Sequential)
#         Conv2d(bias=False) → BatchNorm2d → ReLU


# TODO: class ResidualBlock(nn.Module)
#         two ConvBnRelu layers + identity skip connection


# TODO: class SimpleCNN(nn.Module)
#         __init__: build self.features (conv blocks) + self.classifier (FC head)
#         forward : features → classifier


# TODO: class DeepCNN(nn.Module)
#         same pattern as SimpleCNN but wider/deeper


# TODO: class ResNetCIFAR(nn.Module)
#         stem conv → stage1 (residual blocks) → downsample
#                   → stage2                  → downsample
#                   → stage3 → global avg pool → Linear


# TODO: get_model(name, num_classes=10) -> nn.Module
#         a dict-based registry so adding new architectures is one line


# TODO: count_parameters(model) -> int
