"""MNIST 读取与单张图像预处理。"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parents[1]


def load_mnist(data_dir: Path, train: bool) -> TensorDataset:
    """读取官方训练集或测试集，将全部像素一次性归一化到 [-1, 1]。"""
    dataset = MNIST(root=str(data_dir), train=train, download=True)
    # 原始图像为 uint8 的 [N, 28, 28]，标签为 int64 的 [N]。
    # 预先转换可避免每轮重复做相同的灰度值归一化。
    images = dataset.data.unsqueeze(1).to(torch.float32)
    images.div_(255.0).mul_(2.0).sub_(1.0)
    return TensorDataset(images, dataset.targets)


def load_image(file: Path, invert: bool = False) -> torch.Tensor:
    """将单张图片变为 [1, 1, 28, 28]；白底黑字时显式指定 invert。"""
    with Image.open(file) as image:
        image = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        pixels = np.array(image, dtype=np.float32)

    # MNIST 为黑底白字；训练与预测必须采用相同的像素方向和数值范围。
    if invert:
        pixels = 255.0 - pixels
    pixels = pixels / 255.0 * 2.0 - 1.0
    return torch.from_numpy(pixels).reshape(1, 1, 28, 28)
