"""加载 PyTorch 模型并识别一张手写数字图片。"""

from pathlib import Path

import torch

from .data import load_image
from .model import MultilayerPerceptron


def predict_image(image_path: Path, model_path: Path, invert: bool = False) -> None:
    """读取黑底白字图片，输出预测数字与十类概率。"""
    model = MultilayerPerceptron()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    image = load_image(image_path, invert=invert)
    with torch.inference_mode():
        probabilities = model(image).softmax(dim=1)[0]
    label = probabilities.argmax().item()
    print("预测数字：{}，概率：{:.2%}".format(label, probabilities.max().item()))
    print("各类别概率（0～9）：", [round(value, 6) for value in probabilities.tolist()])
