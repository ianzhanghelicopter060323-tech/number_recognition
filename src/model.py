"""任务书中多层感知器的 PyTorch 实现。"""

import torch
from torch import nn


class MultilayerPerceptron(nn.Module):
    """将 28×28 灰度图像映射为 0～9 共十类的分数。"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输入形状为 [N, 1, 28, 28]，输出形状为 [N, 10]。"""
        x = images.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # CrossEntropyLoss 内部处理 LogSoftmax；训练前不重复做 Softmax。
        return self.fc3(x)
