"""按任务书训练 MNIST 多层感知器，并保存模型、指标和图片。"""

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

from .data import load_mnist
from .model import MultilayerPerceptron

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@torch.inference_mode()
def evaluate(model: nn.Module, reader: DataLoader, device: torch.device) -> Dict[str, float]:
    """按实际样本数统计平均交叉熵与准确率，包含最后一个不足批次。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for images, labels in reader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def draw_train_process(history: List[Dict[str, float]], output_dir: Path) -> None:
    """分别绘制损失和准确率，避免两种量共用同一纵轴。"""
    epochs = [row["epoch"] for row in history]
    fig = Figure(figsize=(11, 4))
    axes = fig.subplots(1, 2)
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(epochs, [row["train_" + metric] for row in history], label="Train")
        axis.plot(epochs, [row["test_" + metric] for row in history], label="Test")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title="MNIST " + metric)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


@torch.inference_mode()
def save_predictions(model: nn.Module, dataset: TensorDataset,
                     device: torch.device, output_dir: Path) -> None:
    """展示测试集前 16 张的真实标签与预测，并导出第一张供独立预测。"""
    model.eval()
    images, labels = dataset.tensors
    probabilities = model(images[:16].to(device)).softmax(dim=1).cpu()
    predictions = probabilities.argmax(dim=1)
    fig = Figure(figsize=(7, 7))
    axes = fig.subplots(4, 4)
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        correct = predictions[index].item() == labels[index].item()
        axis.set_title(
            "True: {}  Pred: {}".format(labels[index].item(), predictions[index].item()),
            color="green" if correct else "red",
        )
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "test_predictions.png", dpi=160)
    plt.close(fig)

    pixels = ((images[0, 0] + 1.0) * 127.5).round().to(torch.uint8).numpy()
    Image.fromarray(pixels).save(output_dir / "sample_digit.png")
    sample = {
        "test_index": 0,
        "true_label": labels[0].item(),
        "predicted_label": predictions[0].item(),
        "probabilities": probabilities[0].tolist(),
    }
    (output_dir / "sample_prediction.json").write_text(
        json.dumps(sample, indent=2), encoding="utf-8"
    )


def train_and_evaluate(args: argparse.Namespace) -> None:
    """按 main.py 提供的参数完成训练、评估、模型保存和绘图。"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step1：训练集打乱，测试集保持顺序；不丢弃不足 128 张的末批。
    train_data = load_mnist(args.data_dir, train=True)
    test_data = load_mnist(args.data_dir, train=False)
    train_reader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_reader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    first_image, first_label = train_data[0]
    print("Train: {}, Test: {}, device: {}".format(len(train_data), len(test_data), device))
    print("First image: shape={}, range=[{}, {}], label={}".format(
        tuple(first_image.shape), first_image.min().item(),
        first_image.max().item(), first_label.item(),
    ))

    # Step2：保留 784→100→100→10 结构、Adam 与每 4000 次更新衰减到 0.1 倍。
    model = MultilayerPerceptron().to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)
    history: List[Dict[str, float]] = []
    best_test_acc = -1.0
    best_epoch = 0
    start = time.perf_counter()

    # Step3：逐批训练，记录真实更新次数、累计样本数及批次均值。
    with (args.output_dir / "batch_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "samples_seen", "epoch", "loss", "accuracy", "learning_rate"])
        step = samples_seen = 0
        for epoch in range(1, args.epochs + 1):
            # 上一轮 evaluate 会进入评估模式，每轮开头必须切回训练模式。
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            epoch_lr = optimizer.param_groups[0]["lr"]
            for images, labels in train_reader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                batch_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                # 这里每个 batch 调用一次，因此衰减单位是更新次数。
                scheduler.step()

                count = labels.size(0)
                correct = (logits.argmax(dim=1) == labels).sum().item()
                total_loss += loss.item() * count
                total_correct += correct
                total_samples += count
                step += 1
                samples_seen += count
                writer.writerow([step, samples_seen, epoch, loss.item(), correct / count, batch_lr])

            # Step4：按总样本数评估，每次更好时立即保存，不受轮数门槛限制。
            test_metrics = evaluate(model, test_reader, device)
            history.append({
                "epoch": epoch,
                "learning_rate": epoch_lr,
                "train_loss": total_loss / total_samples,
                "train_accuracy": total_correct / total_samples,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            })
            if test_metrics["accuracy"] > best_test_acc:
                best_test_acc = test_metrics["accuracy"]
                best_epoch = epoch
                torch.save(model.state_dict(), args.output_dir / "best_model.pt")
            print("Epoch {:02d}/{}  lr={:.6g}  train_loss={:.4f}  train_acc={:.2%}  "
                  "test_loss={:.4f}  test_acc={:.2%}".format(
                      epoch, args.epochs, epoch_lr, history[-1]["train_loss"],
                      history[-1]["train_accuracy"], test_metrics["loss"],
                      test_metrics["accuracy"],
                  ), flush=True)

    torch.save(model.state_dict(), args.output_dir / "last_model.pt")
    with (args.output_dir / "epoch_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    draw_train_process(history, args.output_dir)

    # Step5：重新加载已保存的最优权重，再评估并进行预测展示。
    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=True))
    best_metrics = evaluate(model, test_reader, device)
    save_predictions(model, test_data, device, args.output_dir)
    metrics = {
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "torch_version": str(torch.__version__),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "best_epoch": best_epoch,
        "best_test_accuracy": best_metrics["accuracy"],
        "best_test_loss": best_metrics["loss"],
        "last_test_accuracy": history[-1]["test_accuracy"],
        "elapsed_seconds": time.perf_counter() - start,
        "selection_note": "沿用任务书按测试准确率选择模型，无独立验证集；该测试结果不代表独立泛化评估。",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Best epoch: {}, test accuracy: {:.2%}".format(best_epoch, best_metrics["accuracy"]))
    print("Outputs: {}".format(args.output_dir))


def plot_results(data_dir: Path, output_dir: Path) -> None:
    """读取已有指标和最优模型，重绘训练曲线与预测图，不重新训练。"""
    with (output_dir / "epoch_history.csv").open(encoding="utf-8", newline="") as file:
        history = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
    if not history:
        raise ValueError("逐轮指标文件为空，请先通过 main.py 完成训练")

    model = MultilayerPerceptron()
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    )
    test_data = load_mnist(data_dir, train=False)
    draw_train_process(history, output_dir)
    save_predictions(model, test_data, torch.device("cpu"), output_dir)
    print("已根据现有结果重绘图片：{}".format(output_dir))
