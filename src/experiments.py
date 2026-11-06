import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

# 直接运行本文件时，将项目根目录加入搜索路径，供 src 包内导入使用。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, load_mnist
from src.model import MultilayerPerceptron
from src.train import evaluate
from matplotlib.figure import Figure


def write_json(path: Path, value: object) -> None:
    """保存可直接审阅的指标，拒绝把 NaN 写成有效数值。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


@torch.inference_mode()
def predict_dataset(model: nn.Module, dataset: TensorDataset) -> torch.Tensor:
    """按固定顺序返回全部样本的十类分数，供统计和逐条核验。"""
    model.eval()
    return torch.cat([model(images) for images, _ in DataLoader(dataset, batch_size=128)])


def classification_rows(matrix: np.ndarray) -> List[dict]:
    """由行真实、列预测的矩阵计算指标；无分母的项用 None 表示。"""
    rows = []

    for label in range(10):
        correct = int(matrix[label, label])
        support = int(matrix[label].sum())
        predicted = int(matrix[:, label].sum())

        rows.append({
            "digit": label, "support": support, "predicted": predicted,
            "correct": correct,
            "precision": correct / predicted if predicted else None,
            "recall": correct / support if support else None,
            "f1": 2 * correct / (support + predicted) if support + predicted else None,
        })

    return rows


def error_analysis(data_dir: Path, model_path: Path, output_dir: Path) -> None:
    """分析已有基础模型，输出全量预测、混淆矩阵和索引最早的错误样例。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_mnist(data_dir, train=False)
    images, labels = dataset.tensors

    model = MultilayerPerceptron()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    probabilities = predict_dataset(model, dataset).softmax(1)
    predictions = probabilities.argmax(1)
    matrix = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10).numpy()
    rows = classification_rows(matrix)
    errors = (predictions != labels).nonzero().flatten().tolist()

    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    with (output_dir / "class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for filename, indices in (("predictions.csv", range(len(dataset))), ("errors.csv", errors)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["test_index", "true_label", "predicted_label", "confidence"])
            for i in indices:
                writer.writerow([i, labels[i].item(), predictions[i].item(), probabilities[i].max().item()])

    fig = Figure(figsize=(8, 7))
    axis = fig.add_subplot(111)
    heatmap = axis.imshow(matrix, cmap="Blues")

    for true in range(10):
        for pred in range(10):
            axis.text(pred, true, str(matrix[true, pred]), ha="center", va="center",
                      color="white" if matrix[true, pred] > matrix.max() / 2 else "black", fontsize=8)

    axis.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit",
             ylabel="True digit", title="Basic model: MNIST confusion matrix (counts)")
    fig.colorbar(heatmap, ax=axis)

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)

    fig = Figure(figsize=(10, 10))
    axes = fig.subplots(5, 5)
    for axis in axes.flat:
        axis.axis("off")

    for axis, i in zip(axes.flat, errors[:25]):
        axis.imshow(images[i, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        axis.set_title("#{}  True:{} Pred:{}\np={:.3f}".format(
            i, labels[i].item(), predictions[i].item(), probabilities[i].max().item()), fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "error_examples.png", dpi=160)

    pairs = sorted([(int(matrix[a, b]), a, b) for a in range(10) for b in range(10)
                    if a != b and matrix[a, b] > 0], key=lambda item: (-item[0], item[1], item[2]))
    summary = {
        "samples": len(dataset), "correct": int(matrix.trace()), "errors": len(errors),
        "accuracy": float(matrix.trace() / matrix.sum()), "class_metrics": rows,
        "top_confusions": [{"count": n, "true": a, "predicted": b} for n, a, b in pairs[:10]],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "note": "基础模型按测试集选择；本实验为描述分析，不是独立泛化评估。",
    }
    write_json(output_dir / "summary.json", summary)
    print("A: {}/{} correct; {} errors".format(summary["correct"], len(dataset), len(errors)), flush=True)


def stratified_split(labels: torch.Tensor) -> tuple:
    """按类别比例及最大余数法分配 10,000 个验证样本，划分种子固定为 42。"""
    counts = torch.bincount(labels, minlength=10).numpy()
    quota = counts * (10000 / len(labels))
    allocation = np.floor(quota).astype(int)

    # 余数相同时优先较小的类别编号，保证总数恰好为 10,000。
    order = np.argsort(-(quota - allocation), kind="stable")
    allocation[order[:10000 - int(allocation.sum())]] += 1

    generator = np.random.default_rng(42)
    train, validation = [], []

    for digit in range(10):
        indices = np.flatnonzero(labels.numpy() == digit)
        generator.shuffle(indices)
        validation.extend(indices[:allocation[digit]].tolist())
        train.extend(indices[allocation[digit]:].tolist())

    return np.sort(np.array(train, dtype=np.int64)), np.sort(np.array(validation, dtype=np.int64))


def train_strategy(train: TensorDataset, validation: TensorDataset, seed: int,
                   strategy: str, output_dir: Path) -> dict:
    """完成一次 30 轮训练，仅按验证准确率选择权重，不接触测试集。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model = MultilayerPerceptron()
    initial_hash = hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in model.parameters())).hexdigest()

    reader = DataLoader(train, batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    validation_reader = DataLoader(validation, batch_size=128)

    optimizer = Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    best_accuracy, best_epoch = -1.0, 0
    step = 0
    order_hash = hashlib.sha256()
    start = time.perf_counter()

    with (output_dir / "batch_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "epoch", "samples", "learning_rate", "loss", "accuracy"])

        for epoch in range(1, 31):
            model.train()
            loss_sum, correct_sum, samples = 0.0, 0, 0
            epoch_lr = float(optimizer.param_groups[0]["lr"])

            for images, labels, indices in reader:
                order_hash.update(indices.numpy().tobytes())
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()

                lr = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                if strategy == "step":
                    scheduler.step()

                count = len(labels)
                correct = int((logits.argmax(1) == labels).sum().item())
                loss_sum += loss.item() * count
                correct_sum += correct
                samples += count
                step += 1
                writer.writerow([step, epoch, count, lr, loss.item(), correct / count])

            result = evaluate(model, validation_reader, torch.device("cpu"))
            history.append({"epoch": epoch, "learning_rate": epoch_lr,
                            "train_loss": loss_sum / samples, "train_accuracy": correct_sum / samples,
                            "validation_loss": result["loss"], "validation_accuracy": result["accuracy"]})

            if result["accuracy"] > best_accuracy:
                best_accuracy, best_epoch = result["accuracy"], epoch
                torch.save(model.state_dict(), output_dir / "best_model.pt")

            if epoch % 10 == 0:
                print("B {} seed={} epoch={} validation={:.2%}".format(
                    strategy, seed, epoch, result["accuracy"]), flush=True)

    elapsed = time.perf_counter() - start
    torch.save(model.state_dict(), output_dir / "last_model.pt")
    with (output_dir / "epoch_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    result = {"strategy": strategy, "seed": seed, "best_epoch": best_epoch,
              "best_validation_accuracy": best_accuracy, "training_seconds": elapsed,
              "steps": step, "initial_weights_sha256": initial_hash,
              "training_order_sha256": order_hash.hexdigest()}
    write_json(output_dir / "metrics.json", result)
    return result


def compare_learning_rates(data_dir: Path, output_dir: Path) -> None:
    """固定一次分层划分，完成六次配对训练，再统一评估测试集与汇总。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_mnist(data_dir, train=True)
    images, labels = source.tensors

    train_indices, validation_indices = stratified_split(labels)
    assert len(train_indices) == 50000 and len(validation_indices) == 10000
    assert len(np.unique(np.concatenate([train_indices, validation_indices]))) == 60000
    np.savez(output_dir / "split_indices.npz", train=train_indices, validation=validation_indices)

    train_ids = torch.from_numpy(train_indices)
    validation_ids = torch.from_numpy(validation_indices)
    train = TensorDataset(images[train_ids], labels[train_ids], train_ids)
    validation = TensorDataset(images[validation_ids], labels[validation_ids])

    write_json(output_dir / "protocol.json", {
        "split_seed": 42, "training_seeds": [42, 43, 44], "epochs": 30,
        "batch_size": 128, "initial_learning_rate": 0.005, "step_size": 4000, "gamma": 0.1,
        "train_samples": len(train), "validation_samples": len(validation), "test_samples": 10000,
        "train_class_counts": torch.bincount(labels[train_ids], minlength=10).tolist(),
        "validation_class_counts": torch.bincount(labels[validation_ids], minlength=10).tolist(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "threads": torch.get_num_threads(), "device": "cpu",
        "selection": "max validation accuracy; earliest epoch for ties; test only after all six training runs",
    })

    results = []
    for seed in (42, 43, 44):
        for strategy in ("constant", "step"):
            results.append(train_strategy(train, validation, seed, strategy,
                                          output_dir / "{}_{}".format(strategy, seed)))

        assert results[-2]["initial_weights_sha256"] == results[-1]["initial_weights_sha256"]
        assert results[-2]["training_order_sha256"] == results[-1]["training_order_sha256"]

    # 所有权重已按验证集选定，此后才读取和评估官方测试集，不据此调整设置。
    test = load_mnist(data_dir, train=False)

    model = MultilayerPerceptron()

    for result in results:
        directory = output_dir / "{}_{}".format(result["strategy"], result["seed"])
        model.load_state_dict(torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True))

        logits = predict_dataset(model, test)
        predictions = logits.argmax(1)
        result["test_correct"] = int((predictions == test.tensors[1]).sum().item())
        result["test_accuracy"] = result["test_correct"] / len(test)
        result["test_loss"] = nn.CrossEntropyLoss()(logits, test.tensors[1]).item()

        with (directory / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["test_index", "true_label", "predicted_label"])
            writer.writerows(zip(range(len(test)), test.tensors[1].tolist(), predictions.tolist()))
        write_json(directory / "metrics.json", result)

    aggregates = {}
    for strategy in ("constant", "step"):
        values = [r["test_accuracy"] for r in results if r["strategy"] == strategy]
        aggregates[strategy] = {"mean_test_accuracy": float(np.mean(values)),
                                "sample_std_test_accuracy": float(np.std(values, ddof=1))}

    differences = [results[i+1]["test_accuracy"] - results[i]["test_accuracy"] for i in (0, 2, 4)]
    write_json(output_dir / "summary.json", {
        "runs": results, "aggregates": aggregates, "paired_step_minus_constant": differences,
        "paired_mean": float(np.mean(differences)), "paired_sample_std": float(np.std(differences, ddof=1)),
        "note": "三个训练种子共用同一划分；标准差描述训练随机性，不代表总体置信区间。",
    })

    fig = Figure(figsize=(12, 8))
    axes = fig.subplots(2, 2)

    for result in results:
        directory = output_dir / "{}_{}".format(result["strategy"], result["seed"])
        with (directory / "epoch_history.csv").open() as file:
            history = list(csv.DictReader(file))

        for axis, metric in zip(axes.flat, ("train_loss", "validation_loss", "train_accuracy", "validation_accuracy")):
            axis.plot(range(1, 31), [float(r[metric]) for r in history],
                      color="tab:blue" if result["strategy"] == "constant" else "tab:orange",
                      linestyle={42: "-", 43: "--", 44: ":"}[result["seed"]],
                      label="{} / {}".format(result["strategy"], result["seed"]))
            axis.set(xlabel="Epoch", ylabel=metric, title=metric.replace("_", " "))
            axis.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_curves.png", dpi=160)
    print("B completed: " + json.dumps(aggregates), flush=True)


def main() -> None:
    """直接启动拓展实验，默认依次执行 A、B，也可通过参数单独选择。"""
    parser = argparse.ArgumentParser(description="MNIST 拓展实验：A 错误分析，B 学习率对照")
    parser.add_argument(
        "--experiment", choices=("A", "B", "AB"), default="AB",
        help="要执行的拓展实验，默认 AB",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="MNIST 数据目录")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output", help="实验输出目录")
    parser.add_argument("--model", type=Path, help="A 使用的模型，默认取输出目录的 best_model.pt")
    parser.add_argument("--threads", type=int, default=2, help="CPU 线程数，默认 2")
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("线程数必须大于 0")
    if args.model is not None and args.experiment == "B":
        parser.error("--model 仅用于包含 A 的拓展实验")
    torch.set_num_threads(args.threads)

    if "A" in args.experiment:
        model_path = args.model if args.model is not None else args.output_dir / "best_model.pt"
        error_analysis(args.data_dir, model_path, args.output_dir / "extension_a")

    if "B" in args.experiment:
        compare_learning_rates(args.data_dir, args.output_dir / "extension_b")


if __name__ == "__main__":
    main()
