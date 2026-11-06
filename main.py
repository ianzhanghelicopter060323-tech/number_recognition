"""手写数字识别实验统一入口：基础训练、预测与 A/B 拓展实验。"""

import argparse
from pathlib import Path

import torch

from src.data import ROOT
from src.predict import predict_image
from src.train import plot_results, train_and_evaluate


def main() -> None:
    """默认执行基础实验，也可重绘、预测或执行指定拓展实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true", help="使用已有结果重绘，不训练")
    mode.add_argument("--predict", type=Path, metavar="IMAGE", help="识别单张数字图片")
    mode.add_argument("--experiment", choices=("A", "B", "AB"), help="执行已设计的拓展实验")
    parser.add_argument("--model", type=Path, help="单图预测或实验 A 的模型，默认使用输出目录的 best_model.pt")
    parser.add_argument("--invert", action="store_true", help="预测白底黑字图片时反转灰度值")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数，默认 30")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小，默认 128")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="初始学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--threads", type=int, default=2, help="CPU 计算线程数")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="训练设备")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="数据目录")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output", help="实验输出目录")
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("线程数必须大于 0")
    if args.invert and args.predict is None:
        parser.error("--invert 需与 --predict 一起使用")
    if args.model is not None and args.predict is None and args.experiment not in ("A", "AB"):
        parser.error("--model 需与 --predict 或包含 A 的拓展实验一起使用")
    torch.set_num_threads(args.threads)

    if args.experiment is not None:
        # 拓展 B 使用批准的固定协议，避免误以为基础训练参数已传入对照实验。
        if (args.epochs, args.batch_size, args.learning_rate, args.seed, args.device) != (30, 128, 0.005, 42, "cpu"):
            parser.error("拓展实验固定使用 30 轮、批次 128、初始学习率 0.005、CPU；B 的训练种子为 42/43/44")
        from src.experiments import compare_learning_rates, error_analysis

        if "A" in args.experiment:
            model_path = args.model if args.model is not None else args.output_dir / "best_model.pt"
            error_analysis(args.data_dir, model_path, args.output_dir / "extension_a")
        if "B" in args.experiment:
            compare_learning_rates(args.data_dir, args.output_dir / "extension_b")
    elif args.predict is not None:
        model_path = args.model if args.model is not None else args.output_dir / "best_model.pt"
        predict_image(args.predict, model_path, invert=args.invert)
    elif args.plot_only:
        plot_results(args.data_dir, args.output_dir)
    else:
        if min(args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0:
            parser.error("轮数、批次大小和学习率必须大于 0")
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("当前 PyTorch 无可用 CUDA，请使用 --device cpu")
        train_and_evaluate(args)


if __name__ == "__main__":
    main()
