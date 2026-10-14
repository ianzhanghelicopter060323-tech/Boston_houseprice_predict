"""按 docs/task3.md 顺序执行五步实验，也支持只评估或只预测已有模型。"""

import argparse
import json
import logging
import platform
import random
import sys
from pathlib import Path


# 直接运行 main.py 时没有包上下文；补充项目根目录和包名，保留相对导入。
# 使用 python -m src.paddle_ver.main 时已有包上下文，无需调整。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.paddle_ver"


import matplotlib
import numpy as np
import paddle
import google.protobuf

from .data_prepare import BATCH_SIZE, BUF_SIZE, INFER_BATCH_SIZE, prepare_data
from .evaluate import evaluate
from .predict import predict
from .train import EPOCH_NUM, LEARNING_RATE, train


PROJECT_ROOT = Path(__file__).resolve().parents[2]   # 由入口文件位置定位项目根目录
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paddle_ver"
LOGGER = logging.getLogger(__name__)


def write_json(path, values):
    """以 UTF-8 保存结构化结果，保留中文并拒绝 NaN 等非标准 JSON 数值。"""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def positive_int(value):
    """将命令行参数转换为正整数，用于训练轮数和随机种子。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")

    return number


def positive_float(value):
    """将命令行参数转换为有限正数，避免传入无效学习率。"""
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须为有限正数")

    return number


def main():
    """读取运行配置，依次执行数据准备、训练、评估和预测。"""

    # 解析命令行参数；默认执行完整实验，也可单独评估或预测已有模型。
    parser = argparse.ArgumentParser(description="Paddle 1.8.5 波士顿房价全流程实验")
    parser.add_argument("--stage", choices=["all", "evaluate", "predict"], default="all")
    parser.add_argument("--epochs", type=positive_int, default=EPOCH_NUM)
    parser.add_argument("--learning-rate", type=positive_float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=positive_int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # 创建统一输出目录；单独评估和预测必须先有已训练的模型参数。
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model" / "fit_a_line"
    if args.stage != "all" and not model_path.with_suffix(".pdparams").is_file():
        parser.error("未找到训练模型，请先运行 --stage all")

    # 同时在终端和对应阶段的日志文件中记录运行过程。
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                output_dir / (args.stage + ".log"),
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    # shuffle 使用 Python 随机数；Paddle 的初始化种子在 train 的 guard 内设置。
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 记录实际环境，便于解释相同代码在不同机器上的运行差异。
    environment = {
        "python": platform.python_version(),
        "paddle": paddle.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "protobuf": google.protobuf.__version__,
        "device": "CPU",
    }
    LOGGER.info("Environment: %s", environment)

    # Step1：准备三类 reader，并保存数据来源及首条样本信息。
    train_reader, test_reader, infer_reader, dataset = prepare_data(output_dir)
    LOGGER.info(
        "Dataset: %d train + %d test; first training sample: %s",
        dataset["train_samples"],
        dataset["test_samples"],
        dataset["first_train_sample"],
    )
    write_json(output_dir / "data" / "dataset_info.json", dataset)

    results = {
        "environment": environment,
        "config": {
            "seed": args.seed,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUF_SIZE,
            "infer_batch_size": INFER_BATCH_SIZE,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        "dataset": dataset,
    }

    # Step2～3：train 内创建 Regressor，并完成参数更新和模型保存。
    if args.stage == "all":
        results["training"] = train(
            train_reader, output_dir, args.epochs, args.learning_rate, args.seed
        )

    # Step4：重新加载保存的模型，在全部测试样本上评估。
    if args.stage in ("all", "evaluate"):
        results["evaluation"] = evaluate(test_reader, model_path, output_dir)
        write_json(output_dir / "evaluation.json", results["evaluation"])

    # Step5：重新加载模型，预测测试集前 10 条并保存对比图。
    if args.stage in ("all", "predict"):
        results["inference"] = predict(infer_reader, model_path, output_dir)
        write_json(output_dir / "inference.json", results["inference"])

    # 只在完整运行时更新汇总，单独评估或预测不覆盖原训练配置。
    if args.stage == "all":
        write_json(output_dir / "metrics.json", results)

    LOGGER.info("Experiment finished. Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
