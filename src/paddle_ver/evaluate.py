"""Step4：重新加载模型，以全部测试样本计算评估指标。"""

import csv
import logging
from pathlib import Path

import numpy as np
import paddle.fluid as fluid

from .data_prepare import batch_to_arrays
from .model import load_regressor


LOGGER = logging.getLogger(__name__)


def regression_metrics(ground_truths, predictions):
    """计算逐条预测的 MSE、RMSE、MAE 和 R²，不按批次等权平均。"""

    # 将 (N, 1) 或一维输入统一展开；使用 float64 统计，减少数值舍入。
    truth = np.asarray(ground_truths, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)

    if len(truth) == 0 or truth.shape != predicted.shape:
        raise ValueError("真实值和预测值必须非空且数量相同")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("真实值或预测值包含非有限数值")

    # MSE 为平均平方误差，RMSE 是其平方根，MAE 为平均绝对误差。
    error = predicted - truth
    mse = float(np.mean(error ** 2))

    # R²衡量模型相对于直接预测样本均值的拟合程度。
    # 若真实值全部相同，分母为 0，此时用 None 表示 R²未定义。
    denominator = float(np.sum((truth - truth.mean()) ** 2))

    return {
        "sample_count": len(truth),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else None,
    }


def save_predictions(path, ground_truths, predictions, row_indices=None):
    """价格和误差的单位均为千美元；原始行号从 1 开始。"""

    # 先检查真实值与预测值的数量和有效性，避免保存不完整记录。
    truth = np.asarray(ground_truths).reshape(-1)
    predicted = np.asarray(predictions).reshape(-1)
    regression_metrics(truth, predicted)

    # 打乱后的评估结果没有记录原始行号，用空字段表示未知，不以 0 代替。
    if row_indices is None:
        row_indices = [""] * len(truth)
    if len(row_indices) != len(truth):
        raise ValueError("原始行号数量与样本数不一致")

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample", "source_row", "ground_truth", "prediction", "error", "squared_error"
        ])

        # CSV 样本编号从 1 开始；误差定义为预测值减真实值。
        for index, (source_row, target, result) in enumerate(
            zip(row_indices, truth, predicted), start=1
        ):
            error = float(result) - float(target)
            writer.writerow([index, source_row, float(target), float(result), error, error ** 2])


def evaluate(test_reader, model_path, output_dir):
    """加载已训练模型，完整评估测试集并保存逐条预测结果。"""
    predictions = []     # 每个批次的预测值，形状均为 (批次大小, 1)
    ground_truths = []   # 与预测值顺序对应的真实房价
    batch_sizes = []     # 记录实际批次大小，便于核对末批是否完整

    # 模型只加载一次；评估不需要构建反向传播所需的梯度记录。
    with fluid.dygraph.guard(fluid.CPUPlace()):
        regressor = load_regressor(model_path)
        with fluid.dygraph.no_grad():
            for data in test_reader():
                test_x, test_y = batch_to_arrays(data)
                X = fluid.dygraph.to_variable(test_x)

                # 追加每个批次的结果，不能覆盖之前已读取的样本。
                predictions.append(regressor(X).numpy())
                ground_truths.append(test_y)
                batch_sizes.append(len(data))

    # 先拼接逐条结果，再求 MSE，确保末批 12 条与其他批次按样本数加权。
    # 原任务书 test_costs 被覆盖的写法实际上只统计最后一个 batch。
    predicted = np.concatenate(predictions)
    truth = np.concatenate(ground_truths)
    metrics = regression_metrics(truth, predicted)

    if metrics["sample_count"] != 102:
        raise ValueError("评估应完整覆盖 102 条测试样本")
    metrics["batch_sizes"] = batch_sizes

    # 保存全部测试样本的结果，指标分母是样本总数而非批次数。
    save_predictions(Path(output_dir) / "test_predictions.csv", truth, predicted)
    LOGGER.info(
        "Test, Cost:%.5f (%d samples); RMSE:%.5f; MAE:%.5f; R2:%.5f",
        metrics["mse"],
        metrics["sample_count"],
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
    )

    return metrics
