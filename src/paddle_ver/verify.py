"""离线验收已有实验产物：不重新下载数据，也不重新训练。"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import paddle.fluid as fluid

from .main import DEFAULT_OUTPUT_DIR, write_json
from .model import load_regressor


def _read_json(path):
    """读取实验配置、数据说明或指标文件。"""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path):
    """按表头读取 CSV，每行返回一个字段名称与值对应的字典。"""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check(condition, message):
    """验收条件不满足时立即报错，避免继续使用有问题的产物。"""
    if not condition:
        raise ValueError(message)


def _check_metrics(metrics, truth, prediction):
    """直接用 NumPy 重算，不调用实验的 regression_metrics 函数。"""

    # 使用独立计算路径复核指标，避免评估函数的错误被原样重复。
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    errors = prediction - truth
    _check(metrics["sample_count"] == len(truth), "指标样本数不符")

    mse = float(np.dot(errors, errors) / len(truth))
    np.testing.assert_allclose(metrics["mse"], mse, rtol=1e-6)
    np.testing.assert_allclose(metrics["rmse"], np.sqrt(mse), rtol=1e-6)
    np.testing.assert_allclose(metrics["mae"], np.abs(errors).mean(), rtol=1e-6)

    r2 = 1 - np.dot(errors, errors) / np.sum((truth - truth.mean()) ** 2)
    np.testing.assert_allclose(metrics["r2"], r2, rtol=1e-5, atol=1e-7)


def verify(output_dir):
    """检查数据、训练覆盖、模型重载、逐条预测、评估指标和图像文件。"""
    output_dir = Path(output_dir).resolve()
    metrics = _read_json(output_dir / "metrics.json")

    # 验收原始数据：数量、字段数、有限数值以及文件 MD5 都应一致。
    raw_path = output_dir / "data" / "housing.data"
    raw = np.loadtxt(raw_path)
    _check(raw.shape == (506, 14) and np.isfinite(raw).all(), "原始数据形状或数值错误")
    _check(
        hashlib.md5(raw_path.read_bytes()).hexdigest() == metrics["dataset"]["source_md5"],
        "原始数据 MD5 不一致",
    )

    # 从原始数据重新归一化，逐项核对 NPZ 中的特征、标签及统计参数。
    with np.load(output_dir / "data" / "prepared_data.npz") as data:
        train_x, train_y, test_x, test_y = (
            data[name] for name in ("train_x", "train_y", "test_x", "test_y")
        )
        expected_x = (
            (raw[:, :13] - raw[:, :13].mean(axis=0)) / np.ptp(raw[:, :13], axis=0)
        ).astype(np.float32)

        np.testing.assert_array_equal(np.concatenate([train_x, test_x]), expected_x)
        np.testing.assert_array_equal(
            np.concatenate([train_y, test_y]), raw[:, 13:14].astype(np.float32)
        )
        np.testing.assert_allclose(data["feature_mean"], raw[:, :13].mean(axis=0))
        np.testing.assert_allclose(data["feature_range"], np.ptp(raw[:, :13], axis=0))

    # 验收训练记录：每轮 404 条样本，批次编号及迭代编号连续。
    history = _read_csv(output_dir / "training_history.csv")
    epoch_history = _read_csv(output_dir / "epoch_history.csv")
    epochs = metrics["training"]["epochs"]
    batch_size = metrics["config"]["batch_size"]
    batches_per_epoch = math.ceil(len(train_x) / batch_size)
    _check(
        len(history) == metrics["training"]["iterations"] == epochs * batches_per_epoch,
        "训练迭代数不符",
    )
    _check(len(epoch_history) == epochs, "轮次日志数量不符")

    for epoch in range(epochs):
        rows = history[epoch * batches_per_epoch:(epoch + 1) * batches_per_epoch]
        _check(all(int(row["epoch"]) == epoch for row in rows), "训练轮次编号不符")
        _check(
            [int(row["batch"]) for row in rows] == list(range(batches_per_epoch)),
            "批次编号不连续",
        )
        _check(
            [int(row["iteration"]) for row in rows]
            == list(range(epoch * batches_per_epoch + 1, (epoch + 1) * batches_per_epoch + 1)),
            "训练迭代编号不连续",
        )

        # 最后一批可能不足 30 条，要按实际数量核对并重算加权损失。
        sizes = [int(row["sample_count"]) for row in rows]
        _check(
            sizes == [batch_size] * (batches_per_epoch - 1)
            + [len(train_x) - batch_size * (batches_per_epoch - 1)],
            "训练末批大小不符",
        )
        losses = np.asarray([float(row["mse"]) for row in rows])
        _check(np.isfinite(losses).all(), "训练日志包含非有限数值")
        _check(int(epoch_history[epoch]["epoch"]) == epoch, "轮次汇总编号不符")
        np.testing.assert_allclose(
            float(epoch_history[epoch]["online_train_mse"]),
            np.average(losses, weights=sizes),
        )

    # 重新加载模型，检查参数数量，并计算全部训练集和测试集的输出。
    with fluid.dygraph.guard(fluid.CPUPlace()):
        regressor = load_regressor(output_dir / "model" / "fit_a_line")
        _check(
            sum(int(np.prod(parameter.shape)) for parameter in regressor.parameters()) == 14,
            "线性模型参数应为 13 个权重和 1 个偏置",
        )
        with fluid.dygraph.no_grad():
            loaded_train = regressor(fluid.dygraph.to_variable(train_x)).numpy()
            loaded_test = regressor(fluid.dygraph.to_variable(test_x)).numpy()

    # 保存前后的训练集预测应逐项相同；同时复核最终训练 MSE。
    reference = np.load(output_dir / "data" / "train_predictions_before_reload.npy")
    np.testing.assert_array_equal(loaded_train, reference)
    final_mse = float(np.mean((loaded_train.astype(np.float64) - train_y) ** 2))
    np.testing.assert_allclose(metrics["training"]["final_train_mse"], final_mse)
    _check(final_mse < metrics["training"]["initial_train_mse"], "训练未降低误差")

    # 核对测试 CSV 的覆盖数量，并与重新加载模型的预测对应。
    test_rows = _read_csv(output_dir / "test_predictions.csv")
    _check(len(test_rows) == 102, "测试结果未覆盖全部样本")
    test_truth = np.asarray([float(row["ground_truth"]) for row in test_rows])
    test_prediction = np.asarray([float(row["prediction"]) for row in test_rows])

    # 评估 reader 已打乱，按 (真实值, 预测值) 排序后与重载模型逐项比较。
    saved_pairs = sorted(zip(test_truth, test_prediction))
    expected_pairs = sorted(zip(test_y[:, 0].tolist(), loaded_test[:, 0].tolist()))
    np.testing.assert_allclose(saved_pairs, expected_pairs, rtol=1e-6, atol=1e-6)

    # 检查两份预测 CSV 的编号，以及逐条误差和平方误差。
    for filename, count in [("test_predictions.csv", 102), ("inference_predictions.csv", 10)]:
        rows = _read_csv(output_dir / filename)
        _check(len(rows) == count, "预测记录数量不符")
        _check(
            [int(row["sample"]) for row in rows] == list(range(1, count + 1)),
            "预测样本编号不符",
        )

        for row in rows:
            error = float(row["prediction"]) - float(row["ground_truth"])
            np.testing.assert_allclose(float(row["error"]), error)
            np.testing.assert_allclose(float(row["squared_error"]), error ** 2)

    # 全测试集指标覆盖 30+30+30+12 条样本，单独复核末批是否计入。
    evaluation = _read_json(output_dir / "evaluation.json")
    _check(evaluation["batch_sizes"] == [30, 30, 30, 12], "测试末批 12 条未完整计入")
    _check_metrics(evaluation, test_truth, test_prediction)
    _check_metrics(metrics["evaluation"], test_y, loaded_test)

    # 前 10 条预测必须对应原始文件第 405～414 行，不能从打乱的 reader 取数。
    infer_rows = _read_csv(output_dir / "inference_predictions.csv")
    _check(
        [int(row["source_row"]) for row in infer_rows] == list(range(405, 415)),
        "预测未使用测试集前 10 条",
    )
    np.testing.assert_array_equal(
        [float(row["ground_truth"]) for row in infer_rows], test_y[:10, 0]
    )
    np.testing.assert_allclose(
        [float(row["prediction"]) for row in infer_rows],
        loaded_test[:10, 0],
        rtol=1e-6,
    )
    _check_metrics(_read_json(output_dir / "inference.json"), test_y[:10], loaded_test[:10])
    _check_metrics(metrics["inference"], test_y[:10], loaded_test[:10])

    # 图像文件验收只检查尺寸和内容，不能据此宣称 GUI 弹窗显示正常。
    images = {}
    for name in ("ranges.png", "training_cost.png", "inference_result.png"):
        pixels = mpimg.imread(output_dir / "image" / name)
        _check(
            pixels.shape[0] >= 200 and pixels.shape[1] >= 200 and pixels.std() > 0.01,
            "图像尺寸不足或内容为空",
        )
        images[name] = {"height": pixels.shape[0], "width": pixels.shape[1]}

    # 全部检查通过后，保存验收摘要；发生异常时不会写入通过结果。
    result = {
        "status": "passed",
        "train_samples": len(train_x),
        "test_samples": len(test_x),
        "epochs": epochs,
        "iterations": len(history),
        "reload_max_absolute_error": float(np.max(np.abs(loaded_train - reference))),
        "independent_test_mse": float(np.mean((loaded_test.astype(np.float64) - test_y) ** 2)),
        "images": images,
        "gui_display_verified": False,
    }
    write_json(output_dir / "verification.json", result)

    return result


if __name__ == "__main__":
    # 默认核验完整实验的输出目录，也可通过参数指定其他已有产物目录。
    parser = argparse.ArgumentParser(description="验收已有 Paddle 房价实验产物")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), ensure_ascii=False, indent=2))
