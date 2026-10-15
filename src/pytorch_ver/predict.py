import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

from model_set import Setmodel
import data_prepare as da
from visualization import OUTPUT_DIR, draw_infer_result


class EvaluationMetrics(TypedDict):
    """ 同一组样本在原始房价尺度与归一化尺度上的评价指标。 """
    sample_count: int
    original: Dict[str, float]
    normalized: Dict[str, float]


def _mean_squared_error(
    ground_truths: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> float :
    """ 校验输入并计算 MSE，分母为样本数而不是批次数。 """

    truth = np.asarray(ground_truths, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(truth) == 0 or truth.shape != predicted.shape:
        raise ValueError("真实值和预测值必须非空且数量相同")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("真实值和预测值必须为有限数值")

    return float(np.mean((predicted - truth) ** 2))


def calculate_train_mse(
    ground_truths: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> float :
    """ 从完整数据中取前 404 条，计算最终模型的训练集 MSE。 """
    return _mean_squared_error(ground_truths[:404], predictions[:404])


def calculate_test_mse(
    ground_truths: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> float :
    """ 从完整数据中取后 102 条，计算全部测试样本的 MSE。 """
    return _mean_squared_error(ground_truths[404:], predictions[404:])


def calculate_test_rmse(test_mse: float) -> float :
    """ 对已计算的测试集 MSE 开平方，得到同一房价尺度的 RMSE。 """
    if not np.isfinite(test_mse) or test_mse < 0:
        raise ValueError("测试集 MSE 必须为有限的非负数值")
    return float(np.sqrt(test_mse))


def calculate_first10_mse(
    ground_truths: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> float :
    """ 从完整数据中取第 405～414 行，计算前 10 条测试样本的 MSE。 """
    return _mean_squared_error(ground_truths[404:414], predictions[404:414])


def collect_predictions(
    model: Setmodel,
    data_set: NDArray[np.float64],
) -> NDArray[np.float64] :
    """ 固定最终模型参数，按原始行序返回全部 506 条归一化预测。 """

    test_loader = da.load_data(data_set, 'test')
    train_x = torch.from_numpy(data_set[:404, :13]).float()
    model.eval()
    sample_count = 0
    infer_results = []

    print("============================== start to predict ==============================")
    print("逐条输出的预测值和真实值均为归一化房价。")
    with torch.no_grad() :
        train_results = model(train_x).numpy().astype(np.float64).reshape(-1)

        for x, precise_y in test_loader :
            y_predict = model(x)
            infer_results.extend(y_predict[:, 0].tolist())

            for i, (result, target) in enumerate(zip(y_predict, precise_y)) :
                print(
                    f"No.{sample_count + i}: "
                    f"infer result is {result.item():.5f}, "
                    f"ground truth is {target.item():.5f}"
                )

            sample_count += x.size(0)

    if sample_count != 102:
        raise ValueError("测试集评估必须完整覆盖 102 条样本")
    test_results = np.asarray(infer_results, dtype=np.float64)
    return np.concatenate([train_results, test_results])


def build_evaluation_metrics(
    values: Dict[str, float],
    target_range: float,
) -> Dict[str, EvaluationMetrics] :
    """ 按既有格式整理三组指标，并把原始房价误差换算为归一化误差。 """

    groups: Dict[str, Tuple[int, float, float]] = {
        "train": (404, values["train_mse"], float(np.sqrt(values["train_mse"]))),
        "test": (102, values["test_mse"], values["test_rmse"]),
        "first10": (10, values["first10_mse"], float(np.sqrt(values["first10_mse"]))),
    }
    metrics: Dict[str, EvaluationMetrics] = {}
    for name, (sample_count, mse, rmse) in groups.items():
        metrics[name] = {
            "sample_count": sample_count,
            "original": {"mse": mse, "rmse": rmse},
            "normalized": {
                "mse": mse / target_range ** 2,
                "rmse": rmse / target_range,
            },
        }
    return metrics


def save_predictions(
    path: Path,
    ground_truths: NDArray[np.float64],
    predictions_normalized: NDArray[np.float64],
    source_start: int,
    target_mean: float,
    target_range: float,
) -> None :
    """ 保存两种尺度的逐条预测；原始价格为千美元，行号从 1 开始。 """

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_row", "ground_truth_normalized", "prediction_normalized",
            "ground_truth", "prediction", "error", "squared_error",
        ])
        for i, (target, result_normalized) in enumerate(
            zip(ground_truths, predictions_normalized)
        ):
            result = float(result_normalized) * target_range + target_mean
            error = result - float(target)
            writer.writerow([
                source_start + i,
                (float(target) - target_mean) / target_range,
                float(result_normalized),
                float(target), result, error, error ** 2,
            ])


def save_evaluation_results(
    model_path: Path,
    raw_data: NDArray[np.float64],
    predictions_normalized: NDArray[np.float64],
    metrics: Dict[str, EvaluationMetrics],
    target_mean: float,
    target_range: float,
) -> None :
    """ 保存评价汇总与三组逐条预测，保留既有 JSON、CSV 格式和文件名。 """

    groups: Dict[str, Tuple[int, NDArray[np.float64], NDArray[np.float64]]] = {
        "train": (1, raw_data[:404, 13], predictions_normalized[:404]),
        "test": (405, raw_data[404:, 13], predictions_normalized[404:]),
        "first10": (405, raw_data[404:414, 13], predictions_normalized[404:414]),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (source_start, truth, predicted_normalized) in groups.items():
        filename = "inference_predictions.csv" if name == "first10" else f"{name}_predictions.csv"
        save_predictions(
            OUTPUT_DIR / filename, truth, predicted_normalized,
            source_start, target_mean, target_range,
        )

    # JSON 保留完整精度、单位及模型摘要，便于区分不同模型的评估结果。
    summary = {
        "model_file": str(model_path.relative_to(Path(__file__).resolve().parents[2])),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "normalization": {
            "method": "(value - full_dataset_mean) / full_dataset_range",
            "target_mean": target_mean,
            "target_range": target_range,
            "mse_to_original_factor": target_range ** 2,
        },
        "units": {
            "original_mse": "(1000 USD)^2",
            "original_rmse": "1000 USD",
            "normalized_mse": "dimensionless",
            "normalized_rmse": "dimensionless",
        },
        **metrics,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    with (OUTPUT_DIR / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scope", "sample_count", "mse_1000usd_squared", "rmse_1000usd",
            "mse_normalized", "rmse_normalized",
        ])
        for name, values in metrics.items():
            writer.writerow([
                name, values["sample_count"],
                values["original"]["mse"], values["original"]["rmse"],
                values["normalized"]["mse"], values["normalized"]["rmse"],
            ])


def predict_price(show: bool = False) -> float :
    """ 加载模型，依次调用预测、四项评价、结果保存和绘图函数。 """

    model_path = Path(__file__).resolve().parent / "data" / "price_predict_model_pytorch.pth"
    model = Setmodel()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # 归一化口径沿用训练阶段，标签统计量用于恢复原始房价。
    raw_data = np.loadtxt(da.DATA_PATH, dtype=np.float64)
    if raw_data.shape != (506, 14) or not np.isfinite(raw_data).all():
        raise ValueError("原始数据必须为 506 行、14 列，且所有值均为有限数值")
    target_mean = float(raw_data[:, 13].mean())
    target_range = float(np.ptp(raw_data[:, 13]))
    if target_range <= 0:
        raise ValueError("房价标签的极差必须大于 0")

    predictions_normalized = collect_predictions(model, da.data_scale())
    predictions = predictions_normalized * target_range + target_mean
    ground_truths = raw_data[:, 13]

    # 每项评价使用同一批最终模型预测，分别调用各自的计算函数。
    values = {
        "train_mse": calculate_train_mse(ground_truths, predictions),
        "test_mse": calculate_test_mse(ground_truths, predictions),
        "first10_mse": calculate_first10_mse(ground_truths, predictions),
    }
    values["test_rmse"] = calculate_test_rmse(values["test_mse"])
    metrics = build_evaluation_metrics(values, target_range)
    save_evaluation_results(
        model_path, raw_data, predictions_normalized,
        metrics, target_mean, target_range,
    )

    print(f"404 条训练样本 MSE（千美元）²: {values['train_mse']:.5f}")
    print(f"102 条测试样本 MSE（千美元）²: {values['test_mse']:.5f}")
    print(f"测试集 RMSE（千美元）: {values['test_rmse']:.5f}")
    print(f"前 10 条测试样本 MSE（千美元）²: {values['first10_mse']:.5f}")
    test_mse_normalized = metrics["test"]["normalized"]["mse"]
    print(f"测试集平均误差（MSE，归一化房价）: {test_mse_normalized:.8f}")
    print(f"评价指标与逐条预测数据已保存到: {OUTPUT_DIR}")

    # 绘图函数只接收前 10 条数据，不计算或保存评价指标。
    plot_truths = ((ground_truths[404:414] - target_mean) / target_range).tolist()
    image_path = draw_infer_result(plot_truths, predictions_normalized[404:414].tolist(), show=show)
    print(f"前 10 条测试样本的预测对比图已保存: {image_path}")
    return test_mse_normalized


"""if __name__ == "__main__" :
    predict_price()

"""