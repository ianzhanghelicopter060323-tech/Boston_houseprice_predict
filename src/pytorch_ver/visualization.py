from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "pytorch_ver"
FEATURE_NAMES = (
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
)


def draw_feature_ranges(
    features: NDArray[np.float64],
    output_path: Path = OUTPUT_DIR / "image" / "ranges.png",
    show: bool = False,
) -> Path :
    """ 绘制原始 13 个特征的极差柱状图，输入不包含房价标签。 """

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 13:
        raise ValueError("原始特征形状必须为非空的 (N, 13)")
    if not np.isfinite(features).all():
        raise ValueError("原始特征必须为有限数值")

    # 使用归一化前的数据；归一化后的非恒定特征极差都是 1。
    # 各特征的单位不同，柱高表示数值跨度，不代表对房价的影响大小。
    ranges = np.ptp(features, axis=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(FEATURE_NAMES, ranges, color="steelblue", width=0.65)
    ax.bar_label(bars, labels=[f"{value:.3g}" for value in ranges], padding=3)
    ax.set(
        title="Boston housing: raw feature ranges",
        xlabel="Feature",
        ylabel="Range (max - min; feature-specific units)",
    )
    ax.margins(y=0.15)
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def draw_train_process(
    iters: Sequence[int],
    train_costs: Sequence[float],
    output_path: Path = OUTPUT_DIR / "image" / "training_cost.png",
    show: bool = False,
) -> Path :
    """ 绘制每次参数更新前的批次 MSE，保存图片后可选择显示窗口。 """

    if len(iters) == 0 or len(iters) != len(train_costs):
        raise ValueError("迭代次数与训练损失必须非空且数量相同")
    if not np.isfinite(train_costs).all():
        raise ValueError("训练损失必须为有限数值")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, train_costs, color="red", linewidth=0.8, label="Batch MSE")
    ax.set(
        title="Boston housing training cost",
        xlabel="Iteration (batch update)",
        ylabel="MSE (normalized MEDV)",
    )
    ax.grid(alpha=0.3)
    ax.legend()

    # 英文图例避免依赖中文字体；先保存，再显示并释放画布。
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def draw_infer_result(
    ground_truths: Sequence[float],
    infer_results: Sequence[float],
    output_path: Path = OUTPUT_DIR / "image" / "inference_result.png",
    show: bool = False,
) -> Path :
    """ 绘制真实值与预测值的散点图，越接近 y=x 表示误差越小。 """

    truth = np.asarray(ground_truths, dtype=np.float64).reshape(-1)
    predicted = np.asarray(infer_results, dtype=np.float64).reshape(-1)
    if len(truth) == 0 or truth.shape != predicted.shape:
        raise ValueError("真实值和预测值必须非空且数量相同")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("真实值和预测值必须为有限数值")

    # 使用共同范围画对角线，适应归一化房价中的负数和接近零的值。
    lower = float(min(truth.min(), predicted.min()))
    upper = float(max(truth.max(), predicted.max()))
    padding = max((upper - lower) * 0.08, 0.01)
    limits = (lower - padding, upper + padding)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(limits, limits, color="gray", linestyle="--", label="Perfect prediction")
    ax.scatter(truth, predicted, color="green", label="Test predictions")
    ax.set(
        title=f"Boston housing: first {len(truth)} test samples",
        xlabel="Ground truth (normalized MEDV)",
        ylabel="Prediction (normalized MEDV)",
        xlim=limits,
        ylim=limits,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return output_path
