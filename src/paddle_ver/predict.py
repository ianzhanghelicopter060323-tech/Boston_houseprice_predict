"""Step5：重新加载模型，预测测试集前 10 条并绘制真实值对比图。"""

import logging
from pathlib import Path

import matplotlib

# 使用非交互后端保存图片，终端运行时不依赖图形界面。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid

from .data_prepare import batch_to_arrays
from .evaluate import regression_metrics, save_predictions
from .model import load_regressor


LOGGER = logging.getLogger(__name__)


def draw_infer_result(ground_truths, infer_results, output_path):
    """以真实房价为横轴、预测房价为纵轴，绘制前 10 条样本的对比图。"""
    truth = np.asarray(ground_truths).reshape(-1)
    predicted = np.asarray(infer_results).reshape(-1)

    # 两轴使用共同范围，并留出边距，避免极值样本被边界裁切。
    lower = float(min(truth.min(), predicted.min()))
    upper = float(max(truth.max(), predicted.max()))
    padding = max((upper - lower) * 0.08, 1.0)
    limits = (lower - padding, upper + padding)

    fig, ax = plt.subplots(figsize=(6, 6))
    # 对角线覆盖实际价格范围，避免教程固定 1～19 导致部分点缺少参照。
    ax.plot(limits, limits, color="gray", linestyle="--", label="Perfect prediction")
    ax.scatter(truth, predicted, color="green", label="First 10 test samples")
    ax.set(
        title="Boston housing prediction",
        xlabel="Ground truth (1000 USD)",
        ylabel="Prediction (1000 USD)",
        xlim=limits,
        ylim=limits,
    )
    # 等比例坐标下，点越接近 y=x 直线，表示预测误差越小。
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()

    # 先保存，再释放画布；房价单位为千美元，与原始 MEDV 标签一致。
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def predict(infer_reader, model_path, output_dir):
    """按测试集原始顺序读取一批样本，使用已保存模型进行预测。"""

    # infer_reader 的批次大小为 10，next 只取第一批，不打乱顺序。
    test_data = next(infer_reader())
    test_x, test_y = batch_to_arrays(test_data)

    # 重新加载参数，确保预测使用的是磁盘中的最终模型。
    with fluid.dygraph.guard(fluid.CPUPlace()):
        regressor = load_regressor(model_path)
        with fluid.dygraph.no_grad():
            X = fluid.dygraph.to_variable(test_x)
            fetch_list = regressor(X).numpy()

    # 每行输出一个预测值和真实值；这里只展示前 10 条测试样本。
    for i, (result, target) in enumerate(zip(fetch_list[:, 0], test_y[:, 0])):
        LOGGER.info(
            "No.%d: infer result is %.2f, ground truth is %.2f",
            i, float(result), float(target),
        )

    # 此处的 MSE 只对应当前预测批次，不代表全部测试集的评估误差。
    metrics = regression_metrics(test_y, fetch_list)
    LOGGER.info("First %d test samples, MSE: %.5f", len(test_data), metrics["mse"])

    # 训练集占原始文件前 404 行，因此测试集第一条的原始行号为 405。
    output_dir = Path(output_dir)
    save_predictions(
        output_dir / "inference_predictions.csv",
        test_y,
        fetch_list,
        list(range(405, 405 + len(test_data))),
    )

    # 保存逐条预测数据后，再生成真实值和预测值的散点对比图。
    (output_dir / "image").mkdir(parents=True, exist_ok=True)
    draw_infer_result(test_y, fetch_list, output_dir / "image" / "inference_result.png")

    return metrics
