"""Step3：训练、保存模型，并绘制训练损失曲线。"""

import csv
import logging
from pathlib import Path

import matplotlib

# 使用非交互后端，在终端中直接保存图片；必须放在导入 pyplot 之前。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid

from .data_prepare import batch_to_arrays
from .model import Regressor


EPOCH_NUM = 250          # 完整遍历训练集 250 次
LEARNING_RATE = 0.005    # SGD 每次更新参数的步长，与任务书代码一致
LOGGER = logging.getLogger(__name__)


def draw_train_process(iters, train_costs, output_path):
    """绘制每次参数更新前的批次 MSE，展示训练损失的变化趋势。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, train_costs, color="red", linewidth=0.7, label="training cost")
    ax.set(
        title="Training cost",
        xlabel="Iteration",
        ylabel="Batch MSE (1000 USD)^2",
    )
    ax.grid(alpha=0.3)
    ax.legend()

    # 先调整布局，再保存并释放画布，避免标签裁切及连续绘图占用内存。
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _train_mse(regressor, train_x, train_y):
    """固定参数下计算全训练集误差，区别于更新过程中的批次误差。"""
    regressor.eval()

    # 此处只计算预测结果，不更新参数，也不记录反向传播所需的梯度。
    with fluid.dygraph.no_grad():
        prediction = regressor(fluid.dygraph.to_variable(train_x)).numpy()

    # 恢复训练模式；使用 float64 统计误差，减少求和时的数值舍入。
    regressor.train()
    return float(np.mean((prediction.astype(np.float64) - train_y) ** 2))


def train(
    train_reader,
    output_dir,
    epochs=EPOCH_NUM,
    learning_rate=LEARNING_RATE,
    seed=2026,
):
    """训练线性回归模型，保存参数、逐批损失和逐轮加权误差。"""

    # 准备模型与图像目录，模型路径不包含 .pdparams 后缀。
    output_dir = Path(output_dir)
    model_path = output_dir / "model" / "fit_a_line"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    # 全训练集数组用于计算训练前后的 MSE，不参与额外的参数更新。
    with np.load(output_dir / "data" / "prepared_data.npz") as data:
        train_x, train_y = data["train_x"], data["train_y"]

    history = []       # 每次迭代的编号、轮次、批次、样本数及损失
    epoch_costs = []   # 每轮按样本数加权的训练过程 MSE

    # 在 CPU 动态图环境中创建网络并训练，避免依赖本机 GPU 配置。
    with fluid.dygraph.guard(fluid.CPUPlace()):
        # guard 会创建新 Program，因此要在 guard 内设置初始化随机种子。
        fluid.default_main_program().random_seed = seed
        fluid.default_startup_program().random_seed = seed

        regressor = Regressor()
        initial_mse = _train_mse(regressor, train_x, train_y)
        regressor.train()

        # 优化器与模型一起创建一次，不需要在每个 batch 中重复实例化。
        optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate=learning_rate,
            parameter_list=regressor.parameters(),
        )

        iteration = 0
        for pass_id in range(epochs):
            squared_error_sum = 0.0
            sample_count = 0

            for batch_id, data in enumerate(train_reader()):
                # 将批次数据转换为 float32 数组，再转为动态图变量。
                batch_x, batch_y = batch_to_arrays(data)
                X = fluid.dygraph.to_variable(batch_x)
                y = fluid.dygraph.to_variable(batch_y)

                # 前向计算：先求每条样本的平方误差，再取批次均值作为损失。
                y_predict = regressor(X)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)
                cost_value = float(avg_cost.numpy().reshape(-1)[0])
                if not np.isfinite(cost_value):
                    raise ValueError("训练损失出现非有限数值")

                # 反向传播求梯度，SGD 更新参数，然后清零梯度。
                # Paddle 梯度会累积，清零后下一批次才能独立计算梯度。
                avg_cost.backward()
                optimizer.minimize(avg_cost)
                regressor.clear_gradients()

                # 记录当前批次损失，按实际样本数累计本轮平方误差。
                # 最后一批只有 14 条，不能与 30 条的批次等权平均。
                iteration += 1
                history.append((iteration, pass_id, batch_id, len(data), cost_value))
                squared_error_sum += cost_value * len(data)
                sample_count += len(data)

                if batch_id % 10 == 0:
                    LOGGER.info("Pass:%d,Batch:%d Cost:%.5f", pass_id, batch_id, cost_value)

            # 一轮结束后核对样本数，确认 reader 没有漏掉末尾样本。
            if sample_count != len(train_x):
                raise ValueError("训练轮次未完整覆盖训练集")
            epoch_costs.append(squared_error_sum / sample_count)

        # 全部轮次结束后，计算固定参数的 MSE，并保存最终模型。
        final_mse = _train_mse(regressor, train_x, train_y)
        fluid.save_dygraph(regressor.state_dict(), str(model_path))

        # 验收时可将内存模型输出与重新加载后的输出逐项比较。
        with fluid.dygraph.no_grad():
            reference_predictions = regressor(fluid.dygraph.to_variable(train_x)).numpy()
        np.save(
            output_dir / "data" / "train_predictions_before_reload.npy",
            reference_predictions,
        )

    # 保存逐批和逐轮训练记录；epoch、batch 从 0 开始，iteration 从 1 开始。
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "epoch", "batch", "sample_count", "mse"])
        writer.writerows(history)

    with (output_dir / "epoch_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "online_train_mse"])
        writer.writerows(enumerate(epoch_costs))

    # 训练曲线使用逐批记录，返回的指标同时保留训练前后全训练集误差。
    draw_train_process(
        [row[0] for row in history],
        [row[4] for row in history],
        image_dir / "training_cost.png",
    )
    LOGGER.info(
        "Training MSE: %.5f -> %.5f; model saved: %s.pdparams",
        initial_mse, final_mse, model_path,
    )

    return {
        "epochs": epochs,
        "iterations": len(history),
        "learning_rate": learning_rate,
        "initial_train_mse": initial_mse,
        "final_train_mse": final_mse,
        "first_epoch_online_mse": epoch_costs[0],
        "last_epoch_online_mse": epoch_costs[-1],
    }
