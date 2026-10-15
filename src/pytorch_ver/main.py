import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.sgd import SGD

from model_set import Setmodel
import data_prepare as da
import predict
from visualization import OUTPUT_DIR, draw_feature_ranges, draw_train_process


def train_all(epochs: int, show: bool = False) -> None :
    """ 主函数调用训练 """
    if epochs <= 0:
        raise ValueError("训练轮数必须大于 0")

    # loader准备
    train_loader = da.load_data(da.data_scale(), 'train')

    # 模型准备
    model = Setmodel()

    # optimiezer
    optimizer = SGD(
        model.parameters(),
        lr=0.005
    )

    # loss
    loss_fn = nn.MSELoss()

    # 与任务书一致，每次参数更新记录一个批次损失，而不是只记录每轮均值。
    iters = []
    train_costs = []

    # 训练循环
    model.train()
    for epoch in range(epochs) :
        sample_count = 0 # 本轮训练样本计数
        total_squared_error = 0 # 每轮 epoch 平方误差总和

        for batch_x, batch_y in train_loader :
            y_predict = model(batch_x)

            loss = loss_fn(y_predict, batch_y)
            cost_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算误差
            total_squared_error += cost_value * batch_x.size(0)
            sample_count += batch_x.size(0)

            iters.append(len(iters) + 1)
            train_costs.append(cost_value)

        epoch_mse = total_squared_error / sample_count

        if epoch % 20 == 0 :
            print(f"epoch={epoch}, loss={epoch_mse}")

    # 保存模型参数
    torch.save(
        model.state_dict(),
        Path(__file__).resolve().parent / "data" / "price_predict_model_pytorch.pth"
    )
    print("====================== training done, model param saved ======================")

    # 保存实际绘图数据，便于核对每轮的 14 个批次和整条损失曲线。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "batch_mse_normalized"])
        writer.writerows(zip(iters, train_costs))

    image_path = draw_train_process(iters, train_costs, show=show)
    print(f"训练损失曲线已保存: {image_path}")


if __name__ == "__main__" :
    epochs = 250
    show_plots = False # 有图形界面且需要弹窗时改为 True；两种设置都会保存图片。

    # 数据准备阶段：用全部 506 条原始数据的前 13 列绘制特征极差柱状图。
    raw_data = np.loadtxt(da.DATA_PATH, dtype=np.float64)
    range_image = draw_feature_ranges(raw_data[:, :13], show=show_plots)
    print(f"原始特征极差柱状图已保存: {range_image}")

    # train_all 内部记录训练损失，并调用 draw_train_process 保存训练曲线。
    train_all(epochs=epochs, show=show_plots)

    # predict_price 内部加载训练好的模型，并绘制前 10 条测试样本的对比图。
    predict.predict_price(show=show_plots)

    print(f"实验完成，图表与对应 CSV 已保存到: {OUTPUT_DIR}")
