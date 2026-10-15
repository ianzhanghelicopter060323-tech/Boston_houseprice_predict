import csv
from pathlib import Path

import torch
from torch import nn
from model_set import Setmodel
import data_prepare as da
from visualization import OUTPUT_DIR, draw_infer_result


def predict_price(show: bool = False) -> float :
    """ 预测全部测试样本并返回 MSE，按任务书绘制前 10 条样本的对比图。 """

    # 建立相同的模型
    model = Setmodel()

    # 加载参数
    state_dict = torch.load(
        Path(__file__).resolve().parent / "data" / "price_predict_model_pytorch.pth",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state_dict)

    # 加载loader
    test_loader = da.load_data(da.data_scale(), 'test')

    # 回归任务使用均方误差，不统计分类任务中的正确率。
    model.eval()
    loss_fn = nn.MSELoss(reduction='sum')

    sample_count = 0
    total_squared_error = 0.0

    # 测试 loader 不打乱，保留原始顺序的前 10 条用于任务书中的预测图。
    infer_results = []
    ground_truths = []
    infer_sample_count = 10

    # no_grad 关闭预测过程中的梯度记录。
    print("============================== start to predict ==============================")
    with torch.no_grad() :
        for x, precise_y in test_loader :
            y_predict = model(x)

            remaining = infer_sample_count - len(infer_results)
            if remaining > 0:
                infer_results.extend(y_predict[:remaining, 0].tolist())
                ground_truths.extend(precise_y[:remaining, 0].tolist())

            for i, (result, target) in enumerate(zip(y_predict, precise_y)) :
                print(
                    f"No.{sample_count + i}: "
                    f"infer result is {result.item():.5f}, "
                    f"ground truth is {target.item():.5f}"
                )

            # 先累加逐条平方误差，再除以样本总数。
            # 最后一批只有 12 条，因此不能直接平均各批次的 MSE。
            total_squared_error += loss_fn(y_predict, precise_y).item()
            sample_count += x.size(0)

    test_mse = total_squared_error / sample_count
    # data_scale 同时归一化了标签，因此此处误差也在归一化尺度上。
    print(f"测试集平均误差（MSE，归一化房价）: {test_mse:.5f}")

    # 图中仅展示前 10 条；上面的 MSE 仍统计完整的 102 条测试样本。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "inference_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_row", "ground_truth_normalized", "prediction_normalized"])
        # 训练集占原始文件前 404 行，首条测试样本的行号为 405（从 1 开始）。
        for i, (target, result) in enumerate(zip(ground_truths, infer_results)):
            writer.writerow([405 + i, target, result])

    image_path = draw_infer_result(ground_truths, infer_results, show=show)
    print(f"前 {len(infer_results)} 条测试样本的预测对比图已保存: {image_path}")
    return test_mse

