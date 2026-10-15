# PyTorch 波士顿房价可视化

使用 PyTorch 完成训练和预测，使用 Matplotlib 绘图，不导入 Paddle。绘图函数位于 `visualization.py`，训练和预测入口分别在 `main.py` 与 `predict.py`。

## 运行

在项目根目录使用现有 Python 3.8 环境。依赖见 `requirements-pytorch.txt`，需要安装时执行：

```bash
.venv/bin/python -m pip install -r requirements-pytorch.txt
```

运行数据可视化、训练与预测，生成三张图及训练、预测对应的 CSV：

```bash
.venv/bin/python src/pytorch_ver/main.py
```

默认训练 250 轮，使用 SGD、学习率 0.005，每批 30 条；该命令会更新 `src/pytorch_ver/data/price_predict_model_pytorch.pth`。训练曲线需要本次训练中实际记录的损失，不能由已有模型参数恢复。训练初始化和样本打乱未固定随机种子，不同运行的曲线与误差会变化。

已有模型时，只生成预测图，不重新训练：

```bash
.venv/bin/python src/pytorch_ver/predict.py
```

默认只保存图片，适合终端运行；有图形界面时可调用 `train_all(epochs=250, show=True)` 或 `predict_price(show=True)`，先保存再显示并释放画布。在无图形界面的环境中，可给上述命令加上 `MPLBACKEND=Agg`。Agg 只用于保存图片，不验证窗口显示。

## 图表与数据口径

输出目录由代码位置定位，固定为项目下的 `outputs/pytorch_ver/`：

| 文件 | 内容 |
|---|---|
| `image/ranges.png` | 全部 506 条原始数据中 13 个特征的极差（最大值减最小值）柱状图 |
| `image/training_cost.png` | 横轴为批次迭代次数，纵轴为参数更新前的批次 MSE |
| `training_history.csv` | `iteration` 从 1 开始，`batch_mse_normalized` 与曲线逐点对应 |
| `image/inference_result.png` | 测试集前 10 条，横轴真实值、纵轴预测值，含等比例 `y=x` 参照线 |
| `inference_predictions.csv` | 前 10 条的原始行号、归一化真实值、归一化预测值 |

数据共 506 条，前 404 条训练，后 102 条测试。250 轮训练应有 `250 × 14 = 3500` 个曲线点，最后一批 14 条。预测图使用测试集原始顺序的前 10 条，对应原始文件第 405～414 行。

极差柱状图使用归一化前的特征，不包含房价标签，横轴顺序与原始文件前 13 列一致。各特征的单位不同，柱高体现原始数值跨度，不用于比较特征重要性；归一化后的非恒定特征极差均为 1，不能代替此图。

沿用当前 PyTorch 数据处理：全部 14 列，包括房价标签，都使用完整数据集的 `(值 - 均值) / 极差`。因此坐标标为 `normalized MEDV`，损失是归一化尺度的 MSE，不能直接当作原始房价或与任务书未归一化标签的损失比较。完整数据统计量也包含测试集信息，预处理口径沿用现有实现。

`predict_price()` 仍返回全部 102 条测试样本的 MSE：平方误差总和除以 102，覆盖 `30 + 30 + 30 + 12` 条。这个数值与仅展示前 10 条的散点图样本范围不同。

## 与参考代码的区别

训练曲线保留任务书逐批记录损失的方式；预测图保留前 10 条真实值与预测值散点对比。对角线改为根据真实值和预测值的共同范围绘制，适应归一化后的负值和接近零的数值。图中文字使用英文，避免依赖额外中文字体；训练参数、模型结构与归一化算法沿用现有 PyTorch 实现。

## 本轮验证

在现有 Python 3.8.10、PyTorch 2.4.1+cu121、NumPy 1.23.5、Matplotlib 3.7.5 环境中，以 CPU 和 Agg 后端实际运行 250 轮。已检查 3500 条连续迭代记录、前 10 条预测与原始行号、全部 102 条预测的 MSE，并使用 NumPy 矩阵乘法与磁盘权重独立核对。运行过程中未导入 Paddle，保存后没有遗留打开的画布。

已人工查看两张图片，确认标题、坐标及图例完整；窗口显示未验证。本轮完整运行日志在 `outputs/pytorch_ver/visualization_run.log`，运行前的已有模型保留在 `outputs/pytorch_ver/model_before_visualization.pth`。这两个文件属于本轮验证记录，普通运行入口不会自动备份模型。

补充极差柱状图时，单独读取 506 条原始数据生成 `image/ranges.png`，逐项核验 13 根柱的高度与独立计算的最大值减最小值一致，并查看图片确认标签完整。此次未重新训练，模型文件内容保持一致，未导入 Paddle。
