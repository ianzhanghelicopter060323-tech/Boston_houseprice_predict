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

默认训练 250 轮，使用 SGD、学习率 0.005，每批 30 条；该命令会更新 `src/pytorch_ver/data/price_predict_model_pytorch.pth`。训练曲线需要本次训练中实际记录的损失，不能由已有模型参数恢复。当前 `main.py` 使用 `torch.manual_seed(0)` 设置随机种子。

已有模型时，重新计算评价指标、保存逐条预测并生成预测图，不重新训练：

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
| `metrics.json` | 最终模型在训练集、测试集和前 10 条测试样本上的 MSE、RMSE，包含两种尺度、样本数、单位和模型 SHA256 |
| `metrics.csv` | `train`、`test`、`first10` 三组样本的原始尺度与归一化尺度 MSE、RMSE 汇总 |
| `train_predictions.csv` | 全部 404 条训练样本的逐条预测与误差 |
| `test_predictions.csv` | 全部 102 条测试样本的逐条预测与误差 |
| `inference_predictions.csv` | 前 10 条测试样本的逐条预测与误差 |

数据共 506 条，前 404 条训练，后 102 条测试。250 轮训练应有 `250 × 14 = 3500` 个曲线点，最后一批 14 条。预测图使用测试集原始顺序的前 10 条，对应原始文件第 405～414 行。

极差柱状图使用归一化前的特征，不包含房价标签，横轴顺序与原始文件前 13 列一致。各特征的单位不同，柱高体现原始数值跨度，不用于比较特征重要性；归一化后的非恒定特征极差均为 1，不能代替此图。

沿用当前 PyTorch 数据处理：全部 14 列，包括房价标签，都使用完整数据集的 `(值 - 均值) / 极差`。因此图中坐标标为 `normalized MEDV`，训练曲线是归一化尺度的 MSE。评价时在 `predict.py` 中按 `预测房价 = 归一化预测值 × 标签极差 + 标签均值` 还原预测，再与原始标签计算指标。原始 MSE 单位为（千美元）²，RMSE 单位为千美元；当前标签极差为 45，原始 MSE 为归一化 MSE 的 2025 倍。完整数据统计量也包含测试集信息，预处理口径沿用现有实现。

`predict_price()` 打印 404 条训练样本 MSE、102 条测试样本 MSE、测试集 RMSE，以及前 10 条测试样本 MSE，四项均按原始房价尺度报告。训练集指标由保存后的最终模型在全部训练样本上重新计算，与训练过程中逐批记录的损失不同。函数仍返回归一化尺度的全部测试集 MSE，覆盖 `30 + 30 + 30 + 12` 条。

三份逐条预测 CSV 均包含 `source_row`、`ground_truth_normalized`、`prediction_normalized`、`ground_truth`、`prediction`、`error` 和 `squared_error`。不带 `_normalized` 的房价列为千美元，`error` 定义为预测值减真实值，`squared_error` 是其平方。`metrics.csv` 使用 `mse_1000usd_squared`、`rmse_1000usd` 明确原始单位。

所有评价指标的计算与 JSON/CSV 保存均位于 `predict.py`；`visualization.py` 只使用传入的数据绘图和保存图片。

`predict_price()` 负责组织调用，同文件中的四个函数分别处理四项评价：

| 函数 | 职责 |
|---|---|
| `calculate_train_mse()` | 从完整预测中取前 404 条，计算训练集 MSE |
| `calculate_test_mse()` | 从完整预测中取后 102 条，计算测试集 MSE |
| `calculate_test_rmse()` | 接收已计算的测试集 MSE，计算 RMSE |
| `calculate_first10_mse()` | 取原始第 405～414 行，计算前 10 条测试样本 MSE |

三个 MSE 函数接收按原始顺序排列的完整 506 条真实值与预测值；`collect_predictions()` 负责模型推理，`build_evaluation_metrics()` 整理两种尺度，`save_evaluation_results()` 统一保存 JSON 与 CSV。四项指标在 `predict_price()` 中分别调用，输出格式和归一化测试 MSE 返回值沿用原接口。

## 与参考代码的区别

训练曲线保留任务书逐批记录损失的方式；预测图保留前 10 条真实值与预测值散点对比。对角线改为根据真实值和预测值的共同范围绘制，适应归一化后的负值和接近零的数值。图中文字使用英文，避免依赖额外中文字体；训练参数、模型结构与归一化算法沿用现有 PyTorch 实现。

## 本轮验证

在现有 Python 3.8.10、PyTorch 2.4.1+cu121、NumPy 1.23.5、Matplotlib 3.7.5 环境中，以 CPU 和 Agg 后端实际运行 250 轮。已检查 3500 条连续迭代记录、前 10 条预测与原始行号、全部 102 条预测的 MSE，并使用 NumPy 矩阵乘法与磁盘权重独立核对。运行过程中未导入 Paddle，保存后没有遗留打开的画布。

已人工查看两张图片，确认标题、坐标及图例完整；窗口显示未验证。本轮完整运行日志在 `outputs/pytorch_ver/visualization_run.log`，运行前的已有模型保留在 `outputs/pytorch_ver/model_before_visualization.pth`。这两个文件属于本轮验证记录，普通运行入口不会自动备份模型。

补充极差柱状图时，单独读取 506 条原始数据生成 `image/ranges.png`，逐项核验 13 根柱的高度与独立计算的最大值减最小值一致，并查看图片确认标签完整。此次未重新训练，模型文件内容保持一致，未导入 Paddle。

补充评价指标后，仅加载当前模型运行 `predict_price()`，独立使用 NumPy 和磁盘权重核对训练、测试及前 10 条样本的 MSE/RMSE。已检查三组样本数为 404、102、10，JSON 与汇总 CSV 一致，逐条 CSV 的真实值、预测值、误差、平方误差和原始行号正确，两种尺度的 MSE 比值为 2025。模型文件与绘图模块均未修改，未导入 Paddle；指标随当前模型重新计算，不固定使用历史对比表中的数值。

拆分四项评价函数后，已对照修改前的运行快照：三份逐条预测 CSV 完全一致，汇总指标与返回值在浮点舍入范围内一致；另用训练组、测试前 10 条、其余测试样本具有不同已知误差的数组，核验四个函数的样本范围与计算结果。此次未重新训练。
