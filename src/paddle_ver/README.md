# Paddle 波士顿房价实验

按任务书完成数据准备、13→1 线性回归网络配置、SGD 训练、模型保存与重载、完整测试集评估、前 10 条测试样本预测及可视化。代码保留 Paddle 1.8.5 的 `paddle.fluid` 动态图接口及中文注释。

## 运行

在项目根目录执行，沿用现有 `.venv`（Python 3.8）。依赖版本见根目录 `requirements.txt`，不需要更换框架版本。

```bash
# 必要时安装依赖
.venv/bin/python -m pip install -r requirements.txt

# 完整实验：250 轮、学习率 0.005、随机种子 2026、CPU
.venv/bin/python -m src.paddle_ver.main

# 验收已有产物：不重新下载数据、不重新训练
.venv/bin/python -m src.paddle_ver.verify

# 重新加载已有模型，只做评估或前 10 条预测
.venv/bin/python -m src.paddle_ver.main --stage evaluate
.venv/bin/python -m src.paddle_ver.main --stage predict
```

也支持直接运行脚本，在项目根目录执行：

```bash
.venv/bin/python src/paddle_ver/main.py
```

若希望使用 `python` 命令，先激活项目环境：

```bash
source .venv/bin/activate
python src/paddle_ver/main.py
```

`ModuleNotFoundError: No module named 'paddle'` 表示当前解释器无法找到 Paddle，并不是旧版接口兼容错误。项目现有 `.venv` 已安装所需 Paddle，无需在系统 Python 中重复安装。VS Code 的工作区默认设置已指向 `.venv`；若编辑器已选过其他解释器，按 `Ctrl+Shift+P` → `Python: Select Interpreter`，选择项目 `.venv/bin/python`，再新建终端或使用 `Run Python File in Terminal`。终端使用哪个 Python 可通过 `python -c "import sys; print(sys.executable)"` 确认。

完整运行支持 `--epochs`、`--learning-rate`、`--seed`、`--output-dir`。完整运行会在指定目录重新训练并覆盖对应产物；默认目录根据代码文件位置定位为项目下的 `outputs/paddle_ver`。独立评估和预测不改变训练模型及完整运行的 `metrics.json`。

首次运行由 Paddle 下载 UCI Housing 数据，校验 MD5，并使用其标准缓存；首次运行需要网络。后续默认使用已通过校验的缓存，独立验收只读取本地输出。`--stage evaluate/predict` 仍使用内置数据接口。

## 文件与步骤

| 文件 | 职责 |
|---|---|
| `data_prepare.py` | 内置数据下载、显式输入归一化、shuffle/batch、数据校验与原始/处理后数据保存 |
| `model.py` | `Regressor` 线性层、模型重载 |
| `train.py` | 250 轮训练、清零梯度、保存参数、损失日志与曲线 |
| `evaluate.py` | 全部测试样本的 MSE、RMSE、MAE、R²及逐条结果 |
| `predict.py` | 重载模型，预测测试集前 10 条并绘制对比图 |
| `main.py` | 顺序执行实验、命令行参数、记录环境和指标 |
| `verify.py` | 不重新训练，独立核验数据、轮次、模型重载、指标和图像文件 |

## 数据与评估口径

数据为 506 行、14 列，前 13 列为特征，最后一列 MEDV 为自有住房价格中位数，单位为千美元。实测全部 506 行唯一，无缺失值。内置接口将前 404 条作为训练集、后 102 条作为测试集；训练和评估每批 30 条，shuffle 缓冲区大小为 500。预测按测试集原始顺序读取前 10 条，对应原始文件第 405～414 行（行号从 1 开始）。

与 PyTorch 版相同，13 个输入特征使用完整数据集的 `(x - 均值) / (最大值 - 最小值)` 归一化，再切分为 404 条训练样本和 102 条测试样本。`normalize_features()` 显式从原始文件的前 13 列计算，后续 reader 使用这一结果；均值、极差和归一化数据保存在 NPZ 中。内置 UCI reader 原本已经执行了同一输入变换，这次将该步骤显式实现，未对已归一化的 reader 数据重复处理。

本次对齐范围为输入特征，Paddle 房价标签保留千美元原值；当前 PyTorch 版还归一化了标签，因此两版输出的损失仍需换算到相同房价尺度再比较。这一教学口径使用了完整数据集的统计量，包含测试集特征信息；沿用现有划分与公式，未改为只用训练集统计量。

测试 MSE 是全部 102 条样本的平方误差之和除以 102，覆盖 `30 + 30 + 30 + 12` 条，不丢弃末批。RMSE、MAE 单位为千美元，MSE 单位为（千美元）²，R²无单位。测试结果 CSV 的样本编号是打乱后的遍历顺序，`source_row` 留空；前 10 条预测 CSV 保存原始行号。

损失曲线记录每次参数更新前的批次 MSE，共 3500 次迭代；`epoch_history.csv` 按每批样本数加权统计训练过程误差。`initial_train_mse` 和 `final_train_mse` 是固定模型参数下在全部 404 条训练样本上计算的 MSE，与训练过程中的批次损失含义不同。

## 产物

所有产物位于 `outputs/paddle_ver/`，由 `.gitignore` 忽略，但保留在本地供查看。

| 路径 | 内容 |
|---|---|
| `model/fit_a_line.pdparams` | 13 个权重与 1 个偏置 |
| `data/housing.data` | 原始数据副本 |
| `data/prepared_data.npz` | float32 训练/测试特征、标签及归一化均值/极差 |
| `data/dataset_info.json` | 来源 URL、MD5、划分、样本数、特征名和首条训练样本 |
| `data/train_predictions_before_reload.npy` | 保存模型前的训练集输出，用于模型重载一致性验收 |
| `training_history.csv`、`epoch_history.csv` | 逐批、逐轮训练记录，epoch/batch 从 0 开始 |
| `test_predictions.csv` | 全部 102 条测试样本真实值、预测值、误差和平方误差 |
| `inference_predictions.csv` | 前 10 条测试样本预测及原始行号 |
| `metrics.json` | 完整运行的环境、种子、配置、数据和指标 |
| `evaluation.json`、`inference.json` | 最近一次评估/预测的指标 |
| `all.log`、`evaluate.log`、`predict.log` | 对应实际执行阶段的运行日志 |
| `verification.json` | 独立验收结果，仅在运行验收命令后生成 |
| `image/ranges.png` | 内置接口生成的原始特征极差图，各特征单位不同 |
| `image/training_cost.png` | 训练批次 MSE 曲线 |
| `image/inference_result.png` | 前 10 条测试样本预测对比图，含 y=x 参照线 |

图表使用英文标签，避免缺少中文字体造成乱码；使用 Agg 后端保存并释放画布。已检查图像文件和标签，不验证 GUI 弹窗显示。

## 与任务书的必要差异

| 任务书代码/描述 | 调整及原因 |
|---|---|
| 在每个 batch 中重新创建 SGD | 优化器在训练循环外创建一次，保留 SGD 和学习率 0.005；教程注释中的 0.001 与其代码不符，本实验以代码为准 |
| `test_costs = test_cost.numpy()` 覆盖累计结果 | 拼接全部逐条测试结果计算 MSE，防止只评估最后一个批次或错误地等权平均不同大小的批次 |
| 在每个测试 batch 中加载模型 | 每个评估/预测阶段只加载一次模型，并在 `no_grad` 下计算 |
| 预测文字提到 `load_inference_model` | 按其实际动态图示例使用 `load_dygraph`，与 `.pdparams` 保存方式一致 |
| 散点图对角线固定 1～19 | 使用真实值与预测值共同的实际范围，等比例坐标，使所有点都有参照 |
| `plt.show()` 和当前目录保存模型 | 保存图片和模型到统一输出目录，适配终端运行；不依赖 GUI |
| 未记录随机种子和环境 | 设置 Python、NumPy 和 guard 内 Paddle Program 种子，记录实际依赖、CPU 和参数 |

显式导入 `paddle.reader.shuffle` 对应函数，便于编辑器识别；完整实验入口为 `src.paddle_ver.main` 模块。

## 本次实测

使用 Python 3.8.10、Paddle 1.8.5、NumPy 1.23.5、Matplotlib 3.7.5、Protobuf 3.20.3，在 CPU 上完整训练 250 轮、3500 次更新，seed=2026。

| 指标 | 实测值 |
|---|---:|
| 初始全训练集 MSE | 668.34448 |
| 训练后全训练集 MSE | 28.34614 |
| 全部 102 条测试样本 MSE | 13.95177 |
| 全部测试样本 RMSE | 3.73521 |
| 全部测试样本 MAE | 2.94746 |
| 全部测试样本 R² | 0.48112 |
| 前 10 条测试样本 MSE | 44.33703 |

教程给出的 `Test, Cost:8.60592` 来自有缺陷的评估代码及不同随机条件，不应作为本实验必须达到的结果。前 10 条预测误差也不代表完整测试集误差。数值以本地 JSON/CSV 产物为准；调整种子或参数后会改变结果。

已实测完整入口、独立评估和独立预测，三者均正常结束；后两个阶段执行前后的模型文件内容一致。离线验收通过：模型重载前后的全训练集输出逐项相同，全部测试样本指标与 NumPy 独立重算一致，训练每轮样本数及末批大小正确，CSV 中的误差与平方误差一致，三张图像有效。已人工查看三张图，标签完整；GUI 弹窗未验证。

显式实现输入归一化后，另在临时输出目录核验了全部 506 条特征与 PyTorch 同公式计算结果一致，保存的 NPZ 数组与修改前逐项一致。训练、测试 reader 可重复读取，末批分别保留 14 条和 12 条，前 10 条预测顺序正确。使用已有模型重新评估，测试 MSE 仍为 13.95177，前 10 条 MSE 仍为 44.33703；本次未重新训练或修改模型，标签保留原值。
