"""Step1：使用任务书的 UCI Housing 接口准备数据。"""

import hashlib
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from paddle import batch
from paddle.dataset import uci_housing
from paddle.dataset.common import DATA_HOME
from paddle.reader import shuffle


BUF_SIZE = 500          # 每次最多缓存 500 条样本，再打乱读取顺序
BATCH_SIZE = 30         # 训练和评估时，每个批次最多读取 30 条样本
INFER_BATCH_SIZE = 10   # 预测时，按原始顺序读取测试集的前 10 条样本


def normalize_features(features):
    """按 PyTorch 版公式归一化原始输入特征，返回数据、均值和极差。"""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 13:
        raise ValueError("原始输入特征必须为非空的 (N, 13) 数组，不包含房价标签")
    if not np.isfinite(features).all():
        raise ValueError("原始输入特征必须全部为有限数值")

    # 与 pytorch_ver/data_prepare.py 相同：先减去各列均值，再除以各列极差。
    feature_mean = features.mean(axis=0)
    feature_range = features.max(axis=0) - features.min(axis=0)
    if np.any(feature_range == 0):
        raise ValueError("输入特征存在极差为 0 的常量列，无法按极差归一化")
    normalized = (features - feature_mean) / feature_range
    return normalized, feature_mean, feature_range


@contextmanager
def _working_directory(directory):
    """内置数据接口会写 image/ranges.png，将这一副产物放在输出目录。"""
    previous = Path.cwd()
    os.chdir(directory)

    try:
        yield
    finally:
        # 即使数据加载发生异常，也恢复调用者原来的工作目录。
        os.chdir(previous)


def prepare_data(output_dir):
    """显式归一化 13 个输入特征，保留顺序 8:2 划分和原始房价标签。"""

    # 准备数据输出目录；路径不存在时自动创建。
    output_dir = Path(output_dir).resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 内置接口负责下载和原始极差图，也已归一化输入；后面独立从原始数据计算。
    with _working_directory(output_dir):
        train_data = uci_housing.train()
        test_data = uci_housing.test()

    # reader 是可重复调用的函数；此处先读取全部样本，检查数量和形状。
    train_samples = list(train_data())
    test_samples = list(test_data())
    train_x, train_y = batch_to_arrays(train_samples)
    test_x, test_y = batch_to_arrays(test_samples)

    if train_x.shape != (404, 13) or test_x.shape != (102, 13):
        raise ValueError("UCI Housing 应包含 404 条训练样本、102 条测试样本和 13 个特征")

    # 校验缓存中的原始文件，保存一份未归一化的数据用于追溯。
    raw_path = Path(DATA_HOME) / "uci_housing" / "housing.data"
    raw_bytes = raw_path.read_bytes()
    digest = hashlib.md5(raw_bytes).hexdigest()
    if digest != uci_housing.MD5:
        raise ValueError("原始数据 MD5 与 Paddle 内置数据集不一致")

    raw = np.loadtxt(raw_path)
    if raw.shape != (506, 14) or not np.isfinite(raw).all():
        raise ValueError("原始数据应为 506×14，且所有值均为有限数值")
    shutil.copyfile(raw_path, data_dir / "housing.data")

    # 使用原始文件的前 13 列，避免对内置 reader 已处理的数据再次归一化。
    # 与 PyTorch 版一样，统计量来自完整 506 条数据，然后再顺序划分。
    expected_x, feature_mean, feature_range = normalize_features(raw[:, :13])

    # 对照原始数据检查特征归一化和标签，确认读取过程没有改变样本顺序。
    np.testing.assert_allclose(train_x, expected_x[:404], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(test_x, expected_x[404:], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.concatenate([train_y, test_y]), raw[:, 13:14])

    # 后续训练和预测实际使用这里显式计算的输入，统一转换为 float32。
    train_x = expected_x[:404].astype(np.float32)
    test_x = expected_x[404:].astype(np.float32)
    train_y = raw[:404, 13:14].astype(np.float32)
    test_y = raw[404:, 13:14].astype(np.float32)

    def normalized_train_data():
        """重复读取归一化后的训练输入和未变换的房价标签。"""
        yield from zip(train_x, train_y)

    def normalized_test_data():
        """按原始顺序读取测试样本，供评估和前 10 条预测使用。"""
        yield from zip(test_x, test_y)

    # NPZ 同时保存特征、标签和归一化参数，供训练和离线验收使用。
    np.savez(
        data_dir / "prepared_data.npz",
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        feature_mean=feature_mean,
        feature_range=feature_range,
    )

    # 与参考代码一致：训练和评估都打乱；预测按测试集原始顺序取 10 条。
    # 显式导入的 shuffle 就是 paddle.reader.shuffle，便于编辑器识别。
    train_reader = batch(
        shuffle(normalized_train_data, buf_size=BUF_SIZE),
        batch_size=BATCH_SIZE,
    )
    test_reader = batch(
        shuffle(normalized_test_data, buf_size=BUF_SIZE),
        batch_size=BATCH_SIZE,
    )
    infer_reader = batch(normalized_test_data, batch_size=INFER_BATCH_SIZE)

    # 记录数据来源、字段语义和划分方式，避免只保存结果而丢失计算口径。
    metadata = {
        "source_url": uci_housing.URL,
        "source_md5": digest,
        "total_samples": len(raw),
        "unique_rows": len(np.unique(raw, axis=0)),
        "missing_values": int(np.isnan(raw).sum()),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "feature_count": 13,
        "feature_names": list(uci_housing.feature_names),
        "target": "MEDV: median value of owner-occupied homes in thousands of dollars",
        "split": "first 404 rows for training; last 102 rows for testing",
        "normalization": "(x - full_dataset_mean) / full_dataset_range; target unchanged",
        "normalization_source": "explicit NumPy calculation from raw feature columns",
        "first_train_sample": {
            "features": train_x[0].tolist(),
            "price": float(train_y[0, 0]),
        },
    }

    return train_reader, test_reader, infer_reader, metadata


def batch_to_arrays(data):
    """每条样本由 13 个特征和一个标签组成，转换为动态图所需 float32。"""

    # sample[0] 是房屋特征，sample[1] 是房价；保留标签的第二个维度。
    x = np.asarray([sample[0] for sample in data], dtype=np.float32)
    y = np.asarray([sample[1] for sample in data], dtype=np.float32)

    # X 为 (批次大小, 13)，y 为 (批次大小, 1)，与线性层输入输出对应。
    if x.ndim != 2 or x.shape[1] != 13 or y.shape != (len(x), 1):
        raise ValueError("批次输入形状应为 (N, 13)，标签形状应为 (N, 1)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("批次中存在非有限数值")

    return x, y
