from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray 

from torch.utils.data import dataloader, DataLoader
from torch.utils.data import TensorDataset
from torch import Tensor, Tuple

from typing import Literal


# 数据集路径
DATA_PATH = Path(__file__).resolve().parent / "data" / "housing.data"

def data_scale() -> NDArray[np.float64] :
    """ 处理数据集的导入和放缩 """
    # 加载数据集
    data_set_raw = np.loadtxt(fname=DATA_PATH, dtype=np.float64) # (506, 14)

    # 归一化
    center: NDArray[np.float64] = data_set_raw.mean(axis=0)
    scale: NDArray[np.float64] = (data_set_raw.max(axis=0) - data_set_raw.min(axis=0))
    data_set: NDArray[np.float64] = (data_set_raw - center) / scale

    return data_set

def load_data(data_set: NDArray[np.float64], trainORtest: Literal['train', 'test']) -> "DataLoader[Tuple[Tensor, ...]]" :
    """ 加载训练集、测试集 """

    x_train = data_set[0:404, 0:13]  # 训练集备用（404， 13）
    x_test = data_set[404:506, 0:13] # 测试集备用（102， 13）   
    y_train = data_set[0:404, 13:14] # 训练集备用 (404， 1)
    y_test = data_set[404:506, 13:14] # 训练集备用(102, 1)

    # 将 numpy 数组转为 tensor，统一为模型默认的 float32 类型
    # 训练用
    dataset_tensor_train = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float()
    )
    # 测试用
    dataset_tensor_test = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).float()
    )

    # 训练loader
    train_loader = dataloader.DataLoader(
        dataset=dataset_tensor_train,
        batch_size=30,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    # 测试用loader
    test_loader = dataloader.DataLoader(
        dataset=dataset_tensor_test,
        batch_size=30,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    # 输出方式 'train' or 'test'
    if trainORtest == 'train' :
        return train_loader
    elif trainORtest == 'test' :
        return test_loader
    else :
        ValueError("trainOrtest must be 'train' or 'test'")
