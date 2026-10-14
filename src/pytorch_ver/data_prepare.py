import numpy as np
import torch
from numpy.typing import NDArray 

from torch.utils.data import dataloader, DataLoader
from torch.utils.data import TensorDataset


# 数据集路径
DATA_PATH = "/home/ianzhang/ai_foundation_prog/Bostion_houseprice_predict/src/pytorch_ver/data/housing.data"

def data_scale() -> NDArray[np.float64] :
    """ 处理数据集的导入和放缩 """
    # 加载数据集
    data_set_raw = np.loadtxt(fname=DATA_PATH, dtype=np.float64) # (506, 14)

    # 归一化
    center: NDArray[np.float64] = data_set_raw.mean(axis=0)
    scale: NDArray[np.float64] = (data_set_raw.max(axis=0) - data_set_raw.min(axis=0))
    data_set: NDArray[np.float64] = (data_set_raw - center) / scale

    return data_set

def load_data(data_set: NDArray[np.float64]) -> "tuple[DataLoader[tuple], DataLoader[tuple]]":
    """ 加载训练集、测试集 """

    x_train = data_set[0:405, 0:13]  # 训练集备用（404， 13）
    x_test = data_set[404:506, 0:13] # 测试集备用（102， 13）   
    y_train = data_set[0:405, 13:14] # 训练集备用 (404， 1)
    y_test = data_set[404:506, 13:14] # 训练集备用(102, 1)

    # 将numpy中的矩阵变成tenor并加载备用
    # 训练用
    dataset_tensor_train = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train)
    )
    # 测试用
    dataset_tensor_test = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(y_test)
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
        dataset=dataset_tensor_train,
        batch_size=30,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    return train_loader, test_loader
