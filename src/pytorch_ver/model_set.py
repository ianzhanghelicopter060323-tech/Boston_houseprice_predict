from torch import nn


class Setmodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 单个线性层，不需要激活函数
        self.layers = nn.Linear(
            in_features=13, # 13个特征输入
            out_features=1
        )

    def forward(self, x) :
        return self.layers(x)