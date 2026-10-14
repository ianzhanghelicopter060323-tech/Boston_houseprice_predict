"""Step2：配置任务书中的单层线性回归网络。"""

import paddle.fluid as fluid


class Regressor(fluid.dygraph.Layer):
    """单层线性回归模型，学习 13 个特征的权重和一个偏置。"""

    def __init__(self):
        """初始化全连接层，输出一个房价预测值。"""
        super(Regressor, self).__init__()

        # 13 个房屋特征映射为一个房价；不使用激活函数，即 Y'=wX+b。
        self.linear = fluid.dygraph.Linear(input_dim=13, output_dim=1, act=None)

    def forward(self, inputs):
        """前向计算：将 (N, 13) 的特征转换为 (N, 1) 的预测房价。"""
        return self.linear(inputs)


def load_regressor(model_path):
    """在 dygraph.guard 内调用，只加载一次模型并切换到评估模式。"""

    # model_path 是文件前缀，load_dygraph 会读取对应的 .pdparams 文件。
    # 返回值的第二项为优化器状态，本实验预测时不需要使用。
    state_dict, _ = fluid.load_dygraph(str(model_path))

    # 先创建相同结构的网络，再填入保存的权重和偏置。
    regressor = Regressor()
    regressor.load_dict(state_dict)
    regressor.eval()

    return regressor
