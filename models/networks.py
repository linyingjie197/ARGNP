import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


# 用于创建并返回激活层。它接收三个参数：激活函数类型（act）、是否原地操作（inplace）和负斜率（neg_slope）。根据act的值，它会创建并返回一个ReLU、LeakyReLU或PReLU激活层。
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


# 用于创建并返回归一化层。它接收两个参数：归一化类型（norm）和通道数（nc）。根据norm的值，它会创建并返回一个BatchNorm1d或InstanceNorm1d归一化层。
def norm_layer(norm, nc):
    # helper selecting normalization layer
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)  # 批量归一化通过对每个小批量的数据进行归一化，使得其均值为0，方差为1，从而使模型的训练更加稳定。
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False) #一维的实例归一化  在每个样本的每个特征图上独立地进行归一化
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


# class MLP(nn.Module):
#     def __init__(self, channel_sequence):
#         super().__init__()
#         nb_layers = len(channel_sequence) - 1
#         self.seq = nn.Sequential()
#         for i in range(nb_layers):
#             self.seq.add_module(f"fc{i}", nn.Linear(channel_sequence[i], channel_sequence[i + 1]))
#             if i != nb_layers - 1:
#                 self.seq.add_module(f"ReLU{i}", nn.ReLU(inplace=True))
        
#     def forward(self, x):
#         out = self.seq(x)
#         return out

# 
class MLP(nn.Module): 

    # 包含了一系列的线性层、归一化层和激活层
    def __init__(self, channels, act='leakyrelu', norm=None, bias=True):
        super().__init__()
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))  #为每一层添加一个线性转换层。
            if norm:
                m.append(norm_layer(norm, channels[i]))  # 为每一层添加一个归一化层。
            if act:
                m.append(act_layer(act))  # # 为每一层添加一个归一化层。
        self.body = Seq(*m)   # 将列表m中的所有元素解包，然后传递给Seq(用于创建一个神经网络模块)。
    
    def forward(self, x):
        return self.body(x) # 输入数据将会按照self.body中的层的顺序进行处理。

# class MLP(Seq):
#     def __init__(self, channels, act='leakyrelu', norm=None, bias=True):
#         m = []
#         for i in range(1, len(channels)):
#             m.append(Lin(channels[i - 1], channels[i], bias))
#             if norm:
#                 m.append(norm_layer(norm, channels[i]))
#             if act:
#                 m.append(act_layer(act))

#         super(MLP, self).__init__(*m)
