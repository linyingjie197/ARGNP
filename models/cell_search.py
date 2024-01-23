import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import get_package, get_OPS, OPS
from models.networks import MLP

# 神经网络模块，通过加权求和的方式组合多个候选操作。权重确定每个操作对最终输出的贡献。如果选择了特定的操作，只会将该操作应用于输入。
class Mixed(nn.Module):
    
    def __init__(self, args, type):
        super().__init__()
        self.args       = args
        self.type       = type
        self._ops       = get_OPS(type)  #返回一个操作名称的列表 
        self.candidates = nn.ModuleDict({
            name: get_package(type)(args, OPS[name](args), name, affine = False)
            # name: get_package('v')(args, OPS['V_Max'](args), name, affine = False)
            for name in self._ops
        })   #用于存储候选操作的实例
             #通过字典推导式，它遍历`_ops`中的操作名称。对于每个操作名称，它使用`get_package`函数和相应的参数进行实例化，`get_package`函数使用`type`参数返回相应的包，并使用`OPS[name](args)`，`name`和`affine=False`作为参数来初始化操作。
    #前向传播，在已经确定的操作中更新w（归一化+被选中的操作进行执行），相加得到所有候选操作的加权输出之和
    def forward(self, input: tuple, weight: dict, selected_idx: int = None):
        # 通过迭代`_ops`中的操作，将相应权重乘以操作应用于`input`的输出，并将它们相加，得到所有候选操作的加权输出之和。
        if selected_idx is None or selected_idx == -1:
            weight = weight.softmax(0)  # 对变量 weight 进行 softmax 归一化
            return sum( weight[i] * self.candidates[name](input) for i, name in enumerate(self._ops) )
        # 直接将选定的操作应用于`input`。它使用`selected_idx`从`_ops`中查找选定的操作，并从`candidates`字典中获取相应的操作。然后，它将选定的操作应用于`input`并返回结果。
        else:
            selected_op = self._ops[selected_idx]
            return self.candidates[selected_op](input)
            # candidates字典   返回操作名称（如V-Max）的对应公式结果
        #当所有被选操作均已索引完，计算该组合的加权求和值（其中权值已做归一化处理）


# if __name__ == "__main__":
#     mixed = Mixed()

#     # 想办法在这里直接传入数据
#     # 然后调试
#     # output = mixed()
#     #  output  = func((G, input_v, input_e), arch_para[weight_order], selected)
#     # 








# 加权求和，激活，拼接、归一化等转换得到最终输出值，以此类推，对cell中每个状态顶点进行更新
class Cell(nn.Module):

    # 获取每个cell中的cell结构字典、cell的状态顶点数量、处理顶点特征和边特征输入和输出的转换操作、激活函数、架构加载函数load_arch
    def __init__(self, args, cell_arch, last = False):
        super().__init__()  # 初始化父类
        self.args      = args   # 参数
        self.cell_arch = cell_arch  # 一个表示cell结构的字典
        self.nb_nodes  = max(cell_arch['V'].keys()) - 1  # 计算cell的状态顶点数量（不包括输入顶点）
        total_nodes    = self.nb_nodes + 2   # 表示cell中顶点的总数（包括输入顶点）
        # 分别用于处理节点特征和边特征的转换
        self.trans_output_V = nn.Sequential( MLP((total_nodes*args.ds.node_dim, args.ds.inter_channel_V), "leakyrelu", "batch", False) )
        self.trans_output_E = nn.Sequential( MLP((total_nodes*args.ds.edge_dim, args.ds.inter_channel_E), "leakyrelu", "batch", False) )
        # 添加一个线性层`fin_V`和`fin_E`到`trans_output_V`和`trans_output_E`的末尾，用于最终的特征转换
        if not last:
            self.trans_output_V.add_module( "fin_V", nn.Linear(args.ds.inter_channel_V, args.ds.node_dim, bias=False) )
            self.trans_output_E.add_module( "fin_E", nn.Linear(args.ds.inter_channel_E, args.ds.edge_dim, bias=False) )
        self.activate = nn.LeakyReLU(negative_slope=0.2)

        # 另一种实现方式（手动） 包括了使用线性层和批归一化层对节点特征和边特征进行转换的步骤
        # self.trans_concat_V = nn.Linear(self.nb_nodes * args.ds.node_dim, args.ds.node_dim, bias = True)
        # self.trans_concat_E = nn.Linear(self.nb_nodes * args.ds.edge_dim, args.ds.edge_dim, bias = True)
        # self.batchnorm_V    = nn.BatchNorm1d(args.ds.node_dim)
        # self.batchnorm_E    = nn.BatchNorm1d(args.ds.edge_dim)
        # self.activate       = nn.LeakyReLU(negative_slope=0.2)

        self.load_arch() 

    #  加载cell结构各信息（(Si, Vj, Ek)，每条入边对应的加权求和值），方便地访问和管理cell结构中的连接参数
    def load_arch(self):
        for type in ['V', 'E']: 
            link_para = {}  # 用于存储连接的参数 
            # 键是顶点的索引序号（Si），值是一个列表，表示该顶点的入边（incomings）
            # 循环遍历两个更新图上cell中的各个状态顶点，及其入边
            for Si, incomings in self.cell_arch[type].items():
                # 循环遍历每条入边
                for edge in incomings:
                    # 获取与入边相关的信息（入边的源顶点（Vj）、入边索引（Ek）和该入边是否被选择（selected））。
                    Vj, Ek, selected = edge['Vj'], edge['Ek'], edge['selected']
                    # 通过Mixed类计算每条入边可能进行的（所有操作组合）加权求和值，并将该输入边的参数存储在`link_para`字典中。
                    #  link_para字典的键是元组（Si, Vj, Ek）的字符串表示，值是（该状态顶点）该入边对应的加权求和值。
                    link_para[str((Si, Vj, Ek))] = Mixed(self.args, type)
            # 将`link_para`字典作为`nn.ModuleDict`对象的属性添加到`self`中，属性名称是根据类型动态生成的。
            setattr(self, f'link_para_{type}', nn.ModuleDict(link_para))  


    def forward(self, input):

       # 以下几个由input字典导入的键值均为固定数值
        V0, V1, E0, E1 = input['V0'], input['V1'], input['E0'], input['E1']  # 导入几个初始值，顶点V0，V1和边E0、E1，并初始化
        G, arch_para, cell_topo = input['G'], input['arch_para'], input['cell_topo'] # 定义图、架构参数和cell拓扑结构信息，并初始化
        Vin, Ein = V1, E1  # 以V1、E1作为cell的输入值

        states = {
            'V': [V0, V1], 
            'E': [E0, E1],
        }   # 初始化两个更新图cell的加权和值
        #V0、V1应该是数值而不是变量，在这两个键中对应两个列表值，后续在某个键中添加的值将被纳入该列表内，列表中的每个个值代表cell中每个状态顶点的加权求和值）
 
        # 根据入边的各个信息，调用不同的函数来更新状态变量states中的元素。
        # 循环遍历cell中的各个状态顶点Si（不包括两个输入），并遍历各状态顶点的对应的入边，对每个输入边进行处理，并将处理结果存储在states中的相应位置。
        for Si in range(2, self.nb_nodes + 2): 
            for type in ['V', 'E']: 
                tmp_state = []  
                incomings = cell_topo[type][Si] # 每个更新图的每个状态顶点又有不同的入边
                #循环遍历状态顶点的每个入边
                for edge in incomings:
                    Vj, Ek, weight_order, selected = edge['Vj'], edge['Ek'], edge['weight_order'], edge['selected']
                    input_v = states['V'][Vj]  # 当前状态节点的入边的源顶点信息
                    input_e = states['E'][Ek]  # 当前状态节点的入边信息
                    # 使用getattr函数根据type获取对应的函数func（源自Mixed类方法，计算个输入边的加权求和组合各个操作，得到各架构参数，）
                    func    = getattr(self, f'link_para_{type}')[str((Si, Vj, Ek))]
                    # 基于相应的函数更新该顶点状态，得到该状态顶点在该输入边的输出
                    output  = func((G, input_v, input_e), arch_para[weight_order], selected)
                    tmp_state.append(output)   # 依次将该状态顶点的每个入边加权求和后的输出值添加到tmp_state列表中
                states[type].append(sum(tmp_state))  # 将各个状态顶点对应的所有入边加权求和后的结果相加，存放到states字典对应键值（列表）中
                # print("func set<<<<<<<<<<<<")
                # with open("./cell_vis.txt","a+") as f:
                #     f.write(str(states) + "\n")
                # print("func out>>>>>>>>>>>")
                # print("func set<<<<<<<<<<<<")
                # print(states['V'].shape)
                # print("func out>>>>>>>>>>>")

        # 在两更新图上，将各个状态顶点所得的加权求和值进行激活，再在维度为1上进行拼接、转换为张量
                
        # 对`states['V']`和`states['E']`中的元素应用函数`self.activate()`进行转换，然后将转换后的结果存储回原来的位置`states[type]`。
        for type in ['V', 'E']: 
            states[type] = [self.activate(f) for f in states[type]]
        # 将`states['V']`和`states['E']`中的元素通过拼接转换为张量，然后将拼接后的张量传递给模型中的相应函数进行进一步处理，得到转换后的张量`V`和`E`。
        V = self.trans_output_V(torch.cat(states['V'], dim = 1))
        E = self.trans_output_E(torch.cat(states['E'], dim = 1))

        # V = self.trans_concat_V(torch.cat(states['V'][2:], dim = 1))
        # E = self.trans_concat_E(torch.cat(states['E'][2:], dim = 1))
        # V = self.batchnorm_V(V)  归一化
        # E = self.batchnorm_E(E)   
        # V = self.activate(V)    激活
        # E = self.activate(E)
        # V = F.dropout(V, self.args.ds.dropout, training = self.training)   
        # E = F.dropout(E, self.args.ds.dropout, training = self.training)
        # V = V + Vin    
        # E = E + Ein
        output = {**input}  #将输入字典`input`中的键值对复制到`output`中
        output.update({'V0': V1, 'V1': V, 'E0': E1, 'E1': E})  #这里的update 不懂  
        return output



