import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from collections import namedtuple
from models.cell_search import Cell
from models.operations import V_OPS, E_OPS, get_OPS
from models.networks import MLP
from data import TransInput, TransOutput, get_trans_input
from hydra.utils import get_original_cwd, to_absolute_path

# 定义了一个名为Genotype的命名元组子类，具有两个字段：V和E。
Genotype = namedtuple('Genotype', 'V E')

# 确定动态拓扑结构、各个状态顶点间的权重配置。
def expand_genotype(src_genotypes):
    
    nb_layers = len(src_genotypes)
    nb_nodes  = max( x[0] for x in src_genotypes[0].V) - 1   #cell中的顶点总数
    topology = []   #用于存储网络的拓扑结构
    weight_config = []   #用于存储网络的权重配置
    weight_order = 0    #用于表示权重序列

    # 循环遍历每层cell，在每一层中将状态顶点信息、其相连的前一步状态顶点信息、边信息、操作存储到两个更新图对应的字典中
    for cell_id in range(nb_layers):
        src_type_dict = {'V': {}, 'E': {}}  #创建了一个字典来存储网络各层中cell的顶点信息和关系信息。
        for type in ['V', 'E']:
            src_type = getattr(src_genotypes[cell_id],type) # getattr 用于返回实例化类的变量值
            for Si, Vj, Ek, Op in src_type:
                if Si not in src_type_dict[type]:
                    src_type_dict[type][Si] = []
                src_type_dict[type][Si].append((Vj, Ek, Op))   #将每一层cell中的状态顶点Si的各个信息（其之前相连的状态顶点、边和操作）存储到src_type_dict字典中
       
        # 用于生成一个表示网络拓扑结构的列表topology和一个权重配置列表weight_config。
        # 它遍历每个顶点（顶点和关系），根据src_type_dict中存储的顶点信息和关系信息，构建顶点之间的连接关系，并将连接关系存储在cell_topo字典中。
        cell_topo = {'V': {}, 'E': {}}

        #遍历cell上的每个状态顶点，搭建拓扑结构（每循环一次，在前一次的基础上拓扑结构扩大了一倍））
        for Si in range(2, nb_nodes + 2): 
            for type in ['V', 'E']:
                ops = get_OPS(type)
                cell_topo[type][(Si-1)*2] = [] # 右节点
                cell_topo[type][(Si-1)*2+1] = []  #左节点
                for Vj, Ek, Op in src_type_dict[type][Si]:
                    # 拓扑结构扩展过程（cell基于上一层cell进行扩大）
                    # 扩大过程中，对于偶数序号（奇数序号）的状态顶点做繁殖，有不同的处理方式
                    # Si处于偶数序列时，繁殖未结束，直接确定Si状态节点的操作，并将其添加到cell_topo字典即可
                    nSi = (Si-1) * 2  
                    nVj = Vj if Vj < 2 else (Vj - 1) * 2 + 1 
                    nEk = Ek if Ek < 2 else (Ek - 1) * 2 + 1
                    edge = {
                        'Vj': nVj,
                        'Ek': nEk,
                        'weight_order': -1, 
                        'selected': ops.index(Op), 
                    }
                    cell_topo[type][nSi].append(edge)   
                    
                    # Si处于奇数序列时，繁殖结束，由于后续需要建立局部超网，故无法直接确定出此状态顶点的操作，所以'selected'为-1
                    # 此处建立的拓扑结构不完整，仅连接了当前状态顶点与原顶点之前的各个顶点和边，未建立于原顶点的拓扑结构
                    # 为方便后续操作，建立了列表weight_config，逐步配置每个状态顶点的权重序列，将每个状态节点的操作集长度添加到weight_config列表
                    nSi = (Si-1)*2 + 1
                    nVj = Vj if Vj < 2 else (Vj - 1) * 2 + 1 
                    nEk = Ek if Ek < 2 else (Ek - 1) * 2 + 1
                    edge = {
                        'Vj': nVj, 
                        'Ek': nEk, 
                        'weight_order': weight_order, 
                        'selected': -1,
                    }
                    weight_order += 1
                    weight_config.append(len(ops))
                    cell_topo[type][nSi].append(edge)  
                
                # 此处建立的当前状态顶点与原顶点之间的拓扑结构，
                # 且由于后需要建立局部超网，故'selected'也始终为-1（不确定）
                # 同理 为方便后续操作，建立了列表weight_config，逐步配置每个状态顶点的权重序列，将每个状态节点的操作集长度添加到weight_config列表
                nSi = (Si-1)*2 + 1
                nVj = nEk = (Si-1)*2
                edge = {
                    'Vj': nVj,
                    'Ek': nEk,
                    'weight_order': weight_order,
                    'selected': -1,
                }
                weight_order += 1
                weight_config.append(len(ops))
                cell_topo[type][nSi].append(edge)  
        
        # topology列表中存储的是，各个cell（每繁殖一次为一个cell）中的拓扑结构信息（包括每个cell中每个状态节点Si的Vj, Ek,weight_order,selected）
        topology.append(cell_topo)  
    return topology, weight_config


# if __name__ == '__main__':
#     with open("archs/new_sgas/ZINC/50/cell_geno.txt", "r") as f:
#         src_genotypes = eval(f.read())
#     r1, r2 = expand_genotype(src_genotypes)
#     print(r1)
#     print(r2)
#     import ipdb; ipdb.set_trace()

# 获取纯结构的拓扑字典  构建了一个具有多个层的神经网络单元的拓扑和权重配置
def load_arch_basic(nb_nodes, nb_layers):
    #! 获取纯结构的拓扑字典
    topology = []  # 存储cell拓扑信息的列表
    weight_config = [] # 用于存储权重配置。
    weight_order = 0
    for cell_id in range(nb_layers):
        cell_topo = {}

        for type in ['V', 'E']: 
            type_topo = {}  #用于存储该类型（V,E）的拓扑
            ops = get_OPS(type)

            for Si in range(2, nb_nodes + 2):
                incomings = []  # 用于存储当前Si状态顶点的入边。
                # 循环遍历当前状态顶点Si每个入边的源节点Vj。
                for Vj in range(Si):  
                    edge = {
                        'Vj': Vj, 
                        'Ek': Vj,   # Si入边的信息设为其源顶点Vj的信息
                        'weight_order': weight_order,
                        'selected': -1, 
                    }
                    weight_config.append(len(ops))  # 将ops的长度添加到weight_config列表中
                    incomings.append(edge)  # 将edge字典添加到incomings列表中
                    weight_order += 1 
                type_topo[Si] = incomings  #将每个Si入边信息字典中的键值对赋值给对应的type_topo[Si]
            
            cell_topo[type] = type_topo  #将两个更新图（V E）拓扑信息字典中的键值对分别赋值给对应的cell_topo[type]中
        
        topology.append(cell_topo)   # 将每层cell中（两更新图的）（每个状态顶点的）（每个入边的）拓扑信息字典放入topology列表中
    return topology, weight_config


class Model_Search(nn.Module):

    # 根据已有数据（顶点数量、层数）和之前定义的这两个函数，获取到拓扑结构列表和权重配置列表
    # 此外，获取架构参数（根据后面的函数获取）、cell架构（更新）、损失函数、输入输出数据转换的函数和操作、位置编码
    def __init__(self, args, trans_input_fn, loss_fn):
        super().__init__()
        self.args = args
        self.nb_layers = args.basic.nb_layers
        # 1 根据给定的节点数量和层数，生成一个基本的拓扑结构。返回的结果分别赋值给 self.arch_topo 和 para。
        # 2 如果args.ds.expand_from为空字符串，就调用 load_arch_basic 函数生成基本拓扑结构；
        # 3 否则，将根据 args.ds.expand_from 的路径加载已有的拓扑结构和参数。
        if args.ds.expand_from == "":
            self.arch_topo, para = load_arch_basic(args.basic.nb_nodes, self.nb_layers)   
        else:
            path = os.path.join(get_original_cwd(), args.ds.expand_from)
            with open(path, "r") as f: 
                src_genotypes = eval(f.read())  # 从打开的文件对象 f 中读取内容，并使用 eval() 函数将其解析为可执行的 Python 代码，并将结果赋给变量 src_genotypes。
                print(f"src_genotypes = {src_genotypes}")
                self.arch_topo, para = expand_genotype(src_genotypes)   #基于expand_genotype函数，返回架构的拓扑结构及其参数（权重配置 weight_config）

        self.arch_para      = self.init_arch_para(para)   #初始化架构参数，对权重配置列表做一定处理后，得到架构参数列表（arch_para）
        self.cells          = nn.ModuleList([Cell(args, self.arch_topo[i], last = (i==self.nb_layers-1)) for i in range(self.nb_layers)])
        self.loss_fn        = loss_fn
        
        self.trans_input_fn = trans_input_fn
        self.trans_input    = TransInput(trans_input_fn)
        self.trans_output   = TransOutput(args)# 通过初始化这些对象，可以在类的其他方法中使用它们来转换输入和输出数据，以便符合模型的要求或进行预处理操作。
        # 满足if语句，便创建一个线性层，用于对输入数据进行位置编码。
        # 位置编码是一种在序列数据中引入位置信息的技术，常用于自然语言处理任务或序列建模任务
        
        #new here 
        # self.num_edges  = 3
        # search_space     = 9
        # self._arch_parameters = nn.Parameter(1e-3*torch.randn(self.num_edges, search_space) )


        if args.ds.pos_encode > 0:
            self.position_encoding = nn.Linear(args.ds.pos_encode, args.ds.node_dim)

    # 导入变量G, V, E、arch_topo（该变量源自之前定义的两个函数）
    # 更新output（即更新cells中的每个操作组合）
    def forward(self, input):
        # print("开始传播了。。。。。。")
        input = self.trans_input(input)
        G, V, E   = input['G'], input['V'], input['E']
        # 如果输入中没有提供 'arch_topo' 的值或值为 None，则使用默认的 self.arch_topo 值；否则，使用输入中指定的值。
        if "arch_topo" not in input or input['arch_topo'] is None: 
            arch_topo = self.arch_topo
        else: 
            arch_topo = input['arch_topo']
        
        # 在特定条件下对变量 V 进行位置编码操作
        if self.args.ds.pos_encode > 0:
            V = V + self.position_encoding(G.ndata['pos_enc'].float().cuda()) # 表示获取图 G 的节点数据中 'pos_enc' 的值，并将其转换为浮点型并移动到 GPU 上进行计算。然后，通过将位置编码结果与原始的 V 相加，更新了 V 的值。
        
        output = {'G': G, 'V0': V, 'V1': V, 'E0': E, 'E1': E, 'arch_para': self.arch_para}
        # 在output字典中加入新键'cell_topo'，并将其键值进行更新
        for i, cell in enumerate(self.cells):  #此处基于索引遍历self.cells中的模型
            output['cell_topo'] = arch_topo[i]  # 使用索引 i 来更新 output 字典中的 'cell_topo' 键的值，以便与 arch_topo 列表中相应索引位置的值保持一致。

            # call function cell !!!
            # 对cell更新处理
            output = cell(output)   # output变为了Cell类中的input进行模型搭建，以此对output进行更新
        
        output.update({'V': output['V1'], 'E': output['E1']}) # 基于for循环中的私有变量output（Cell类中的），相应获取最终更新
        output = self.trans_output(output)
        return output

    # 初始化架构参数
    def init_arch_para(self, para):
        arch_para = []  # 用于存储初始化后的拓扑结构参数
        # `para`参数是一个列表，其中包含了每个拓扑结构参数的长度。len(ops)
        for plen in para:
            # 1 创建一个长度为 `plen` 的张量，并使用 `torch.rand(plen)` 生成一个取值范围在 0 到 1 之间的随机张量。
            # 2 将生成的随机张量乘以 1e-3，以缩小其范围。
            # 3 使用 `Variable` 函数将随机张量转换为可求导（requires_grad = True）的张量，并将其存储在 GPU 上 
            arch_para.append(Variable(1e-3 * torch.rand(plen).cuda(), requires_grad = True))
        # return arch_para
        return arch_para
     
        
    
        # total = len(para)
        # arch_para = []
        # if 'V' in self.args.optimizer.fix:
        #     requires_grad = False
        #     eps = 1
        # else:
        #     requires_grad = True
        #     eps = 1e-3
        
        # for plen in para[:total//2]:
        #     arch_para.append(Variable(eps * torch.rand(plen).cuda(), requires_grad = requires_grad))

        # if 'E' in self.args.optimizer.fix:
        #     requires_grad = False
        #     eps = 1
        # else:
        #     requires_grad = True
        #     eps = 1e-3
        
        # for plen in para[total//2:]:
        #     arch_para.append(Variable(eps * torch.rand(plen).cuda(), requires_grad = requires_grad))
        # return arch_para

    # 创建一个新的模型对象model_new，并将当前对象的架构参数复制到新模型中，并返回新模型对象。
    def new(self):
        model_new = Model_Search(self.args, get_trans_input(self.args), self.loss_fn).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)  # 将y.data的内容复制到x.data中
        return model_new
    
    # 将 alphas 中的数值拷贝到 self.arch_parameters() 中对应的参数中，实现参数的更新。
    def load_alpha(self, alphas):
        for x, y in zip(self.arch_parameters(), alphas):
            x.data.copy_(y.data)

    # arch_parameters 方法用于获取模型的架构参数
    def arch_parameters(self):
        print("arch args = ",self.arch_para)
        return self.arch_para
            

    # 属于某个类中的成员函数
    def _loss(self, input, targets):
        scores = self.forward(input)  # 计算输入数据 input 的预测分数
        return self.loss_fn(scores, targets)  # 计算预测分数 scores 和目标值 targets 之间的损失值