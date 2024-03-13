import os
import sys
import dgl 
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from data import *
# from torch.utils.data import DataLoader 
from models.model_search import *
# 肯定要运行search 才能调试 or 自己编写测试函数
from utils.utils import *
from searcher.darts import DARTS
from searcher.sgas import SGAS
# from architect.architect import Architect
from utils.plot_genotype import plot_genotypes
from utils.visualize import *

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import warnings
warnings.filterwarnings('ignore')
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

class Searcher(object): # 等同于 class Searcher :  因为类接括号写法代表继承  所有类都继承于object 所以写了跟不写一样    空了你得看看python的多态等等昂
    def __init__(self, args):

        self.args = args
        self.console = Console()

        #初始化设置
        self.console.log('=> [1] Initial settings')
        np.random.seed(args.basic.seed)
        torch.manual_seed(args.basic.seed) #  设置 CPU 的随机种子
        torch.cuda.manual_seed(args.basic.seed)  # 设置 CUDA（即 GPU）的随机种子，args.basic.seed是一个整数，它指定了随机种子的值。
        cudnn.benchmark = True   #启用cudnn库的自动调优功能
        cudnn.enabled   = True   #表示启用cudnn库加速深度学习模型的计算。

        # 初始化模型
        self.console.log('=> [2] Initial models')
        self.metric    = load_metric(args)   # 表示将参数args用于加载评价指标(metric)，不同的数据集指标不同。
        self.loss_fn   = get_loss_fn(args).cuda() # 将参数args用于获取对应损失函数
        self.model     = Model_Search(args, get_trans_input(args), self.loss_fn).cuda() # 使用参数args获得相应的模型
        # print(self.model) 


        self.console.log(f'[red]=> Supernet Parameters: {count_parameters_in_MB(self.model):.4f} MB') # 计算Supernet的参数量

        #准备数据集
        self.console.log(f'=> [3] Preparing dataset')
        self.dataset     = load_data(args)  #使用参数args加载数据，并赋值给self.dataset变量
        if args.ds.pos_encode > 0:
            self.console.log(f'[red]==> [3.1] Adding positional encodings')
            self.dataset._add_positional_encodings(args.ds.pos_encode)     #给self.dataset的数据添加位置编码，标记数据集中各个数据的位置
        self.search_data = self.dataset.train
        self.val_data    = self.dataset.val
        self.test_data   = self.dataset.test #从获取到的数据集中分配训练、验证和测试数据
        self.load_dataloader()       #加载数据加载器。将数据集转换为数据加载器，方便数据的批量加载和训练
       

        #初始化优化器
        self.console.log(f'=> [4] Initial optimizer')
        #创建了一个SGD优化器的实例，并将其赋值给 self.optimizer 变量。
        self.optimizer   = torch.optim.SGD(              
            params       = self.model.parameters(),   #params 参数指定了需要进行优化的参数集合
            lr           = args.optimizer.lr,         #lr 参数指定了学习率（learning rate），这是 SGD 算法的一个重要超参数，用于控制参数更新的步长。
            momentum     = args.optimizer.momentum,   #momentum 参数指定了动量（momentum），这是 SGD 算法的一个调整项，用于加速梯度下降的过程。
            weight_decay = args.optimizer.weight_decay #weight_decay 参数指定了权重衰减，也称为 L2 正则化，用于控制模型的复杂度，防止过拟合。
        )      

        #设置学习率调度器，可以自动地在训练过程中动态地调整学习率，采用的是余弦退火调度器（避免模型陷入局部最优点）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
            optimizer  = self.optimizer,   #指定了需要调度学习率的优化器对象
            T_max      = float(args.basic.search_epochs),   #指定了一个周期的步数，在这个周期内，学习率会由较大的值线性地下降到较小的值
            eta_min    = args.optimizer.lr_min      #指定了学习率的最小值
        )
        
        #设置搜索策略
        #根据search_mode的值来选择相应的搜索算法或优化器。
        if args.optimizer.search_mode == 'darts':
            SEARCHER = DARTS
        elif args.optimizer.search_mode == 'sgas':
            SEARCHER = SGAS
        else:
            raise Exception("Unknown Search Mode!") # 如果没有对应策略，抛出异常

       
        # 创建了一个SEARCHER的实例，SEARCHER的构造函数会接受一个包含以下参数的字典作为输入
        self.searcher = SEARCHER({
            "args": self.args,    #包含各种参数的对象
            "model_search": self.model,       #模型对象
            "arch_queue": self.arch_queue,    #存储架构描述的队列对象
            "para_queue": self.para_queue,    #存储参数的队列对象
            "loss_fn": self.loss_fn,          #损失函数对象
            "metric": self.metric,            #度量指标对象
            "optimizer": self.optimizer,      #优化器对象
        }) 


    # 定义数据加载器，读取数据并创建数据加载器
    def load_dataloader(self):

        num_search  = int(len(self.search_data) * self.args.basic.data_clip)   # 训练集中部分数据的大小
        indices      = list(range(num_search))                                  # 将这些数据标记为0~num_search的列表 
        split       = int(np.floor(self.args.basic.portion * num_search))      # 拆分数据的大小，floor意为向下取整
        self.console.log(f'=> Para set size: {split}, Arch set size: {num_search - split}') 
        # Arch set size 可能指的是 batch_size，即用于训练的每个小批量数据的大小设置。
        
        # 将数据集划分成两个部分，并分别创建两个数据加载器，以进行不同的训练或测试任务。
        self.para_queue = torch.utils.data.DataLoader( # 这里就是自带的，写法比较臃肿
            dataset     = self.search_data,
            batch_size  = self.args.ds.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), # 从数据集中随机选择训练样本，并仅使用前split个样本
            pin_memory  = True,  # 指定是否将加载的数据存储到固定的内存中。
            num_workers = self.args.basic.nb_workers,  # 指定加载数据的线程数。
            collate_fn  = self.dataset.collate     # 指定如何将样本整理成一个batch的函数
        ) 

        self.arch_queue = torch.utils.data.DataLoader(
            dataset     = self.search_data,
            batch_size  = self.args.ds.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),  #使用后split个样本进行训练。
            pin_memory  = True,
            num_workers = self.args.basic.nb_workers,
            collate_fn  = self.dataset.collate
        )

        # 创建一个验证数据加载器
        if self.val_data is not None:
            num_valid = int(len(self.val_data) * self.args.basic.data_clip)  #验证数据集中的样本数量
            indices   = list(range(num_valid))   #将验证数据集切分为指定数量的样本，创建包含这些样本索引的列表


            self.val_queue  = torch.utils.data.DataLoader(
                dataset     = self.val_data,
                batch_size  = self.args.ds.batch,
                sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),#使用指定数量的样本进行训练
                pin_memory  = True,
                num_workers = self.args.basic.nb_workers,
                collate_fn  = self.dataset.collate
            )
        
        #创建一个测试数据加载器
        num_test = int(len(self.test_data) * self.args.basic.data_clip)
        indices  = list(range(num_test))

        self.test_queue = torch.utils.data.DataLoader(
            dataset     = self.test_data,
            batch_size  = self.args.ds.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            pin_memory  = True,
            num_workers = self.args.basic.nb_workers,
            collate_fn  = self.dataset.collate
        )

    # 在每个epoch中进行搜索和评估，并将结果输出到控制台。
    def run(self):

        self.console.log(f'=> [4] Search & Train')
        for i_epoch in range(self.args.basic.search_epochs):   #训练循环将在每个epoch中运行
            self.scheduler.step()   #更新学习率
            self.lr = self.scheduler.get_lr()[0]   #获取当前学习率，并使用索引0从返回的列表中获取学习率的值
            #检查i_epoch是否是报告频率的倍数。
            print("<<<<<<<<<")
            print(f"epoch = {i_epoch}")
            # print(f"epoch = {i_epoch} , {self.args.visualize.report_freq}")
            if i_epoch % self.args.visualize.report_freq == 0:
                # 获取当前基因型并生成基因型图表。
                # todo report genotype   可能需要根据实际情况修改代码。
                geno = self.searcher.genotypes()   #使用搜索器对象的 genotypes() 方法来获取基因型信息
                print("对应基因型为：----")
                plot_genotypes(self.args, i_epoch, geno)#  将基因型绘制成图表
                print( geno )
                print("----")

            #调用了searcher对象的search方法，并传入一个字典作为参数。search方法将使···用这些参数执行搜索操作
            search_result = self.searcher.search({"lr": self.lr, "epoch": i_epoch})   
            #在控制台上查看每个训练轮次的搜索结果的损失值和指标值。
            self.console.log(f"[green]=> [{i_epoch}] search result - loss: {search_result['loss']:.4f} - metric : {search_result['metric']:.4f}")
            

            with torch.no_grad():   #上下文管理器，它用于在运行推理时禁用梯度计算。
                #通过infer()函数分别计算验证集和测试集的结果并进行输出  
                if self.val_data is not None:
                    val_result    = self.infer(self.val_queue)
                    self.console.log(f"[green]=> [{i_epoch}] valid result  - loss: {val_result['loss']:.4f} - metric : {val_result['metric']:.4f}")

                test_result   = self.infer(self.test_queue)
                self.console.log(f"[underline][red]=> [{i_epoch}] test  result  - loss: {test_result['loss']:.4f} - metric : {test_result['metric']:.4f}")

    #推导损失值与指标值
    def infer(self, dataloader):

        self.model.eval()  #模型设为评估模式
        epoch_loss   = 0   #初始化损失值
        epoch_metric = 0   #初始化指标值
        desc         = '=> inferring'       #用于在进度条中显示推导的进程
        device       = torch.device('cuda') #使用CUDA加速

        # 创建一个进度条t，接收数据加载器作为参数，并设置进度条的描述为desc，禁止进度条显示完毕后关闭。
        with tqdm(dataloader, desc = desc, leave = False) as t:
            #使用一个迭代器（enumerate(t)）循环遍历进度条t，在每次循环中，迭代器会返回一个元组(i_step, (batch_graphs, batch_targets))
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)
                E = batch_graphs.edata['feat'].to(device)  
                batch_targets = batch_targets.to(device)    #将图数据（G，V，E）和目标数据转移到GPU上
                batch_scores  = self.model({'G': G, 'V': V, 'E': E, 'arch_topo': self.searcher.arch_topo})   #推导图数据，计算每一批次的预测分数
                loss          = self.loss_fn(batch_scores, batch_targets)   #计算每批次预测分数与目标数据之间的损失值

                epoch_loss   += loss.detach().item()   # detach()方法用于切断计算图中的梯度流，item()方法用于获取标量值。
                epoch_metric += self.metric(batch_scores, batch_targets)   #通过模型得到的预测分数和目标值来计算指标的方法
                # set_postfix()用于设置进度条的提示信息。这里用来显示每个epoch的平均损失值和指标值
                t.set_postfix(loss   = epoch_loss / (i_step + 1),     
                              metric = epoch_metric / (i_step + 1))

        return {'loss'   : epoch_loss / (i_step + 1), 
                'metric' : epoch_metric / (i_step + 1)}


#使用Hydra库加载配置文件并执行某个应用程序的例子，同时在终端输出配置的内容
@hydra.main(config_path = 'conf', config_name = 'defaults')  #使用装饰器标识为Hydra应用程序的入口点
def app(args):
    OmegaConf.set_struct(args, False)   #将args对象转换为非结构化的配置对象
    #使用Console和Syntax类以及Panel.fit函数创建一个漂亮的终端输出，将args对象转换为YAML格式，并使用monokai主题和行号进行显示。
    console = Console()                 
    vis = Syntax(OmegaConf.to_yaml(args), "yaml", theme="monokai", line_numbers=True)
    richPanel = Panel.fit(vis)
    console.print(richPanel)

    #创建一个文件夹路径
    data_path = os.path.join(get_original_cwd(), args.ds.arch_save, args.ds.data)
    Path(data_path).mkdir(parents = True, exist_ok = True)     
    #使用open函数创建一个名为"configs.txt"的文件，并将args对象的字符串表示写入文件中
    with open(os.path.join(data_path, "configs.txt"), "w") as f:
        f.write(str(args))

    #通过对象args，执行Searcher类中的run方法
    Searcher(args).run()

#调用app()函数来运行整个应用程序。
if __name__ == '__main__':
    app()