import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import copy

# 一个用于在 PyTorch 中连接多个张量的函数。
# 把x先拉成一行，然后把所有的x摞起来，变成n行
def _concat(xs): 
    return torch.cat([x.view(-1) for x in xs])


class Architect(object): 

    def __init__(self, model, args): 
        self.args                 = args
        self.network_momentum     = args.optimizer.momentum
        self.network_weight_decay = args.optimizer.weight_decay
        self.model                = model
        self.optimizer            = torch.optim.Adam(
            params       = self.model.arch_parameters(),
            lr           = args.optimizer.arch_lr,
            betas        = (0.5, 0.999),
            weight_decay = args.optimizer.arch_weight_decay
        )

        """
    我们更新梯度就是theta = theta + v + weight_decay * theta 
      1.theta就是我们要更新的参数
      2.weight_decay*theta为正则化项用来防止过拟合
      3.v的值我们分带momentum和不带momentum：
        普通的梯度下降：v = -dtheta * lr 其中lr是学习率，dx是目标函数对x的一阶导数
        带momentum的梯度下降：v = lr*(-dtheta + v * momentum)
    """

    # 对应公式6第一项的w − ξ*dwLtrain(w, α)，更新w
    # 不直接用外面的optimizer来进行w的更新，而是自己新建一个unrolled_model展开
    def _compute_unrolled_model(self, input, target, eta, network_optimizer): 
        loss  = self.model._loss(input, target)  # Ltrain
        theta = _concat(self.model.parameters()).data  # 把参数整理成一行代表一个参数的形式,得到我们要更新的参数theta
        try: 
            # momentum*v,  用的就是Network进行w更新的momentum
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except: 
            moment = torch.zeros_like(theta)  #  不加momentum
            # 前一部分是loss对参数theta求梯度，self.network_weight_decay*theta就是正则项
            dtheta = _concat(torch.autograd.grad(
                            outputs      = loss,
                            inputs       = self.model.parameters())
                        ).data + self.network_weight_decay*theta   
            # 对参数进行更新，等价于optimizer.step()
            unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta)) 
        return unrolled_model 

    # 执行一步训练或优化过程。根据参数 unrolled 的取值，选择不同的反向传播计算方法，并通过优化器更新模型的参数。
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch, unrolled,entropy_reg=None): 
        self.optimizer.zero_grad()   #清除上一步的残余更新参数值
        if unrolled:
            # 用论文的提出的方法
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer,entropy_reg=entropy_reg)
        else:   #不用论文提出的双层优化，只是简单的对α求导
            # if self.args.search_mode == 'darts_1':
            #     self._backward_step(input_valid, target_valid)
            # elif self.args.search_mode == 'train':
            #     self._backward_step(input_train, target_train)
            
            # 调用 _backward_step 方法执行反向传播计算
            self._backward_step(input_train, target_train, epoch, entropy_reg=entropy_reg)
        for weights in self.model.arch_parameters():
            # 如果某个架构参数不需要进行梯度计算，则将其梯度置为 None
            if not weights.requires_grad:
                weights.grad = None
        self.optimizer.step()  #应用梯度：根据反向传播得到的梯度进行参数的更新， 这些参数的梯度是由loss.backward()得到的，optimizer存了这些参数的指针
                               #因为这个optimizer是针对alpha的优化器，所以他存的都是alpha的参数。

    # 用于在标准训练过程中执行反向传播步骤
    def _backward_step(self, input_valid, target_valid, epoch, entropy_reg=None): 
        #here new
        weights = 0 + 50*epoch/100
        # print(f"打印model的内部方法：{dir(self.model.arch_parameters)}")
        ssr_normal = self.mlc_loss(self.model.arch_parameters())
        #new here
        loss = self.model._loss(input_valid, target_valid, entropy_reg=entropy_reg) + weights*ssr_normal
        # loss = self.model._loss(input_valid, target_valid) + weights*ssr_normal
        # loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    # 更新α，计算DARTS公式六：dαLval(w',α) ，其中w' = w − ξ*dwLtrain(w, α)
    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer,entropy_reg=None): 
        # w'，更新w
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)
        # Lval
        # new here
        # unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid,entropy_reg=entropy_reg)

        unrolled_loss.backward()
        # dαLval(w',α)
        dalpha         = [v.grad for v in unrolled_model.arch_parameters()]   #架构参数的梯度
        # dw'Lval(w',α)
        vector         = [v.grad.data for v in unrolled_model.parameters()]   #模型参数的梯度向量 
        #计算DARTS公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)  #隐式梯度

        # 公式六减公式八 dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
        # 通过减去学习率缩放后的隐式梯度来 更新架构参数的梯度值
        for g, ig in zip(dalpha, implicit_grads): 
            g.data.sub_(eta, ig.data)

        # 对α进行更新
        # 将更新后的梯度值赋给模型的架构参数的 grad 属性
        for v, g in zip(self.model.arch_parameters(), dalpha): 
            if  v.grad is None: 
                v.grad = Variable(g.data)
            else: 
                v.grad.data.copy_(g.data)

    # 对应optimizer.step()，对新建的模型的参数进行更新
    def _construct_model_from_theta(self, theta): 
        model_new  = self.model.new() # 创建与原模型相同结构的新模型
        model_dict = self.model.state_dict()  # 获取原模型的参数字典

        params, offset = {}, 0 # 用于存储构建出的新模型的参数 
        # 遍历原模型的每个参数 v 和其对应的名称 k
        for k, v in self.model.named_parameters(): 
            v_length  = np.prod(v.size())   # 计算参数 v 的元素个数 v_length，即该参数的形状大小的乘积
            params[k] = theta[offset: offset+v_length].view(v.size())  # 从参数向量 theta 中提取对应参数的部分，并将其视图（view）显示为与参数 v 相同的形状
            offset   += v_length # 更新偏移量 offset

        assert offset == len(theta)  # 确保最终的偏移量 offset 等于参数向量 theta 的长度，以确保所有参数都被正确处理。
        model_dict.update(params)    # 将构建出的新模型的参数更新到 model_dict 中。
        model_new.load_state_dict(model_dict)  # 将更新后的参数字典加载到新模型 model_new 中。
        if not model_new.args.disable_cuda: 
            model_new.cuda = model_new.cuda()  # 将新模型移动到 CUDA 设备上。
        return model_new # 返回构建出的新模型 
    
    # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon   w- = w-dw'Lval(w',α)*epsilon
    def _hessian_vector_product(self, vector, input, target, r=1e-2):  #  vector就是dw'Lval(w',α)
        R = r / _concat(vector).norm()  # epsilon
        # dαLtrain(w+,α)
        for p, v in zip(self.model.parameters(), vector):   
            p.data.add_(R, v)            # 将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
        loss    = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())  

        # dαLtrain(w-,α)
        for p, v in zip(self.model.parameters(), vector): 
            p.data.sub_(2*R, v)    # 将模型中所有的w'更新成w-=w-dw'Lval(w',α)*epsilon
        loss    = self.model._loss(input, target)   
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())   #计算损失对模型的架构参数的梯度。（恢复参数值后计算得到的）

        # 将模型的参数从w-恢复成w
        for p, v in zip(self.model.parameters(), vector): 
            p.data.add_(R, v)   #   #w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]  


# #here new
    # def mlc_loss(self, arch_param):

    #     # 1. 长度统一
    #     max_length = max(t.size(0) for t in arch_param)
    #     padded_tensors = [F.pad(t, (0, max_length - t.size(0))) for t in arch_param]

    #     # 2. 拼接张量
    #     y_pred_neg = torch.stack(padded_tensors)

    #     neg_loss = torch.logsumexp(y_pred_neg, dim=0)
    #     aux_loss = torch.mean(neg_loss)
    #     # print(f"nef loss = {neg_loss}")
    #     return aux_loss
        
    def mlc_loss(self, arch_param):
         # 1. 长度统一
        max_length = max(t.size(0) for t in arch_param)
        padded_tensors = [F.pad(t, (0, max_length - t.size(0))) for t in arch_param]

          # 2. 拼接张量
        y_pred_neg = torch.stack(padded_tensors)

        neg_loss = torch.log(torch.exp(y_pred_neg))
        aux_loss = torch.mean(neg_loss)
        return aux_loss
    
    