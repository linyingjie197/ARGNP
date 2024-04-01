import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# 一个用于在 PyTorch 中连接多个张量的函数。
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

    # 计算在当前模型参数基础上，根据输入和目标计算损失，并根据网络优化器的动量信息以及学习率进行参数更新，最后构建一个新的模型并返回。
    def _compute_unrolled_model(self, input, target, eta, network_optimizer): 
        loss  = self.model._loss(input, target)  # 计算了输入 input 在模型 self.model 上的损失值。这里假设模型有一个名为 _loss 的方法用于计算损失。
        theta = _concat(self.model.parameters()).data  #  通过 _concat 函数将模型的参数拼接成一个一维张量，并获取其数据部分（即参数的值）。这里假设 _concat 是一个函数，用于将多个张量拼接在一起。
        try: 
            # 遍历模型的参数，并通过 network_optimizer 获取每个参数的动量缓存。这里假设network_optimizer是一个网络优化器对象，它存储了参数的状态信息。
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except: 
            moment = torch.zeros_like(theta)  # 如果无法获取到动量缓存，则使用全零张量作为动量。
            # 通过计算损失对模型参数的梯度，并将梯度张量拼接成一维张量，然后加上权重衰减项乘以参数 theta，得到了参数变化量 dtheta
            dtheta = _concat(torch.autograd.grad(
                            outputs      = loss,
                            inputs       = self.model.parameters())
                        ).data + self.network_weight_decay*theta   
            unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta)) #根据更新后的参数构建一个新的模型 unrolled_model
        return unrolled_model 

    # 执行一步训练或优化过程。根据参数 unrolled 的取值，选择不同的反向传播计算方法，并通过优化器更新模型的参数。
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch, unrolled): 
        self.optimizer.zero_grad()
        if unrolled:
            # 调用 _backward_step_unrolled 方法执行反向传播计算。（这个方法可能是用于执行unrolled训练过程的自定义实现，它会在训练和验证数据上进行一次前向传播和反向传播，并更新模型的参数。） 
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else: 
            # if self.args.search_mode == 'darts_1':
            #     self._backward_step(input_valid, target_valid)
            # elif self.args.search_mode == 'train':
            #     self._backward_step(input_train, target_train)
            
            # 调用 _backward_step 方法执行反向传播计算
            self._backward_step(input_train, target_train, epoch)
        for weights in self.model.arch_parameters():
            # 如果某个架构参数不需要进行梯度计算，则将其梯度置为 None
            if not weights.requires_grad:
                weights.grad = None
        self.optimizer.step()  #调用优化器的 step 方法，根据参数的梯度更新参数值，完成一步优化。

    # 用于在标准训练过程中执行反向传播步骤
    def _backward_step(self, input_valid, target_valid, epoch): 
        #here new
        weights = 0 + 50*epoch/100
        # print(f"打印model的内部方法：{dir(self.model.arch_parameters)}")
        ssr_normal = self.mlc_loss(self.model.arch_parameters())
        loss = self.model._loss(input_valid, target_valid) + weights*ssr_normal
        # loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    # 用于在unrolled模型上执行反向传播步骤
    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer): 
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha         = [v.grad for v in unrolled_model.arch_parameters()]   #架构参数的梯度
        vector         = [v.grad.data for v in unrolled_model.parameters()]   #模型参数的梯度向量 
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)  #隐式梯度

        # 通过减去学习率缩放后的隐式梯度来 更新架构参数的梯度值
        for g, ig in zip(dalpha, implicit_grads): 
            g.data.sub_(eta, ig.data)

        # 将更新后的梯度值赋给模型的架构参数的 grad 属性
        for v, g in zip(self.model.arch_parameters(), dalpha): 
            if  v.grad is None: 
                v.grad = Variable(g.data)
            else: 
                v.grad.data.copy_(g.data)

    # 用于根据给定的参数向量 theta 构建一个新的模型
    # 实现了根据给定参数向量构建新模型的功能，可以用于在某些情况下，根据优化得到的参数向量构建一个新的模型来进行后续的操作。
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
    
    # 计算向量 vector 关于输入 input 和目标 target 的 Hessian-向量积
    # 使用了两次损失函数的梯度计算，并根据梯度差异和缩放因子进行了计算和归一化。
    def _hessian_vector_product(self, vector, input, target, r=1e-2): 
        R = r / _concat(vector).norm()  # 计算缩放因子 
        for p, v in zip(self.model.parameters(), vector):   
            p.data.add_(R, v)           # 对模型的每个参数 p 和向量 vector 进行逐元素的相加并按比例缩放，使用缩放因子 R。这将改变模型的参数值。
        loss    = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())  #计算损失对模型的架构参数的梯度（返回一个梯度张量的元组，每个元素对应一个架构参数的梯度）

        for p, v in zip(self.model.parameters(), vector): 
            p.data.sub_(2*R, v)    # 对模型的每个参数 p 和向量 vector 进行逐元素的相减并按比例缩放，使用缩放因子 2*R。这将恢复模型的参数值为原始值。
        loss    = self.model._loss(input, target)   
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())   #计算损失对模型的架构参数的梯度。（恢复参数值后计算得到的）

        for p, v in zip(self.model.parameters(), vector): 
            p.data.add_(R, v)   #  再次对模型的每个参数 p 和向量 vector 进行逐元素的相加并按比例缩放，使用缩放因子 R。这将恢复模型的参数值为原始值。

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]  #计算梯度差异，并将其除以两倍的缩放因子 2*R。返回的是一个梯度差异的列表，每个元素对应一个架构参数的梯度差异。


# #here new
    def mlc_loss(self, arch_param):

        # 1. 长度统一
        max_length = max(t.size(0) for t in arch_param)
        padded_tensors = [F.pad(t, (0, max_length - t.size(0))) for t in arch_param]

        # 2. 拼接张量
        y_pred_neg = torch.stack(padded_tensors)

        neg_loss = torch.logsumexp(y_pred_neg, dim=0)
        aux_loss = torch.mean(neg_loss)
        # print(f"nef loss = {neg_loss}")
        return aux_loss
        
    # def mlc_loss(self, arch_param):
    #     y_pred_neg = arch_param
    #     neg_loss = torch.log(torch.exp(y_pred_neg))
    #     aux_loss = torch.mean(neg_loss)
    #     return aux_loss
    
    