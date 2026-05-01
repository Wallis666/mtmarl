"""HARL 工具函数模块。"""

import copy
import math

import torch
import torch.nn as nn


def init_device(args):
    """初始化设备。

    参数:
        args: 参数字典
    返回:
        device: torch 设备对象
    """
    if args["cuda"] and torch.cuda.is_available():
        print("选择使用 GPU...")
        device = torch.device("cuda:0")
        if args["cuda_deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("选择使用 CPU...")
        device = torch.device("cpu")
    torch.set_num_threads(args["torch_threads"])
    return device


def get_active_func(activation_func):
    """获取激活函数。

    参数:
        activation_func: 激活函数名称字符串
    返回:
        对应的激活函数实例
    """
    if activation_func == "sigmoid":
        return nn.Sigmoid()
    elif activation_func == "tanh":
        return nn.Tanh()
    elif activation_func == "relu":
        return nn.ReLU()
    elif activation_func == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_func == "selu":
        return nn.SELU()
    elif activation_func == "hardswish":
        return nn.Hardswish()
    elif activation_func == "identity":
        return nn.Identity()
    else:
        assert False, "不支持的激活函数！"


def get_init_method(initialization_method):
    """获取初始化方法。

    参数:
        initialization_method: 初始化方法名称字符串
    返回:
        对应的初始化方法
    """
    return nn.init.__dict__[initialization_method]


# pylint: disable-next=invalid-name
def huber_loss(e, d):
    """Huber 损失函数。"""
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


# pylint: disable-next=invalid-name
def mse_loss(e):
    """均方误差损失函数。"""
    return e**2 / 2


def update_linear_schedule(
    optimizer,
    epoch,
    total_num_epochs,
    initial_lr,
):
    """线性衰减学习率。

    参数:
        optimizer: 优化器
        epoch: 当前轮次
        total_num_epochs: 总轮次数
        initial_lr: 初始学习率
    """
    learning_rate = initial_lr - (
        initial_lr * ((epoch - 1) / float(total_num_epochs))
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def init(
    module,
    weight_init,
    bias_init,
    gain=1,
):
    """初始化模块权重和偏置。

    参数:
        module: 神经网络模块
        weight_init: 权重初始化方法
        bias_init: 偏置初始化方法
        gain: 增益系数
    返回:
        初始化后的模块
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    """克隆模块 N 次。"""
    return nn.ModuleList(
        [copy.deepcopy(module) for _ in range(N)]
    )


def get_grad_norm(parameters):
    """获取梯度范数。"""
    sum_grad = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        sum_grad += parameter.grad.norm() ** 2
    return math.sqrt(sum_grad)
