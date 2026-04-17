"""离散动作空间工具函数模块。"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def onehot_from_logits(logits, eps=0.0):
    """根据 logits 使用 epsilon 贪心策略返回独热编码样本。

    参数:
        logits: 一批 logits
        eps: epsilon 贪心的概率阈值
    返回:
        独热编码的动作
    """
    # 获取当前策略下最优动作的独热编码
    argmax_acs = (
        logits == logits.max(1, keepdim=True)[0]
    ).float()
    if eps == 0.0:
        return argmax_acs
    # 获取随机动作的独热编码
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [
                np.random.choice(
                    range(logits.shape[1]),
                    size=logits.shape[0],
                )
            ]
        ],
        requires_grad=False,
    )
    # 使用 epsilon 贪心策略在最优动作和随机动作间选择
    return torch.stack(
        [
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(
                torch.rand(logits.shape[0])
            )
        ]
    )


def sample_gumbel(
    shape,
    device,
    eps=1e-20,
    tens_type=torch.FloatTensor,
):
    """从 Gumbel(0, 1) 分布中采样。

    参数:
        shape: 采样形状
        device: 计算设备
        eps: 数值稳定性的小常数
        tens_type: 张量类型
    返回:
        Gumbel 分布采样结果
    """
    U = Variable(
        tens_type(*shape).uniform_(),
        requires_grad=False,
    ).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(
    logits,
    temperature,
    device,
):
    """从 Gumbel-Softmax 分布中采样。

    参数:
        logits: 未归一化的对数概率
        temperature: 温度参数
        device: 计算设备
    返回:
        Gumbel-Softmax 采样结果
    """
    y = logits + sample_gumbel(
        logits.shape,
        tens_type=type(logits.data),
        device=device,
    )
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(
    logits,
    device,
    temperature=1.0,
    hard=False,
):
    """从 Gumbel-Softmax 分布中采样，可选离散化。

    参数:
        logits: [batch_size, n_class] 未归一化对数概率
        device: 计算设备
        temperature: 非负标量温度参数
        hard: 若为 True 则取 argmax，但对软采样 y
              进行反向传播
    返回:
        [batch_size, n_class] Gumbel-Softmax 分布
        的采样结果。若 hard=True 则返回独热编码，
        否则返回各类别概率之和为 1 的概率分布
    """
    y = gumbel_softmax_sample(
        logits, temperature, device=device,
    )
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
