"""循环神经网络模块。"""

import torch
import torch.nn as nn

from baselines.utils.model import get_init_method


class RNNLayer(nn.Module):
    """循环神经网络层。"""

    def __init__(
        self,
        inputs_dim,
        outputs_dim,
        recurrent_n,
        initialization_method,
    ):
        super(RNNLayer, self).__init__()
        self.recurrent_n = recurrent_n
        self.initialization_method = initialization_method

        self.rnn = nn.GRU(
            inputs_dim,
            outputs_dim,
            num_layers=self.recurrent_n,
        )
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                init_method = get_init_method(
                    initialization_method,
                )
                init_method(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(
        self,
        x,
        hxs,
        masks,
    ):
        """前向传播。"""
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (
                    hxs
                    * masks.repeat(
                        1,
                        self.recurrent_n,
                    ).unsqueeze(-1)
                )
                .transpose(0, 1)
                .contiguous(),
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x 是一个 (T, N, -1) 的张量，
            # 已被展平为 (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # 还原展平
            x = x.view(T, N, x.size(1))

            # 对掩码做同样的处理
            masks = masks.view(T, N)

            # 找出序列中哪些步骤的掩码为零
            # 始终假设 t=0 处有零以简化逻辑
            has_zeros = (
                (masks[1:] == 0.0)
                .any(dim=-1)
                .nonzero()
                .squeeze()
                .cpu()
            )

            # +1 以修正 masks[1:]
            if has_zeros.dim() == 0:
                # 处理标量情况
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (
                    (has_zeros + 1).numpy().tolist()
                )

            # 将 t=0 和 t=T 添加到列表中
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # 可以将掩码中没有零的步骤一起处理，
                # 这样更快
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (
                    hxs
                    * masks[start_idx]
                    .view(1, -1, 1)
                    .repeat(self.recurrent_n, 1, 1)
                ).contiguous()
                rnn_scores, hxs = self.rnn(
                    x[start_idx:end_idx],
                    temp,
                )
                outputs.append(rnn_scores)

            # x 是一个 (T, N, -1) 的张量
            x = torch.cat(outputs, dim=0)

            # 展平
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs
