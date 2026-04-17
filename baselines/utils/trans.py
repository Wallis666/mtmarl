"""张量变换工具函数。"""

def _t2n(value):
    """将torch.Tensor转换为numpy.ndarray。"""
    return value.detach().cpu().numpy()


def _flatten(
    T,
    N,
    value,
):
    """将张量的前两个维度展平。"""
    return value.reshape(T * N, *value.shape[2:])


def _sa_cast(value):
    """用于缓冲区数据操作。

    将张量从 (episode_length, n_rollout_threads, *dim)
    转置为 (n_rollout_threads, episode_length, *dim)，
    然后将前两个维度合并为一个维度。
    """
    return value.transpose(1, 0, 2).reshape(
        -1, *value.shape[2:]
    )


def _ma_cast(value):
    """用于缓冲区数据操作。

    将张量从
    (episode_length, n_rollout_threads, num_agents, *dim)
    转置为
    (n_rollout_threads, num_agents, episode_length, *dim)，
    然后将前三个维度合并为一个维度。
    """
    return value.transpose(1, 2, 0, 3).reshape(
        -1, *value.shape[3:]
    )
