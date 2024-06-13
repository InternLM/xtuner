import torch


class RunningStates:
    # adopt from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py  # noqa: E501
    def __init__(self, epsilon: float = 1e-4):
        self.mean = torch.tensor(0, dtype=torch.float32)
        self.var = torch.tensor(0, dtype=torch.float32)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        x_var, x_mean = torch.var_mean(x.cpu(), unbiased=False)
        x_count = x.shape[0]
        self.update_from_moments(x_mean, x_var, x_count)

    def update_from_other(self, other: 'RunningStates'):
        self.update_from_moments(other.mean, other.var, other.count)

    def update_from_moments(self, mean: torch.Tensor, var: torch.Tensor,
                            count: int):
        delta = mean - self.mean
        tot_count = self.count + count
        m_a = self.var * self.count
        m_b = var * count
        m_2 = m_a + m_b + delta**2 * self.count * count / (self.count + count)
        new_var = m_2 / (self.count + count)

        self.mean += delta * count / tot_count
        self.var = new_var
        self.count = tot_count

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load_state_dict(self, states):
        self.mean = states['mean']
        self.var = states['var']
        self.count = states['count']
