import math
from abc import ABC
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from sample_factory.algo.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.model_utils import ModelModule, create_mlp, nonlinearity
from sample_factory.utils.typing import Config


class Critic(ModelModule, ABC):
    pass


class MlpCritic(Critic):
    def __init__(self, cfg: Config, critic_input_size: int):
        super().__init__(cfg)
        self.critic_input_size = critic_input_size
        self.critic_out_size = 1
        critic_layers: List[int] = cfg.critic_mlp_layers
        activation = nonlinearity(cfg)
        self.mlp = create_mlp(critic_layers, critic_input_size, activation)
        if len(critic_layers) > 0:
            self.mlp = torch.jit.script(self.mlp)

        mlp_out_size = calc_num_elements(self.mlp, (critic_input_size,))
        self.critic_linear = nn.Linear(mlp_out_size, self.critic_out_size)

    def forward(self, core_output):
        return self.critic_linear(self.mlp(core_output))


class ValueParameterizationContinuousNonAdaptiveStddev(nn.Module):
    """Use a single learned parameter for action stddevs."""

    def __init__(self, cfg, core_out_size):
        super().__init__()
        self.cfg = cfg

        # calculate only value means using the critic neural network
        self.distribution_linear = nn.Linear(core_out_size, 1)
        # stddev is a single learned parameter
        initial_stddev = torch.empty([1])
        initial_stddev.fill_(math.log(self.cfg.initial_stddev))
        self.learned_stddev = nn.Parameter(initial_stddev, requires_grad=True)

    def forward(self, actor_core_output: Tensor):
        value_means = self.distribution_linear(actor_core_output)
        batch_size = value_means.shape[0]
        value_stddevs = self.learned_stddev.repeat(batch_size, 1)
        value_distribution_params = torch.cat((value_means, value_stddevs), dim=1)
        value_distribution = ContinuousActionDistribution(params=value_distribution_params)
        return value_distribution_params, value_distribution


class ParametrizedCritic(Critic):
    def __init__(self, cfg: Config, critic_input_size: int):
        super().__init__(cfg)
        self.critic_input_size = critic_input_size
        self.critic_out_size = 2
        critic_layers: List[int] = cfg.critic_mlp_layers
        activation = nonlinearity(cfg)
        self.mlp = create_mlp(critic_layers, critic_input_size, activation)
        if len(critic_layers) > 0:
            self.mlp = torch.jit.script(self.mlp)

        mlp_out_size = calc_num_elements(self.mlp, (critic_input_size,))
        self.critic_parametrization = ValueParameterizationContinuousNonAdaptiveStddev(cfg, mlp_out_size)

    def forward(self, core_output):
        value_distribution_params, self.last_value_distribution = self.critic_parametrization(self.mlp(core_output))
        values = self.last_value_distribution.sample()
        return values


def default_make_critic_func(cfg: Config, critic_input_size: int) -> Critic:
    if cfg.critic_deterministic:
        return MlpCritic(cfg, critic_input_size)
    else:
        return ParametrizedCritic(cfg, critic_input_size)
