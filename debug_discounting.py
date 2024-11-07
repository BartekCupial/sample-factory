from typing import Optional, Union

import torch
from torch import Tensor

buff = torch.load("tmp.pth")
strategy_steps = buff["normalized_obs"]["strategy_steps"].squeeze()
rewards = buff["rewards"]
dones = buff["dones"].float()
values = buff["values"]
valids = buff["valids"].float()


def calculate_conditioned_discounts(
    x: Tensor, dones: Tensor, valids: Tensor, discount: float, x_last: Optional[Tensor] = None
) -> Tensor:
    """
    Computing cumulative product of discounts for the trajectory conditioning on x, taking episode termination into consideration.
    """
    if x_last is None:
        x_last = x[-1].clone().fill_(1.0)

    cumulative = x_last

    conditioned_discounts = torch.zeros_like(x)
    i = 0
    while i <= len(x) - 1:
        # do not discount invalid steps so we can entirely skip a part of the trajectory
        # x should be already multiplied by valids
        discount_valid = discount * valids[i] + (1.0 - valids[i])
        cumulative = discount_valid ** x[i] * cumulative * (1.0 - dones[i])
        conditioned_discounts[i] = cumulative
        i += 1

    return conditioned_discounts


def calculate_discounted_sum_torch(
    x: Tensor, dones: Tensor, valids: Tensor, discount: Union[float, Tensor], x_last: Optional[Tensor] = None
) -> Tensor:
    """
    Computing cumulative sum (of something) for the trajectory, taking episode termination into consideration.
    """
    if x_last is None:
        x_last = x[-1].clone().fill_(0.0)

    cumulative = x_last

    discounted_sum = torch.zeros_like(x)
    i = len(x) - 1
    while i >= 0:
        # do not discount invalid steps so we can entirely skip a part of the trajectory
        # x should be already multiplied by valids
        if isinstance(discount, float):
            discount_valid = discount * valids[i] + (1 - valids[i])
        else:
            discount_valid = discount[i] * valids[i] + (1 - valids[i])
        cumulative = x[i] + discount_valid * cumulative * (1.0 - dones[i])
        discounted_sum[i] = cumulative
        i -= 1

    return discounted_sum


def gae_advantages_conditioned(
    rewards: Tensor, dones: Tensor, values: Tensor, valids: Tensor, γ: float, λ: float, strategy_steps: Tensor
) -> Tensor:
    strategy_steps = strategy_steps.transpose(0, 1)  # [E, T] -> [T, E]
    rewards = rewards.transpose(0, 1)  # [E, T] -> [T, E]
    dones = dones.transpose(0, 1).float()  # [E, T] -> [T, E]
    values = values.transpose(0, 1)  # [E, T+1] -> [T+1, E]
    valids = valids.transpose(0, 1).float()  # [E, T+1] -> [T+1, E]

    assert len(rewards) == len(dones)
    assert len(rewards) + 1 == len(values)

    # section 3 in GAE paper: calculating advantages
    deltas = (rewards - values[:-1]) * valids[:-1] + (1 - dones) * (γ * values[1:] * valids[1:])

    γ = calculate_conditioned_discounts(strategy_steps[:-1], dones=dones, valids=valids[:-1], discount=γ)
    advantages = calculate_discounted_sum_torch(deltas, dones, valids[:-1], γ * λ)

    # transpose advantages back to [E, T] before creating a single experience buffer
    advantages.transpose_(0, 1)
    return advantages


advantages = gae_advantages_conditioned(rewards, dones, values, valids, 0.99, 0.95, strategy_steps)
