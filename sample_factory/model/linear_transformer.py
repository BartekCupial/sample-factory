import math
from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.ray_models.base_model import BaseModel

# TODO: handle packedsequence


class Phi(nn.Module):
    def forward(self, x):
        return torch.nn.functional.elu(x) + 1

class SumAggregation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        return x.cumsum(dim=1).clamp(-1e20, 1e20) + memory


class LinearAttentionBlock(nn.Module):
    """
    The building block from the Linear Transformers are Secretly RNNs Paper. This is
    a form of linear transformer.

    Inputs:
        input_size: Size of input feature dim
        hidden_size: Size of key/query/value space
        feed_forward: Whether to apply a perceptron to the output
        residual: Whether to apply a residual connection from input to output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feed_forward=True,
        residual=True,
    ):
        super().__init__()
        self.key = nn.Linear(input_size, hidden_size, bias=False)
        self.query = nn.Linear(input_size, hidden_size, bias=False)
        self.value = nn.Linear(input_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(input_size)
        self.phi = Phi()
        self.S_aggregator = SumAggregation()
        self.Z_aggregator = SumAggregation()
        self.feed_forward = feed_forward
        self.residual = residual

        if self.feed_forward:
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True)
            )
        if self.residual:
            self.shortcut = nn.Linear(input_size, hidden_size)

    def forward(
        self, x: torch.Tensor, state: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Input:
            x: [B, T, F]
            state: Tuple[
                [B, 1, D, D],
                [B, 1, D]
            ]
        Output:
            y: [B, T, D]
            state: Tuple[
                [B, 1, D, D],
                [B, 1, D]
            ]
        """

        x = self.norm(x)
        K = self.phi(self.key(x))
        Q = self.phi(self.query(x))
        V = self.value(x)
        S, Z = state
        B, T, F = K.shape

        # S = sum(K V^T)
        outer_prod = torch.einsum("bti, btj -> btij", K, V).reshape(B, T, F * F)

        # Okay, so we can handle T steps at the same time... Cool!
        S = self.S_aggregator(
            outer_prod,
            S.reshape(B, 1, F * F),
        ).reshape(B, T, F, F)
        # Z = sum(K)
        Z = self.Z_aggregator(K, Z.reshape(B, 1, F))
        # numerator = Q^T S
        numerator = torch.einsum("bti, btil -> btl", Q, S)
        # denominator = Q^T Z

        # TODO: what is going on here?
        # TODO: I think this is a bug...
        denominator = torch.einsum("bti, btl -> bt", Q, Z).reshape(B, T, 1) + 1e-5
        # output = (Q^T S) / (Q^T Z)
        output = numerator / denominator

        if self.feed_forward:
            output = self.ff(output)

        if self.residual:
            output = output + self.shortcut(x)

        state = [S, Z]

        return output, state


class DeepLinearAttention(nn.Module):
    """A multi-layer version of linear attention."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int,
        num_layers: int,
        name: str,
        **custom_model_kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.num_layers = num_layers

        assert self.num_layers >= 1

        # They do this to have a constant hidden_size between layers, so that
        # each model has the same "storage".
        self.h = d_model

        core = [
            LinearAttentionBlock(
                input_size=input_size,
                hidden_size=self.h,
                feed_forward=True,
            )
        ]
        for _ in range(self.num_layers - 1):
            core.append(
                LinearAttentionBlock(
                    input_size=self.h,
                    hidden_size=self.h,
                    feed_forward=True,
                )
            )
        self.core = nn.ModuleList(core)
        self.unmap = nn.Linear(self.h, output_size)

    def forward(self, z, state):

        z = z.permute(1, 0, 2)

        state = state.split(self.num_layers, dim=1)
        for idx, layer_state in enumerate(state):
            state[idx] = layer_state.split([self.h * self.h], dim=1)

        z, state = self.forward_memory(z, state)

        # TODO: repack state

        z = z.permute(1, 0, 2)

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
    ) -> Tuple[TensorType, List[TensorType]]:

        # TODO: is the state size as we want?

        B, T, _ = z.shape
        for i, cell in enumerate(self.core):
            z, (s0, s1) = cell(z, [state[2 * i], state[2 * i + 1]])
            state[2 * i] = s0
            state[2 * i + 1] = s1
        z = self.unmap(z)
        return z, [s[:, -1:] for s in state]
