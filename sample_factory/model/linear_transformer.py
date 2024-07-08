import math
from typing import List, Tuple

import gymnasium as gym
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# TODO: handle packedsequence

MAX_EP_LEN = 10_000

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

def get_embedding(position, d_model):
    position = position.unsqueeze(-1)
    # Changed log from 10_000.0 to max_len, improves accuracy for hard labyrinth
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=position.device).float() * (-math.log(MAX_EP_LEN) / d_model)
    ).view(1, 1, -1)
    sine = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    return sine, cos

def positional_encoding(position, d_model):
    sine, cos = get_embedding(position, d_model)

    # [B, T, D]
    pe = torch.zeros(position.shape[0], position.shape[1], d_model)
    pe[:, :, 0::2] = sine
    pe[:, :, 1::2] = cos
    return pe

def rotate(x, position, d_model):
    sine, cos = get_embedding(position, d_model)
    sine = sine.repeat(1, 1, 2)
    cos = cos.repeat(1, 1, 2)
    y1, y2 = torch.chunk(x, chunks=2, dim=-1)
    x_rotated = torch.cat([-y2, y1], dim=-1)
    return x * cos + x_rotated * sine


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
        denominator = torch.einsum("bti, bti -> bt", Q, Z).reshape(B, T, 1) + 1e-5
        # output = (Q^T S) / (Q^T Z)
        output = numerator / denominator

        if self.feed_forward:
            output = self.ff(output)

        if self.residual:
            output = output + self.shortcut(x)

        S = S.reshape(B, T, F * F)
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
        embedding_type: str,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_type = embedding_type

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
        if isinstance(z, torch.nn.utils.rnn.PackedSequence):
            z, lens = pad_packed_sequence(z, batch_first=False, padding_value=0.0)
            to_repack = True
        else:
            to_repack = False
        z = z.permute(1, 0, 2)

        T = z.shape[1]


        # [batch_size]
        timesteps = state[:, -1]
        state = state[:, :-1]

        # [B, -1] -> [B, num_layers, -1]
        state = state.reshape(state.shape[0], self.num_layers, -1)
        # [B, num_layers, -1] -> [num_layers, B, -1]
        state = state.transpose(0, 1)

        state = list(state.chunk(self.num_layers, dim=0))
        for idx, layer_state in enumerate(state):
            # ([B, 1, H * H + H])
            layer_state = layer_state.transpose(0, 1)
            # ([B, 1, H * H], [B, 1, H])
            state[idx] = list(layer_state.split([self.h * self.h, self.h], dim=2))

        # [B, T]
        current_timesteps = timesteps.view(-1, 1) + torch.arange(T, device=timesteps.device).view(1, -1)
        if self.embedding_type == "sine":
            emb = positional_encoding(current_timesteps, z.shape[-1])
            z = z + emb.to(z.device)
        elif self.embedding_type == "rope":
            z = rotate(z, current_timesteps, z.shape[-1])
        elif self.embedding_type == "none":
            pass
            

        z, state = self.forward_memory(z, state)

        state = [torch.cat([layer_state[0], layer_state[1]], dim=1)
                 for layer_state in state]
        # [num_layers, B, -1]
        state = torch.stack(state, dim=0)
        state = state.transpose(0, 1)
        state = state.reshape(state.shape[0], -1)
        timesteps = timesteps + T
        state = torch.cat([state, timesteps.view(-1, 1)], dim=1)

        z = z.permute(1, 0, 2)
        if to_repack:
            z = pack_padded_sequence(z, lens, batch_first=False, enforce_sorted=False)
        return z, state

    def forward_memory(
        self,
        z,
        state: List,
    ) -> Tuple:

        B, T, _ = z.shape
        for i, cell in enumerate(self.core):
            z, (s0, s1) = cell(z, [state[i][0], state[i][1]])
            state[i][0] = s0
            state[i][1] = s1
        z = self.unmap(z)
        return z, [[s[0][:, -1], s[1][:, -1]] for s in state]
