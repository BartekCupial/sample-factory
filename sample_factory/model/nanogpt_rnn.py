"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from sample_factory.model.nanogpt import GPT, init_weights

class GRUCell(nn.Module):
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
        config,
    ):
        super().__init__()
        self.cell = nn.GRU(config.d_model, config.d_model, batch_first=True)

    def init_weights(self):
        self.apply(init_weights)

    # TODO: what about the number of heads here?
    def forward(
        self, x: torch.Tensor, state: torch.Tensor, context_mask=None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x, state = self.cell(x, state.contiguous().unsqueeze(0))
        return x, state.squeeze(0)


class GRUGPT(GPT):
    def __init__(self, config):
        self.time_mixing_cls = GRUCell
        self.window_transformer = False
        super().__init__(config)

        # TODO: the identity part is a bit stupid, let's fix it later.
        assert config.attention_type == "none"

        if self.config.embedding_type == "rope":
            raise ValueError("RecurrentGPT does not support RoPE embeddings")

    def _unpack_context(self, state, new_seqlen=None, device=None):
        # [batch_size]
        timesteps = state[:, -1]
        state = state[:, :-1]

        current_timesteps = timesteps.view(-1, 1) + torch.arange(new_seqlen, device=timesteps.device).view(1, -1)

        # [B, -1] -> [B, num_layers, -1]
        state = state.reshape(state.shape[0], self.config.num_layers, -1)
        # [B, num_layers, -1] -> [num_layers, B, -1]
        state = state.transpose(0, 1)

        # state = list(state.chunk(self.config.num_layers, dim=0))
        mask = torch.ones_like(current_timesteps, dtype=torch.bool)
        return state, mask, current_timesteps

    def _pack_context(self, state, state_mask, state_timesteps):
        # [num_layers, B, -1]
        # state = torch.stack(state, dim=0)
        # [B, num_layers, -1]
        state = state.transpose(0, 1)
        # [B, -1]
        state = state.reshape(state.shape[0], -1)
        # Take last timestep from each batch
        state_timesteps = state_timesteps[:, -1]
        state = torch.cat([state, state_timesteps.view(-1, 1)], dim=1)
        return state

