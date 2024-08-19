from abc import ABC

import torch
from torch import nn

from sample_factory.model.linear_transformer import DeepLinearAttention
from sample_factory.model.mamba import CustomMamba, InferenceParams
from sample_factory.model.model_utils import ModelModule
from sample_factory.model.nanogpt import AutoregressiveGPT, ContextWindowGPT, GPTConfig
from sample_factory.utils.typing import Config


class ModelCore(ModelModule, ABC):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.core_output_size = -1  # to be overridden in derived classes

    def get_out_size(self) -> int:
        return self.core_output_size


class ModelCoreRNN(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False
        self.is_mamba = False
        self.is_nanogpt = False
        self.is_linear_transformer = False

        cfg.rnn_input_size = input_size

        if cfg.rnn_type == "gru":
            assert cfg.rnn_d_model == cfg.rnn_d_output, "d_model = d_output in RNNs"
            self.core = nn.GRU(input_size, cfg.rnn_d_output, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            assert cfg.rnn_d_model == cfg.rnn_d_output, "d_model = d_output in RNNs"
            self.core = nn.LSTM(input_size, cfg.rnn_d_output, cfg.rnn_num_layers)
        elif cfg.rnn_type == "mamba":
            self.is_mamba = True
            self.core = CustomMamba(input_size,
                                    output_size=cfg.rnn_d_output,
                                    d_model=cfg.rnn_d_model,
                                    d_state=cfg.mamba_state_size,
                                    d_conv=cfg.mamba_conv_size,
                                    expand=cfg.mamba_expand,
                                    num_layers=cfg.rnn_num_layers,
                                    use_complex=cfg.mamba_use_complex,
                                    selective=cfg.mamba_selective_ssm)
        elif cfg.rnn_type == "nanogpt":
            self.is_nanogpt = True
            GPT_cfg = GPTConfig(
                input_size=input_size,
                output_size=cfg.rnn_d_output,
                num_layers=cfg.rnn_num_layers,
                d_model=cfg.rnn_d_model,
                n_head=cfg.nanogpt_n_head,
                dropout=cfg.nanogpt_dropout,
                block_size=cfg.nanogpt_block_size,
                embedding_type=cfg.nanogpt_embedding_type,
                relative_timesteps=cfg.nanogpt_relative_timesteps,
                context_len=cfg.rollout,
                constant_context=cfg.nanogpt_constant_context,
                attention_type=cfg.nanogpt_attention_type,
                two_layer_norms=cfg.nanogpt_two_layer_norms,
            )
            if cfg.nanogpt_recurrent_mode:
                self.core = AutoregressiveGPT(GPT_cfg)
            else:
                self.core = ContextWindowGPT(GPT_cfg)
        elif cfg.rnn_type == "xlstm":
            from xlstm import (
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
                mLSTMBlockConfig,
                mLSTMLayerConfig,
                sLSTMBlockConfig,
                sLSTMLayerConfig,
                FeedForwardConfig,
            )

            cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                    )
                ),
                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="vanilla",
                        num_heads=4,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),
                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                ),
                context_length=256,
                num_blocks=cfg.rnn_num_layers,
                embedding_dim=cfg.rnn_d_model,
                slstm_at=[1],

            )

            self.core = xLSTMBlockStack(cfg)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.core_output_size = cfg.rnn_d_output
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output) or head_output.ndim == 3

        rnn_states = rnn_states * self.cfg.decay_hidden_states

        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if not self.is_nanogpt and not self.is_linear_transformer:
            if self.rnn_num_layers > 1:
                rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
                rnn_states = rnn_states.permute(1, 0, 2)
            else:
                rnn_states = rnn_states.unsqueeze(0)

        if self.is_mamba or self.is_gru or self.is_nanogpt or self.is_linear_transformer:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            # Just give zeros to LSTM
            h, c = torch.split(rnn_states, self.cfg.rnn_d_output, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if not self.is_nanogpt and not self.is_linear_transformer:
            if self.rnn_num_layers > 1:
                new_rnn_states = new_rnn_states.permute(1, 0, 2)
                new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
            else:
                new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


class ModelCoreIdentity(ModelCore):
    """A noop core (no recurrency)."""

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    # noinspection PyMethodMayBeStatic
    def forward(self, head_output, fake_rnn_states):
        return head_output, fake_rnn_states


def default_make_core_func(cfg: Config, core_input_size: int) -> ModelCore:
    if cfg.use_rnn:
        core = ModelCoreRNN(cfg, core_input_size)
    else:
        core = ModelCoreIdentity(cfg, core_input_size)

    return core
