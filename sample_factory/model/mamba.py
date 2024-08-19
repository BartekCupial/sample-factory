import math
from dataclasses import dataclass, field
from functools import partial
from typing import Optional
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sample_factory.model.mamba_simple import Mamba, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states


class CustomMamba(nn.Module):
    def __init__(self, input_size: int, output_size: int, d_model: int,
                 d_state: int, d_conv: int, expand: int, num_layers: int = 1,
                 use_complex: bool = False, selective: bool = True):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_layers = num_layers
        self.use_complex = use_complex

        ssm_cfg = {
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "use_complex": use_complex,
            "selective": selective,
        }

        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, output_size)

        self.core = MixerModel(d_model, n_layer=num_layers, ssm_cfg=ssm_cfg)

    def forward(self, x, rnn_states):
        # states -> [num_layers, batch_size, d_state]

        # Handle rnn_states
        inference_params = InferenceParams(max_seqlen=3,
                                           max_batch_size=rnn_states.shape[1],
                                           seqlen_offset=2)
        rnn_states = rnn_states.reshape(self.num_layers, rnn_states.shape[1], -1, self.d_conv + self.d_state)
        rnn_states = rnn_states.contiguous()
        conv_state = rnn_states[..., :self.d_conv]
        rnn_state = rnn_states[..., self.d_conv:]

        if self.use_complex:
            conv_state = conv_state.real

        inference_params.key_value_memory_dict = {
            layer_idx: (conv_state[layer_idx], rnn_state[layer_idx])
            for layer_idx in range(self.num_layers)
        }

        # Process input
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lens = pad_packed_sequence(x, batch_first=False, padding_value=0.0)
            to_repack = True
        else:
            to_repack = False

        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        output_xs = []
        for seq_idx in range(x.shape[1]):
            current_x = self.core(x[:, seq_idx].unsqueeze(1), inference_params=inference_params)
            output_xs += [current_x]
        x = torch.cat(output_xs, dim=1)
        x = x.permute(1, 0, 2)
        x = self.output_projection(x)

        if to_repack:
            x = pack_padded_sequence(x, lens, batch_first=False, enforce_sorted=False)

        conv_state = torch.stack(
            list(inference_params.key_value_memory_dict[layer_idx][0]
                 for layer_idx in range(self.num_layers)),
            dim=0
        )
        rnn_state = torch.stack(
            list(inference_params.key_value_memory_dict[layer_idx][1]
                 for layer_idx in range(self.num_layers)),
            dim=0
        )

        if self.use_complex:
            conv_state = torch.complex(conv_state, torch.zeros_like(conv_state))

        new_rnn_states = torch.cat((conv_state, rnn_state), dim=-1)
        new_rnn_states = new_rnn_states.reshape(self.num_layers, new_rnn_states.size(1), -1)

        return x, new_rnn_states
