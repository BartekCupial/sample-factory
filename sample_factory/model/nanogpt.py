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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


RecurrentBlockCache = namedtuple("RecurrentBlockCache", ["rg_lru_state", "conv1d_state"])

def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


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


class Identity(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def init_weights(self):
        pass

    def forward(self, x, state, *args, **kwargs):
        return x, state


class LinearAttention(nn.Module):
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
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.phi = Phi() if config.attention_type == "elu" else nn.Identity()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.S_aggregator = SumAggregation()
        self.Z_aggregator = SumAggregation()

    def init_weights(self):
        self.apply(init_weights)

    # TODO: what about the number of heads here?
    def forward(
        self, x: torch.Tensor, state: List[torch.Tensor], context_mask=None
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
        # 3 * [B, T, F]
        Q, K, V = self.c_attn(x).split(self.d_model, dim=2)

        K = self.phi(K)
        Q = self.phi(Q)
        V = V

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

        denominator = torch.einsum("bti, bti -> bt", Q, Z).reshape(B, T, 1) + 1e-5
        # output = (Q^T S) / (Q^T Z)
        output = numerator / denominator

        S = S.reshape(B, T, F * F)
        state = [S[:, -1], Z[:, -1]]

        return output, state


class RoPE(nn.Module):
    # features are paired x_i, x_{i + d_head/2}
    def __init__(self, dhead, length):
        super().__init__()
        self.dhead = dhead
        self.length = length
        angle_exponents = torch.arange(0, dhead, 2) / dhead
        angles = torch.pow(1 / 10000, angle_exponents).reshape(1, -1)
        angle_per_token = angles * torch.arange(0, length).reshape(-1, 1)
        self.register_buffer("sin", torch.sin(angle_per_token).repeat(1, 2))
        self.register_buffer("cos", torch.cos(angle_per_token).repeat(1, 2))

    def forward(self, x):
        y1, y2 = torch.chunk(x, chunks=2, dim=-1)
        x_rotated = torch.cat([-y2, y1], dim=-1)
        return x * self.cos + x_rotated * self.sin


class SineEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        self.div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) *
                                  -(math.log(10000.0) / self.d_model)))

    def forward(self, position, seq_length):
        # position - [bs, seqlen]
        pe = torch.zeros(position.shape[0], seq_length, self.d_model, device=position.device)
        position = position.unsqueeze(-1)  # [bs, seqlen, 1]

        pe[..., 0::2] = torch.sin(position.float() * self.div_term.to(position.device))
        pe[..., 1::2] = torch.cos(position.float() * self.div_term.to(position.device))
        return pe  # [bs, seqlen, d_model]


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.context_len = config.context_len
        self.softmax_attention = config.attention_type == "softmax"

        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        if config.constant_context:
            constant_context_bias = torch.triu(torch.ones(config.block_size, config.block_size),
                                               diagonal=-config.context_len + 1)
            bias = bias * constant_context_bias
        bias = bias.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence

        if config.embedding_type == "rope":
            self.rope = RoPE(config.d_model // config.n_head, config.block_size)
            self.use_rope = True
        else:
            self.use_rope = False

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x, state=None, mask=None):
        # Context is a dummy variable for compatibility with the LinearAttention module

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        if mask is not None:
            causal_mask = self.bias[:,:,:T,:T] == 1
            full_mask = torch.logical_and(mask, causal_mask)
            # If all tokens in a row are masked, then set True to avoid NaNs
            empty_rows = torch.logical_not(torch.any(full_mask, dim=-1))
            full_mask[empty_rows] = True
        else:
            full_mask = None

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=full_mask, dropout_p=self.dropout if self.training else 0, is_causal=full_mask is None)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # (B, nh, T, T)
            if full_mask is None:
                full_mask = causal_mask

            att = att.masked_fill(torch.logical_not(full_mask), float('-inf'))
            if self.softmax_attention:
                att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, None


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def init_weights(self):
        init_weights(self.c_fc)
        init_weights(self.c_proj)


class Block(nn.Module):

    def __init__(self, config, block_idx, time_mixing_cls, use_layer_norm=True):
        super().__init__()

        if use_layer_norm:
            self.ln_1 = LayerNorm(config.d_model, bias=config.bias) if config.two_layer_norms else nn.Identity()
            self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()

        self.time_mixing = time_mixing_cls(config)
        self.mlp = MLP(config)
        self.block_idx = block_idx

    def forward(self, x, state, context_mask=None):
        current_state = None
        if state is not None:
            current_state = state[self.block_idx]

        y, current_state = self.time_mixing(self.ln_1(x), current_state, context_mask)

        if state is not None:
            state[self.block_idx] = current_state

        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, state

    def init_weights(self):
        self.mlp.init_weights()
        self.time_mixing.init_weights()


@dataclass
class GPTConfig:
    block_size: int = 1024
    num_layers: int = 12
    n_head: int = 12
    input_size: int = 128
    output_size: int = 128
    d_model: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    context_len: int = 32
    embedding_type: str = "table"  # table, linear, rope, sine
    relative_timesteps: bool = True  # use absolute timestep idx or relative (t=0 is start of the trajectory vs t=0 is the start of the chunk)
    constant_context: bool = False
    attention_type: str = "softmax"  # softmax, linear
    two_layer_norms: bool = True


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        if self.config.embedding_type == "table":
            wpe = nn.Embedding(config.block_size, config.d_model)
        elif self.config.embedding_type == "linear":
            wpe = nn.Linear(1, config.d_model)
        elif self.config.embedding_type == "sine":
            wpe = SineEmbedding(config.d_model)

        # Only linear and sine embeddings support absolute timesteps
        assert config.relative_timesteps or config.embedding_type in ["linear", "sine"]

        transformer_modules = {
            "input_projection": nn.Linear(config.input_size, config.d_model, bias=config.bias),
            "output_projection": nn.Linear(config.d_model, config.output_size, bias=config.bias),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config, block_idx, self.time_mixing_cls) for block_idx in range(config.num_layers)]),
            "ln_f": LayerNorm(config.d_model, bias=config.bias),
        }

        if self.config.embedding_type in ["table", "linear"]:
            transformer_modules["wpe"] = wpe
        elif self.config.embedding_type == "sine":
            self.wpe = wpe

        self.transformer = nn.ModuleDict(transformer_modules)
        self.init_weights()

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.embedding_type in ["table", "linear"]:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def init_weights(self):
        # init all weights
        init_weights(self.transformer["input_projection"])
        init_weights(self.transformer["output_projection"])

        for block in self.transformer["h"]:
            block.init_weights()

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layers))

    def forward(self, x, context):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lens = pad_packed_sequence(x, batch_first=False, padding_value=0.0)
            to_repack = True
        else:
            to_repack = False

        # x: [seq_len, batch_size, ...] and we would like [batch_size, seq_len, ...]
        x = x.permute(1, 0, 2)
        context, context_mask, context_timesteps = self._unpack_context(context, x.shape[1], x.device)

        x, context = self._forward(x, context, context_mask, context_timesteps)

        context = self._pack_context(context, context_mask, context_timesteps)

        x = x.permute(1, 0, 2)

        if to_repack:
            x = pack_padded_sequence(x, lens, batch_first=False, enforce_sorted=False)
        return x, context

    def _get_positional_embedding(self, context, context_mask, context_timesteps):
        if self.config.relative_timesteps:
            # TODO: this won't work if we have multiple sequences in the same batch
            timesteps = context_mask.cumsum(-1)
        else:
            timesteps = context_timesteps  # [bs, seqlen]

        if self.config.embedding_type == "table":
            pos_emb = self.transformer.wpe(timesteps.long())
        elif self.config.embedding_type == "linear":
            pos_emb = self.transformer.wpe(timesteps.unsqueeze(-1))
        elif self.config.embedding_type == "sine":
            pos_emb = self.wpe(timesteps, timesteps.shape[-1])
        elif self.config.embedding_type == "rope":
            pos_emb = torch.zeros(context.shape[0], context.shape[1], self.config.d_model,
                                  device=context.device, dtype=context.dtype)
        elif self.config.embedding_type == "none":
            pos_emb = 0
        return pos_emb  # [bs, seqlen, d_model]

    def _forward(self, x, context, context_mask, context_timesteps):
        new_batch_size, new_seqlen, _ = x.size()
        # if isinstance(context, list):
        #     old_batch_size = context[0][0].shape[0]
        # elif isinstance(context, ReurrentBlockCache):
        #     old_batch_size = context.state[0].shape[0]
        # else:
        #     old_batch_size = context.shape[0]

        # assert old_batch_size == new_batch_size, f"Batch size mismatch: {old_batch_size} != {new_batch_size}"
        # forward the GPT model itself
        x = self.transformer.input_projection(x)  # shape (b, t, d_model)

        if self.window_transformer:
            context = torch.cat((context, x), dim=1)
            state = None
        else:
            state = context

        pos_emb = self._get_positional_embedding(context, context_mask, context_timesteps)
        # TODO: a separate function so that we can apply the rotational encoding if needed?
        if self.window_transformer:
            x = self.transformer.drop(context + pos_emb)
        else:
            x = self.transformer.drop(x + pos_emb)

        # Reshape the context mask to fit the attention layers
        context_mask_2d = (context_mask.unsqueeze(2) * context_mask.unsqueeze(1)).unsqueeze(1)
        for idx, block in enumerate(self.transformer.h):
            x, state = block(x, state, context_mask_2d)
        x = self.transformer.ln_f(x)
        x = self.transformer.output_projection(x)  # shape (b, t, output_siz)

        if self.window_transformer:
            x = x[:, -new_seqlen:]
        else:
            context = state

        # Return updated hidden_states
        return x, context

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.num_layers, cfg.n_head, cfg.d_model//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def _unpack_context(self, context, new_seqlen, device):
        raise NotImplementedError

    def _pack_context(self, context, context_mask, context_timesteps):
        raise NotImplementedError


class ContextWindowGPT(GPT):
    def __init__(self, config):
        self.time_mixing_cls = CausalSelfAttention
        self.window_transformer = True
        super().__init__(config)
        assert config.attention_type in ["linear", "softmax"]


    def _unpack_context(self, context, new_seqlen, device):
        # reshape and unpack context:
        # [batch_size, seq_len, d_model + 2] -> three tensors:
        # tokens: [batch_size, seq_len, d_model],
        # token mask: [batch_size, seq_len],
        # token timestep: [batch_size, seq_len]
        context = context.view(-1, self.config.block_size, self.config.d_model + 2)
        context_mask = context[:, :, -2].bool()  # [batch_size, seq_len]
        context_timesteps = context[:, :, -1]  # [batch_size, seq_len]
        context = context[:, :, :-2]

        # token mask for the newtokens should be 1
        ones_row = torch.ones([context_mask.shape[0], new_seqlen],
                              device=context_mask.device, dtype=context_mask.dtype)
        context_mask = torch.cat([context_mask, ones_row], dim=-1)

        # create new timesteps for the new tokens
        new_timesteps = torch.arange(new_seqlen, dtype=torch.float, device=device).view(1, -1) + 1
        new_timesteps = new_timesteps + context_timesteps[:, -1].view(-1, 1)

        context_timesteps = torch.cat([context_timesteps, new_timesteps], dim=-1)

        # We need to make room for the new tokens
        context = context[:, new_seqlen:]
        context_timesteps = context_timesteps[:, new_seqlen:]
        context_mask = context_mask[:, new_seqlen:]

        return context, context_mask, context_timesteps

    def _pack_context(self, context, context_mask, context_timesteps):
        context = torch.cat([context, context_mask.unsqueeze(-1), context_timesteps.unsqueeze(-1)], dim=-1)
        context = context.view(-1, self.config.block_size * (self.config.d_model + 2))
        return context


class AutoregressiveGPT(GPT):
    def __init__(self, config):
        self.time_mixing_cls = LinearAttention
        self.window_transformer = False
        super().__init__(config)

        assert config.attention_type in ["identity", "linear", "elu"]

        # TODO: implement something rope-like for LinearTransformer
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

        state = list(state.chunk(self.config.num_layers, dim=0))
        mask = torch.ones_like(current_timesteps, dtype=torch.bool)
        for idx, layer_state in enumerate(state):
            # ([B, 1, H * H + H])
            layer_state = layer_state.transpose(0, 1)
            # ([B, 1, H * H], [B, 1, H])
            state[idx] = list(layer_state.split([self.config.d_model * self.config.d_model, self.config.d_model], dim=2))
        return state, mask, current_timesteps

    def _pack_context(self, state, state_mask, state_timesteps):
        state = [torch.cat([layer_state[0], layer_state[1]], dim=1)
                 for layer_state in state]
        # [num_layers, B, -1]
        state = torch.stack(state, dim=0)
        state = state.transpose(0, 1)
        state = state.reshape(state.shape[0], -1)
        # Take last timestep from each batch
        state_timesteps = state_timesteps[:, -1]
        state = torch.cat([state, state_timesteps.view(-1, 1)], dim=1)
        return state

class IdentityGPT(GPT):
    def __init__(self, config):
        self.time_mixing_cls = Identity
        self.window_transformer = False
        super().__init__(config)

        assert config.attention_type == "none"

        # TODO: implement something rope-like for LinearTransformer
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
