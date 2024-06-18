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
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    def __init__(self, seq_length, d_model):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model

        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        self.div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) *
                                  -(math.log(10000.0) / self.d_model)))

    def forward(self, position):
        # position - [bs, seqlen]
        pe = torch.zeros(position.shape[0], self.seq_length, self.d_model, device=position.device)
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

    def forward(self, x, mask=None):
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

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, use_layer_norm=True):
        super().__init__()

        if use_layer_norm:
            self.ln_1 = LayerNorm(config.d_model, bias=config.bias) if config.two_layer_norms else nn.Identity()
            self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, context_mask=None):
        x = x + self.attn(self.ln_1(x), context_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


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
    constant_context: bool = True
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
            wpe = SineEmbedding(config.block_size, config.d_model)

        # Only linear and sine embeddings support absolute timesteps
        assert config.relative_timesteps or config.embedding_type in ["linear", "sine"]

        transformer_modules = {
            "input_projection": nn.Linear(config.input_size, config.d_model, bias=config.bias),
            "output_projection": nn.Linear(config.d_model, config.output_size, bias=config.bias),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            "ln_f": LayerNorm(config.d_model, bias=config.bias),
        }

        if self.config.embedding_type in ["table", "linear"]:
            transformer_modules["wpe"] = wpe
        elif self.config.embedding_type == "sine":
            self.wpe = wpe

        self.transformer = nn.ModuleDict(transformer_modules)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    def forward(self, x, context):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lens = pad_packed_sequence(x, batch_first=False, padding_value=0.0)
            to_repack = True
        else:
            to_repack = False

        # x: [seq_len, batch_size, ...] and we would like [batch_size, seq_len, ...]
        x = x.permute(1, 0, 2)
        context, context_mask, context_timesteps = self._unpack_context(context, x.shape[1], x.device)

        x, context, context_mask, context_timesteps = self._forward(
                x, context, context_mask, context_timesteps)

        context = torch.cat([context, context_mask.unsqueeze(-1), context_timesteps.unsqueeze(-1)], dim=-1)
        context = context.view(-1, self.config.block_size * (self.config.d_model + 2))

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
            pos_emb = self.wpe(timesteps)
        elif self.config.embedding_type == "rope":
            pos_emb = torch.zeros(context.shape[0], context.shape[1], self.config.d_model,
                                  device=context.device, dtype=context.dtype)
        return pos_emb  # [bs, seqlen, d_model]



    def _forward(self, x, context, context_mask, context_timesteps):
        new_batch_size, new_seqlen, _ = x.size()
        old_batch_size, old_seqlen, _ = context.size()

        assert old_batch_size == new_batch_size, f"Batch size mismatch: {old_batch_size} != {new_batch_size}"
        # forward the GPT model itself
        x = self.transformer.input_projection(x)  # shape (b, t, d_model)

        context = torch.cat((context, x), dim=1)

        pos_emb = self._get_positional_embedding(context, context_mask, context_timesteps)

        # Reshape the context mask to fit the attention layers
        context_mask_2d = (context_mask.unsqueeze(2) * context_mask.unsqueeze(1)).unsqueeze(1)

        x = self.transformer.drop(context + pos_emb)
        for idx, block in enumerate(self.transformer.h):
            x = block(x, context_mask_2d)
        x = self.transformer.ln_f(x)
        x = self.transformer.output_projection(x)  # shape (b, t, output_siz)

        x = x[:, -new_seqlen:]

        # Return updated hidden_states
        return x, context, context_mask, context_timesteps

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
