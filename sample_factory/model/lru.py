
import math
import einops
import torch

from torch import nn

from sample_factory.model.nanogpt import GPT, RecurrentBlockCache

_MAX_SQRT_GRADIENT = 1000.0


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Returns the GELU activation function with the same approximation as JAX."""
    return nn.functional.gelu(x, approximate="tanh")


def rnn_param_init(
    tensor: torch.Tensor,
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Initializes the `A` parameter of the RG-LRU uniformly on a ring."""
    with torch.no_grad():
        # Proportional to area in a ring.
        # 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + 1e-8)
        tensor.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
        tensor.log_().mul_(0.5)

        if transform == "softplus":
            # Inverse transform.
            # jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
            return tensor.neg_().exp_().sub_(1.0).log_()
        else:
            raise NotImplementedError()


# TODO: add an option to use RMSNorm instead of layernorm!
class RMSNorm(nn.Module):
    """RMS Norm."""

    def __init__(
        self,
        width: int,
        eps: float = 1e-6,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the RMSNorm.

        Args:
          width: The number of dimensions of the input and output.
          eps: Small constant added to the square root when normalizing.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialzation.
          dtype: What dtype to use for initialziation.
        """
        super().__init__()
        self.width = width
        self.eps = eps

        # Parameters.
        self.scale = nn.Parameter(torch.empty(
            [self.width], device=device, dtype=dtype
        ))

        # Initialization
        self.init_weights()

    def init_weights(self) -> None:
        """Resets the parameters of the module."""
        torch.nn.init.zeros_(self.scale)

    def forward(self, x):
        """Calls the RMSNorm."""
        var = torch.mean(torch.square(x), axis=-1, keepdims=True)
        normed_x = x * torch.rsqrt(var + self.eps)

        scale = torch.reshape(self.scale, [1 for _ in range(x.ndim - 1)] + [-1])

        return normed_x * (scale + 1)


class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer."""

    def __init__(
        self,
        width: int,
        num_blocks: int,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the BlockDiagonalLinear.

        Args:
          width: The number of dimensions of the input and output.
          num_blocks: The number of diagonal blocks in the layer.
          w_init_variance_scale: A parameters that scales the variance of the
            initialization of the weights.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialzation.
          dtype: What dtype to use for initialziation.
        """
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.block_width = self.width // self.num_blocks

        # Parameters.
        self.w = nn.Parameter(torch.empty(
            [self.num_blocks, self.block_width, self.block_width],
            device=device,
            dtype=dtype
        ))
        self.b = nn.Parameter(torch.empty(
            [self.num_blocks, self.block_width], device=device, dtype=dtype
        ))

    def init_weights(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.w)
        torch.nn.init.zeros_(self.b)

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weight `w` of the layer."""
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(self, x):
        """Calls the BlockDiagonalLinear."""
        # Split x to blocks.
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias.
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

        # Flatten the output.
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
        return grad_output / torch.sqrt(clipped_x_times_4)


def rnn_scan(x, a, reset, h0, acc_dtype: torch.dtype = torch.float32):
    """Runs the recurrence of a linear RNN.

    Args:
      x: The input sequence.
      a: The diagonal of the recurrence matrix `A`.
      reset: Indicator of document boundaries, e.g. when to reset the hidden state
        of the RNN.
      h0: The initial hidden state.
      acc_dtype: The data type for the accumulation.

    Returns:
      The output of the linear recurrence.
    """
    assert x.ndim == 3
    assert a.shape == x.shape[-a.ndim :]
    assert a.dtype == x.dtype
    assert type(a) is type(x)
    assert h0 is None or h0.dtype == acc_dtype

    # Multiply `a` by the reset.
    a = a * ~reset[..., None]

    if x.shape[1] == 1:
        # Using scan in sampling mode.
        if h0 is None:
            return x, x[:, 0].type(acc_dtype)

        else:
            y = a.type(acc_dtype) * h0[:, None] + x.type(acc_dtype)
            return y.type(x.dtype), y[:, -1]

    else:
        # Using scan in linear mode.
        if h0 is not None:
            h_t = h0
        else:
            h_t = torch.zeros(x[:, 0].shape, dtype=acc_dtype, device=x.device)

        y = torch.zeros_like(x)
        for t in range(x.shape[1]):
            h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
            y[:, t] = h_t.type(x.dtype)

    return y, h_t


class RGLRU(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the RG-LRU.

        Args:
          width: The number of dimensions of the input and output.
          num_heads: The number of diagonal blocks in the input and A gate layers.
          w_init_variance_scale: Initialization parameter for the
            BlockDiagonalLinear layers of the gates. See the `BlockDiagonalLinear`
            layer for details.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialzation.
          dtype: What dtype to use for initialziation.
        """
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters and layers.
        self.a_param = nn.Parameter(torch.empty(
            [self.width], device=device, dtype=dtype
        ))
        self.input_gate = BlockDiagonalLinear(
            width=self.width,
            num_blocks=self.num_heads,
            w_init_variance_scale=w_init_variance_scale,
            device=device,
            dtype=dtype,
        )
        self.a_gate = BlockDiagonalLinear(
            width=self.width,
            num_blocks=self.num_heads,
            w_init_variance_scale=self.w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

    def init_weights(self) -> None:
        """Resets the parameters of the module."""
        self.input_gate.init_weights()
        self.a_gate.init_weights()
        self.a_param_init(self.a_param)

    def a_param_init(self, w: torch.Tensor) -> torch.Tensor:
        """Initializes the `A` parameter of the RG-LRU."""
        return rnn_param_init(w, min_rad=0.9, max_rad=0.999)

    def forward(self, x, cache=None, return_cache: bool = True):
        """Calls the RG-LRU.

        Args:
          x: Sequence of input activations.
          cache: The previous hidden state of the RG-LRU.
          return_cache: Whether to compute and return the updated cache.

        Returns:
          Output of the block together with the updated hidden state.
        """

        bs, l, _ = x.shape

        reset = torch.zeros((bs, l), dtype=torch.bool, device=x.device)
        if cache is None:
            reset[:, 0] = True
        else:
            reset[:, 0] = (cache == 0).all(dim=1)

        # Gates for x and a.
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))

        # Compute the parameter `A` of the recurrence.
        log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
        a = torch.exp(log_a)
        a_square = torch.exp(2 * log_a)

        # Gate the input.
        gated_x = x * gate_x

        # Apply gamma normalization to the input. We need to clip the derivatives of
        # `sqrt` in order to prevent NaNs during training in bfloat16.
        multiplier = SqrtBoundDerivative.apply(1 - a_square)
        # Do not apply the multiplier at the reset positions.
        multiplier = reset[..., None] + ~reset[..., None] * multiplier
        normalized_x = gated_x * multiplier.type(x.dtype)

        y, last_h = rnn_scan(
            x=normalized_x,
            a=a,
            reset=reset,
            h0=cache,
        )

        if not return_cache:
            return y, None

        return y, last_h


class Conv1D(nn.Module):
    """A 1D temporal convolution layer."""

    def __init__(
        self,
        width: int,
        temporal_width: int,
        w_init_variance_scale: float = 0.01,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the Conv1D.

        Args:
          width: The number of features for both inputs and outputs.
          temporal_width: The size of the temporal receptive field of the
            convolution. In other words, how much back in time the convolution can
            look to produce an output.
          w_init_variance_scale: A parameter that scales the variance of the
            initialization of the weights.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialzation.
          dtype: What dtype to use for initialziation.
        """
        super().__init__()
        self.width = width
        self.temporal_width = temporal_width
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters.
        self.w = nn.Parameter(torch.empty(
            [self.temporal_width, self.width], device=device, dtype=dtype
        ))
        self.b = nn.Parameter(torch.empty([width], device=device, dtype=dtype))

    def init_weights(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.w)
        torch.nn.init.zeros_(self.b)

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weight matrix `w` of the Conv1D."""
        std = math.sqrt(self.w_init_variance_scale / self.temporal_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(self, x, cache=None, return_cache=True):
        """Calls the Conv1D.

        Args:
          x: Sequence of input activations.
          cache: The cache containing the previous `self.temporal_width-1` inputs
            This is set to `None` in training mode.
          return_cache: Whether to compute and return the updated cache.

        Returns:
          The output of the convolution and the updated state.
        """
        output_len = x.shape[1]

        # TODO: I think this is fine, but maybe we should check it again
        if cache is not None:
            # 1. Decoding mode:
            # - We have access to the previous `self.temporal_width - 1` inputs.
            x = self._concatenate_with_cache(x, cache)
            prompt_len = self.temporal_width - 1
            cache_dtype = cache.dtype
        else:
            # 1. Training mode:
            # - The full sequence length need to be output.
            prompt_len = 0
            cache_dtype = x.dtype

        # 3. Perform the convolution:
        # - Initialize an accumulator for the convolution output.
        convolution_output = 0.0

        # - We cannot look back by more than the total sequence length
        #   ("valid" convolution).
        temporal_width = min(self.temporal_width, prompt_len + output_len)

        # - The convolution is implemented as a manual loop so that we can
        #   incorporate the window masking further below.
        for temporal_shift in range(temporal_width):
            start_idx, end_idx = self._convolution_window_indices(
                prompt_len=prompt_len,
                shift_back=temporal_shift,
                output_len=output_len,
            )
            x_window = x[:, start_idx:end_idx]

            x_window = self._pad_window(x_window, output_len)

            # - Select w for this temporal shift, and expand on the batch and time
            #   dimensions.
            w = self.w[self.temporal_width - temporal_shift - 1][None, None, :]

            # - Accumulate the convolution result.
            convolution_output += x_window * w

        # - Add the bias of the convolution.
        convolution_output += self.b[None, None]

        if not return_cache:
            return convolution_output, None

        # 4. Store the new (potentially padded) cache for future decoding.
        new_cache = x[:, 1 - self.temporal_width :].type(cache_dtype)
        new_cache = self._pad_cache(new_cache)

        return convolution_output, new_cache

    def _concatenate_with_cache(self, x, cache):
        """Concatenates the current input `x` with the previous cache for decoding.

        Args:
          x: The current input activations (shape: [batch_size, 1, width]).
          cache: Cached tensor storing previous inputs (shape: [batch_size,
            temporal_width - 1, width]).

        Returns:
          The concatenated input sequence
          (shape: [batch_size, temporal_width, width]).
        """
        b, num_tokens, d = x.shape
        assert cache.shape == (b, self.temporal_width - 1, d)
        # assert num_tokens == 1
        return torch.concatenate([cache.type(x.dtype), x], dim=1)

    def _convolution_window_indices(
        self,
        *,
        prompt_len: int,
        shift_back: int,
        output_len: int,
    ) -> tuple[int, int]:
        """Calculates the start and end indices for the convolution window.

        Args:
          prompt_len: Length of the prompt (zero in training mode).
          shift_back: By how much the window should be shifted backwards.
          output_len: Sequence length of the output (sequence length in training
            mode, one in decoding mode).

        Returns:
          start_idx: The starting index for the convolution window.
          end_idx: The ending index for the convolution window.
        """
        start_idx = max(prompt_len - shift_back, 0)
        end_idx = prompt_len + output_len - shift_back
        return start_idx, end_idx

    def _pad_window(self, window, output_len: int):
        """Left-pads the window if it is shorter than the output sequence length."""
        batch_size, window_len, width = window.shape
        padding_len = output_len - window_len
        padding = torch.zeros(
            (batch_size, padding_len, width),
            dtype=window.dtype,
            device=window.device,
        )
        return torch.concatenate([padding, window], dim=1)

    def _pad_cache(self, state):
        """Left-pads the state if it is shorter than the temporal width."""
        b, state_seq_len, d = state.shape
        padding_len = self.temporal_width - state_seq_len - 1
        padding = torch.zeros(
            (b, padding_len, d),
            dtype=state.dtype,
            device=state.device,
        )
        return torch.concatenate([padding, state], dim=1)


class LRURecurrentBlock(nn.Module):
    """Griffin and Hawk's recurrent block."""

    def __init__(
        self,
        config,
        # TODO: do something about the final_w_init_variance_scale
        final_w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the recurrent block.

        Args:
          width: The width of the block.
          num_heads: The number of RG-LRU heads/blocks to use.
          lru_width: Internal dimension to be projected into for RG-LRU to operate
            on.
          conv1d_temporal_width: The temporal width of the 1d convolution.
          final_w_init_variance_scale: The scale for the initialization of the last
            layer of the block.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialziation.
        """
        super().__init__()

        self.width = config.d_model
        self.num_heads = config.n_head
        # TODO: possibly have something different here?
        self.lru_width = config.d_model
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # Hardcoded for now
        self.conv1d_temporal_width = 4

        # TODO: do we want nn.Linear?
        # Layers.
        self.linear_y = nn.Linear(
            in_features=self.width,
            out_features=self.lru_width,
            device=device,
            dtype=dtype,
        )
        self.linear_x = nn.Linear(
            in_features=self.width,
            out_features=self.lru_width,
            device=device,
            dtype=dtype,
        )
        self.linear_out = nn.Linear(
            in_features=self.lru_width,
            out_features=self.width,
            device=device,
            dtype=dtype,
        )
        self.conv_1d = Conv1D(
            width=self.lru_width,
            temporal_width=self.conv1d_temporal_width,
            device=device,
            dtype=dtype,
        )
        self.rg_lru = RGLRU(
            width=self.lru_width,
            num_heads=self.num_heads,
            device=device,
            dtype=dtype,
        )

    def init_weights(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.linear_x.weight)
        torch.nn.init.zeros_(self.linear_x.bias)
        self.w_init_(self.linear_y.weight)
        torch.nn.init.zeros_(self.linear_y.bias)
        self.out_w_init_(self.linear_out.weight)
        torch.nn.init.zeros_(self.linear_out.bias)
        self.conv_1d.init_weights()
        self.rg_lru.init_weights()

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weights of the linear x and y layers of the block."""
        torch.nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))

    def out_w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weights of the last layer of the block."""
        std = math.sqrt(self.final_w_init_variance_scale / self.lru_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(self, x, cache, mask, return_cache=True):
        """Calls the recurrent block.

        Args:
          x: Sequence of input activations.
          cache: Optional cache with the previous state of the RG-LRU and Conv1D.
          return_cache: Whether to compute and return the updated cache.

        Returns:
          Output of the block together with the updated cache. If `cache` is None
          than the returned updated cache is empty initialized and filled in from
          the input sequence.
        """
        # y branch.
        y = self.linear_y(x)
        y = gelu(y)

        # x branch.
        x = self.linear_x(x)
        x, conv1d_state = self.conv_1d(
            x=x,
            cache=cache.conv1d_state,
            return_cache=return_cache,
        )
        x, rg_lru_state = self.rg_lru(
            x=x,
            cache=cache.rg_lru_state,
            return_cache=return_cache,
        )

        # Join branches.
        x = x * y
        x = self.linear_out(x)

        if not return_cache:
            return x, None

        return x, RecurrentBlockCache(conv1d_state=conv1d_state, rg_lru_state=rg_lru_state)


class LRU(GPT):
    def __init__(self, config):
        self.time_mixing_cls = LRURecurrentBlock
        self.window_transformer = False
        super().__init__(config)

        self.conv_width = 4  # hardcoded for now

        assert config.attention_type == "none", "LRU does not support attention"

        if self.config.embedding_type == "rope":
            raise ValueError("LRU does not support RoPE embeddings")

    def _unpack_context(self, state, new_seqlen=None, device=None):
        # [batch_size]
        timesteps = state[:, -1]
        state = state[:, :-1]

        current_timesteps = timesteps.view(-1, 1) + torch.arange(new_seqlen, device=timesteps.device).view(1, -1)

        # [B, -1] -> [B, num_layers, -1]
        state = state.reshape(state.shape[0], self.config.num_layers, -1)
        # [B, num_layers, -1] -> [num_layers, B, -1]
        state = state.transpose(0, 1)

        # [num_layers, B, -1] -> [num_layers] *  [1, B, -1]
        state = list(state.chunk(self.config.num_layers, dim=0))
        mask = torch.ones_like(current_timesteps, dtype=torch.bool)
        for idx, layer_state in enumerate(state):
            # ([B, -1])
            layer_state = layer_state.squeeze(0)
            # ([B, d_model], [B,  d_model * (conv_width - 1)])
            lru_state, conv_state = layer_state.split(
                [self.config.d_model,
                 self.config.d_model * (self.conv_width - 1)], dim=1)

            conv_state = conv_state.view(-1, self.conv_width - 1, self.config.d_model)
            state[idx] = RecurrentBlockCache(rg_lru_state=lru_state,
                                             conv1d_state=conv_state)
        return state, mask, current_timesteps

    def _pack_context(self, state, state_mask, state_timesteps):
        batch_size = state[0].rg_lru_state.shape[0]
        state = [torch.cat([layer_state.rg_lru_state,
                            layer_state.conv1d_state.view(batch_size, -1)], dim=1)
                 for layer_state in state]
        # [num_layers, B, -1]
        state = torch.stack(state, dim=0)
        # [B, num_layers, -1]
        state = state.transpose(0, 1)
        # [B, -1]
        state = state.reshape(state.shape[0], -1)
        # Take last timestep from each batch
        state_timesteps = state_timesteps[:, -1]
        state = torch.cat([state, state_timesteps.view(-1, 1)], dim=1)
        return state
