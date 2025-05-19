import torch
import torch.nn as nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace, RunningMeanStd
# from gymnasium.wrappers.normalize import RunningMeanStd
from sample_factory.model.model_utils import orthogonal_init


class SimBaEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace, hidden_dim: int, num_blocks: int, use_max_pool: bool, expansion: int = 4):
        
        super().__init__(cfg)
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        out_size = 0

        for obs_key in self.obs_keys:
            shape = obs_space[obs_key].shape

            if len(shape) == 1:
                self.encoders[obs_key] = SimBaEncoderMLP(obs_space[obs_key].shape[0], hidden_dim, num_blocks, expansion)
            elif len(shape) > 1:
                self.encoders[obs_key] = SimBaCNN(obs_space[obs_key], hidden_dim, num_blocks, use_max_pool, expansion)
            else:
                raise NotImplementedError(f"Unsupported observation space {obs_space}")

            # self.encoders[obs_key] = encoder_fn(obs_space[obs_key], hidden_dim, num_blocks, expansion)
            out_size += self.encoders[obs_key].get_out_size()

        self.encoder_out_size = out_size

    def forward(self, obs_dict):
        if len(self.obs_keys) == 1:
            key = self.obs_keys[0]
            return self.encoders[key](obs_dict[key])

        encodings = []
        for key in self.obs_keys:
            x = self.encoders[key](obs_dict[key])
            encodings.append(x)

        return torch.cat(encodings, 1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


class SimBaConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # GroupNorm with num_groups=1 is equivalent to LayerNorm
        self.layer_norm = nn.GroupNorm(1, in_channels)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

        # Add projection layer if channels change
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.layer_norm(x)
        out = self.conv_block(out)

        if self.projection is not None:
            identity = self.projection(identity)

        return identity + out


class SimBaCNN(nn.Module):
    def __init__(
        self,
        obs_space,
        hidden_dim=64,
        num_blocks=2,
        use_max_pool=False,
        expansion=2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_max_pool = use_max_pool
        in_channels = obs.space["screen_image"].shape[0]

        assert in_channels & (in_channels - 1) == 0, "in_channels must be power of 2"
        assert hidden_dim & (hidden_dim - 1) == 0, "hidden_dim must be power of 2"
        assert hidden_dim >= in_channels, "hidden_dim must be >= in_channels"
        assert not use_max_pool or (use_max_pool and num_blocks <= 4)

        # Calculate number of doublings needed
        current_channels = in_channels
        self.blocks = []

        # Initial convolution to project to hidden dimension
        self.initial_conv = orthogonal_init(
            nn.Conv2d(in_channels, current_channels * 2, kernel_size=3, padding=0, bias=False),
            gain=1.0,
        )
        current_channels *= 2

        # SimBa residual blocks
        self.blocks = []
        for i in range(num_blocks):
            next_channels = min(current_channels * 2, hidden_dim)
            self.blocks.append(SimBaConvBlock(current_channels, next_channels * expansion, next_channels))
            if self.use_max_pool:
                self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = next_channels
        self.blocks = nn.ModuleList(self.blocks)

        # Post-layer normalization
        # GroupNorm with num_groups=1 is equivalent to LayerNorm
        self.post_norm = nn.GroupNorm(1, current_channels)

        # Global average pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Initial projection
        x = self.initial_conv(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Post normalization
        x = self.post_norm(x)

        # Global pooling
        x = self.pooling(x)
        x = x.view(x.size(0), -1)

        return x

    def get_out_size(self):
        return self.hidden_dim


class SimBaMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.ln = nn.GroupNorm(1, dim)
        self.fc1 = orthogonal_init(nn.Linear(dim, hidden_dim), gain=1.0)
        self.act = nn.ELU(inplace=True)
        self.fc2 = orthogonal_init(nn.Linear(hidden_dim, dim), gain=1.0)
    
    def forward(self, x):
        identity = x
        out = self.ln(x)
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        return out + identity
        

class SimBaEncoderMLP(nn.Module):
    def __init__(self, obs_dim, hidden_dim: int, num_blocks: int, expansion: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.input_projection = orthogonal_init(nn.Linear(obs_dim, self.hidden_dim))
        # self.norm = RunningMeanStdInPlace((self.hidden_dim,))
        # self.norm = RunningMeanStd((self.hidden_dim,))

        self.blocks = nn.ModuleList(
            [SimBaMLPBlock(self.hidden_dim, expansion*self.hidden_dim) for _ in range(num_blocks)]
        )
        # self.output_projection = orthogonal_init(nn.Linear(curr_dim/2, obs_dim))
        self.post_ln = nn.GroupNorm(1, self.hidden_dim)

    def forward(self, x):
        out = x
        out = self.input_projection(out)
        # μ, σ, clip = self.norm.forward(out)
        # out = out.sub(μ).mul(1 / σ).clamp(-clip, clip)

        for block in self.blocks:
            out = block(out)
        out = self.post_ln(out)
        return out

    def get_out_size(self):
        return self.hidden_dim


class SimBaActorEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = SimBaEncoder(
            cfg=cfg,
            obs_space=obs_space,
            hidden_dim=cfg.actor_hidden_dim,
            num_blocks=cfg.actor_depth,
            use_max_pool=cfg.actor_use_max_pool,
            expansion=cfg.actor_expansion
        )

    def forward(self, x):
        return self.model(x)

    def get_out_size(self):
        return self.model.get_out_size()


class SimBaCriticEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = SimBaEncoder(
            cfg=cfg,
            obs_space=obs_space,
            hidden_dim=cfg.critic_hidden_dim,
            num_blocks=cfg.critic_depth,
            use_max_pool=cfg.critic_use_max_pool,
            expansion=cfg.critic_expansion,
        )

    def forward(self, x):
        return self.model(x)

    def get_out_size(self):
        return self.model.get_out_size()