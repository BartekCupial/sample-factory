import torch
import torch.nn as nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace, RunningMeanStd
# from gymnasium.wrappers.normalize import RunningMeanStd
from sample_factory.model.model_utils import orthogonal_init


class BROEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace, hidden_dim: int, num_blocks: int):
        
        super().__init__(cfg)
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        out_size = 0

        for obs_key in self.obs_keys:
            shape = obs_space[obs_key].shape

            if len(shape) == 1:
                self.encoders[obs_key] = BROEncoderMLP(obs_space[obs_key].shape[0], hidden_dim, num_blocks)
            elif len(shape) > 1:
                raise NotImplementedError(f"Conv encoder not implemented yet")
                # self.encoders[obs_key] = BROCNN(obs_space[obs_key], ...)
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


# class BROConvBlock(nn.Module):
#     def __init__(self, ...):
#         super().__init__()
#         ...

#     def forward(self, x):
        

# class BROCNN(nn.Module):
#     def __init__(
#         self,
#         obs_space,
#         ...,
#     ):
#         super().__init__()
#         ...

#     def forward(self, x):
#         ...

#     def get_out_size(self):
#         ...

class BROMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.ln1 = nn.GroupNorm(1, dim)
        self.fc1 = orthogonal_init(nn.Linear(dim, hidden_dim), gain=1.0)
        self.ln2 = nn.GroupNorm(1, hidden_dim)
        self.fc2 = orthogonal_init(nn.Linear(hidden_dim, dim), gain=1.0)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.ln1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.ln2(out)
        out = self.fc2(out)
        return out + identity

class BROEncoderMLP(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 hidden_dim: int = 512,
                 num_blocks: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.stem = nn.Sequential(
            orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=1.0),
            nn.GroupNorm(1, hidden_dim),
            nn.ELU(inplace=True)
        )

        self.blocks = nn.ModuleList([BROMLPBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.post_ln = nn.GroupNorm(1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.post_ln(x)
        return x

    def get_out_size(self) -> int:
        return self.hidden_dim



class BROActorEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = BROEncoder(
            cfg=cfg,
            obs_space=obs_space,
            hidden_dim=cfg.actor_hidden_dim,
            num_blocks=cfg.actor_depth,
        )

    def forward(self, x):
        return self.model(x)

    def get_out_size(self):
        return self.model.get_out_size()


class BROCriticEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = BROEncoder(
            cfg=cfg,
            obs_space=obs_space,
            hidden_dim=cfg.critic_hidden_dim,
            num_blocks=cfg.critic_depth,
        )

    def forward(self, x):
        return self.model(x)

    def get_out_size(self):
        return self.model.get_out_size()