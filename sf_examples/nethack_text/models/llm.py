from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import model_device
from sample_factory.utils.normalize import ObservationNormalizer


class PolicyLLM(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super().__init__()
        self.dummy_model = nn.Linear(10, 10)

        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space

        # we make normalizers a part of the model, so we can use the same infrastructure
        # to load/save the state of the normalizer (running mean and stddev statistics)
        self.obs_normalizer: ObservationNormalizer = ObservationNormalizer(obs_space, cfg)

        self.returns_normalizer: Optional[RunningMeanStdInPlace] = None
        if cfg.normalize_returns:
            returns_shape = (1,)  # it's actually a single scalar but we use 1D shape for the normalizer
            self.returns_normalizer = RunningMeanStdInPlace(returns_shape)
            # comment this out for debugging (i.e. to be able to step through normalizer code)
            self.returns_normalizer = torch.jit.script(self.returns_normalizer)

        self.last_action_distribution = None  # to be populated after each forward step

    def device_for_input_tensor(self, input_tensor_name: str) -> Optional[torch.device]:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.int64

    def normalize_obs(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.obs_normalizer(obs)

    def model_to_device(self, device):
        pass

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        # TODO: for now training is turned off, maybe we will implement this later
        raise NotImplementedError()

    def forward_core(self, head_output, rnn_states):
        # TODO: for now training is turned off, maybe we will implement this later
        raise NotImplementedError()

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        # TODO: for now training is turned off, maybe we will implement this later
        raise NotImplementedError()

    def forward(self, normalized_obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        raise NotImplementedError()
