from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

from sample_factory.algo.utils.action_distributions import is_continuous_action_space, sample_actions_log_probs
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace, running_mean_std_summaries
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.configurable import Configurable
from sample_factory.model.action_parameterization import (
    ActionParameterizationContinuousNonAdaptiveStddev,
    ActionParameterizationDefault,
)
from sample_factory.model.model_utils import model_device
from sample_factory.utils.normalize import ObservationNormalizer
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.utils.utils import log

from gymnasium.wrappers.normalize import RunningMeanStd
import copy


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class ActorCritic(nn.Module, Configurable):
    def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        nn.Module.__init__(self)
        Configurable.__init__(self, cfg)
        self.action_space = action_space
        self.encoders = []

        # we make normalizers a part of the model, so we can use the same infrastructure
        # to load/save the state of the normalizer (running mean and stddev statistics)
        self.obs_normalizer: ObservationNormalizer = ObservationNormalizer(obs_space, cfg)

        self.returns_normalizer: Optional[RunningMeanStdInPlace] = None
        if cfg.normalize_returns:
            returns_shape = (1,)  # it's actually a single scalar but we use 1D shape for the normalizer
            self.returns_normalizer = RunningMeanStdInPlace(returns_shape)
            # comment this out for debugging (i.e. to be able to step through normalizer code)
            self.returns_normalizer = torch.jit.script(self.returns_normalizer)

            if self.cfg.with_rnd:
                # Separate normalizer that keeps stats for intrinsic returns
                self.int_returns_normalizer = RunningMeanStdInPlace(returns_shape)
                self.int_returns_normalizer = torch.jit.script(self.int_returns_normalizer)

                self.discounted_reward = RewardForwardFilter(self.cfg.int_gamma)
                self.reward_rms = RunningMeanStd()

        self.last_action_distribution = None  # to be populated after each forward step

    def get_action_parameterization(self, decoder_output_size: int):
        if not self.cfg.adaptive_stddev and is_continuous_action_space(self.action_space):
            action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
                self.cfg,
                decoder_output_size,
                self.action_space,
            )
        else:
            action_parameterization = ActionParameterizationDefault(self.cfg, decoder_output_size, self.action_space)

        return action_parameterization

    def model_to_device(self, device):
        for module in self.children():
            # allow parts of encoders/decoders to be on different devices
            # (i.e. text-encoding LSTM for DMLab is faster on CPU)
            if hasattr(module, "model_to_device"):
                module.model_to_device(device)
            else:
                module.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        device = self.encoders[0].device_for_input_tensor(input_tensor_name)
        if device is None:
            device = model_device(self)
        return device

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return self.encoders[0].type_for_input_tensor(input_tensor_name)

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.cfg.policy_init_gain

        if hasattr(layer, "bias") and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        if self.cfg.policy_initialization == "orthogonal":
            if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
            else:
                # LSTMs and GRUs initialize themselves
                # should we use orthogonal/xavier for LSTM cells as well?
                # I never noticed much difference between different initialization schemes, and here it seems safer to
                # go with default initialization,
                pass
        elif self.cfg.policy_initialization == "xavier_uniform":
            if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
            else:
                pass
        elif self.cfg.policy_initialization == "torch_default":
            # do nothing
            pass

        self.initial_state = self.state_dict()  # Save initial state for L2 init loss

    def normalize_obs(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.obs_normalizer(obs)

    def summaries(self) -> Dict:
        # Can add more summaries here, like weights statistics
        s = self.obs_normalizer.summaries()
        if self.returns_normalizer is not None:
            for k, v in running_mean_std_summaries(self.returns_normalizer).items():
                s[f"returns_{k}"] = v
        return s

    def action_distribution(self):
        return self.last_action_distribution

    def _maybe_sample_actions(self, sample_actions: bool, result: TensorDict) -> None:
        if sample_actions:
            # for non-trivial action spaces it is faster to do these together
            actions, result["log_prob_actions"] = sample_actions_log_probs(self.last_action_distribution)
            assert actions.dim() == 2  # TODO: remove this once we test everything
            result["actions"] = actions.squeeze(dim=1)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward_core(self, head_output, rnn_states):
        raise NotImplementedError()

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        raise NotImplementedError()

    def forward(self, normalized_obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        raise NotImplementedError()


class ActorCriticSharedWeights(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(obs_space, action_space, cfg)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.encoders = [self.encoder]  # a single shared encoder

        self.core = model_factory.make_model_core_func(cfg, self.encoder.get_out_size())
        self.cores = [self.core]

        self.decoder = model_factory.make_model_decoder_func(cfg, self.core.get_out_size())
        self.decoders = [self.decoder]

        decoder_out_size: int = self.decoder.get_out_size()

        self.critic = model_factory.make_model_critic_func(cfg, self.decoder.get_out_size())
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)

        self.with_rnd = cfg.with_rnd
        self.apply(self.initialize_weights)

        # RND Networks
        if self.with_rnd:
            # TODO: should these use encoder architcture?
            # self.target_network = model_factory.make_model_encoder_func(cfg, obs_space)
            # self.predictor_network = model_factory.make_model_encoder_func(cfg, obs_space)

            # Copied from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
            def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
                torch.nn.init.orthogonal_(layer.weight, std)
                torch.nn.init.constant_(layer.bias, bias_const)
                return layer

            # Prediction network
            self.predictor_network = nn.Sequential(
                layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                nn.LeakyReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(7 * 7 * 64, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
            )

            # Target network
            self.target_network = nn.Sequential(
                layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                nn.LeakyReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(7 * 7 * 64, 512)),
            )

            # Critic that estimates intrisic rewards
            self.int_critic = model_factory.make_model_critic_func(cfg, self.decoder.get_out_size())

            # Freeze target network
            for param in self.target_network.parameters():
                param.requires_grad = False

        self.n_params = self.get_n_params()

    def get_n_params(self):
        self.n_params_encoders = 0
        self.n_params_cores = 0
        self.n_params_decoders = 0

        for encoder in self.encoders:
            self.n_params_encoders += sum(p.numel() for p in encoder.parameters())

        for core in self.cores:
            self.n_params_cores += sum(p.numel() for p in core.parameters())

        for decoder in self.decoders:
            self.n_params_decoders += sum(p.numel() for p in decoder.parameters())

        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic(decoder_output).squeeze()

        result = TensorDict(values=values)

        if self.with_rnd:
            int_values = self.int_critic(decoder_output).squeeze()
            result["int_values"] = int_values

        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output)

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result

# Use the same networks as in CleanRL's RND
class CleanRLActorCritic(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        cfg.encoder_conv_mlp_layers = [256, 448]
        super().__init__(obs_space, action_space, cfg)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.encoders = [self.encoder]  # a single shared encoder

        self.core = model_factory.make_model_core_func(cfg, self.encoder.get_out_size())
        self.cores = [self.core]

        self.decoder = model_factory.make_model_decoder_func(cfg, self.core.get_out_size())
        self.decoders = [self.decoder]

        decoder_out_size: int = self.decoder.get_out_size()
        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer
        self.extra_layer_critic = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.extra_layer_actor = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.01), nn.ReLU())
        
        self.critic = model_factory.make_model_critic_func(cfg, self.decoder.get_out_size())
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)

        self.with_rnd = cfg.with_rnd
        self.apply(self.initialize_weights)

        # RND Networks
        if self.with_rnd:
            # TODO: should these use encoder architcture?
            # self.target_network = model_factory.make_model_encoder_func(cfg, obs_space)
            # self.predictor_network = model_factory.make_model_encoder_func(cfg, obs_space)

            # Copied from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
            def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
                torch.nn.init.orthogonal_(layer.weight, std)
                torch.nn.init.constant_(layer.bias, bias_const)
                return layer

            # Prediction network
            self.predictor_activations = {}
            self.predictor_last_linear_layer = None
            self.predictor_network = nn.Sequential(
                layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                nn.LeakyReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(7 * 7 * 64, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
            )

            self.register_hooks()

            # Target network
            self.target_network = nn.Sequential(
                layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                nn.LeakyReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(7 * 7 * 64, 512)),
            )

            # Critic that estimates intrisic rewards
            self.int_critic = model_factory.make_model_critic_func(cfg, self.decoder.get_out_size())

            # Freeze target network
            for param in self.target_network.parameters():
                param.requires_grad = False

        self.n_params = self.get_n_params()

    def get_n_params(self):
        self.n_params_encoders = 0
        self.n_params_cores = 0
        self.n_params_decoders = 0

        for encoder in self.encoders:
            self.n_params_encoders += sum(p.numel() for p in encoder.parameters())

        for core in self.cores:
            self.n_params_cores += sum(p.numel() for p in core.parameters())

        for decoder in self.decoders:
            self.n_params_decoders += sum(p.numel() for p in decoder.parameters())

        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def register_hooks(self):
        for name, layer in self.predictor_network.named_modules():
            if isinstance(layer, nn.Linear):
                self.predictor_last_linear_layer = name
                layer.register_forward_hook(self.save_activations_hook(name, True))
            elif isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.save_activations_hook(name, False))

    def save_activations_hook(self, layer_name, is_linear):
        def hook(module, input, output):
            if is_linear:
                self.predictor_activations["predictor_mlp_" + layer_name] = output
            else:
                self.predictor_activations["predictor_conv_" + layer_name] = output
        return hook

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        decoder_output = self.decoder(core_output)
        extra_layer_output_critic = self.extra_layer_critic(decoder_output)
        values = self.critic(extra_layer_output_critic).squeeze()

        result = TensorDict(values=values)

        if self.with_rnd:
            int_values = self.int_critic(extra_layer_output_critic).squeeze()
            result["int_values"] = int_values

        if values_only:
            return result

        extra_layer_output_actor = self.extra_layer_actor(decoder_output)
        action_distribution_params, self.last_action_distribution = self.action_parameterization(extra_layer_output_actor)

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result

class ActorCriticSeparateWeights(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(obs_space, action_space, cfg)

        self.actor_encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.actor_core = model_factory.make_model_core_func(cfg, self.actor_encoder.get_out_size())

        self.critic_encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.critic_core = model_factory.make_model_core_func(cfg, self.critic_encoder.get_out_size())

        self.encoders = [self.actor_encoder, self.critic_encoder]
        self.cores = [self.actor_core, self.critic_core]

        self.core_func = self._core_rnn if self.cfg.use_rnn else self._core_empty

        self.actor_decoder = model_factory.make_model_decoder_func(cfg, self.actor_core.get_out_size())
        self.critic_decoder = model_factory.make_model_decoder_func(cfg, self.critic_core.get_out_size())
        self.decoders = [self.actor_decoder, self.critic_decoder]

        self.critic = model_factory.make_model_critic_func(cfg, self.critic_decoder.get_out_size())
        self.action_parameterization = self.get_action_parameterization(self.critic_decoder.get_out_size())

        self.with_rnd = cfg.with_rnd
        self.apply(self.initialize_weights)

        # RND Networks
        if self.with_rnd:
            # TODO: should these use encoder architcture?
            # self.target_network = model_factory.make_model_encoder_func(cfg, obs_space)
            # self.predictor_network = model_factory.make_model_encoder_func(cfg, obs_space)

            # Copied from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
            def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
                torch.nn.init.orthogonal_(layer.weight, std)
                torch.nn.init.constant_(layer.bias, bias_const)
                return layer

            # Prediction network
            self.predictor_network = nn.Sequential(
                layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                nn.LeakyReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(7 * 7 * 64, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
            )

            # Target network
            self.target_network = nn.Sequential(
                layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
                nn.LeakyReLU(),
                layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                nn.LeakyReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(7 * 7 * 64, 512)),
            )

            # Critic head that estimates intrisic rewards
            self.int_critic = model_factory.make_model_critic_func(cfg, self.critic_decoder.get_out_size())

            # Freeze target network
            for param in self.target_network.parameters():
                param.requires_grad = False

        self.n_params = self.get_n_params()

    def get_n_params(self):
        self.n_params_encoders = 0
        self.n_params_cores = 0
        self.n_params_decoders = 0

        for encoder in self.encoders:
            self.n_params_encoders += sum(p.numel() for p in encoder.parameters())

        for core in self.cores:
            self.n_params_cores += sum(p.numel() for p in core.parameters())

        for decoder in self.decoders:
            self.n_params_decoders += sum(p.numel() for p in decoder.parameters())

        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def _core_rnn(self, head_output, rnn_states):
        """
        This is actually pretty slow due to all these split and cat operations.
        Consider using shared weights when training RNN policies.
        """
        num_cores = len(self.cores)

        rnn_states_split = rnn_states.chunk(num_cores, dim=1)

        if isinstance(head_output, PackedSequence):
            # We cannot chunk PackedSequence directly, we first have to to unpack it,
            # chunk, then pack chunks again to be able to process then through the cores.
            # Finally we have to return concatenated outputs so we repeat the proces,
            # but this time using concatenation - unpack, cat and pack.

            unpacked_head_output, lengths = pad_packed_sequence(head_output)
            unpacked_head_output_split = unpacked_head_output.chunk(num_cores, dim=2)
            head_outputs_split = [
                pack_padded_sequence(unpacked_head_output_split[i], lengths, enforce_sorted=False)
                for i in range(num_cores)
            ]

            unpacked_outputs, new_rnn_states = [], []
            for i, c in enumerate(self.cores):
                output, new_rnn_state = c(head_outputs_split[i], rnn_states_split[i])
                unpacked_output, lengths = pad_packed_sequence(output)
                unpacked_outputs.append(unpacked_output)
                new_rnn_states.append(new_rnn_state)

            unpacked_outputs = torch.cat(unpacked_outputs, dim=2)
            outputs = pack_padded_sequence(unpacked_outputs, lengths, enforce_sorted=False)
        else:
            head_outputs_split = head_output.chunk(num_cores, dim=1)
            rnn_states_split = rnn_states.chunk(num_cores, dim=1)

            outputs, new_rnn_states = [], []
            for i, c in enumerate(self.cores):
                output, new_rnn_state = c(head_outputs_split[i], rnn_states_split[i])
                outputs.append(output)
                new_rnn_states.append(new_rnn_state)

            outputs = torch.cat(outputs, dim=1)

        new_rnn_states = torch.cat(new_rnn_states, dim=1)

        return outputs, new_rnn_states

    @staticmethod
    def _core_empty(head_output, fake_rnn_states):
        """Optimization for the feed-forward case."""
        return head_output, fake_rnn_states

    def forward_head(self, normalized_obs_dict: Dict):
        head_outputs = []
        for enc in self.encoders:
            head_outputs.append(enc(normalized_obs_dict))

        return torch.cat(head_outputs, dim=1)

    def forward_core(self, head_output, rnn_states):
        return self.core_func(head_output, rnn_states)

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        # second core output corresponds to the critic
        critic_decoder_output = self.critic_decoder(core_outputs[1])
        values = self.critic(critic_decoder_output).squeeze()

        result = TensorDict(values=values)

        if self.with_rnd:
            int_values = self.int_critic(critic_decoder_output).squeeze()
            result["int_values"] = int_values

        if values_only:
            # this can be further optimized - we don't need to calculate actor head/core just to get values
            return result

        # first core output corresponds to the actor
        actor_decoder_output = self.actor_decoder(core_outputs[0])
        action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_decoder_output)

        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result


def default_make_actor_critic_func(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    from sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()

    if cfg.cleanrl_actor_critic:
        return CleanRLActorCritic(model_factory, obs_space, action_space, cfg)
    elif cfg.actor_critic_share_weights:
        return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
    else:
        return ActorCriticSeparateWeights(model_factory, obs_space, action_space, cfg)


def create_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    # check if user specified custom actor/critic creation function
    from sample_factory.algo.utils.context import global_model_factory

    make_actor_critic_func = global_model_factory().make_actor_critic_func
    return make_actor_critic_func(cfg, obs_space, action_space)
