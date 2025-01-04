from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import torch
import torch.nn.functional as F
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
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


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

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        raise NotImplementedError()

    def forward(
        self, normalized_obs_dict, rnn_states, values_only: bool = False, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
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

        self.decoder = model_factory.make_model_decoder_func(cfg, self.core.get_out_size())
        decoder_out_size: int = self.decoder.get_out_size()

        self.critic_linear = nn.Linear(decoder_out_size, 1)
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(
            decoder_output, action_mask
        )

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(
        self, normalized_obs_dict, rnn_states, values_only=False, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True, action_mask=action_mask)
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

        self.critic_linear = nn.Linear(self.critic_decoder.get_out_size(), 1)
        self.action_parameterization = self.get_action_parameterization(self.critic_decoder.get_out_size())

        self.apply(self.initialize_weights)

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

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        # second core output corresponds to the critic
        critic_decoder_output = self.critic_decoder(core_outputs[1])
        values = self.critic_linear(critic_decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            # this can be further optimized - we don't need to calculate actor head/core just to get values
            return result

        # first core output corresponds to the actor
        actor_decoder_output = self.actor_decoder(core_outputs[0])
        action_distribution_params, self.last_action_distribution = self.action_parameterization(
            actor_decoder_output, action_mask
        )

        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(
        self, normalized_obs_dict, rnn_states, values_only=False, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True, action_mask=action_mask)
        result["new_rnn_states"] = new_rnn_states
        return result


INSTRUCTION_PROMPT = """
You are an agent playing MiniHack. The following are the possible high-level strategies you can take in the game, followed by a short description of each strategy:

{skill_list}

Each observation in the game is character-based. Here is a legend for what each character represents in the observation:
    @: the player
    #: a corridor
    +: a closed door
    |: a vertical wall
    -: a horizontal wall
    .: the floor
    <: stairs leading up
    >: stairs leading down

Please output the strategy you would like to take when prompted with an observation in the following format:
STRATEGY: <your_strategy> 
Note that you can only pick from the strategies given above.
"""


class LMActorCriticSeparateWeights(ActorCriticSeparateWeights):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(model_factory, obs_space, action_space, cfg)

        skill_list = ""
        for _, s in enumerate(cfg.strategies, 1):
            skill_list += f"- {s}\n"
        self.system_prompt = INSTRUCTION_PROMPT.format(skill_list=skill_list)

        self.critic_encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.critic_core = model_factory.make_model_core_func(cfg, self.critic_encoder.get_out_size())

        self.encoders = [self.critic_encoder]
        self.cores = [self.critic_core]

        self.core_func = self._core_rnn if self.cfg.use_rnn else self._core_empty

        self.critic_decoder = model_factory.make_model_decoder_func(cfg, self.critic_core.get_out_size())
        self.decoders = [self.critic_decoder]

        self.critic_linear = nn.Linear(self.critic_decoder.get_out_size(), 1)
        self.action_parameterization = self.get_action_parameterization(self.critic_decoder.get_out_size())

        self.apply(self.initialize_weights)

        # NOTE: needs to come *after* weight initialization so pre-trained weights aren't lost!
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_name)
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # TODO: I'm still not entirely sure if this is the right thing to do

        self.actor = AutoModelForCausalLM.from_pretrained(
            cfg.lm_model_name,
            # attn_implementation="flash_attention_2", # NOTE: messes up generations for some reason - turn off for now
            torch_dtype=torch.bfloat16,
        )

        # This one is kept frozen
        self.base_actor = AutoModelForCausalLM.from_pretrained(
            cfg.lm_model_name,
            # attn_implementation="flash_attention_2", # NOTE: messes up generations for some reason - turn off for now
            torch_dtype=torch.bfloat16,
        )

        self.generation_config = GenerationConfig(
            bos_token_id=128000,
            do_sample=True,
            eos_token_id=[128001, 128008, 128009],
            temperature=0.7,
            top_p=0.9,
        )

        self.pad_token_id = self.actor.config.eos_token_id[0]

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        # second core output corresponds to the critic
        critic_decoder_output = self.critic_decoder(core_outputs[0])
        values = self.critic_linear(critic_decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            # this can be further optimized - we don't need to calculate actor head/core just to get values
            return result

        return result

    def _actions_and_logits(self, tokenized_messages):
        generated_sequence = self.actor.generate(
            input_ids=tokenized_messages["input_ids"],
            attention_mask=tokenized_messages["attention_mask"],
            max_new_tokens=self.cfg.lm_max_act_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            # **generate_kwargs # TODO: check if anything else needed here
        )

        logits = generated_sequence.scores
        generated_tokens = generated_sequence.sequences[:, tokenized_messages["input_ids"].shape[1] :]

        return generated_tokens, logits

    def _log_probs_from_logits(self, logits, generated_tokens):
        logits = torch.stack(logits)  # T x B x V
        logits = logits.permute(1, 0, 2)  # Rearrange to B x T x V
        log_softmaxed_logits = F.log_softmax(logits, dim=-1)  # Shape: B x T x V
        gathered_log_probs = log_softmaxed_logits.gather(
            dim=2, index=generated_tokens.unsqueeze(-1)
        )  # Shape: B x T x 1
        gathered_log_probs = gathered_log_probs.squeeze(-1)  # Shape: B x T

        padding_mask = generated_tokens != self.pad_token_id  # Shape: B x T
        masked_log_probs = gathered_log_probs.masked_fill(~padding_mask, 0.0)

        total_log_prob = masked_log_probs.sum(dim=1)  # Shape: B

        return total_log_prob

    def _pad_generated_tokens(self, generated_tokens):
        # pad to max so that all actions have the same number of tokens
        cur_length = generated_tokens.shape[1]
        max_length = self.cfg.lm_max_act_tokens
        if cur_length < max_length:
            padding = torch.full(
                (generated_tokens.shape[0], max_length - cur_length),
                self.pad_token_id,
                dtype=generated_tokens.dtype,
                device=generated_tokens.device,
            )
            generated_tokens = torch.cat([generated_tokens, padding], dim=1)

        return generated_tokens

    def forward(
        self, normalized_obs_dict, rnn_states, values_only=False, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True, action_mask=action_mask)

        # (1) generate actions and (2) record log probabilities
        if not values_only:
            tokenized_messages = {
                "input_ids": normalized_obs_dict["tokenized_tty_chars_input_ids"].long(),
                "attention_mask": normalized_obs_dict["tokenized_tty_chars_attn_mask"].long(),
            }
            generated_tokens, logits = self._actions_and_logits(tokenized_messages)
            log_probs = self._log_probs_from_logits(logits, generated_tokens)
            generated_tokens = self._pad_generated_tokens(generated_tokens)

            # NOTE: decoding is only for debugging
            # generated_tokens_list = generated_tokens.cpu().numpy().tolist()
            # for sequence in generated_tokens_list:
            #     text = self.tokenizer.decode(
            #         sequence,
            #         skip_special_tokens=True,
            #         clean_up_tokenization_spaces=True,
            #     )
            #     print(text)

            result["log_prob_actions"] = log_probs
            result["actions"] = generated_tokens
            # NOTE: action logits aren't used for LM model, so we just set them to zeros
            result["action_logits"] = torch.zeros((log_probs.shape[0], self.action_space.n), device=log_probs.device)

        result["new_rnn_states"] = new_rnn_states
        return result

    def log_prob(self, obs, actions, base_model: bool = False):
        model = self.actor if not base_model else self.base_actor

        input_ids = torch.cat([obs["tokenized_tty_chars_input_ids"], actions], dim=1).long()
        attn_mask = torch.cat([obs['tokenized_tty_chars_attn_mask'], torch.ones_like(actions)], dim=1).long()

        # forward pass
        position_ids = attn_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn_mask == 0, 1)
        x = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "position_ids": position_ids,
        }
        outputs = model(**x)

        # TODO: potentially move this to __init__
        model._prepare_special_tokens(self.generation_config, True, input_ids.device)

        logits_processor = model._get_logits_processor(
            generation_config=self.generation_config,
            input_ids_seq_length=input_ids.shape[1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=[],
            device=input_ids.device,
            model_kwargs=None,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        )

        # Take only the newly generated tokens
        logits = outputs.logits[:, obs["tokenized_tty_chars_input_ids"].shape[1] - 1 : -1].float() 

        for t in range(logits.shape[1]):
            logits[:, t, :] = logits_processor(input_ids, logits[:, t, :])

        log_softmaxed_logits = F.log_softmax(logits, dim=-1)  # Shape: B x T x V
        gathered_log_probs = log_softmaxed_logits.gather(dim=2, index=actions.long().unsqueeze(-1))  # Shape: B x T x 1
        gathered_log_probs = gathered_log_probs.squeeze(-1)  # Shape: B x T

        padding_mask = actions.long() != self.pad_token_id
        masked_log_probs = gathered_log_probs.masked_fill(~padding_mask, 0.0)

        return masked_log_probs.sum(dim=1) # Shape: B

def default_make_actor_critic_func(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    from sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()
    obs_space = obs_space_without_action_mask(obs_space)

    if cfg.actor_critic_share_weights:
        return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
    else:
        return ActorCriticSeparateWeights(model_factory, obs_space, action_space, cfg)


def create_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    # check if user specified custom actor/critic creation function
    from sample_factory.algo.utils.context import global_model_factory

    make_actor_critic_func = global_model_factory().make_actor_critic_func
    return make_actor_critic_func(global_model_factory(), obs_space, action_space, cfg)
    # return make_actor_critic_func(cfg, obs_space, action_space)


def obs_space_without_action_mask(obs_space: ObsSpace) -> ObsSpace:
    if isinstance(obs_space, gym.spaces.Dict) and "action_mask" in obs_space.spaces:
        spaces = obs_space.spaces.copy()
        del spaces["action_mask"]
        obs_space = gym.spaces.Dict(spaces)

    return obs_space
