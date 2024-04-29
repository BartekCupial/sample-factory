import torch
from vllm import LLM, SamplingParams

from sample_factory.algo.utils.tensor_dict import TensorDict
from sf_examples.nethack_text.models.llm import PolicyLLM
from sf_examples.nethack_text.utils.wrappers.tokenizer import NLETokenizer


class VLLM(PolicyLLM):
    def __init__(self, cfg, obs_space, action_space):
        super().__init__(cfg, obs_space, action_space)

        self.model = LLM(model=cfg.model_path, max_num_batched_tokens=cfg.max_num_batched_tokens)
        self.sampling_params = SamplingParams(
            max_tokens=4096,
            seed=cfg.seed,
            stop_token_ids=[NLETokenizer.nethack_obs_start_token_id],
        )

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        # TODO: vllm doesn't support AMD cpu
        device = "cuda"
        return device

    def forward(self, normalized_obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        bs, seq_len = normalized_obs_dict["prompt"].shape

        prompts = normalized_obs_dict["prompt"]
        attention_masks = normalized_obs_dict["attention_mask"]
        short_prompts = []
        for prompt, mask in zip(prompts, attention_masks):
            short_prompts.append(prompt[mask.bool()].tolist())

        outputs = self.model.generate(
            sampling_params=self.sampling_params,
            prompt_token_ids=short_prompts,
            use_tqdm=False,
        )

        actions = torch.ones((bs, self.action_space.shape[0])) * -1
        # embed outputs into padded actions
        for i, output in enumerate(outputs):
            # output.outputs[0].cumulative_logprob
            token_ids = output.outputs[0].token_ids
            actions[i, -len(token_ids) :] = torch.tensor(token_ids)
        action_logits = torch.zeros((bs, self.action_space.shape[0] * 2))
        log_prob_actions = torch.zeros((bs, 1))
        values = torch.zeros((bs, 1))

        result = TensorDict(
            actions=actions,
            action_logits=action_logits,
            log_prob_actions=log_prob_actions,
            values=values,
            new_rnn_states=rnn_states,
        )

        return result
