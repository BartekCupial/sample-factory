import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria

from sample_factory.algo.utils.tensor_dict import TensorDict
from sf_examples.nethack_text.models.llm import PolicyLLM
from sf_examples.nethack_text.utils.wrappers.tokenizer import NLETokenizer


class UnrollLengthCriteria(StoppingCriteria):
    def __init__(self, unroll_length, stop_token_id, num_return_sequences):
        assert isinstance(unroll_length, int)
        self.unroll_length = unroll_length
        self.stop_token_id = stop_token_id
        self.counts_per_sequence = torch.zeros((num_return_sequences,))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            if input_ids[i][-1] == self.stop_token_id:
                self.counts_per_sequence[i] += 1
                if self.counts_per_sequence[i] >= self.unroll_length:
                    sequences_should_be_stopped.append(True)
                    continue
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)


class HuggingFace(PolicyLLM):
    def __init__(self, cfg, obs_space, action_space):
        super().__init__(cfg, obs_space, action_space)

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_path)

        self.action_generation_config = GenerationConfig(
            max_new_tokens=4096,
            decoder_start_token_id=0,
            eos_token_id=self.model.config.eos_token_id,
            pad_token=self.model.config.pad_token_id,
            num_beams=cfg.num_beams,
        )
        self.stopping_criteria = UnrollLengthCriteria(
            unroll_length=cfg.unroll_length,
            stop_token_id=NLETokenizer.nethack_obs_start_token_id,
            num_return_sequences=1,
        )

    def model_to_device(self, device):
        self.model.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        device = self.model.device
        return device

    def forward(self, normalized_obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        bs, seq_len = normalized_obs_dict["prompt"].shape

        outputs = self.model.generate(
            normalized_obs_dict["prompt"],
            attention_mask=normalized_obs_dict["attention_mask"],
            generation_config=self.action_generation_config,
            stopping_criteria=[self.stopping_criteria],
        )
        outputs = outputs[..., seq_len:]

        actions = torch.ones((bs, self.action_space.shape[0])) * -1
        # embed outputs into padded actions
        for i, output in enumerate(outputs):
            actions[i, -len(output) :] = output
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
