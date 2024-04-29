import difflib

import gym
import gym.spaces
import numpy as np
from transformers import AutoTokenizer

from sf_examples.nethack_text.utils.action_textmap import nle_action_textmap

special_tokens_interaction_history = {
    "action": "<|action|>",
    "observation": "<|observation|>",
}

ACTION_TOKEN = special_tokens_interaction_history["action"]
OBSERVATION_TOKEN = special_tokens_interaction_history["observation"]


class NLETokenizer(gym.Wrapper):
    nethack_obs_start_token_id = 30001
    nethack_obs_end_token_id = 30002
    nethack_obs_start_diff_token_id = 30004
    nethack_pad_token = 30005

    def __init__(self, env: gym.Env, tokenizer, nethack_anchor_every=1, max_ctx_tokens=8000, unroll_length=1):
        super().__init__(env)

        self.unroll_length = unroll_length
        self.nethack_anchor_every = nethack_anchor_every
        self._max_ctx_tokens = max_ctx_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        vocab_size = len(self.tokenizer.get_vocab())

        self._observations = None
        self._actions = None
        self._token_buffer = None
        self._anchor_obs = None

        # self.action_space = gym.spaces.Discrete(vocab_size)
        longest_action = max(map(len, self.tokenizer(list(nle_action_textmap.values())).data["input_ids"]))
        self.action_space = gym.spaces.Box(low=0, high=vocab_size, shape=(longest_action,), dtype=np.int32)
        self.observation_space = gym.spaces.Dict(
            {
                "prompt": gym.spaces.Box(low=0, high=vocab_size, shape=(max_ctx_tokens,), dtype=np.int32),
                "attention_mask": gym.spaces.Box(low=0, high=1, shape=(max_ctx_tokens,), dtype=np.int32),
            }
        )

        self.last_text_action = None
        self.last_text_observation = None

    @staticmethod
    def diff_strings(s1, s2, diffs_delimiter="\n"):
        s1 = s1.splitlines(keepends=False)
        s2 = s2.splitlines(keepends=False)
        differences = difflib.unified_diff(s1, s2, n=0)
        diff_str = ""
        for d in differences:
            if d == "--- \n" or d == "+++ \n" or d[0] == "@":
                continue
            diff_str += d + diffs_delimiter
        return diff_str

    def append_observation(self, observation):
        obs = observation.strip()
        i = len(self._observations)
        self._observations.append(obs)
        assert len(self._observations) == len(self._actions) + 1

        obs = obs.strip()
        if i % self.nethack_anchor_every == 0:
            # Anchor the observation (encode full observation)
            self._anchor_obs = obs
            last_obs = obs
            tokens_obs = self.tokenizer.encode(obs, add_special_tokens=False)
            tokens_obs = [self.nethack_obs_start_token_id] + tokens_obs
        else:
            diff_obs = self.diff_strings(self._anchor_obs, obs)  # todo: implement diff_strings
            last_obs = diff_obs
            tokens_obs = self.tokenizer.encode(diff_obs)
            tokens_obs = [self.nethack_obs_start_diff_token_id] + tokens_obs

        self.last_text_observation = "\033[92m <>" + last_obs + "</>\033[0m"
        # print("\033[92m <>" + last_obs + "</>\033[0m", file=sys.stderr)

        tokens_obs += [self.nethack_obs_end_token_id]

        self._token_buffer.extend(tokens_obs)

    def append_action(self, action):
        # print in red
        action = action.strip()
        self.last_text_action = "\033[91m <>" + action + "</>\033[0m"
        # print("\033[91m <>" + action + "</>\033[0m", file=sys.stderr)
        action = action.strip()
        self._actions.append(action)
        assert len(self._observations) == len(self._actions)
        tokens_action = self.tokenizer.encode(action, add_special_tokens=False)

        self._token_buffer.extend(tokens_action)

    def return_tokenized(self):
        assert self._token_buffer[-1] == self.nethack_obs_end_token_id
        query_tokens = self._token_buffer[-self._max_ctx_tokens :]
        query_tokens = np.array(query_tokens)

        # 0 in attention mask where we have pad tokens
        attention_mask = np.ones_like(query_tokens)
        attention_mask[np.where(query_tokens == self.nethack_pad_token)[0]] = 0

        return query_tokens, attention_mask

    def reset(self, **kwargs):
        self.last_text_action = None
        self.last_text_observation = None

        self._observations = []
        self._actions = []
        self._token_buffer = [self.nethack_pad_token] * self._max_ctx_tokens + [self.tokenizer.bos_token_id]
        self._anchor_obs = None
        obs = self.env.reset(**kwargs)

        self.append_observation(obs["prompt"])
        query_tokens, attention_mask = self.return_tokenized()

        return dict(prompt=query_tokens, attention_mask=attention_mask)

    def preprocess_action(self, token_ids):
        # remove padding
        token_ids = token_ids[token_ids != -1]
        token_ids = token_ids.astype(int)

        suffix = self.tokenizer.decode(token_ids)
        suffix = "<|action|>" + suffix.strip()

        assert self.unroll_length == 1, "unroll_length > 1 not supported yet"

        actions = []
        pred_obs = []
        while len(actions) < self.unroll_length:
            saction = suffix[suffix.find(ACTION_TOKEN) + len(ACTION_TOKEN) :]
            action = saction[: saction.find(OBSERVATION_TOKEN)].replace(ACTION_TOKEN, "").strip()
            if "<" in action:
                action = action[: action.find("<")]
            actions += [action]

            suffix = suffix[suffix.find(OBSERVATION_TOKEN) :]
            obs = suffix[len(OBSERVATION_TOKEN) : suffix.find(ACTION_TOKEN)]
            pred_obs += [obs]
            suffix = suffix[suffix.find(ACTION_TOKEN) :]

        return actions[-1]

    def step(self, action):
        action = self.preprocess_action(action)

        obs, reward, done, info = self.env.step(action)

        self.append_action(action)
        self.append_observation(obs["prompt"])
        query_tokens, attention_mask = self.return_tokenized()

        return dict(prompt=query_tokens, attention_mask=attention_mask), reward, done, info

    def render(self, mode="human", **kwargs):
        print(self.last_text_observation)
        print(self.last_text_action)
        self.env.render(mode, **kwargs)
        print("\n")
