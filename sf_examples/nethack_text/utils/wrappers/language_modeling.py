from gym import Wrapper, spaces
from nle.env import NLE
from nle.nethack.actions import ACTIONS
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv

from sf_examples.nethack_text.utils.action_textmap import (
    nle_action_textmap,
    nle_comp_preqs,
    nle_obs_preqs,
    special_tokens_interaction_history,
)
from sf_examples.nethack_text.utils.instruction_encode_templates import encode_instruction_example


class NLELMWrapper(Wrapper):
    def __init__(
        self,
        env,
        prediction_target="action",
        observation=False,
        kphist=False,
        random_template=True,
        include_interleave_in_prompt=False,
    ):
        super().__init__(env)
        assert isinstance(env, NLE), "Only NLE environments are supported"

        self._kphist = kphist
        self._observation = observation
        # Space is meant to be abstract!
        # self.action_space = spaces.Space()
        # self.observation_space = spaces.Dict(
        #     {
        #         "text_glyphs": spaces.Space(),
        #         "text_message": spaces.Space(),
        #         "text_blstats": spaces.Space(),
        #         "text_inventory": spaces.Space(),
        #         "text_cursor": spaces.Space(),
        #     }
        # )
        self._instruction = ""

        self._interleaving_token = special_tokens_interaction_history["action"]
        self._final_completion_prefix = None
        self._hist_preq = nle_obs_preqs["prev_action_seq"]
        self.random_template = random_template

        self._action_map = {nle_action_textmap[str(action)]: i for i, action in enumerate(ACTIONS)}  # pm change_here
        self._reverse_action_map = {v: k for (k, v) in self._action_map.items()}
        self._nle_language = NLELanguageObsv()
        self._obs_preqs = nle_obs_preqs
        self._comp_preqs = nle_comp_preqs

        self.include_interleave_in_prompt = include_interleave_in_prompt

    @property
    def action_map(self):
        return self._action_map

    @property
    def reverse_action_map(self):
        return self._reverse_action_map

    @property
    def nle_language(self):
        return self._nle_language

    @property
    def obs_preqs(self):
        return self._obs_preqs

    @property
    def comp_preqs(self):
        return self._comp_preqs

    @property
    def instruction(self):
        return self._instruction

    @property
    def interleaving_token(self):
        return self._interleaving_token

    @property
    def spec(self):
        return self.env.spec

    def action_map_fn(self, action):
        if isinstance(action, str):
            return self.action_map[action]
        else:
            return self._reverse_action_map[action]

    def strategy_map_fn(self, strategy):
        return (
            strategy.replace("visit and search", "open_visit_search")
            .replace(" ", "_")
            .replace("fight", "fight2")
            .replace("explore", "explore_gather_identify")
        )

    def nle_obsv_to_language(self, nle_obsv):
        """Translate NLE Observation into a language observation.
        Args:
            nle_obsv (dict): NLE observation from the base environment
        Returns:
            (dict): language observation
        """
        glyphs = nle_obsv["glyphs"]
        blstats = nle_obsv["blstats"]
        tty_cursor = nle_obsv["tty_cursor"]
        inv_strs = nle_obsv["inv_strs"]
        inv_letters = nle_obsv["inv_letters"]
        tty_chars = nle_obsv["tty_chars"]
        return {
            "txt_glyphs": self.nle_language.text_glyphs(glyphs, blstats).decode("latin-1"),
            "txt_message": self.nle_language.text_message(tty_chars).decode("latin-1"),
            "txt_blstats": self.nle_language.text_blstats(blstats).decode("latin-1"),
            "txt_inventory": self.nle_language.text_inventory(inv_strs, inv_letters).decode("latin-1"),
            "txt_cursor": self.nle_language.text_cursor(glyphs, blstats, tty_cursor).decode("latin-1"),
        }

    def promptify_nle_obsv(self, nle_obsv, history=None):
        obs = self.nle_obsv_to_language(nle_obsv)

        if self._observation:
            query = "\n".join(
                [
                    "%s[\n%s\n]" % (self.obs_preqs[key], obs[key])
                    for key in (
                        "txt_blstats",
                        "txt_glyphs",
                        "txt_message",
                        "txt_inventory",
                        "txt_cursor",
                    )
                ]
            )
        else:
            inter = "\n"
            query = inter.join(
                [
                    "%s\n%s" % (self.obs_preqs[key], obs[key])
                    for key in (
                        "txt_blstats",
                        "txt_glyphs",
                        "txt_message",
                        "txt_inventory",
                        "txt_cursor",
                    )
                ]
            )

        if not self._kphist:
            if history == " ":
                query = "%s\n" % (self._hist_preq,) + "\n\n" + query
            elif history is not None:
                query = "%s\n%s" % (self._hist_preq, history) + "\n\n" + query
        else:
            if history == []:
                query = "%s\n" % (self._hist_preq,) + "\n\n" + query
            elif history is not None:
                prev_action_seqs = "".join(
                    ["%s%s" % (self._comp_preqs["action"], prev_action) for prev_action in history]
                )
                query = "%s\n%s" % (self._hist_preq, prev_action_seqs) + "\n\n" + query

        query = "\n" + query

        out = encode_instruction_example(
            self.instruction,
            query,
            " ",
            random_template=self.random_template,
            eos_token=None,
        )["prompt"]

        if self._final_completion_prefix is not None:
            out = out.strip() + self._final_completion_prefix

        if self.include_interleave_in_prompt:
            out += self._interleaving_token

        return out, obs["txt_glyphs"], obs["txt_message"]

    def reset(self, history=None, **kwargs):
        nle_obsv = self.env.reset(**kwargs)
        prompt, txt_glyphs, txt_message = self.promptify_nle_obsv(nle_obsv, history=history)
        nle_obsv["prompt"] = prompt
        nle_obsv["txt_glyphs"] = txt_glyphs
        nle_obsv["txt_message"] = txt_message
        return nle_obsv

    def step(self, action, output_caction=False, history=None):
        c_action = self.action_map[action]
        nle_obsv, reward, done, info = self.env.step(c_action)
        prompt, txt_glyphs, txt_message = self.promptify_nle_obsv(nle_obsv, history=history)
        nle_obsv["prompt"] = prompt
        nle_obsv["txt_glyphs"] = txt_glyphs
        nle_obsv["txt_message"] = txt_message
        if output_caction:
            return nle_obsv, reward, done, info, c_action
        return nle_obsv, reward, done, info
