from pathlib import Path
from typing import Optional

from nle.env.tasks import (
    NetHackChallenge,
    NetHackEat,
    NetHackGold,
    NetHackOracle,
    NetHackScore,
    NetHackScout,
    NetHackStaircase,
    NetHackStaircasePet,
)

from sample_factory.algo.utils.gymnasium_utils import patch_non_gymnasium_env
from sf_examples.nethack_text.utils.wrappers import (
    BlstatsInfoWrapper,
    GymV21CompatibilityV0,
    NLELMWrapper,
    NLETimeLimit,
    NLETokenizer,
    TaskRewardsInfoWrapper,
    TtyrecInfoWrapper,
)

NETHACK_ENVS = dict(
    nethack_staircase=NetHackStaircase,
    nethack_score=NetHackScore,
    nethack_pet=NetHackStaircasePet,
    nethack_oracle=NetHackOracle,
    nethack_gold=NetHackGold,
    nethack_eat=NetHackEat,
    nethack_scout=NetHackScout,
    nethack_challenge=NetHackChallenge,
)


def nethack_env_by_name(name):
    if name in NETHACK_ENVS.keys():
        return NETHACK_ENVS[name]
    else:
        raise Exception("Unknown NetHack env")


def make_nethack_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    env_class = nethack_env_by_name(env_name)

    observation_keys = (
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        "glyphs",
        "inv_strs",
        "inv_letters",
    )

    if cfg.gameloaddir:
        # gameloaddir can be either list or a single path
        if isinstance(cfg.gameloaddir, list):
            # if gameloaddir is a list we will pick only one element from this list
            if env_config:
                # based on env_id
                idx = env_config["env_id"] % len(cfg.gameloaddir)
                gameloaddir = cfg.gameloaddir[idx]
            else:
                # if no env_id use first element
                gameloaddir = cfg.gameloaddir[0]
        else:
            # if gameliaddir is a single path
            assert isinstance(cfg.gameloaddir, (str, Path))
            gameloaddir = cfg.gameloaddir
    else:
        gameloaddir = None

    kwargs = dict(
        character=cfg.character,
        max_episode_steps=cfg.max_episode_steps,
        observation_keys=observation_keys,
        penalty_step=cfg.penalty_step,
        penalty_time=cfg.penalty_time,
        penalty_mode=cfg.fn_penalty_step,
        savedir=cfg.savedir,
        save_ttyrec_every=cfg.save_ttyrec_every,
        gameloaddir=gameloaddir,
    )
    if env_name == "challenge":
        kwargs["no_progress_timeout"] = cfg.no_progress_timeout

    if env_name in ("staircase", "pet", "oracle"):
        kwargs.update(reward_win=cfg.reward_win, reward_lose=cfg.reward_lose)
    # else:  # print warning once
    # warnings.warn("Ignoring cfg.reward_win and cfg.reward_lose")
    if cfg.state_counter is not None:
        kwargs.update(state_counter=cfg.state_counter)

    env = env_class(**kwargs)
    env = NLELMWrapper(env, observation=True, random_template=False, include_interleave_in_prompt=False)
    env = NLETokenizer(
        env,
        tokenizer=cfg.tokenizer_path,
        unroll_length=cfg.unroll_length,
        nethack_anchor_every=cfg.nethack_anchor_every,
        max_ctx_tokens=cfg.max_ctx_tokens,
    )

    # add TimeLimit.truncated to info
    env = NLETimeLimit(env)

    if cfg.add_stats_to_info:
        env = BlstatsInfoWrapper(env)
        env = TaskRewardsInfoWrapper(env)
        if cfg.save_ttyrec_every != 0:
            env = TtyrecInfoWrapper(env)

    # convert gym env to gymnasium one, due to issues with render NLE in reset
    gymnasium_env = GymV21CompatibilityV0(env=env)

    # preserving potential multi-agent env attributes
    if hasattr(env, "num_agents"):
        gymnasium_env.num_agents = env.num_agents
    if hasattr(env, "is_multiagent"):
        gymnasium_env.is_multiagent = env.is_multiagent
    env = gymnasium_env

    env = patch_non_gymnasium_env(env)

    if env_config is not None:
        if env_config["env_id"] == 0 and cfg.debug_render:
            env.render_mode = "human"

    if render_mode:
        env.render_mode = render_mode

    return env
