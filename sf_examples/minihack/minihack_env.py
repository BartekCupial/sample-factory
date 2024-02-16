import json
from typing import Optional

import gym
import minihack  # NOQA: F401
from nle import nethack

from sample_factory.algo.utils.gymnasium_utils import patch_non_gymnasium_env
from sample_factory.utils.utils import is_module_available, log
from sf_examples.minihack.utils.gym_compatibility import GymV21CompatibilityV0
from sf_examples.minihack.utils.timelimit import TimeLimit


def minihack_available():
    return is_module_available("minihack")


class MiniHackSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


def get_minihack_env_specs():
    specs = gym.envs.registry.all()

    return [
        MiniHackSpec("_".join(spec.id.split("-")[:-1]), spec.id) for spec in specs if spec.id.startswith("MiniHack")
    ]


MINIHACK_ENVS = get_minihack_env_specs()


def minihack_env_by_name(name):
    for cfg in MINIHACK_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown MiniHack env")


def make_minihack_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    mujoco_spec = minihack_env_by_name(env_name)

    observation_keys = (
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        # ALSO AVAILABLE (OFF for speed)
        # "specials",
        # "colors",
        # "chars",
        # "glyphs",
        # "inv_glyphs",
        # "inv_strs",
        # "inv_letters",
        # "inv_oclasses",
    )

    actions = tuple(nethack.CompassDirection) + (nethack.Command.OPEN,)

    kwargs = dict(
        observation_keys=observation_keys,
        actions=actions,
        character=cfg.character,
        max_episode_steps=cfg.max_episode_steps,
        penalty_step=cfg.penalty_step,
        penalty_time=cfg.penalty_time,
        penalty_mode=cfg.fn_penalty_step,
        savedir=cfg.savedir,
        save_ttyrec_every=cfg.save_ttyrec_every,
    )

    env = gym.make(mujoco_spec.env_id, **kwargs)

    # wrap NLE with timeout
    env = TimeLimit(env)

    # convert gym env to gymnasium one, due to issues with render NLE in reset
    gymnasium_env = GymV21CompatibilityV0(env=env)
    # preserving potential multi-agent env attributes
    if hasattr(env, "num_agents"):
        gymnasium_env.num_agents = env.num_agents
    if hasattr(env, "is_multiagent"):
        gymnasium_env.is_multiagent = env.is_multiagent
    env = gymnasium_env

    env = patch_non_gymnasium_env(env)

    if render_mode:
        env.render_mode = render_mode

    return env
