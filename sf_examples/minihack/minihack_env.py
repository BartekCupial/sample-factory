import json
from typing import Optional

import gym

from sample_factory.utils.utils import is_module_available, log


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
log.info(json.dumps(MINIHACK_ENVS, indent=4))


def minihack_env_by_name(name):
    for cfg in MINIHACK_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown MiniHack env")


def make_minihack_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    mujoco_spec = minihack_env_by_name(env_name)
    env = gym.make(mujoco_spec.env_id, render_mode=render_mode)

    # TODO: add wrappers
    # TODO: observation keys???
    # TODO: reward shaping???

    return env
