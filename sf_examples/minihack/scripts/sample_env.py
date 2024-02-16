import sys

from sample_factory.algo.utils.rl_utils import make_dones
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log
from sf_examples.minihack.train_minihack import parse_minihack_args, register_minihack_components


def main():
    register_minihack_components()
    cfg = parse_minihack_args()
    env = create_env(cfg.env, cfg=cfg, render_mode="human")

    for i in range(10):
        env.reset()
        done = False
        j = 0
        while not done:
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            done = make_dones(terminated, truncated)
            j += 1
    log.info("Done!")


if __name__ == "__main__":
    sys.exit(main())
