from pathlib import Path

import gymnasium as gym


class TtyrecInfoWrapper(gym.Wrapper):
    def step(self, action):
        ttyrec = self.env.unwrapped.nethack._ttyrec
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            info["episode_extra_stats"]["ttyrecname"] = Path(ttyrec).name

        return obs, reward, terminated, truncated, info
