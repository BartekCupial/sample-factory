import time
import json
from pathlib import Path
from collections import deque
from typing import Dict, Tuple, List

import pandas as pd
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.algo.sampling.sampling_utils import record_episode_statistics_wrapper_stats


def visualize_policy_inputs(normalized_obs: Dict[str, Tensor]) -> None:
    """
    Display actual policy inputs after all wrappers and normalizations using OpenCV imshow.
    """
    import cv2

    if "obs" not in normalized_obs.keys():
        return

    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]
    if obs.dim() != 3:
        # this function is only for RGB images
        return

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    obs = obs.cpu().numpy()
    # convert to uint8
    obs = cv2.normalize(
        obs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
    )  # this will be different frame-by-frame but probably good enough to give us an idea?
    scale = 5
    obs = cv2.resize(obs, (obs.shape[1] * scale, obs.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("policy inputs", obs)
    cv2.waitKey(delay=1)


def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
    render_start = time.time()

    if cfg.save_video:
        need_video_frame = len(video_frames) < cfg.video_frames or cfg.video_frames < 0 and num_episodes == 0
        if need_video_frame:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    else:
        if not cfg.no_render:
            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
            current_delay = render_start - last_render_start
            time_wait = target_delay - current_delay

            if time_wait > 0:
                # log.info("Wait time %.3f", time_wait)
                time.sleep(time_wait)

            try:
                env.render()
            except (gym.error.Error, TypeError) as ex:
                debug_log_every_n(1000, f"Exception when calling env.render() {str(ex)}")

    return render_start


def _print_eval_summaries(cfg, eval_stats):
    for policy_id in range(cfg.num_policies):
        results = {}
        for key, stat in eval_stats.items():
            stat_value = np.mean(stat[policy_id])

            if "/" in key:
                # custom summaries have their own sections in tensorboard
                avg_tag = key
                min_tag = f"{key}_min"
                max_tag = f"{key}_max"
            elif key in ("reward", "len"):
                # reward and length get special treatment
                avg_tag = f"{key}/{key}"
                min_tag = f"{key}/{key}_min"
                max_tag = f"{key}/{key}_max"
            else:
                avg_tag = f"policy_stats/avg_{key}"
                min_tag = f"policy_stats/avg_{key}_min"
                max_tag = f"policy_stats/avg_{key}_max"

            results[avg_tag] = float(stat_value)

            # for key stats report min/max as well
            if key in ("reward", "true_objective", "len"):
                results[min_tag] = float(min(stat[policy_id]))
                results[max_tag] = float(max(stat[policy_id]))

        log.info(json.dumps(results, indent=4))


def _save_eval_results(cfg, eval_stats):
    for policy_id in range(cfg.num_policies):
        data = {}
        for key, stat in eval_stats.items():
            data[key] = stat[policy_id]

        csv_output_dir = Path(experiment_dir(cfg=cfg))
        if cfg.csv_folder_name is not None:
            csv_output_dir = csv_output_dir / cfg.csv_folder_name
        csv_output_dir.mkdir(exist_ok=True, parents=True)
        csv_output_path = csv_output_dir / f"eval_p{policy_id}.csv"

        data = pd.DataFrame(data)
        data.to_csv(csv_output_path)


def _log_to_wandb(cfg, eval_stats):
    import wandb
    from sample_factory.utils.wandb_utils import init_wandb

    init_wandb(cfg)

    for policy_id in range(cfg.num_policies):
        results = {}
        for key, stat in eval_stats.items():
            stat_value = np.mean(stat[policy_id])

            if "/" in key:
                # custom summaries have their own sections in tensorboard
                avg_tag = key
                min_tag = f"{key}_min"
                max_tag = f"{key}_max"
            elif key in ("reward", "len"):
                # reward and length get special treatment
                avg_tag = f"{key}/{key}"
                min_tag = f"{key}/{key}_min"
                max_tag = f"{key}/{key}_max"
            else:
                avg_tag = f"policy_stats/avg_{key}"
                min_tag = f"policy_stats/avg_{key}_min"
                max_tag = f"policy_stats/avg_{key}_max"

            results[avg_tag] = float(stat_value)

            # for key stats report min/max as well
            if key in ("reward", "true_objective", "len"):
                results[min_tag] = float(min(stat[policy_id]))
                results[max_tag] = float(max(stat[policy_id]))
        
        # Log to wandb
        wandb.log(results, step=policy_id)


def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    cfg.no_render = True
    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    obs, infos = env.reset()
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = []
    num_episodes = 0
    last_episode_duration = 0

    def handle_episode_stats(stats, policy_id=0):
        # heavily based on the `_episodic_stats_handler` from `Runner`
        episode_number = stats.get("episode_number", 0)
        log.debug(
            f"Episode {episode_number} / {cfg.max_num_episodes} ended after {stats['len']:.1f} steps. Return: {stats['reward']:.1f}. True objective {stats['true_objective']:.1f}"
        )

        for _, key, value in iterate_recursively(stats):
            if key not in policy_avg_stats:
                policy_avg_stats[key] = [[] for _ in range(cfg.num_policies)]

            if isinstance(value, np.ndarray) and value.ndim > 0:
                policy_avg_stats[key][policy_id].extend(value)
            else:
                policy_avg_stats[key][policy_id].append(value)

    policy_avg_stats: Dict[str, List[List]] = dict()
    with torch.no_grad():
        while not max_frames_reached(num_frames):
            for _ in range(render_action_repeat):
                last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                obs, rew, terminated, truncated, infos = env.step([env.action_space.sample()])
                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                if "env_steps" in infos[0]["episode_extra_stats"]:
                    last_episode_duration += infos[0]["episode_extra_stats"]["env_steps"]
                else:
                    # multiply by frameskip to get the episode lenghts matching the actual number of simulated steps
                    last_episode_duration += env_info.frameskip if cfg.summaries_use_frameskip else 1

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective)

                        stats = dict(
                            reward=rew,
                            true_objective=true_objective,
                            len=last_episode_duration,
                            episode_number=num_episodes,
                            episode_extra_stats=infos[agent_i].get("episode_extra_stats", dict()),
                        )

                        episode_wrapper_stats = record_episode_statistics_wrapper_stats(infos[agent_i])
                        if episode_wrapper_stats is not None:
                            wrapper_rew, wrapper_len = episode_wrapper_stats
                            stats["RecordEpisodeStatistics_reward"] = wrapper_rew
                            stats["RecordEpisodeStatistics_len"] = wrapper_len

                        handle_episode_stats(stats, policy_id=agent_i)

                        last_episode_duration = 0
                        episode_reward[agent_i] = 0
                        num_episodes += 1

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()

    _print_eval_summaries(cfg, policy_avg_stats)
    _save_eval_results(cfg, policy_avg_stats)
    _log_to_wandb(cfg, policy_avg_stats)