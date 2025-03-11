from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.misc import EPISODIC
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import log, static_vars

from typing import Dict, Optional
import cv2

import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from collections import deque


def make_plot(y):
    num_rooms, num_steps = y.shape
    x = np.linspace(0, num_steps, num_steps)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(x, y, labels=[f"Room {i+1}" for i in range(num_rooms)])
    ax.legend(loc="upper left")
    ax.set_title("Room Visitation Over Time")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Proportion of Visits")
    return fig


@static_vars(stackplot=dict(), heatmaps=dict())
def montezuma_extra_episodic_stats_processing(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
    episode_stats = msg[EPISODIC].get("episode_extra_stats", {})

    heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    stackplot = montezuma_extra_episodic_stats_processing.stackplot
    
    if policy_id not in heatmaps:
        heatmaps[policy_id] = dict()

    if policy_id not in stackplot:
        stackplot[policy_id] = []
    
    for stat_key, stat_value in episode_stats.items():
        if "heatmap" in stat_key:
            room_id = stat_key.split("_")[1]

            if room_id not in heatmaps[policy_id]:
                heatmaps[policy_id][room_id] = []

            heatmaps[policy_id][room_id].append(stat_value)

        elif "visitation_frequency" == stat_key:
            stackplot[policy_id].append(stat_value)

def montezuma_extra_summaries(runner: Runner, policy_id: PolicyID, env_steps: int, summary_writer: SummaryWriter) -> None:
    heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    cfg = runner.cfg
    # saved = montezuma_extra_summaries.saved

    if policy_id not in heatmaps:
        return

    log.debug(f"Hello from the heatmap saver! Env_steps is {env_steps}")
    
    policy_avg_stats = runner.policy_avg_stats

    # we can access Learners like this
    # learner = runner.learners[policy_id].learner
    
    for room_id in heatmaps[policy_id].keys():
        mean_heatmap = np.mean(heatmaps[policy_id][room_id], axis=0)

        if env_steps > 100_000:
            log.debug(f"Watch out, saving!!")

            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.imshow(mean_heatmap, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Heatmap for Room {room_id}')

            try:
                # we need to save normalized heatmaps, they look much better!
                normalized_heatmap = mean_heatmap.astype(np.float32)
                normalized_heatmap = (normalized_heatmap - normalized_heatmap.min()) / (np.ptp(normalized_heatmap) + 1e-8)
                normalized_heatmap = np.uint8(255 * normalized_heatmap)

                # Save the heatmap as a PNG file:
                cv2.imwrite("/homeplaceholder/images/heatmap_{room_id}.png", normalized_heatmap)
            except Exception as e:
                log.debug(f"Error: {e}")

            # this will log to wandb
            # summary_writer.add_image(f"heatmaps/room_{room_id}", mean_heatmap[np.newaxis, :], env_steps)

            # doesn't work for now -- adding fig with writer not implemented
            # summary_writer.add_image(f"heatmaps/room_{room_id}", fig, env_steps)
            plt.close(fig)

            log.debug(f"Saved a heatmap")

        target_objective_stat = "montezuma_heatmaps_{room_id}"

        if target_objective_stat not in policy_avg_stats:
            policy_avg_stats[target_objective_stat] = [deque(maxlen=1) for _ in range(cfg.num_policies)]
        policy_avg_stats[target_objective_stat][policy_id].append(mean_heatmap)