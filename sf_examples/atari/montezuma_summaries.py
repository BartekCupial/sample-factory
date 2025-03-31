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
import datetime
import os

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
                heatmaps[policy_id][room_id] = deque(maxlen=runner.cfg.stats_avg)

            heatmaps[policy_id][room_id].append(stat_value)
            # log.debug(f"[CHECK] heatmap has {len(heatmaps)} keys, policy one has {len(heatmaps[policy_id])} keys, room one {room_id} has {len(heatmaps[policy_id][room_id])} keys")

        elif "visitation_frequency" == stat_key:
            stackplot[policy_id].append(stat_value)

    # I don't think it's needed (it seems to be updating correctly without this as well)
    # montezuma_extra_episodic_stats_processing.heatmaps = heatmaps    

@static_vars(last_saved=0, date_time=None)
def montezuma_extra_summaries(runner: Runner, policy_id: PolicyID, env_steps: int, summary_writer: SummaryWriter) -> None:
    if montezuma_extra_summaries.date_time is None:
        montezuma_extra_summaries.date_time = str(datetime.datetime.now().strftime('%d_%m_%Y-%H_%M'))

    last_saved = montezuma_extra_summaries.last_saved
    heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    cfg = runner.cfg

    if policy_id not in heatmaps:
        return

    policy_avg_stats = runner.policy_avg_stats

    # we can access Learners like this
    # learner = runner.learners[policy_id].learner

    if env_steps-last_saved>cfg.heatmap_save_freq:
        for room_id in heatmaps[policy_id].keys():
            mean_heatmap = np.mean(heatmaps[policy_id][room_id], axis=0)
            normalized_heatmap = mean_heatmap.astype(np.float32)
            normalized_heatmap = (normalized_heatmap - normalized_heatmap.min()) / (np.ptp(normalized_heatmap) + 1e-8)
            normalized_heatmap = np.uint8(255 * normalized_heatmap)

            # save the heatmap as a PNG file:
            if cfg.save_heatmaps_locally:
                log.debug(f"Saving heatmaps at /homeplaceholder/images, env_steps is {env_steps}, last_saved at {last_saved}")
                os.makedirs(f"/homeplaceholder/images", exist_ok = True)
                fig, ax = plt.subplots(figsize=(6, 6))
                cax = ax.imshow(normalized_heatmap, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Heatmap for Room {room_id}, env_step {env_steps}')
                fig.savefig(f"/homeplaceholder/images/heatmap_{room_id}__{env_steps}.png")
                plt.close(fig)

            # this will log to wandb
            if cfg.log_heatmaps_to_wandb:
                log.debug(f"Logging heatmaps to wandb, env_steps is {env_steps}, last_saved at {last_saved}")
                summary_writer.add_image(f"heatmaps/room_{room_id}", normalized_heatmap[np.newaxis, :], env_steps)

            # doesn't work for now -- adding fig with writer not implemented
            # summary_writer.add_image(f"heatmaps/room_{room_id}", fig, env_steps)
            montezuma_extra_summaries.last_saved = env_steps

        target_objective_stat = "montezuma_heatmaps_{room_id}"

        if target_objective_stat not in policy_avg_stats:
            policy_avg_stats[target_objective_stat] = [deque(maxlen=1) for _ in range(cfg.num_policies)]
        policy_avg_stats[target_objective_stat][policy_id].append(normalized_heatmap)