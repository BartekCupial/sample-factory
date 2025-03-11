from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.misc import EPISODIC
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import log, static_vars

from typing import Dict, Optional

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
        log.debug(f"policy_id not in heatmaps, setting dict()")
        heatmaps[policy_id] = dict()

    if policy_id not in stackplot:
        log.debug(f"policy_id not in stachplot, setting []")
        stackplot[policy_id] = []
    
    for stat_key, stat_value in episode_stats.items():
        if "heatmap" in stat_key:
            room_id = stat_key.split("_")[1]

            if room_id not in heatmaps[policy_id]:
                heatmaps[policy_id][room_id] = []

            heatmaps[policy_id][room_id].append(stat_value)
            # log.debug(f"Found a heatmap: {stat_value.mean()}, now len is {len(heatmaps[policy_id][room_id])}")

        elif "visitation_frequency" == stat_key:
            stackplot[policy_id].append(stat_value)

# @static_vars(saved=False)
def montezuma_extra_summaries(runner: Runner, policy_id: PolicyID, env_steps: int, summary_writer: SummaryWriter) -> None:
    heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    cfg = runner.cfg
    # saved = montezuma_extra_summaries.saved

    if policy_id not in heatmaps:
        log.debug(f"[EX] Policy id not in heatmaps, returning")
        return

    log.debug(f"Hello from the heatmap saver! Env_steps is {env_steps}")
    
    policy_avg_stats = runner.policy_avg_stats
    
    for room_id in heatmaps[policy_id].keys():
        mean_heatmap = np.mean(heatmaps[policy_id][room_id], axis=0)

        if env_steps > 10_000_000:
            log.debug(f"Watch out, saving!!")

            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.imshow(mean_heatmap, cmap='viridis', interpolation='nearest')
            # fig.colorbar(cax, ax=ax, shrink=0.8, label='Intensity')
            ax.set_title(f'Heatmap for Room {room_id}')
            # ax.set_xlabel('')
            # ax.set_ylabel('')

            # summary_writer.add_image(f"heatmaps/room_{room_id}", mean_heatmap[np.newaxis, :], env_steps)
            summary_writer.add_image(f"heatmaps/room_{room_id}", fig, env_steps)
            plt.close(fig)

            log.debug(f"Saved a heatmap")
            # saved=True

        target_objective_stat = "montezuma_heatmaps_{room_id}"

        if target_objective_stat not in policy_avg_stats:
            policy_avg_stats[target_objective_stat] = [deque(maxlen=1) for _ in range(cfg.num_policies)]
        policy_avg_stats[target_objective_stat][policy_id].append(mean_heatmap)