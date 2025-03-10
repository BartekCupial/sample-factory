from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.misc import EPISODIC
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import log, static_vars

from typing import Dict, Optional

import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


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
    heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    stackplot = montezuma_extra_episodic_stats_processing.stackplot

    episode_stats = msg[EPISODIC].get("episode_extra_stats", {})
    
    if policy_id not in heatmaps:
        heatmaps[policy_id] = dict()

    if policy_id not in stackplot:
        stackplot[policy_id] = dict()
    
    h = heatmaps[policy_id]
    s = stackplot[policy_id]
    
    for stat_key, stat_value in episode_stats.items():
        if "heatmap" in stat_key:
            pass
        elif "visitation_frequency" in stat_key:
            pass

@static_vars(saved=False)
def montezuma_extra_summaries(runner: Runner, policy_id: PolicyID, env_steps: int, summary_writer: SummaryWriter) -> None:
    # if env_steps > 0 and env_steps % 500000 == 0:
    #     heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    #     heatmaps = heatmaps[policy_id]
    #     for k, v in heatmaps.items():
    #         summary_writer.add_image(f"policy_stats/montezuma_heatmap_{k}", v, env_steps)
    heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    if policy_id not in heatmaps:
        return
    
    for room in heatmaps[policy_id].keys():
        summary_writer.add_image(f"heatmaps/room_{room}", heatmaps[policy_id][room], env_steps)
    log.debug(f"Saved!!!")



