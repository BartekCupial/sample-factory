from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.misc import EPISODIC
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import log, static_vars

from typing import Dict, Optional

import numpy as np
from tensorboardX import SummaryWriter


@static_vars(new_level_returns=dict(), heatmaps=dict(), dummy=dict())
def montezuma_extra_episodic_stats_processing(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
    # heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    # episode_heatmaps = msg[EPISODIC].get("episode_heatmaps", {})
    # log.debug(episode_heatmaps)
    # heatmaps[policy_id] = episode_heatmaps
    dummy = montezuma_extra_episodic_stats_processing.dummy
    heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    episode_stats = msg[EPISODIC].get("episode_extra_stats", {})
    for stat_key, stat_value in episode_stats.items():
        new_level_returns = montezuma_extra_episodic_stats_processing.new_level_returns
        if policy_id not in new_level_returns:
            new_level_returns[policy_id] = dict()

        new_level_returns[policy_id][stat_key] = stat_value

        if "room_count" in episode_stats:
            dummy[policy_id] = episode_stats["room_count"] + np.random.rand(1)
            log.debug(f"dummy {dummy}")

        if "heatmaps" in episode_stats:
            heatmaps[policy_id] = episode_stats["heatmaps"]
            log.debug("Heatmap added!")


@static_vars(saved=False)
def montezuma_extra_summaries(runner: Runner, policy_id: PolicyID, env_steps: int, summary_writer: SummaryWriter) -> None:
    # if env_steps > 0 and env_steps % 500000 == 0:
    #     heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    #     heatmaps = heatmaps[policy_id]
    #     for k, v in heatmaps.items():
    #         summary_writer.add_image(f"policy_stats/montezuma_heatmap_{k}", v, env_steps)
    new_level_returns = montezuma_extra_episodic_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        return
    
    # dummy = montezuma_extra_episodic_stats_processing.dummy
    # dummy = dummy[policy_id]
    # summary_writer.add_scalar(f"heatmaps/dummy", dummy, env_steps)
    saved = montezuma_extra_summaries.saved
    heatmaps = montezuma_extra_episodic_stats_processing.heatmaps
    if policy_id in heatmaps and not saved:
        for room in heatmaps[policy_id].keys():
            summary_writer.add_image(f"heatmaps/room_{room}", heatmaps[policy_id][room], env_steps)
        log.debug(f"Saved!!!")
        saved=True



