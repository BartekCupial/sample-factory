from typing import Optional

import minihack  # NOQA: F401
from minihack.envs import (
    boxohack,
    corridor,
    exploremaze,
    fightcorridor,
    hidenseek,
    keyroom,
    mazewalk,
    memento,
    minigrid,
    river,
    room,
    skills_lava,
    skills_quest,
    skills_simple,
    skills_wod,
    sokoban,
)
from nle import nethack

from sample_factory.algo.utils.gymnasium_utils import patch_non_gymnasium_env
from sample_factory.utils.utils import is_module_available, log
from sf_examples.minihack.utils.gym_compatibility import GymV21CompatibilityV0
from sf_examples.minihack.utils.timelimit import TimeLimit


def minihack_available():
    return is_module_available("minihack")


MINIHACK_ENVS = dict(
    # MiniHack Room
    small_room=room.MiniHackRoom5x5,
    small_room_random=room.MiniHackRoom5x5Random,
    small_room_dark=room.MiniHackRoom5x5Dark,
    small_room_monster=room.MiniHackRoom5x5Monster,
    small_room_trap=room.MiniHackRoom5x5Trap,
    small_room_ultimate=room.MiniHackRoom5x5Ultimate,
    big_room=room.MiniHackRoom15x15,
    big_room_random=room.MiniHackRoom15x15Random,
    big_room_dark=room.MiniHackRoom15x15Dark,
    big_room_monster=room.MiniHackRoom15x15Monster,
    big_room_trap=room.MiniHackRoom15x15Trap,
    big_room_ultimate=room.MiniHackRoom15x15Ultimate,
    # MiniHack Corridor
    corridor2=corridor.MiniHackCorridor2,
    corridor3=corridor.MiniHackCorridor3,
    corridor5=corridor.MiniHackCorridor5,
    # MiniHack KeyRoom
    keyroom_small_fixed=keyroom.MiniHackKeyRoom5x5Fixed,
    keyroom_small=keyroom.MiniHackKeyRoom5x5,
    keyroom_small_dark=keyroom.MiniHackKeyRoom5x5Dark,
    keyroom_big=keyroom.MiniHackKeyRoom15x15,
    keyroom_big_dark=keyroom.MiniHackKeyRoom15x15Dark,
    # MiniHack MazeWalk
    mazewalk_small=mazewalk.MiniHackMazeWalk9x9,
    mazewalk_small_mapped=mazewalk.MiniHackMazeWalk9x9Premapped,
    mazewalk_big=mazewalk.MiniHackMazeWalk15x15,
    mazewalk_big_mapped=mazewalk.MiniHackMazeWalk15x15Premapped,
    mazewalk_huge=mazewalk.MiniHackMazeWalk45x19,
    mazewalk_huge_mapped=mazewalk.MiniHackMazeWalk45x19Premapped,
    # MiniHack Fight Corridor
    fight_corridor=fightcorridor.MiniHackFightCorridor,
    fight_corridor_dark=fightcorridor.MiniHackFightCorridorDark,
    # MiniHack River
    river=river.MiniHackRiver,
    river_lava=river.MiniHackRiverLava,
    river_monster=river.MiniHackRiverMonster,
    river_monsterlava=river.MiniHackRiverMonsterLava,
    river_narrow=river.MiniHackRiverNarrow,
    # MiniHack Memento
    memento_short=memento.MiniHackMementoShortF2,
    memento=memento.MiniHackMementoF2,
    memento_hard=memento.MiniHackMementoF4,
    # MiniHack Hide&Seek
    hidenseek=hidenseek.MiniHackHideAndSeek,
    hidenseek_mapped=hidenseek.MiniHackHideAndSeekMapped,
    hidenseek_lava=hidenseek.MiniHackHideAndSeekLava,
    hidenseek_big=hidenseek.MiniHackHideAndSeekBig,
    # MiniHack ExploreMaze
    explore_easy=exploremaze.MiniHackExploreMazeEasy,
    explore_easy_map=exploremaze.MiniHackExploreMazeEasyMapped,
    explore_hard=exploremaze.MiniHackExploreMazeHard,
    explore_hard_map=exploremaze.MiniHackExploreMazeHardMapped,
    # MiniHack MultiRooms
    multiroom_2=minigrid.MiniHackMultiRoomN2,
    multiroom_4=minigrid.MiniHackMultiRoomN4,
    multiroom_6=minigrid.MiniHackMultiRoomN6,
    multiroom_2_locked=minigrid.MiniHackMultiRoomN2Locked,
    multiroom_4_locked=minigrid.MiniHackMultiRoomN4Locked,
    multiroom_6_locked=minigrid.MiniHackMultiRoomN6Locked,
    multiroom_2_lava=minigrid.MiniHackMultiRoomN2Lava,
    multiroom_4_lava=minigrid.MiniHackMultiRoomN4Lava,
    multiroom_6_lava=minigrid.MiniHackMultiRoomN6Lava,
    multiroom_2_monster=minigrid.MiniHackMultiRoomN2Monster,
    multiroom_4_monster=minigrid.MiniHackMultiRoomN4Monster,
    multiroom_6_monster=minigrid.MiniHackMultiRoomN6Monster,
    multiroom_2_extreme=minigrid.MiniHackMultiRoomN2Extreme,
    multiroom_4_extreme=minigrid.MiniHackMultiRoomN4Extreme,
    multiroom_6_extreme=minigrid.MiniHackMultiRoomN6Extreme,
    # MiniHack Boxoban
    boxoban_unfiltered=boxohack.MiniHackBoxobanUnfiltered,
    boxoban_hard=boxohack.MiniHackBoxobanHard,
    boxoban_medium=boxohack.MiniHackBoxobanMedium,
    # MiniHack Simple Skills
    mini_eat=skills_simple.MiniHackEat,
    mini_pray=skills_simple.MiniHackPray,
    mini_sink=skills_simple.MiniHackSink,
    mini_read=skills_simple.MiniHackRead,
    mini_zap=skills_simple.MiniHackZap,
    mini_puton=skills_simple.MiniHackPutOn,
    mini_wear=skills_simple.MiniHackWear,
    mini_wield=skills_simple.MiniHackWield,
    mini_locked=skills_simple.MiniHackLockedDoor,
    # MiniHack Simple Skills (Fixed versions)
    mini_eat_fixed=skills_simple.MiniHackEatFixed,
    mini_pray_fixed=skills_simple.MiniHackPrayFixed,
    mini_sink_fixed=skills_simple.MiniHackSinkFixed,
    mini_read_fixed=skills_simple.MiniHackReadFixed,
    mini_zap_fixed=skills_simple.MiniHackZapFixed,
    mini_puton_fixed=skills_simple.MiniHackPutOnFixed,
    mini_wear_fixed=skills_simple.MiniHackWearFixed,
    mini_wield_fixed=skills_simple.MiniHackWieldFixed,
    mini_locked_fixed=skills_simple.MiniHackLockedDoorFixed,
    # MiniHack Simple Skills (Fixed versions)
    mini_eat_distr=skills_simple.MiniHackEatDistr,
    mini_pray_distr=skills_simple.MiniHackPrayDistr,
    mini_sink_distr=skills_simple.MiniHackSinkDistr,
    mini_read_distr=skills_simple.MiniHackReadDistr,
    mini_zap_distr=skills_simple.MiniHackZapDistr,
    mini_puton_distr=skills_simple.MiniHackPutOnDistr,
    mini_wear_distr=skills_simple.MiniHackWearDistr,
    mini_wield_distr=skills_simple.MiniHackWieldDistr,
    # WoD
    wod_easy=skills_wod.MiniHackWoDEasy,
    wod_medium=skills_wod.MiniHackWoDMedium,
    wod_hard=skills_wod.MiniHackWoDHard,
    wod_pro=skills_wod.MiniHackWoDPro,
    # MiniHack Lava Crossing
    lava=skills_lava.MiniHackLC,
    lava_lev=skills_lava.MiniHackLCLevitate,
    lava_lev_potion_inv=skills_lava.MiniHackLCLevitatePotionInv,
    lava_lev_potion_pick=skills_lava.MiniHackLCLevitatePotionPickup,
    lava_lev_ring_inv=skills_lava.MiniHackLCLevitateRingInv,
    lava_lev_ring_pick=skills_lava.MiniHackLCLevitateRingPickup,
    # MiniHack Quest
    quest_easy=skills_quest.MiniHackQuestEasy,
    quest_medium=skills_quest.MiniHackQuestMedium,
    quest_hard=skills_quest.MiniHackQuestHard,
    # MiniHack Sokoban
    sokoban_1a=sokoban.MiniHackSokoban1a,
    sokoban_1b=sokoban.MiniHackSokoban1b,
    sokoban_2a=sokoban.MiniHackSokoban2a,
    sokoban_2b=sokoban.MiniHackSokoban2b,
    sokoban_3a=sokoban.MiniHackSokoban3a,
    sokoban_3b=sokoban.MiniHackSokoban3b,
    sokoban_4a=sokoban.MiniHackSokoban4a,
    sokoban_4b=sokoban.MiniHackSokoban4b,
)


def minihack_env_by_name(name):
    if name in MINIHACK_ENVS.keys():
        return MINIHACK_ENVS[name]
    raise Exception("Unknown MiniHack env")


def make_minihack_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    env_class = minihack_env_by_name(env_name)

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

    env = env_class(**kwargs)

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
