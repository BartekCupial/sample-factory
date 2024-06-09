# %%
import ast
import glob
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from os.path import join
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nle.dataset as nld
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import wandb
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import EnvInfo, extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict, clone_tensordict, stack_tensordicts
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import ensure_dir_exists, experiment_dir, log
from sf_examples.nethack.algo.learning.learner import DatasetLearner
from sf_examples.nethack.datasets.actions import ACTION_MAPPING
from sf_examples.nethack.datasets.dataset import load_nld_aa_large_dataset
from sf_examples.nethack.datasets.render import render_screen_image
from sf_examples.nethack.datasets.roles import Alignment, Race, Role
from sf_examples.nethack.train_nethack import make_nethack_actor_critic, parse_nethack_args, register_nethack_components

# %%
milestones_path = "/home/bartek/2024-05-29-monk-appo-ks-t-layernorm-pretrain-critic-lr-groups_jnve_3/train_dir/default_experiment/checkpoint_p0/milestones"

# %%
character = "mon-hum-neu-mal"
batch_size = 64
num_workers = 8
device = "cuda"

# %%
if character == "@":
    role, race, align = None, None, None
else:
    role, race, align = character.split("-")[:3]
    role, race, align = Role(role), Race(race), Alignment(align)

dataset = load_nld_aa_large_dataset(
    dataset_name="autoascend",
    data_path="/nle/nld-aa/nle_data",
    db_path="/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db",
    seq_len=32,
    batch_size=batch_size,
    role=role,
    race=race,
    align=align,
)

tp = ThreadPoolExecutor(max_workers=num_workers)

env_name = "challenge"
register_nethack_components()

cfg = parse_nethack_args(
    [
        f"--env={env_name}",
        "--use_pretrained_checkpoint=False",
        "--load_checkpoint_kind=latest",
        "--train_dir=/home/bartek/2024-05-29-monk-appo-ks-t-layernorm-pretrain-critic-lr-groups_jnve_3/train_dir",
        "--teacher_path=train_dir/amzn-AA-BC_pretrained",
        "--dataset_num_workers=16",
        f"--dataset_batch_size={32*42}",
        "--dataset_rollout=32",
        "--dataset_num_splits=1",
        "--restart_behavior=overwrite",
        "--db_path=/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db",
    ]
)

cfg = load_from_checkpoint(cfg)

env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
env_info = extract_env_info(env, cfg)

obs_space = env_info.obs_space
action_space = env.action_space

# %%


def get_dataset_scores(dataset_name, dbfilename=nld.db.DB):
    sql_args = (dataset_name,)

    sql = """
    SELECT games.gameid, games.points
    FROM games
    INNER JOIN datasets ON games.gameid=datasets.gameid
    WHERE datasets.dataset_name=?"""

    with nld.db.connect(dbfilename) as conn:
        scores = dict(list(conn.execute(sql, sql_args)))
    return scores


class CustomLearner:
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_id,
        devuce,
        checkpoint_path,
    ) -> None:
        self.cfg = cfg
        self.env_info = env_info
        self.policy_id = policy_id
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.best_performance = -1e9

        # initialize the Torch modules
        if self.cfg.seed is None:
            log.info("Starting seed is not provided")
        else:
            log.info("Setting fixed seed %d", self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        log.debug("Initializing actor-critic model on device %s", self.device)

        # trainable torch module
        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        log.debug("Created Actor Critic model with architecture:")
        log.debug(self.actor_critic)
        self.actor_critic.model_to_device(self.device)

        self.load_from_checkpoint(self.policy_id)

        dataset_batch_size = self.cfg.dataset_batch_size // self.cfg.dataset_rollout
        self.dataset = self._get_dataset()
        self.dataset_scores = self._get_dataset_scores()
        self.tp = ThreadPoolExecutor(max_workers=self.cfg.dataset_num_workers)

        def _make_sing_iter(dataset, dataset_scores):
            dataset = iter(dataset)

            def _iter():
                prev_actions = np.zeros((dataset_batch_size, 1))
                prev_timestamps = np.ones((dataset_batch_size, 1)) * -1
                prev_scores = np.zeros((dataset_batch_size, 1))

                while True:
                    batch = next(dataset)

                    screen_image = render_screen_image(
                        tty_chars=batch["tty_chars"],
                        tty_colors=batch["tty_colors"],
                        tty_cursor=batch["tty_cursor"],
                        threadpool=self.tp,
                    )
                    batch["screen_image"] = screen_image
                    batch["actions"] = ACTION_MAPPING[batch["keypresses"]]
                    batch["prev_actions"] = np.concatenate([prev_actions, batch["actions"][:, :-1].copy()], axis=1)
                    prev_actions = np.expand_dims(batch["actions"][:, -1].copy(), -1)

                    # dones are broken in NLD-AA, so we just rewrite them with always done at last step
                    # see: https://github.com/facebookresearch/nle/issues/355
                    timestamp_diff = batch["timestamps"] - np.concatenate(
                        [prev_timestamps, batch["timestamps"][:, :-1].copy()], axis=1
                    )
                    # override dones which may or may not be set correctly in dataset
                    batch["done"][:] = 0
                    batch["done"][np.where(timestamp_diff != 1)] = 1
                    prev_timestamps = np.expand_dims(batch["timestamps"][:, -1].copy(), -1)

                    scores_diff = batch["scores"] - np.concatenate(
                        [prev_scores, batch["scores"][:, :-1].copy()], axis=1
                    )
                    # we need to use relu since when trajectory ends we will see negative rewards
                    batch["rewards"] = scores_diff * (scores_diff > 0)
                    batch["clipped_rewards"] = np.clip(batch["rewards"], 0, 10)
                    prev_scores = np.expand_dims(batch["scores"][:, -1].copy(), -1)

                    # add reward to go
                    reward_to_go = (
                        dataset_scores[batch["gameids"].flatten()].reshape(batch["gameids"].shape) - batch["scores"]
                    )
                    batch["reward_to_go"] = reward_to_go

                    # ensure that we don't overrite data
                    normalized_batch = prepare_and_normalize_obs(self.actor_critic, batch)
                    normalized_batch = clone_tensordict(TensorDict(normalized_batch))

                    yield normalized_batch

            return iter(_iter())

        self.rnn_states = [
            torch.zeros((dataset_batch_size, get_rnn_size(self.cfg)), dtype=torch.float32, device=self.device)
            for _ in range(self.cfg.dataset_num_splits)
        ]
        self.idx = 0
        self.prev_idx = 0

        self._iterators = []
        self._results = []
        for _ in range(self.cfg.dataset_num_splits):
            it = _make_sing_iter(self.dataset, self.dataset_scores)
            self._iterators.append(it)
            self._results.append(self.tp.submit(next, it))

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f"checkpoint_p{policy_id}")
        return ensure_dir_exists(checkpoint_dir)

    @staticmethod
    def get_checkpoints(checkpoints_dir, pattern="checkpoint_*"):
        checkpoints = glob.glob(join(checkpoints_dir, pattern))
        return sorted(checkpoints)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning("No checkpoints found")
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                # noinspection PyBroadException
                try:
                    log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
                    return checkpoint_dict
                except Exception:
                    log.exception(f"Could not load from checkpoint, attempt {attempt}")

    def _load_state(self, checkpoint_dict, load_progress=True):
        self.actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)

    def load_from_checkpoint(self, policy_id: PolicyID, load_progress: bool = True) -> None:
        checkpoints = [str(Path(self.checkpoint_dir(self.cfg, policy_id)) / self.checkpoint_path)]
        # name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        # checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern=f"{name_prefix}_*")
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug("Did not load from checkpoint, starting from scratch!")
        else:
            log.debug("Loading model from checkpoint")

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            self._load_state(checkpoint_dict, load_progress=load_progress)

    def _get_dataset(self):
        if self.cfg.character == "@":
            role, race, align = None, None, None
        else:
            role, race, align = self.cfg.character.split("-")[:3]
            role, race, align = Role(role), Race(race), Alignment(align)

        dataset = load_nld_aa_large_dataset(
            dataset_name=self.cfg.dataset_name,
            data_path=self.cfg.data_path,
            db_path=self.cfg.db_path,
            seq_len=self.cfg.dataset_rollout,
            batch_size=self.cfg.dataset_batch_size // self.cfg.dataset_rollout,
            role=role,
            race=race,
            align=align,
        )

        return dataset

    def result(self):
        return self._results[self.idx].result()

    def step(self):
        fut = self.tp.submit(next, self._iterators[self.idx])
        self._results[self.idx] = fut
        self.prev_idx = self.idx
        self.idx = (self.idx + 1) % self.cfg.dataset_num_splits

    def _get_dataset_minibatch(self) -> TensorDict:
        normalized_batch = self.result()
        self.step()
        return normalized_batch

    def _calculate_dataset_outputs(self, mb: TensorDict):
        rnn_state = self.rnn_states[self.prev_idx]

        model_outputs = []
        seq_len = mb["actions"].shape[1]
        for i in range(seq_len):
            # we split the forward since we want to use teacher from kickstarter
            head_outputs = self.actor_critic.forward_head(mb[:, i])
            core_outputs, new_rnn_state = self.actor_critic.forward_core(head_outputs, rnn_state)
            outputs = self.actor_critic.forward_tail(core_outputs, values_only=False, sample_actions=False)

            not_done = (1.0 - mb["done"][:, i].float()).unsqueeze(-1)
            rnn_state = new_rnn_state * not_done
            model_outputs.append(outputs)

        # update rnn_states for next iteration
        self.rnn_states[self.prev_idx] = rnn_state.detach()

        model_outputs = stack_tensordicts(model_outputs, dim=1)

        return model_outputs

    def _get_dataset_scores(self):
        dataset_scores = get_dataset_scores(self.cfg.dataset_name, self.cfg.db_path)

        gameids = np.array(list(map(int, dataset_scores.keys()))).max()
        max_scores = np.zeros(gameids + 1)
        for key, value in dataset_scores.items():
            max_scores[int(key)] = value

        return max_scores


# %%
for checkpoint_path in Path(milestones_path).iterdir():
    learner = CustomLearner(cfg, env_info, 0, "cuda", checkpoint_path)

    from collections import defaultdict

    rewards = defaultdict(lambda: torch.tensor([], device="cuda"))
    clipped_rewards = defaultdict(lambda: torch.tensor([], device="cuda"))
    timestamps = defaultdict(lambda: torch.tensor([], device="cuda"))
    denormalized_values = defaultdict(lambda: torch.tensor([], device="cuda"))
    try:
        while True:
            dataset_mb = learner._get_dataset_minibatch()
            unique_gameids = np.unique(dataset_mb["gameids"].cpu().numpy())
            with torch.no_grad():
                dataset_mb_results = learner._calculate_dataset_outputs(dataset_mb)
            values = dataset_mb_results["values"]
            learner.actor_critic.returns_normalizer(values, denormalize=True)  # in place
            values = values.detach()  # needed because of memory learks

            for gameid in unique_gameids:
                gameid = int(gameid)
                gameid_mask = dataset_mb["gameids"] == gameid
                rewards[gameid] = torch.cat([rewards[gameid], dataset_mb["rewards"][gameid_mask]])
                clipped_rewards[gameid] = torch.cat(
                    [clipped_rewards[gameid], dataset_mb["clipped_rewards"][gameid_mask]]
                )
                timestamps[gameid] = torch.cat([timestamps[gameid], dataset_mb["timestamps"][gameid_mask]])
                denormalized_values[gameid] = torch.cat([denormalized_values[gameid], values[gameid_mask]])
            print(len(unique_gameids))
    except Exception as e:
        print(e)

    sum_rewards = dict(map(lambda kv: (kv[0], torch.sum(kv[1]).item()), rewards.items()))
    sum_clipped_rewards = dict(map(lambda kv: (kv[0], torch.sum(kv[1]).item()), clipped_rewards.items()))

    Path(checkpoint_path.stem).mkdir(exist_ok=True, parents=True)

    rewards = dict(rewards)
    torch.save(rewards, f"{checkpoint_path.stem}/artifacts/rewards.pt")

    clipped_rewards = dict(clipped_rewards)
    torch.save(clipped_rewards, f"{checkpoint_path.stem}/artifacts/clipped_rewards.pt")

    timestamps = dict(timestamps)
    torch.save(timestamps, f"{checkpoint_path.stem}/artifacts/timestamps.pt")

    denormalized_values = dict(denormalized_values)
    torch.save(denormalized_values, f"{checkpoint_path.stem}/artifacts/denormalized_values.pt")
