from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple

import nle.dataset as nld
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS, memory_stats
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import alloc_trajectory_tensors
from sample_factory.algo.utils.tensor_dict import TensorDict, clone_tensordict, stack_tensordicts, tensor_dict_to_cpu
from sample_factory.algo.utils.tensor_utils import ensure_torch_tensor
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import log
from sf_examples.nethack.datasets.actions import ACTION_MAPPING
from sf_examples.nethack.datasets.dataset import load_nld_aa_large_dataset
from sf_examples.nethack.datasets.render import render_screen_image
from sf_examples.nethack.datasets.roles import Alignment, Race, Role
from sf_examples.nethack.models.kickstarter import KickStarter
from sf_examples.nethack.models.utils import freeze_selected, unfreeze_selected


class DatasetLearner(Learner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        super().__init__(
            cfg,
            env_info,
            policy_versions_tensor,
            policy_id,
            param_server,
        )

        self.dataset: nld.TtyrecDataset = None
        self.tp = None

        self.rnn_states = None
        self._iterators = None
        self._results = None

        self.supervised_loss_func: Optional[Callable] = None
        self.distillation_loss_func: Optional[Callable] = None
        self.kickstarting_loss_func: Optional[Callable] = None

        self.models_frozen = dict(zip(self.cfg.freeze.keys(), [False] * len(self.cfg.freeze)))

    def init(self) -> InitModelData:
        init_model_data = super().init()

        if self.cfg.use_dataset:
            assert self.cfg.rollout == self.cfg.dataset_rollout

            dataset_batch_size = self.cfg.dataset_batch_size // self.cfg.dataset_rollout
            self.dataset = self._get_dataset()
            self.tp = ThreadPoolExecutor(max_workers=self.cfg.dataset_num_workers)

            def _make_sing_iter(dataset):
                dataset = iter(dataset)

                def _iter():
                    prev_actions = np.zeros((dataset_batch_size, 1))
                    prev_timestamps = np.ones((dataset_batch_size, 1)) * -1

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
                it = _make_sing_iter(self.dataset)
                self._iterators.append(it)
                self._results.append(self.tp.submit(next, it))

        use_supervised_loss = self.cfg.supervised_loss_coeff > 0.0
        use_distillation_loss = self.cfg.distillation_loss_coeff > 0.0
        use_kickstarting_loss = self.cfg.kickstarting_loss_coeff > 0.0
        use_distillation_loss = False
        use_supervised_loss = False

        assert (
            sum(
                [
                    use_supervised_loss,
                    use_distillation_loss,
                    use_kickstarting_loss,
                ]
            )
            <= 1
        ), f"only one regularization loss allowed at the time, use_kickstarting_loss: {use_kickstarting_loss}, use_distillation_loss: {use_distillation_loss}, use_supervised_loss: {use_supervised_loss}."

        assert (
            use_supervised_loss or use_distillation_loss
        ) == self.cfg.use_dataset, (
            "If either 'use_supervised_loss' or 'use_distillation_loss' is true, then 'use_dataset' must also be true."
        )

        self.supervised_loss_func = self._supervised_loss if use_supervised_loss else lambda *_: 0.0
        self.distillation_loss_func = self._distillation_loss if use_distillation_loss else lambda *_: 0.0
        self.kickstarting_loss_func = self._kickstarting_loss if use_kickstarting_loss else lambda *_: 0.0

        return init_model_data

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

    def _supervised_loss(self, mb_results, mb, num_invalids: int):
        outputs = mb_results["action_logits"].flatten(0, 1)
        targets = mb["actions"].flatten(0, 1).long()
        supervised_loss = F.cross_entropy(outputs, targets, reduction="none")
        # supervised_loss = masked_select(supervised_loss, valids, num_invalids)
        supervised_loss *= self.cfg.supervised_loss_coeff
        supervised_loss = supervised_loss.mean()

        return supervised_loss

    def _distillation_loss(self, mb_results, mb, num_invalids: int):
        outputs = mb_results["action_logits"].flatten(0, 1)
        targets = mb_results["kick_action_logits"].flatten(0, 1)
        # we want to be equivalent to reduction "batchmean",
        # first we will sum div on every single distribution and leave the batch intact
        # after masked_select we will average what is left
        distillation_loss = F.kl_div(
            F.log_softmax(outputs, dim=-1),
            F.log_softmax(targets, dim=-1),
            log_target=True,
            reduction="none",
        ).sum(axis=1)
        # distillation_loss = masked_select(distillation_loss, valids, num_invalids)
        distillation_loss *= self.cfg.distillation_loss_coeff
        distillation_loss = distillation_loss.mean()

        return distillation_loss

    def _kickstarting_loss(self, result, valids, num_invalids: int):
        # we want to be equivalent to reduction "batchmean",
        # first we will sum div on every single distribution and leave the batch intact
        # after masked_select we will average what is left
        kickstarting_loss = F.kl_div(
            F.log_softmax(result["action_logits"], dim=-1),
            F.log_softmax(result["kick_action_logits"], dim=-1),
            log_target=True,
            reduction="none",
        ).sum(axis=1)
        kickstarting_loss = masked_select(kickstarting_loss, valids, num_invalids)
        kickstarting_loss *= self.cfg.kickstarting_loss_coeff
        kickstarting_loss = kickstarting_loss.mean()

        return kickstarting_loss

    def _calculate_losses(
        self,
        mb: AttrDict,
        num_invalids: int,
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        # PPO clipping
        clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
        # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
        clip_ratio_low = 1.0 / clip_ratio_high
        clip_value = self.cfg.ppo_clip_value

        if not self.cfg.behavioral_clone:
            detach_critic = self.cfg.aux_train and not self.cfg.actor_critic_share_weights
            mb_results = self._compute_model_outputs(mb, num_invalids, detach_critic=detach_critic)
            # we want action distribution (last) of the same shape as mb
            action_distribution = self.actor_critic.action_distribution()

            with self.timing.add_time("ppo_losses"):
                ratio = mb_results["ratio"]
                adv = mb_results["adv"]
                adv_mean = mb_results["adv_mean"]
                adv_std = mb_results["adv_std"]
                valids = mb_results["valids"]
                values = mb_results["values"]
                targets = mb_results["targets"]

                # noinspection PyTypeChecker
                policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
                exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
                kl_old, kl_loss = self.kl_loss_func(
                    self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
                )
                old_values = mb["values"]
                value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)

            with self.timing.add_time("kickstarting_loss"):
                kickstarting_loss = self.kickstarting_loss_func(
                    mb_results["result"],
                    mb_results["valids"],
                    num_invalids,
                )
        else:
            action_distribution = None
            ratio = torch.tensor([0.0])
            policy_loss = 0.0
            exploration_loss = 0.0
            kl_old, kl_loss = torch.tensor([0.0]), 0.0
            value_loss = 0.0
            kickstarting_loss = 0.0
            adv, adv_mean, adv_std = 0.0, 0.0, 0.0
            values = torch.tensor([0.0]).to(self.device)

        with self.timing.add_time("prepare_dataset_batch"):
            if self.cfg.use_dataset:
                dataset_mb = self._get_dataset_minibatch()
                dataset_mb_results = self._calculate_dataset_outputs(dataset_mb)
                dataset_num_invalids = 0
            else:
                dataset_mb = None
                dataset_mb_results = None
                dataset_num_invalids = 0

        with self.timing.add_time("supervised_loss"):
            supervised_loss = self.supervised_loss_func(
                dataset_mb_results,
                dataset_mb,
                dataset_num_invalids,
            )

        with self.timing.add_time("distillation_loss"):
            distillation_loss = self.distillation_loss_func(
                dataset_mb_results,
                dataset_mb,
                dataset_num_invalids,
            )

        action_distribution = (
            action_distribution if action_distribution is not None else self.actor_critic.action_distribution()
        )
        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=values,
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
        )
        regularizer_loss = supervised_loss + distillation_loss + kickstarting_loss
        regularizer_loss_summaries = dict(
            supervised_loss=to_scalar(supervised_loss),
            distillation_loss=to_scalar(distillation_loss),
            kickstarting_loss=to_scalar(kickstarting_loss),
        )

        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
            regularizer_loss,
            regularizer_loss_summaries,
        )

    def _train(
        self,
        gpu_buffer: TensorDict,
        batch_size: int,
        experience_size: int,
        num_invalids: int,
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

            with timing.add_time("freeze_model"):
                if isinstance(self.actor_critic, KickStarter):
                    freeze_selected(self.env_steps, self.cfg, self.actor_critic.student, self.models_frozen)
                else:
                    freeze_selected(self.env_steps, self.cfg, self.actor_critic, self.models_frozen)

            with timing.add_time("unfreeze_model"):
                if isinstance(self.actor_critic, KickStarter):
                    unfreeze_selected(self.env_steps, self.cfg, self.actor_critic.student, self.models_frozen)
                else:
                    unfreeze_selected(self.env_steps, self.cfg, self.actor_critic, self.models_frozen)

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        loss_summaries,
                        regularizer_loss,
                        regularizer_loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    # noinspection PyTypeChecker
                    actor_loss: Tensor = policy_loss + exploration_loss + kl_loss
                    critic_loss = value_loss
                    loss: Tensor = actor_loss + critic_loss + regularizer_loss

                    epoch_actor_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(loss),
                            to_scalar(policy_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                            to_scalar(kl_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # if kl_old is not None it is already calculated above
                    if kl_old is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = get_action_distribution(
                            self.actor_critic.action_space,
                            mb.action_logits,
                        )
                        kl_old = action_distribution.kl_divergence(old_action_distribution)
                        kl_old = masked_select(kl_old, mb.valids, num_invalids)

                    kl_old_mean = float(kl_old.mean().item())
                    recent_kls.append(kl_old_mean)
                    if kl_old.numel() > 0 and kl_old.max().item() > 100:
                        log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

                actual_lr = self.curr_lr
                curr_policy_version = self.train_step  # policy version before the weight update
                if self.env_steps >= self.cfg.warmup:
                    # update the weights
                    with timing.add_time("update"):
                        # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                        for p in self.actor_critic.parameters():
                            p.grad = None

                        loss.backward()

                        if self.cfg.max_grad_norm > 0.0:
                            with timing.add_time("clip"):
                                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                        if num_invalids > 0:
                            # if we have masked (invalid) data we should reduce the learning rate accordingly
                            # this prevents a situation where most of the data in the minibatch is invalid
                            # and we end up doing SGD with super noisy gradients
                            actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                        self._apply_lr(actual_lr)

                        with self.param_server.policy_lock:
                            self.optimizer.step()

                        self.cfg.kickstarting_loss_coeff *= self.cfg.kickstarting_loss_decay
                        self.cfg.distillation_loss_coeff *= self.cfg.distillation_loss_decay
                        self.cfg.supervised_loss_decay *= self.cfg.supervised_loss_decay

                        if self.cfg.substitute_regularization_with_exploration:
                            self.cfg.exploration_loss_coeff = max(
                                max(self.cfg.min_kickstarting_loss_coeff - self.cfg.kickstarting_loss_coeff, 0),
                                self.cfg.exploration_loss_coeff,
                            )
                            self.cfg.exploration_loss_coeff = max(
                                max(self.cfg.min_distillation_loss_coeff - self.cfg.distillation_loss_coeff, 0),
                                self.cfg.exploration_loss_coeff,
                            )
                            self.cfg.exploration_loss_coeff = max(
                                max(self.cfg.min_supervised_loss_coeff - self.cfg.supervised_loss_coeff, 0),
                                self.cfg.exploration_loss_coeff,
                            )
                        else:
                            self.cfg.kickstarting_loss_coeff = max(
                                self.cfg.kickstarting_loss_coeff, self.cfg.min_kickstarting_loss_coeff
                            )
                            self.cfg.distillation_loss_coeff = max(
                                self.cfg.distillation_loss_coeff, self.cfg.min_distillation_loss_coeff
                            )
                            self.cfg.supervised_loss_decay = max(
                                self.cfg.supervised_loss_coeff, self.cfg.min_supervised_loss_coeff
                            )

                        num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        # dont have a better way to do this then to modify record summaries
                        for key, value in regularizer_loss_summaries.items():
                            stats_and_summaries[key] = value

                        stats_and_summaries["kickstarting_loss_coeff"] = self.cfg.kickstarting_loss_coeff
                        stats_and_summaries["distillation_loss_coeff"] = self.cfg.distillation_loss_coeff
                        stats_and_summaries["supervised_loss_decay"] = self.cfg.supervised_loss_decay

                        del summary_vars
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        return stats_and_summaries

    def train(self, batch: TensorDict) -> Optional[Dict]:
        if self.cfg.save_milestones_ith > 0 and self.env_steps // self.cfg.save_milestones_ith > self.checkpoint_steps:
            self.save_milestone()
            self.checkpoint_steps = self.env_steps // self.cfg.save_milestones_ith

        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("prepare_batch"):
            if self.env_steps >= self.cfg.skip_train:
                buff, experience_size, num_invalids = self._prepare_batch(batch)

                if self.cfg.aux_train:
                    aux_buff = tensor_dict_to_cpu(buff)
                    self.aux_batch.append((aux_buff, experience_size, num_invalids))
            else:
                experience_size = batch["dones"].shape[0] * batch["dones"].shape[1]
                num_invalids = 0

        if num_invalids >= experience_size:
            if self.cfg.with_pbt:
                log.warning("No valid samples in the batch, with PBT this must mean we just replaced weights")
            else:
                log.error(f"Learner {self.policy_id=} received an entire batch of invalid data, skipping...")
            return None
        else:
            with self.timing.add_time("train"):
                if self.env_steps >= self.cfg.skip_train:
                    train_stats = self._train(
                        buff,
                        self.cfg.batch_size,
                        experience_size,
                        num_invalids,
                    )

                    if self.cfg.aux_train and len(self.aux_batch) == self.cfg.aux_train_frequency:
                        with self.timing.add_time("aux_train"):
                            aux_buff, aux_experience_size, aux_num_invalids = self._prepare_aux_batch(self.aux_batch)
                            aux_train_stats = self._aux_train(
                                aux_buff, self.cfg.aux_batch_size, aux_experience_size, aux_num_invalids, True
                            )
                            # update train stats with aux_train_stats
                            if aux_train_stats is not None:
                                if train_stats is not None:
                                    train_stats.update(**aux_train_stats)
                                else:
                                    train_stats = aux_train_stats
                            self.aux_batch = []
                else:
                    train_stats = None

            # multiply the number of samples by frameskip so that FPS metrics reflect the number
            # of environment steps actually simulated
            if self.cfg.behavioral_clone:
                self.env_steps += self.cfg.dataset_batch_size
            else:
                if self.cfg.summaries_use_frameskip:
                    self.env_steps += experience_size * self.env_info.frameskip
                else:
                    self.env_steps += experience_size

            stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
            if train_stats is not None:
                if train_stats is not None:
                    stats[TRAIN_STATS] = train_stats
                stats[STATS_KEY] = memory_stats("learner", self.device)

            return stats
