import ast
import copy
import sys
from os.path import join
from typing import Callable

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.context import global_model_factory, sf_global_context
from sample_factory.cfg.arguments import load_from_path, parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.actor_critic import ActorCritic, default_make_actor_critic_func
from sample_factory.model.encoder import Encoder
from sample_factory.train import run_rl
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.utils.utils import log, str2bool
from sf_examples.atari.algo.learning.learner import DatasetLearner
from sf_examples.atari.atari_params import atari_override_defaults
from sf_examples.atari.atari_utils import ATARI_ENVS, make_atari_env
from sf_examples.atari.models.kickstarter import KickStarter


def register_atari_envs():
    for env in ATARI_ENVS:
        register_env(env.name, make_atari_env)


def register_atari_components():
    sf_global_context().learner_cls = DatasetLearner
    register_atari_envs()
    global_model_factory().register_actor_critic_factory(make_atari_actor_critic)


def parse_atari_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_learner(parser)
    add_extra_params_general(parser)
    atari_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def add_extra_params_learner(parser):
    """
    Specify any additional command line arguments for NetHack evaluation.
    """
    # TODO: add help
    p = parser
    p.add_argument("--use_dataset", type=str2bool, default=False)
    p.add_argument("--behavioral_clone", type=str2bool, default=False)
    p.add_argument("--data_path", type=str, default="/nle/nld-aa/nle_data")
    p.add_argument("--dataset_name", type=str, default="autoascend")
    p.add_argument("--dataset_num_splits", type=int, default=2)
    p.add_argument("--dataset_warmup", type=int, default=0)
    p.add_argument("--dataset_rollout", type=int, default=32)
    p.add_argument("--dataset_batch_size", type=int, default=1024)
    p.add_argument("--dataset_num_workers", type=int, default=8)
    p.add_argument("--dataset_shuffle", type=str2bool, default=True, help="for debugging purposes")
    p.add_argument("--reset_on_rollout_boundary", type=str2bool, default=False)


def make_atari_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    create_model = default_make_actor_critic_func

    use_distillation_loss = cfg.distillation_loss_coeff > 0.0
    use_kickstarting_loss = cfg.kickstarting_loss_coeff > 0.0

    if use_distillation_loss or use_kickstarting_loss:
        student = create_model(cfg, obs_space, action_space)
        if cfg.use_pretrained_checkpoint:
            if not cfg.actor_critic_share_weights:
                load_pretrained_checkpoint_from_shared_weights(
                    student, cfg, cfg.model_path, cfg.load_checkpoint_kind, create_model, obs_space, action_space
                )
            else:
                load_pretrained_checkpoint(
                    student, cfg.model_path, cfg.load_checkpoint_kind, normalize_returns=cfg.normalize_returns
                )

        # because there can be some missing parameters in the teacher config
        # we will get the default values and override the default_cfg with what teacher had in the config
        teacher_cfg = load_from_path(join(cfg.teacher_path, "config.json"))
        default_cfg = parse_atari_args(argv=[f"--env={cfg.env}"], evaluation=False)
        default_cfg.__dict__.update(dict(teacher_cfg))

        if not cfg.actor_critic_share_weights:
            # because of the way how we handle rnn_states we need the teacher
            # and student to use the same rnn_size.
            # ActorCriticSeparateWeights has 2x rnn_size the SharedWeights version.
            # This is the reason behind making the teacher SeparateWeights.
            default_cfg.actor_critic_share_weights = False
            teacher = create_model(default_cfg, obs_space, action_space)
            load_pretrained_checkpoint_from_shared_weights(
                teacher, default_cfg, cfg.teacher_path, cfg.load_checkpoint_kind, create_model, obs_space, action_space
            )
        else:
            teacher = create_model(default_cfg, obs_space, action_space)
            load_pretrained_checkpoint(
                teacher, cfg.teacher_path, cfg.load_checkpoint_kind, normalize_returns=default_cfg.normalize_returns
            )

        model = KickStarter(student, teacher, run_teacher_hs=cfg.run_teacher_hs)
        log.debug("Created kickstarter")
    else:
        model = create_model(cfg, obs_space, action_space)
        if cfg.use_pretrained_checkpoint:
            if not cfg.actor_critic_share_weights:
                load_pretrained_checkpoint_from_shared_weights(
                    model, cfg, cfg.model_path, cfg.load_checkpoint_kind, create_model, obs_space, action_space
                )
            else:
                load_pretrained_checkpoint(
                    model, cfg.model_path, cfg.load_checkpoint_kind, normalize_returns=cfg.normalize_returns
                )
            log.debug("Loading model from pretrained checkpoint")

    return model


def load_pretrained_checkpoint(model, checkpoint_dir: str, checkpoint_kind: str, normalize_returns: bool = True):
    name_prefix = dict(latest="checkpoint", best="best")[checkpoint_kind]
    checkpoints = Learner.get_checkpoints(join(checkpoint_dir, "checkpoint_p0"), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, "cpu")

    student_params = dict(
        filter(
            lambda x: x[0].startswith("student"),
            checkpoint_dict["model"].items(),
        )
    )

    if len(student_params) > 0:
        # means that the pretrained checkpoint was a KickStarter, we only want to load the student
        student_params = dict(map(lambda x: (x[0].removeprefix("student."), x[1]), student_params.items()))
        checkpoint_dict["model"] = student_params

    if not normalize_returns:
        del checkpoint_dict["model"]["returns_normalizer.running_mean"]
        del checkpoint_dict["model"]["returns_normalizer.running_var"]
        del checkpoint_dict["model"]["returns_normalizer.count"]
    else:
        checkpoint_dict["model"]["returns_normalizer.running_mean"][:] = 0
        checkpoint_dict["model"]["returns_normalizer.running_var"][:] = 1
        checkpoint_dict["model"]["returns_normalizer.count"][:] = 1

    # TODO: handle loading linear critic
    model.load_state_dict(checkpoint_dict["model"], strict=False)


def load_pretrained_checkpoint_from_shared_weights(
    model: ActorCritic,
    cfg: Config,
    checkpoint_dir: str,
    checkpoint_kind: str,
    create_model: Callable,
    obs_space: ObsSpace,
    action_space: ActionSpace,
):
    # since our pretrained checkpoints have shared weights we load them in that format
    # then create temporary model with separate actor and critic with modules from pretrained model
    # we finally use load_state_dict to ensure that the shapes match
    cfg.actor_critic_share_weights = True
    model_shared = create_model(cfg, obs_space, action_space)
    load_pretrained_checkpoint(model_shared, checkpoint_dir, checkpoint_kind, normalize_returns=cfg.normalize_returns)
    cfg.actor_critic_share_weights = False
    tmp_model: ActorCritic = create_model(cfg, obs_space, action_space)

    tmp_model.obs_normalizer = copy.deepcopy(model_shared.obs_normalizer)
    tmp_model.returns_normalizer = copy.deepcopy(model_shared.returns_normalizer)
    tmp_model.actor_encoder = copy.deepcopy(model_shared.encoder)
    tmp_model.actor_core = copy.deepcopy(model_shared.core)
    tmp_model.actor_decoder = copy.deepcopy(model_shared.decoder)
    tmp_model.action_parameterization = copy.deepcopy(model_shared.action_parameterization)

    if cfg.init_critic_from_actor:
        tmp_model.critic_encoder = copy.deepcopy(model_shared.encoder)
        tmp_model.critic_core = copy.deepcopy(model_shared.core)
        tmp_model.critic_decoder = copy.deepcopy(model_shared.decoder)
    # tmp_model.critic_linear = copy.deepcopy(model_shared.critic_linear)

    model.load_state_dict(tmp_model.state_dict(), strict=False)


def add_extra_params_general(parser):
    """
    Specify any additional command line arguments for NetHack.
    """
    # TODO: add help
    p = parser
    p.add_argument("--exp_tags", type=str, default="local")
    p.add_argument("--exp_point", type=str, default="point-A")
    p.add_argument("--group", type=str, default="group2")
    p.add_argument("--use_pretrained_checkpoint", type=str2bool, default=False)
    p.add_argument("--model", type=str, default="ChaoticDwarvenGPT5")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--supervised_loss_coeff", type=float, default=0.0)
    p.add_argument("--kickstarting_loss_coeff", type=float, default=0.0)
    p.add_argument("--distillation_loss_coeff", type=float, default=0.0)
    p.add_argument("--supervised_loss_decay", type=float, default=1.0)
    p.add_argument("--kickstarting_loss_decay", type=float, default=1.0)
    p.add_argument("--distillation_loss_decay", type=float, default=1.0)
    p.add_argument("--min_supervised_loss_coeff", type=float, default=0.0)
    p.add_argument("--min_kickstarting_loss_coeff", type=float, default=0.0)
    p.add_argument("--min_distillation_loss_coeff", type=float, default=0.0)
    p.add_argument("--substitute_regularization_with_exploration", type=str2bool, default=False)
    p.add_argument("--exploration_coeff_on_supervised_loss_coeff", type=float, default=0.0)
    p.add_argument("--exploration_coeff_on_kickstarting_loss_coeff", type=float, default=0.0)
    p.add_argument("--exploration_coeff_on_distillation_loss_coeff", type=float, default=0.0)
    p.add_argument("--teacher_path", type=str, default=None)
    p.add_argument("--run_teacher_hs", type=str2bool, default=False)
    p.add_argument("--add_stats_to_info", type=str2bool, default=True)
    p.add_argument("--capture_video", type=str2bool, default=False)
    p.add_argument("--capture_video_ith", type=int, default=100)
    p.add_argument("--freeze", type=ast.literal_eval, default={})
    p.add_argument("--unfreeze", type=ast.literal_eval, default={})
    p.add_argument("--freeze_batch_norm", type=str2bool, default=False)
    p.add_argument("--skip_train", type=int, default=-1)
    p.add_argument("--target_batch_size", type=int, default=128)
    p.add_argument("--optim_step_every_ith", type=int, default=1)
    p.add_argument("--init_critic_from_actor", type=str2bool, default=True)
    p.add_argument("--critic_layer_norm", type=str2bool, default=False)
    p.add_argument("--encoder_conv_scale", type=int, default=1)
    p.add_argument(
        "--critic_learning_rate",
        type=float,
        default=None,
        help="this parameter doesn't work with with lr_scheduler, it will be overwritten.",
    )
    p.add_argument("--remove_critic", type=str2bool, default=False)


def main():  # pragma: no cover
    """Script entry point."""
    register_atari_components()
    cfg = parse_atari_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
