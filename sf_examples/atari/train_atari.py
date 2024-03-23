import ast
import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.atari.atari_params import atari_override_defaults
from sf_examples.atari.atari_utils import ATARI_ENVS, make_atari_env
from sf_examples.atari.algo.learning.learner import DatasetLearner
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.utils.utils import str2bool


def register_atari_envs():
    for env in ATARI_ENVS:
        register_env(env.name, make_atari_env)


def register_atari_components():
    sf_global_context().learner_cls = DatasetLearner
    register_atari_envs()


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


def main():  # pragma: no cover
    """Script entry point."""
    register_atari_components()
    cfg = parse_atari_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
