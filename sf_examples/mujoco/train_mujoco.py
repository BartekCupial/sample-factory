import sys
import ast
from typing import Callable


from sample_factory.cfg.arguments import load_from_path, parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.mujoco.mujoco_params import add_mujoco_env_args, mujoco_override_defaults
from sf_examples.mujoco.mujoco_utils import MUJOCO_ENVS, make_mujoco_env
from sample_factory.algo.utils.context import global_model_factory, sf_global_context
from sample_factory.model.encoder import Encoder, MultiInputEncoder
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.utils.utils import log, str2bool
from sample_factory.model.actor_critic import (
    ActorCritic,
    ActorCriticSeparateWeights,
    ActorCriticSharedWeights,
    # obs_space_without_action_mask,
)
from sf_examples.mujoco.models import (
    SimBaActorEncoder,
    SimBaCriticEncoder,
    BROActorEncoder,
    BROCriticEncoder,
)

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
    p.add_argument("--model", type=str, default="default")
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
    p.add_argument("--actor_depth", type=int, default=1)
    p.add_argument("--actor_hidden_dim", type=int, default=64)
    p.add_argument("--actor_expansion", type=int, default=4)
    p.add_argument("--actor_use_max_pool", type=str2bool, default=False)
    p.add_argument("--critic_depth", type=int, default=1)
    p.add_argument("--critic_hidden_dim", type=int, default=64)
    p.add_argument("--critic_expansion", type=int, default=4)
    p.add_argument("--critic_use_max_pool", type=str2bool, default=False)

class ActorCriticDifferentEncoders(ActorCriticSeparateWeights):
    def __init__(self, model_factory, obs_space, action_space, cfg):
        super().__init__(model_factory, obs_space, action_space, cfg)
        if cfg.model == "simba":
            self.actor_encoder = SimBaActorEncoder(cfg, obs_space)
            self.critic_encoder = SimBaCriticEncoder(cfg, obs_space)
        elif cfg.model == "bro":
            self.actor_encoder = BROActorEncoder(cfg, obs_space)
            self.critic_encoder = BROCriticEncoder(cfg, obs_space)

        self.actor_core = model_factory.make_model_core_func(cfg, self.actor_encoder.get_out_size())
        self.critic_core = model_factory.make_model_core_func(cfg, self.critic_encoder.get_out_size())

        self.encoders = [self.actor_encoder, self.critic_encoder]
        self.cores = [self.actor_core, self.critic_core]

        self.core_func = self._core_rnn if self.cfg.use_rnn else self._core_empty

        self.actor_decoder = model_factory.make_model_decoder_func(cfg, self.actor_core.get_out_size())
        self.critic_decoder = model_factory.make_model_decoder_func(cfg, self.critic_core.get_out_size())
        self.decoders = [self.actor_decoder, self.critic_decoder]

        self.critic_linear = orthogonal_init(nn.Linear(self.critic_decoder.get_out_size(), 1), gain=1.0)
        self.action_parameterization = self.get_action_parameterization(self.actor_decoder.get_out_size())

        self.encoder_outputs_sizes = [encoder.get_out_size() for encoder in self.encoders]
        self.rnn_hidden_sizes = [core.core.hidden_size * 2 for core in self.cores]
        self.core_outputs_sizes = [decoder.get_out_size() for decoder in self.decoders]


def make_mujoco_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    from sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()
    # obs_space = obs_space_without_action_mask(obs_space)

    if cfg.model in ["simba", "bro"]:
        if cfg.actor_critic_share_weights:
            return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
        else:
            return ActorCriticDifferentEncoders(model_factory, obs_space, action_space, cfg)
    elif cfg.model == "default":
        if cfg.actor_critic_share_weights:
            return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
        else:
            return ActorCriticSeparateWeights(model_factory, obs_space, action_space, cfg)
    else:
        raise NotImplementedError

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



def make_mujoco_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    if cfg.model == "default":
        return MultiInputEncoder(cfg, obs_space)
    elif cfg.model == "simba":
        return SimBaActorEncoder(cfg, obs_space)
    elif cfg.model == "bro":
        return BROActorEncoder(cfg, obs_space)


def register_mujoco_components():
    for env in MUJOCO_ENVS:
        register_env(env.name, make_mujoco_env)
    global_model_factory().register_encoder_factory(make_mujoco_encoder)
    global_model_factory().register_actor_critic_factory(make_mujoco_actor_critic)


def parse_mujoco_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_general(parser)
    add_mujoco_env_args(partial_cfg.env, parser)
    mujoco_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_mujoco_components()
    cfg = parse_mujoco_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
