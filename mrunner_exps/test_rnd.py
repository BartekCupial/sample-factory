# from mrunner.helpers.specification_helper import create_experiments_helper

# name = globals()["script"][:-3]

# # params for all exps
# config = {
#     "exp_tags": [name],
#     "train_for_env_steps": 1_000_000,
#     "group": "monk-APPO-KLAA-T",
#     "num_workers": 1,
#     "num_envs_per_worker": 8,
#     "worker_num_splits": 1,
#     "rollout": 128,  # like in the paper
#     "async_rl": False,
#     "restart_behavior": "overwrite",
#     "save_milestones_ith": 10_000_000,
#     # Wandb settings
#     "wandb_user": "e-dobrowolska",
#     "wandb_project": "atari",
#     "wandb_group": "test plasticity new ranks",
#     "wandb_tags": [name],
#     "batch_size": 1024,
#     "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
#     # "with_wandb": True,
#     "serial_mode": False,
#     "use_pretrained_checkpoint": False,
#     "kickstarting_loss_coeff": 0.0,
#     "load_checkpoint_kind": "best",
#     "reward_scale": 0.01,
#     "device": "cpu",
#     # "decorrelate_envs_on_one_worker": False,
#     "normalize_input": False,
# }

# # params different between exps
# atari_games = ["montezuma"]

# params_grid = []

# for atari_game in atari_games:
#     for learning_rate in [0.0001]:
#         params_grid += [
#             {
#                 "seed": [3],
#                 "learning_rate": [learning_rate],
#                 "env": [f"atari_{atari_game}"],
#                 # "cleanrl_actor_critic": [True],
#                 "actor_critic_share_weights": [False],
#                 "delta": [0.99],
#                 "with_rnd": [True],
#                 "gamma": [0.999],  # extrinsic gamma
#                 "num_epochs": [8],
#                 "repeat_action_probability": [0.25],
#                 # "use_shrink_perturb": [True],
#                 "freq_shrink_perturb": [150_000],
#                 "modules_to_perturb": [["predictor_network", "int_critic"]],
#                 "l2_init_loss_coeff": [0.01],
#                 # "log_heatmaps_to_wandb": [True],
#                 # "save_heatmaps_locally": [True],
#                 "heatmap_save_freq": [100_000],
#                 "policy_initialization": ["orthogonal"],
#                 "env_frameskip": [3],
#                 # "decoder_mlp_layers": [[512, 512]],
#             },
#         ]

# experiments_list = create_experiments_helper(
#     experiment_name=name,
#     project_name="atari",
#     with_neptune=False,
#     script="python3 mrunner_run.py",
#     python_path=".",
#     tags=[name],
#     base_config=config,
#     params_grid=params_grid,
#     mrunner_ignore=".mrunnerignore",
# )

from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "train_for_env_steps": 500_000_000,
    "num_workers": 4,
    "num_envs_per_worker": 8,
    "num_batches_per_epoch": 16,
    "worker_num_splits": 2,  #??
    "rollout": 128,  #??
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "ideas-ncbr",
    "wandb_project": "atari plasticity_ed",
    "wandb_group": "plasticity, reproduce paper, new",
    "wandb_tags": [name],
    "with_wandb": False,
    "device": "cpu",
}

# params different between exps
# atari_games = ["breakout", "montezuma", "phoenix", "namethisgame"]
atari_games = ["phoenix"]

params_grid = []

for atari_game in atari_games:
    params_grid += [
        {
            "seed": list(range(1)),
            "with_rnd": [False],

            # Check async
            "async_rl": [True],

            # paper's params: env
            "env": [f"atari_{atari_game}"],
            "normalize_input": [False],
            "env_frameskip": [3],

            # paper's params: model
            "actor_critic_share_weights": [True],
            "encoder_conv_mlp_layers": [[512]],
            "nonlinearity": ["relu"],

            # paper's params: training
            "batch_size": [4096],
            "learning_rate": [0.00025],
            "lr_schedule": ["constant"],
            "gamma": [0.99],
            "gae_lambda": [0.95],
            "ppo_clip_ratio": [0.1],  # "We use unbiased clip(x, 1+e, 1/(1+e)) instead of clip(x, 1+e, 1-e) in the paper"
            "exploration_loss_coeff": [0.01],
            "value_loss_coeff": [0.5],
            "max_grad_norm": [0.5],
            "optimizer": ["adam"],
            "num_epochs": [8],
            "normalize_returns": [True],
            "repeat_action_probability": [0.25],

            # paper's params: plasticity
            "delta": [0.99],
        },
    ]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="atari",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
