from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-KLAA-T",
    "num_workers": 8,
    "num_envs_per_worker": 16,
    "worker_num_splits": 2,
    "rollout": 128,
    "async_rl": True,
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "ideas-ncbr",
    "wandb_project": "atari",
    "wandb_group": "plasticity, breakout + montezuma",
    "wandb_tags": [name],
    "batch_size": 256,
    "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,
    "use_pretrained_checkpoint": False,
    "kickstarting_loss_coeff": 0.0,
    "skip_train": 5_000_000,
    "load_checkpoint_kind": "best",
    "reward_scale": 0.01,
}

# params different between exps
atari_games = ["breakout", "montezuma"]

params_grid = []

for atari_game in atari_games:
    for learning_rate in [1e-4]:
        params_grid += [
            {
                "seed": list(range(1)),
                "learning_rate": [learning_rate],
                "env": [f"atari_{atari_game}"],
                "actor_critic_share_weights": [True],
                "delta": [0.99],
                "encoder_conv_mlp_layers": [[1024], [1024,1024]],
                "decoder_mlp_layers": [[], [1024]],
                "num_epochs": [2,8],
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