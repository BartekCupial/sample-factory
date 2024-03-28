from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "atari_breakout",
    "exp_tags": [name],
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-KLAA-T",
    "num_workers": 8,
    "num_envs_per_worker": 16,
    "worker_num_splits": 2,
    "rollout": 32,
    "async_rl": True,
    "restart_behavior": "overwrite",
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_atari",
    "wandb_group": "gmum",
    "wandb_tags": [name],
    "batch_size": 256,
    "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,
    "kickstarting_loss_coeff": 0.0,
    "actor_critic_share_weights": False,
    "device": "cpu",
}

atari_games = ["breakout", "qbert", "montezuma", "upndown"]
# params different between exps

params_grid = []

for atari_game in atari_games:
    params_grid += [
        {
            "seed": list(range(1)),
            "learning_rate": [1e-4, 1e-3],
            # "value_loss_coeff": [1e-2, 1e-1],
            "reward_scale": [1e-2, 1e-1],
            "env": [f"atari_{atari_game}"],
        },
    ]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="atari_sf",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
