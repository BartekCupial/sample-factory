from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],

    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-KLAA-T",

    "num_workers": 4,
    "num_envs_per_worker": 16,
    "worker_num_splits": 2,
    "rollout": 32,
    "async_rl": True,
    "restart_behavior": "overwrite",

    "save_milestones_ith": 10_000_000,

    # Wandb settings
    "wandb_user": "e-dobrowolska",
    "wandb_project": "atari",
    "wandb_group": "bc",
    "wandb_tags": [name],

    "use_dataset": True,
    "dataset_rollout": 32,
    "dataset_num_workers": 8,
    "supervised_loss_coeff": 1.0,
    "behavioral_clone": True,
    "num_epochs": 1,

    "batch_size": 32,
    "dataset_batch_size": 128,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,
}


atari_games = ["breakout"]

# params different between exps
params_grid = []

for atari_game in atari_games:
    params_grid += [
        {
            "seed": list(range(1)),
            "dataset_name": [f"atari_2B_atari_{atari_game}_1111/atari_{atari_game}_dataset.npz"],
            "env": [f"atari_{atari_game}"],
            "learning_rate": [1e-4, 5e-4, 1e-3],
        }
    ]

print(params_grid)

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
