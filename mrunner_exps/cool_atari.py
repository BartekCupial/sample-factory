from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "train_for_env_steps": 500_000,
    "num_workers": 4,
    "num_envs_per_worker": 8,
    "num_batches_per_epoch": 16,
    "worker_num_splits": 2,  #??
    "rollout": 128,  #??
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "e-dobrowolska",
    "wandb_project": "atari",
    "wandb_group": "montezuma rooms tracker",
    "wandb_tags": [name],
    "with_wandb": True,
}

# params different between exps
atari_games = ["montezuma"]

params_grid = []

for atari_game in atari_games:
    params_grid += [
        {
            "seed": [25],
            "env": [f"atari_{atari_game}"],
            "count_montezuma_rooms": [True],
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
