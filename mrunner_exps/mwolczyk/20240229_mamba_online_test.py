from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO",
    "character": "mon-hum-neu-mal",
    "num_workers": 16,
    "num_envs_per_worker": 30,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 4096,  # this equals bs = 128, 128 * 32 = 4096
    "async_rl": True,
    "serial_mode": False,
    "save_milestones_ith": 10_000_000,
    "wandb_user": "rahid",
    "wandb_project": "sp_nethack",
    "wandb_group": "rahid",
    "with_wandb": True,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "rnn_type": ["mamba"],
        "rnn_size": [256, 512],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "rnn_num_layers": [1, 2, 3],
        "mamba_use_complex": [False],
        "mamba_model_size": [128, 256],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="sf2_nethack",
    with_neptune=False,
    script="./run.sh",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
