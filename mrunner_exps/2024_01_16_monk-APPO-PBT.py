from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-PBT",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-PBT",
    "character": "mon-hum-neu-mal",
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 128,  # this equals bs = 128, 128 * 32 = 4096
    "async_rl": True,
    "serial_mode": False,
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_nethack",
    "wandb_group": "gmum",
    "with_wandb": True,
    "use_prev_action": True,
    "gamma": 1.0,
    "reward_scale": 0.01,
    "learning_rate": 0.0001,
    "num_policies": 8,
    "pbt_replace_reward_gap": 0.1,
    "pbt_replace_reward_gap_absolute": 0.1,
    "pbt_period_env_steps": 5000000,
    "with_pbt": True,
    "pbt_start_mutation": 100000000,
    "pbt_optimize_gamma": True,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(5)),
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="sf2_nethack",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)