from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "atari_breakout",
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
    "wandb_user": "rahid",
    "wandb_project": "atari_sf",
    "wandb_group": "rahid",
    "wandb_tags": [name],

    "batch_size": 32,
    "dataset_batch_size": 128,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,

    "use_pretrained_checkpoint": True,
    "model_path": "/home/maciejwolczyk/breakout_checkpoint/default_experiment/",
    "kickstarting_loss_coeff": 0.0,
    "learning_rate": 1e-4,
    "value_loss_coeff": 1e-4,
    "skip_train": 1_000_000,
    "freeze": {"encoder": 0, "core": 0, "decoder": 0, "action_parameterization": 0},
    "unfreeze": {"encoder": 2_000_000, "core": 2_000_000, "decoder": 2_000_000, "action_parameterization": 2_000_000} ,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
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
