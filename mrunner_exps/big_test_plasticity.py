from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-T",
    "train_for_env_steps": 4_000_000_000,
    "group": "monk-APPO-T",
    "character": "mon-hum-neu-mal",
    "num_workers": 16,
    "num_envs_per_worker": 30,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 4096,  # this equals bs = 128, 128 * 32 = 4096
    "async_rl": True,
    "serial_mode": False,
    "save_milestones_ith": 10_000_000,
    "with_wandb": True,
    "wandb_user": "ideas-ncbr",
    "wandb_project": "nethack_plasticity",
    "wandb_group": "plasticity, large exp v2",
    "heartbeat_reporting_interval": 1800,

}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "delta": [0.99],
        "decoder_mlp_layers": [[], [512]],
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