from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    # "device": "cpu",
    "restart_behavior": "overwrite",
    "env": "challenge",
    "train_for_env_steps": 2_000_000_000,
    "character": "mon-hum-neu-mal",
    "num_workers": 32,
    # "num_workers": 2,
    "num_envs_per_worker": 2,
    "worker_num_splits": 2,
    "rollout": 32,
    # "batch_size": 128,
    "batch_size": 4096,
    "async_rl": True,
    "serial_mode": False,
    "use_prev_action": True,
    "model": "ScaledNet",
    "num_batches_per_epoch": 2,
    "use_resnet": True,
    "rnn_size": 512,
    "h_dim": 512,
    "gamma": 0.999,
    "heartbeat_interval": 600,
    "heartbeat_reporting_interval": 1200,
    "learning_rate": 0.0001,
    "wandb_user": "bartosz-m-smoczynski",
    "wandb_project": "sf2_nethack",
    "wandb_group": "barsmo-team-org",
    "with_wandb": True,
}

params_grid = [
    {
        # "seed": list(range(1)),
        "seed": list(range(3)),
    }
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
