from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "nethack_challenge",
    "train_for_env_steps": 2_000_000_000,
    "character": "mon-hum-neu-mal",
    "num_workers": 32,
    "num_envs_per_worker": 16,
    "worker_num_splits": 2,
    "async_rl": True,
    "serial_mode": False,
    "wandb_user": "bartekcupial",
    "wandb_project": "sample_factory_nethack",
    "wandb_group": "gmum",
    "with_wandb": True,
    "use_prev_action": True,
    "model": "ScaledNet",
    "use_resnet": True,
    "rnn_size": 512,
    "h_dim": 512,
    "gamma": 0.999,
    "heartbeat_interval": 600,
    "heartbeat_reporting_interval": 1200,
    "learning_rate": 0.0001,
    "rollout": 32,
    "num_batches_per_epoch": 2,
    "batch_size": 4096,
}

params_grid = [
    {
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
