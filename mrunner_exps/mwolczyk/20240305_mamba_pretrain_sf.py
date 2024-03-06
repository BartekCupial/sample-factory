from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-KLAA-T",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-KLAA-T",
    "character": "mon-hum-neu-mal",
    "num_workers": 4,
    "num_envs_per_worker": 16,
    "worker_num_splits": 2,
    "rollout": 32,
    "async_rl": True,
    "serial_mode": False,
    "save_milestones_ith": 10_000_000,
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_nethack",
    "wandb_group": "gmum",
    "with_wandb": True,
    # "dataset_rollout": 32,
    # "dataset_batch_size": 8192,  # this equals bs = 256, 256 * 32 = 8192
    # "distillation_loss_coeff": 0.2,
    # "teacher_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/@-AA-BC/pretrained_use_prev_action",
    "run_teacher_hs": False,
    "use_prev_action": True,
    "model": "ScaledNet",
    "use_resnet": True,
    "rnn_size": 512,
    "use_dataset": True,
    "dataset_rollout": 32,
    "dataset_num_workers": 8,
    "supervised_loss_coeff": 1.0,
    "behavioral_clone": True,

    # Athena
    # "db_path": "/ttyrecs/ttyrecs.db",
    # "dataset_name": "autoascend",
    # "batch_size": 4096,
    # "dataset_batch_size": 16384,  # this equals bs = 512, 512 * 32 = 16384

    # Local
    "db_path": "/home/maciejwolczyk/Repos/ttyrecs.db",
    "dataset_name": "nld-aa-taster-v1",
    "batch_size": 8,
    "dataset_batch_size": 256,  # this equals bs = 512, 512 * 32 = 16384
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        # "serial_mode": [True],
        "with_wandb": [False],
        "restart_behavior": ["overwrite"],
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
