from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-KLAA-T",
    "train_for_env_steps": 2_000_000_000,
    "group": "sine-rotate-large-scale",
    "character": "mon-hum-neu-mal",
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "worker_num_splits": 2,
    "rollout": 32,
    "async_rl": True,
    "save_milestones_ith": 10_000_000,
    "wandb_user": "rahid",
    "wandb_project": "sp_nethack",
    "wandb_group": "rahid",
    "wandb_tags": [name],
    # "dataset_rollout": 32,
    # "dataset_batch_size": 8192,  # this equals bs = 256, 256 * 32 = 8192
    # "distillation_loss_coeff": 0.2,
    # "teacher_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/@-AA-BC/pretrained_use_prev_action",
    "run_teacher_hs": False,
    "use_prev_action": True,
    "model": "ChaoticDwarvenGPT5",
    "use_resnet": True,
    "rnn_size": 512,
    "use_dataset": True,
    "dataset_rollout": 32,
    "dataset_num_workers": 8,
    "supervised_loss_coeff": 1.0,
    "behavioral_clone": True,
    "restart_behavior": "overwrite",
    "process_seq_in_batch_mode": False,
    "decorrelate_envs_on_one_worker": True,

    # Athena
    "db_path": "/ttyrecs/ttyrecs.db",
    "dataset_name": "autoascend",
    "batch_size": 128,
    "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,

    # Local
    # "db_path": "/home/maciejwolczyk/Repos/ttyrecs.db",
    # "dataset_name": "nld-aa-taster-v1",
    # "batch_size": 4,
    # "dataset_batch_size": 16,  # this equals bs = 512, 512 * 32 = 16384
    # "with_wandb": True,
    # "serial_mode": True,
}

# params different between exps
base_params_grid = [
    {
        "seed": list(range(3)),
        "rnn_type": ["lstm", "mamba", "nanogpt"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "rnn_size": [512],
        "decoder_mlp_layers": [[512, 512], []],
        "rnn_num_layers": [1, 3],
        "model": ["ChaoticDwarvenGPT5"],
        "process_seq_in_batch_mode": [True],
    },
]


# WARNING: here we "translate" the batch_size treated as num trajectories
# to batch size treated as num samples
params_grid = []
for grid in base_params_grid:
    for rollout in [32]:
        new_grid = grid.copy()
        new_grid["rollout"] = [rollout]
        # new_grid["nanogpt_block_size"] = [64, 256]
        new_grid["dataset_rollout"] = [rollout]
        new_grid["dataset_batch_size"] = [config["dataset_batch_size"] * rollout]
        new_grid["batch_size"] = [config["batch_size"] * rollout]

        params_grid.append(new_grid)

print(params_grid)

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
