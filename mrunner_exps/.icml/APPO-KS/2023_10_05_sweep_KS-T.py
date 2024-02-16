from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-KS",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-KS",
    "character": "mon-hum-neu-mal",
    "num_workers": 16,
    "num_envs_per_worker": 30,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 4096,  # this equals bs = 128, 128 * 32 = 4096
    "async_rl": True,
    "serial_mode": False,
    "save_milestones_ith": 10_000_000,
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_nethack",
    "wandb_group": "gmum",
    "with_wandb": True,
    "kickstarting_loss_coeff": 0.2,
    "teacher_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/@-AA-BC/pretrained",
    "run_teacher_hs": False,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(3)),
        "kickstarting_loss_coeff": [0.2, 0.1, 0.05, 0.01, 0.005],
        "use_pretrained_checkpoint": [True],
        "model_path": ["/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/@-AA-BC/pretrained"],
        "teacher_path": ["/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/@-AA-BC/pretrained"],
    },
    {
        "seed": list(range(3)),
        "kickstarting_loss_coeff": [0.2, 0.1, 0.05, 0.01, 0.005],
        "use_pretrained_checkpoint": [True],
        "model_path": ["/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/monk-AA-BC/pretrained"],
        "teacher_path": ["/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/monk-AA-BC/pretrained"],
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