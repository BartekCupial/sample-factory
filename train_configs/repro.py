import os

from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]


# params for all exps
config = {
    "exp_tag": name,
    "run_script": "sf_examples.nethack.train_nethack",
    "wandb_user": "bartekcupial",
    "wandb_project": "nle_simba",
    "wandb_group": "ideas-ncbr",
    "with_wandb": True,
    "env": "nethack_challenge",
    "batch_size": 4096,
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "worker_num_splits": 2,
    "rollout": 32,
    "character": "mon-hum-neu-mal",
    "model": "ChaoticDwarvenGPT5",
    "rnn_size": 512,
    "experiment": "nethack_monk",
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
    }
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle_simba",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    env={
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
    },
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
