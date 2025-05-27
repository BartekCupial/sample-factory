from mrunner.helpers.specification_helper import create_experiments_helper
import json

# name = globals()["script"][:-3]
import os
name = os.path.splitext(os.path.basename(__file__))[0]

# params for all exps
config = {
    "train_for_env_steps": 1_000_000,
    "num_workers": 16,
    "num_envs_per_worker": 16,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 1024,  # this equals bs = 128, 128 * 32 = 4096
    "async_rl": True,
    "serial_mode": False,
    "restart_behavior": "overwrite",
    "with_wandb": True,
    "wandb_user": "e-dobrowolska",
    "wandb_project": "atari",
    "wandb_group": "cool atari",
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "env": ["breakout"],
        "actor_critic_share_weights": [True],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="helios_atari",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
from mrunner.helpers.client_helper import get_configuration

exps = []

for i, exp in enumerate(experiments_list):
    curr_config = {"project_name": exp.project, "unique_name": exp.unique_name, "name": exp.name}
    params = exp.parameters
    run_script = params.pop("run_script", "sf_examples.atari.train_atari")
    key_pairs = [f"--{key}={value}" for key, value in params.items()]
    cmd = ["python", "-m", run_script] + key_pairs
    curr_config["command"] = " ".join(cmd)
    exps.append(curr_config)


with open("config.jsonl", "w") as f:
    for item in exps:
        f.write(json.dumps(item) + "\n")