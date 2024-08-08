from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_script": "sf_examples.atari.train_atari",
    "wandb_group": "atari_setup",

    "env": "atari_breakout",
    "train_for_env_steps": 2_000_000_000,

     "rollout": 4,
    # "use_rnn": True,

    "with_wandb": True,
    "wandb_user": "rahid",
    "wandb_project": "sf2_atari",
    "wandb_group": "rahid",
    "wandb_tags": [name],
    "recurrence": -1,
}

params_grid = [
    # RNN
    {
        "seed": list(range(3)),
        "env_frameskip": [1],
        "env_framestack": [1],
        "rollout": [128], 
        "recurrence": [4, 8],
        "use_rnn": [True],
        "rnn_type": ["gru"],
        "learning_rate": [1e-4, 3e-4, 1e-3],
    },
    # Non-markovian MLP
    {
        "seed": list(range(3)),
        "env_frameskip": [1],
        "env_framestack": [1],
        "rollout": [128], 
        "learning_rate": [1e-4, 3e-4, 1e-3],
    },
    # Standard setup
    {
        "seed": list(range(3)),
        "env_frameskip": [4],
        "env_framestack": [4],
        "rollout": [128],
        "learning_rate": [1e-4, 3e-4, 1e-3],
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
