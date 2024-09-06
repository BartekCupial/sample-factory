from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_script": "sf_examples.atari.train_atari",
    "wandb_group": "atari_setup",

    # Wandb params
    "with_wandb": True,
    "wandb_user": "ideas-ncbr",
    "wandb_project": "recurrent_rl",
    "wandb_tags": [name],

    # Experiment setup
    "env": "atari_qbert",
    "train_for_env_steps": 100_000_000,
    "recurrence": -1,
    "batch_size": 1024,
    "env_frameskip": 1,
    "env_framestack": 1,
    "num_epochs": 1,

    # Rnn settings
    "roleout": 128,
    "use_rnn": True,
    "rnn_d_output": 256,
    "rnn_d_model": 256,
}

params_grid = [
    # RNN
    {
        "seed": list(range(3)),
        "use_rnn": ["False"],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "recurrence": [4],
        "rollout": [4, 32],
        "num_envs_per_worker": [1, 4],
    },
    # RNN
    {
        "seed": list(range(3)),
        "use_rnn": ["False"],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "recurrence": [4],
        "rollout": [128],
        "num_envs_per_worker": [1],
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
