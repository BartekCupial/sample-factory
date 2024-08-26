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
    "train_for_env_steps": 2_000_000_000,
    "batch_size": 1024,
    "env_frameskip": 1,
    "env_framestack": 1,
    "num_envs_per_worker": 4,

    # Rnn settings
    "use_rnn": True,
    "rnn_type": "nanogpt",
    "rnn_d_output": 256,
    "rnn_d_model": 256,
    "rollout": 128,
    "recurrence": 4,
    "policy_initialization": "torch_default",

}

params_grid = [
    # RNN
    {
        "seed": list(range(3)),
        "rnn_type": ["gru"],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "rnn_num_layers": [3, 1],
    },
    {
        "seed": list(range(3)),
        "env_frameskip": [4, 1],
        "env_framestack": [4],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "use_rnn": [False],
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
