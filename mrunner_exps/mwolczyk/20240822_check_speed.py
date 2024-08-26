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
    "num_envs_per_worker": 1,
    "serial_mode": True,

    # Rnn settings
    "use_rnn": True,
    "rnn_type": "nanogpt",
    "rnn_d_output": 256,
    "rnn_d_model": 256,
    "rollout": 128,
    "recurrence": 2,
    "policy_initialization": "torch_default",

}

params_grid = [
    {
        "seed": list(range(1)),
        "nanogpt_time_mixing": ["identity"],
        "learning_rate": [1e-4],
        "rnn_num_layers": [3],
        "nanogpt_n_head": [2],
        "nanogpt_dropout": [0.],
        "nanogpt_embedding_type": ["none"],
        "nanogpt_attention_type": ["none"],
        "rnn_d_output": [64],
        "rnn_d_model": [64],
        "nanogpt_block_size": [1, 64, 128, 256, 2048],
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
