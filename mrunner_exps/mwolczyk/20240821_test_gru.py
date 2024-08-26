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

    # Rnn settings
    "use_rnn": True,
    "rnn_d_output": 256,
    "rnn_d_model": 256,
    "rollout": 128,

}

params_grid = [
    # Rollout: 4
    {
        "seed": list(range(3)),
        "rnn_type": ["nanogpt"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "rnn_num_layers": [3],
        "nanogpt_n_head": [2],
        "nanogpt_dropout": [0.],
        "nanogpt_embedding_type": ["none"],
        "nanogpt_relative_timesteps": [True],
        "nanogpt_constant_context": [False],
        "recurrence": [4],
        "rollout": [4, 16, 128],
        "nanogpt_block_size": [4],
        "nanogpt_attention_type": ["none"],
        "nanogpt_time_mixing": ["gru"],
        "policy_initialization": ["torch_default"],
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
