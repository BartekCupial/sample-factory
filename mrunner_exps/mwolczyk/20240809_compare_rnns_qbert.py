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
    "recurrence": -1,
    "batch_size": 1024,
    "env_frameskip": 1,
    "env_framestack": 1,

    # Rnn settings
    "rollout": 128,
    "use_rnn": True,
    "rnn_d_output": 256,
    "rnn_d_model": 256,
}

params_grid = [
    # Standard setup
    {
        "seed": list(range(3)),
        "env_frameskip": [4],
        "env_framestack": [4],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "use_rnn": [False],
    },
    # RNN
    {
        "seed": list(range(3)),
        "rnn_type": ["gru", "lstm"],
        "learning_rate": [1e-5, 5e-5, 1e-4],
    },
    # Linear transformer
    {
        "seed": list(range(3)),
        "rnn_type": ["linear_transformer"],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "rnn_num_layers": [3],
    },
    # NanoGPT
    {
        "seed": list(range(3)),
        "rnn_type": ["nanogpt"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "rnn_num_layers": [3],

        "nanogpt_n_head": [8],
        "nanogpt_dropout": [0.],
        "nanogpt_embedding_type": ["rope"],
        "nanogpt_relative_timesteps": [True],
        "nanogpt_constant_context": [False],
        "nanogpt_block_size": [256],
        "nanogpt_attention_type": ["linear", "softmax"],
    },
    # Mamba, selective
    {
        "seed": list(range(3)),
        "rnn_type": ["mamba"],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "rnn_num_layers": [3],
        "max_grad_norm": [4.],
        "mamba_use_complex": [False],
        "mamba_selective_ssm": [True]
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
