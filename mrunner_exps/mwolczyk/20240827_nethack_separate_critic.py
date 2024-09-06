from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO",
    "character": "mon-hum-neu-mal",

    "num_workers": 8,
    "num_envs_per_worker": 4,
    "worker_num_splits": 2,

    "restart_behavior": "overwrite",
    "async_rl": True,
    "save_milestones_ith": 10_000_000,
    "wandb_user": "ideas-ncbr",
    "wandb_project": "recurrent_rl",
    "model": "ScaledNet",
    "heartbeat_interval": 600,
    "heartbeat_reporting_interval": 1200,

    # Rnn settings
    "use_rnn": True,
    "rnn_type": "nanogpt",
    "rnn_d_output": 256,
    "rnn_d_model": 256,
    "rollout": 128,
    "recurrence": 32,
    "policy_initialization": "torch_default",

    # Athena
    "batch_size": 4096,  # this equals bs = 128, 128 * 32 = 4096
    "with_wandb": True,

    # Local
    # "batch_size": 256,  # this equals bs = 128, 128 * 32 = 4096
    # "with_wandb": True,
}

# params different between exps
params_grid = [
    # Window transformer
    {
        "seed": list(range(3)),
        "nanogpt_time_mixing": ["window_attention"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "rnn_num_layers": [3],
        "nanogpt_n_head": [2],
        "nanogpt_dropout": [0.],
        "nanogpt_embedding_type": ["none", "sine", "rope"],
        "nanogpt_relative_timesteps": [True],
        "nanogpt_block_size": [128],
        "nanogpt_attention_type": ["softmax"],
        "actor_critic_share_weights": [False, True],
    },
    # LRU + GRU
    {
        "seed": list(range(3)),
        "nanogpt_time_mixing": ["gru", "lru"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "rnn_num_layers": [3],
        "nanogpt_n_head": [2],
        "nanogpt_dropout": [0.],
        "nanogpt_embedding_type": ["none"],
        "nanogpt_attention_type": ["none"],
        "actor_critic_share_weights": [False, True],
    },
    # Autoregressive transformer
    {
        "seed": list(range(3)),
        "nanogpt_time_mixing": ["autoregressive_attention"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "rnn_num_layers": [3],
        "nanogpt_n_head": [2],
        "nanogpt_dropout": [0.],
        "nanogpt_embedding_type": ["sine"],
        "nanogpt_relative_timesteps": [False],
        "nanogpt_attention_type": ["linear"],
        "actor_critic_share_weights": [False, True],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="sf2_nethack",
    with_neptune=False,
    script="./run.sh",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
