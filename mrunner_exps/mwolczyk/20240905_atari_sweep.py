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
    "train_for_env_steps": 100_000_000,
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

    # Separate critic:
    "actor_critic_share_weights": False,

}

params_grid = [
    # MLP 1/4 frames
    {
        "env": ["atari_mspacman", "atari_privateye", "atari_riverraid", "atari_pong"],
        "seed": list(range(3)),
        "env_frameskip": [1],
        "env_framestack": [1],
        "learning_rate": [1e-4, 5e-4],
        "use_rnn": [False],
        "actor_critic_share_weights": [True, False],
    },
    # GRU
    # {
    #     "seed": list(range(3)),
    #     "rnn_type": ["gru"],
    #     "learning_rate": [1e-4, 5e-4, 1e-3],
    #     "rnn_num_layers": [3, 1],
    #     "actor_critic_share_weights": [True, False],
    # },
    # # LRU + GRU
    {
        "env": ["atari_mspacman", "atari_privateye", "atari_riverraid", "atari_pong"],
        "seed": list(range(3)),
        "nanogpt_time_mixing": ["gru"],
        "learning_rate": [1e-4, 5e-4],
        "rnn_num_layers": [3],
        "nanogpt_n_head": [2],
        "nanogpt_dropout": [0.],
        "nanogpt_embedding_type": ["none"],
        "nanogpt_attention_type": ["none"],
        "actor_critic_share_weights": [True, False],
    },
    # Window transformer
    {
        "env": ["atari_mspacman", "atari_privateye", "atari_riverraid", "atari_pong"],
        "seed": list(range(3)),
        "nanogpt_time_mixing": ["window_attention"],
        "learning_rate": [1e-4, 5e-4],
        "rnn_num_layers": [3],
        "nanogpt_n_head": [2],
        "nanogpt_dropout": [0.],
        "nanogpt_embedding_type": ["sine", "rope"],
        "nanogpt_relative_timesteps": [True],
        "nanogpt_block_size": [128],
        "nanogpt_attention_type": ["softmax"],
        "actor_critic_share_weights": [True, False],
    },
    # # Autoregressive transformer
    # {
    #     "seed": list(range(3)),
    #     "nanogpt_time_mixing": ["autoregressive_attention"],
    #     "learning_rate": [1e-4, 5e-4, 1e-3],
    #     "rnn_num_layers": [3],
    #     "nanogpt_n_head": [2],
    #     "nanogpt_dropout": [0.],
    #     "nanogpt_embedding_type": ["sine"],
    #     "nanogpt_relative_timesteps": [False],
    #     "nanogpt_attention_type": ["linear"],
    # },
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
