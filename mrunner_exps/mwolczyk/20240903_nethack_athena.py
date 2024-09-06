from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO",
    "train_for_env_steps": 200_000_000,
    "group": "monk-APPO",
    "character": "mon-hum-neu-mal",

    "num_workers": 16,
    "num_envs_per_worker": 30,
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
    {
        "seed": list(range(3)),
        "rnn_type": ["lstm"],
        "serial_mode": [True, False],
        "async_rl": [True],
        "learning_rate": [5e-5],
        "rnn_num_layers": [1],
        "actor_critic_share_weights": [True],
        "rnn_d_model": [512],
        "rnn_d_output": [512],
        "decoder_mlp_layers": ["[512, 512]"],
        "rollout": [32],
        "recurrence": [-1],
    },
    # {
    #     "seed": list(range(3)),
    #     "rnn_type": ["lstm"],
    #     "serial_mode": [True],
    #     "async_rl": [True],
    #     "learning_rate": [5e-5],
    #     "rnn_num_layers": [1],
    #     "actor_critic_share_weights": [True],
    #     "rnn_d_model": [512],
    #     "rnn_d_output": [512],
    #     "decoder_mlp_layers": ["[512, 512]"],
    #     "rollout": [32],
    #     "recurrence": [-1],
    # },
    # {
    #     "seed": list(range(3)),
    #     "rnn_type": ["lstm"],
    #     "serial_mode": [True],
    #     "async_rl": [True],
    #     "learning_rate": [5e-5],
    #     "rnn_num_layers": [1],
    #     "actor_critic_share_weights": [True],
    #     "rnn_d_model": [512],
    #     "rnn_d_output": [512],
    #     "decoder_mlp_layers": ["[512, 512]"],
    #     "rollout": [32],
    #     "recurrence": [-1],
    #     "policy_initialization": ["orthogonal"],
    # },
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
