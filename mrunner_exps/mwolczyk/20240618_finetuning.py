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
    "num_workers": 16,
    "num_envs_per_worker": 30,
    "worker_num_splits": 2,
    "rollout": 32,

    "restart_behavior": "overwrite",
    "async_rl": True,
    "serial_mode": True,
    "save_milestones_ith": 10_000_000,
    "wandb_user": "rahid",
    "wandb_project": "sp_nethack",
    "wandb_group": "rahid",
    "model": "ScaledNet",
    "heartbeat_interval": 600,
    "heartbeat_reporting_interval": 1200,
    "process_seq_in_batch_mode": True,

    # Athena
    # "batch_size": 4096,  # this equals bs = 128, 128 * 32 = 4096
    # "with_wandb": True,

    # Local
    "batch_size": 64,  # this equals bs = 128, 128 * 32 = 4096
    "with_wandb": True,

    # Fine-tuning stuff
    "use_pretrained_checkpoint": True,
    "load_checkpoint_kind": "best",

}

# params different between exps
params_grid = [
    # {
    #     "seed": list(range(5)),
    #     "rnn_type": ["lstm"],
    #     "learning_rate": [1e-5, 5e-5],
    #     "rnn_size": [512],
    # },
    {
        "seed": list(range(3)),
        "rnn_type": ["mamba"],
        "rnn_size": [512],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "rnn_num_layers": [3],
        "rollout": [32, 64],
        "max_grad_norm": [4.],
        "mamba_use_complex": [False],
        "mamba_model_size": [256],
        "mamba_selective_ssm": [True],
        # Finetuning params
        "model_path": [f"./default_experiment/"],
        "actor_critic_share_weights": [True],
        "use_prev_action": [True],

        # "init_critic_from_actor": [True, False],
        # "critic_mlp_layers": [[512, 512], [512], []],
        # "critic_layer_norm": [True, False],
        # "critic_learning_rate": [learning_rate * 10, learning_rate * 5, learning_rate],
        # "freeze": [{"actor_encoder": 0, "actor_core": 0, "actor_decoder": 0, "action_parameterization": 0}],
        # "unfreeze": [
        #     {
        #         "actor_encoder": 10_000_000,
        #         "actor_core": 10_000_000,
        #         "actor_decoder": 10_000_000,
        #         "action_parameterization": 10_000_000,
        #     }
        # ],
        "freeze": [{"encoder": 0, "core": 0, "decoder": 0, "action_parameterization": 0}],
        "unfreeze": [
            {"encoder": 50_000_000, "core": 50_000_000, "decoder": 50_000_000, "action_parameterization": 50_000_000}
        ],
    }
    # {
    #     "seed": list(range(5)),
    #     "rnn_type": ["nanogpt"],
    #     "rnn_size": [512],
    #     "learning_rate": [1e-4, 5e-4, 1e-3],
    #     "rollout": [16, 32, 64],
    #     "nanogpt_model_size": [256],
    #     "rnn_num_layers": [3],
    #     "nanogpt_n_head": [8],
    #     "nanogpt_dropout": [0.],
    #     "nanogpt_embedding_type": ["rope"],
    #     "nanogpt_relative_timesteps": [True],
    #     "nanogpt_constant_context": [False],
    #     "nanogpt_block_size": [256],
    #     "nanogpt_attention_type": ["linear"],
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
