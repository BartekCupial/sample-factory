from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "atari_breakout",
    "exp_tags": [name],
    "train_for_env_steps": 100_000_000,
    "group": "monk-APPO-KLAA-T",
    "num_workers": 8,
    "num_envs_per_worker": 16,
    "worker_num_splits": 2,
    "rollout": 128,
    "async_rl": True,
    "restart_behavior": "overwrite",
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_atari",
    "wandb_group": "gmum",
    "wandb_tags": [name],
    "batch_size": 256,
    "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,
    "use_pretrained_checkpoint": True,
    "skip_train": 5_000_000,
    "device": "cpu",
    "load_checkpoint_kind": "best",
    "reward_scale": 0.01,
}

# params different between exps
atari_games = ["breakout", "qbert", "montezuma", "upndown"]

params_grid = []

for atari_game in atari_games:
    for learning_rate in [1e-4]:
        params_grid += [
            {
                "seed": list(range(1)),
                "learning_rate": [learning_rate],
                "model_path": [f"/atari_checkpoints/{atari_game}/default_experiment/"],
                "teacher_path": [f"/atari_checkpoints/{atari_game}/default_experiment/"],
                "env": [f"atari_{atari_game}"],
                "actor_critic_share_weights": [False],
                "init_critic_from_actor": [True, False],
                "critic_mlp_layers": [[512, 512], [512], []],
                "critic_layer_norm": [True, False],
                "critic_learning_rate": [learning_rate * 10, learning_rate * 5, learning_rate],
                "freeze": [{"actor_encoder": 0, "actor_core": 0, "actor_decoder": 0, "action_parameterization": 0}],
                "unfreeze": [
                    {
                        "actor_encoder": 10_000_000,
                        "actor_core": 10_000_000,
                        "actor_decoder": 10_000_000,
                        "action_parameterization": 10_000_000,
                    }
                ],
                "kickstarting_loss_coeff": [0.01, 0.001, 0.0],
            },
            {
                "seed": list(range(1)),
                "learning_rate": [learning_rate],
                "model_path": [f"/atari_checkpoints/{atari_game}/default_experiment/"],
                "teacher_path": [f"/atari_checkpoints/{atari_game}/default_experiment/"],
                "env": [f"atari_{atari_game}"],
                "actor_critic_share_weights": [True],
                "critic_mlp_layers": [[512, 512], [512], []],
                "critic_layer_norm": [True, False],
                "critic_learning_rate": [learning_rate * 10, learning_rate * 5, learning_rate],
                "freeze": [{"encoder": 0, "core": 0, "decoder": 0, "action_parameterization": 0}],
                "unfreeze": [
                    {
                        "encoder": 10_000_000,
                        "core": 10_000_000,
                        "decoder": 10_000_000,
                        "action_parameterization": 10_000_000,
                    }
                ],
                "kickstarting_loss_coeff": [0.01, 0.001, 0.0],
            },
        ]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="atari_sf",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
