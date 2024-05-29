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
    "rollout": 512,
    "async_rl": True,
    "restart_behavior": "overwrite",
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "e-dobrowolska",
    "wandb_project": "atari",
    "wandb_group": "no critc fin",
    "wandb_tags": [name],
    "batch_size": 1024,
    "dataset_batch_size": 1024,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,
    "use_pretrained_checkpoint": True,
    "kickstarting_loss_coeff": 0.0,
    "skip_train": 5_000_000,
    "device": "cpu",
    "load_checkpoint_kind": "best",
    "reward_scale": 0.01,
}

# params different between exps
atari_games = ["breakout"]

params_grid = []

for atari_game in atari_games:
    for learning_rate in [1e-5]:
        params_grid += [
            {
                "seed": list(range(3)),
                "learning_rate": [learning_rate],
                "model_path": [f"/atari_checkpoints/{atari_game}/default_experiment/"],
                "env": [f"atari_{atari_game}"],
                "actor_critic_share_weights": [False],
                "init_critic_from_actor": [False],
                "critic_mlp_layers": [[512, 512], [512], []],
                "critic_layer_norm": [True, False],
                "gae_lambda": [1.0],
                "normalize_returns": [False],
                "value_loss_coeff": [0.0],
                "critic_learning_rate": [learning_rate * 5],
                "freeze": [{"actor_encoder": 0, "actor_core": 0, "actor_decoder": 0, "action_parameterization": 0}],
                "unfreeze": [
                    {
                        "actor_encoder": 50_000_000,
                        "actor_core": 50_000_000,
                        "actor_decoder": 50_000_000,
                        "action_parameterization": 50_000_000,
                    }
                ],
            },
        ]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="atari",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
