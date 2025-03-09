from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "train_for_env_steps": 1_000_000_000,
    "group": "monk-APPO-KLAA-T",
    "num_workers": 8,
    "num_envs_per_worker": 16,  # 8*16=128 - like in the paper
    "worker_num_splits": 2,
    "rollout": 128,  # like in the paper
    "async_rl": True,
    "restart_behavior": "overwrite",
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "ideas-ncbr",
    "wandb_project": "atari",
    "wandb_group": "montezuma rnd v4",
    "wandb_tags": [name],
    "batch_size": 4096,  # like in ClearRL
    "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,
    "use_pretrained_checkpoint": False,
    "kickstarting_loss_coeff": 0.0,
    "load_checkpoint_kind": "best",
}

# params different between exps
atari_games = ["montezuma"]

params_grid = []

for atari_game in atari_games:
    for learning_rate in [1e-4]:
        params_grid += [
            {
                "seed": [1],
                "learning_rate": [learning_rate],
                "async_rl": [True],
                "env": [f"atari_{atari_game}"],
                "actor_critic_share_weights": [True],
                "delta": [0.99],
                "with_rnd": [True],
                "gamma": [0.999],  # extrinsic gamma
                "gae_lambda": [0.95],
                "num_epochs": [4],
                "exploration_loss_coeff": [0.001],
                "repeat_action_probability": [0.25],
                "encoder_mlp_layers": [[448, 448]],  # like in ClearRL
                "lr_schedule": ["linear_decay"],  # like in ClearRL
                "adam_eps": [1e-5],  # like in ClearRL

                # Warm-up workaround
                "freeze": [{"encoder": 0, "core": 0, "decoder": 0, "predictor_network": 0, "critic": 0, "int_critic": 0}],
                "unfreeze": [{"encoder": 6400, "core": 6400, "decoder": 6400, "predictor_network": 6400, "critic": 6400, "int_critic": 6400}],
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