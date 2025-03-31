from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "train_for_env_steps": 500_000,
    "group": "monk-APPO-KLAA-T",
    "num_workers": 8,
    "num_envs_per_worker": 16,
    "worker_num_splits": 2,
    "rollout": 128,  # like in the paper
    "async_rl": True,
    "restart_behavior": "overwrite",
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "e-dobrowolska",
    "wandb_project": "atari",
    "wandb_group": "test plasticity log",
    "wandb_tags": [name],
    "batch_size": 256,
    "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": False,
    "serial_mode": False,
    "use_pretrained_checkpoint": False,
    "kickstarting_loss_coeff": 0.0,
    "load_checkpoint_kind": "best",
    "reward_scale": 0.01,
    "device": "cpu",
}

# params different between exps
atari_games = ["montezuma"]

params_grid = []

for atari_game in atari_games:
    for learning_rate in [0.0001]:
        params_grid += [
            {
                "seed": list(range(1)),
                "learning_rate": [learning_rate],
                "env": [f"atari_{atari_game}"],
                "cleanrl_actor_critic": [False],
                "actor_critic_share_weights": [False],
                "delta": [0.99],
                "with_rnd": [True],
                "gamma": [0.999],  # extrinsic gamma
                "num_epochs": [4],
                "repeat_action_probability": [0.25],
                "use_shrink_perturb": [True],
                "freq_shrink_perturb": [150_000],
                "modules_to_perturb": [["predictor_network", "int_critic"]],
                "l2_init_loss_coeff": [0.01],
                "log_heatmaps_to_wandb": [True],
                "save_heatmaps_locally": [True],
                "heatmap_save_freq": [100_000],
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