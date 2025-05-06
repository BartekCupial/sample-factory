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
    "wandb_project": "atari plasticity_ed",
    "wandb_group": "rnd params, corrected",
    "wandb_tags": [name],
    "batch_size": 4096,  # like in CleanRL
    "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "use_pretrained_checkpoint": False,
    "kickstarting_loss_coeff": 0.0,
    "load_checkpoint_kind": "best",
}

# params different between exps
atari_game = "montezuma"

params_grid = [
    {
        "seed": list(range(10)),
        "learning_rate": [1e-4],
        "async_rl": [True],
        "env": [f"atari_{atari_game}"],
        "cleanrl_actor_critic": [True],
        "delta": [0.99],
        "with_rnd": [True],
        "gamma": [0.999],  # extrinsic gamma
        "gae_lambda": [0.95],
        "num_epochs": [4],
        "exploration_loss_coeff": [0.001],
        "repeat_action_probability": [0.25],
        "lr_schedule": ["linear_decay"],  # like in CleanRL
        "lr_adaptive_min": [0.00005],
        "adam_eps": [1e-5],  # like in CleanRL
        "skip_train": [6400],
        "use_shrink_perturb": [False],
        "l2_init_loss_coeff": [0.0],
        "int_gamma": [0.99, 0.999],
        "int_coeff": [1],
        "ext_coeff": [2],
        "log_heatmaps_to_wandb": [True],
        "save_heatmaps_locally": [True],
        "heatmap_save_freq": [5_000_000],
    },
    {
        "seed": list(range(10)),
        "learning_rate": [1e-4],
        "async_rl": [True],
        "env": [f"atari_{atari_game}"],
        "cleanrl_actor_critic": [True],
        "delta": [0.99],
        "with_rnd": [True],
        "gamma": [0.999],  # extrinsic gamma
        "gae_lambda": [0.95],
        "num_epochs": [4],
        "exploration_loss_coeff": [0.001],
        "repeat_action_probability": [0.25],
        "lr_schedule": ["linear_decay"],  # like in CleanRL
        "lr_adaptive_min": [0.00005],
        "adam_eps": [1e-5],  # like in CleanRL
        "skip_train": [6400],
        "use_shrink_perturb": [False],
        "l2_init_loss_coeff": [0.0],
        "int_gamma": [0.99, 0.999],
        "int_coeff": [2],
        "ext_coeff": [1],
        "log_heatmaps_to_wandb": [True],
        "save_heatmaps_locally": [True],
        "heatmap_save_freq": [5_000_000],
    },
    {
        "seed": list(range(10)),
        "learning_rate": [1e-4],
        "async_rl": [True],
        "env": [f"atari_{atari_game}"],
        "cleanrl_actor_critic": [True],
        "delta": [0.99],
        "with_rnd": [True],
        "gamma": [0.999],  # extrinsic gamma
        "gae_lambda": [0.95],
        "num_epochs": [4],
        "exploration_loss_coeff": [0.001],
        "repeat_action_probability": [0.25],
        "lr_schedule": ["linear_decay"],  # like in CleanRL
        "lr_adaptive_min": [0.00005],
        "adam_eps": [1e-5],  # like in CleanRL
        "skip_train": [6400],
        "use_shrink_perturb": [False],
        "l2_init_loss_coeff": [0.0],
        "int_gamma": [0.99, 0.999],
        "int_coeff": [3],
        "ext_coeff": [1],
        "log_heatmaps_to_wandb": [True],
        "save_heatmaps_locally": [True],
        "heatmap_save_freq": [5_000_000],
    },
    {
        "seed": list(range(10)),
        "learning_rate": [1e-4],
        "async_rl": [True],
        "env": [f"atari_{atari_game}"],
        "cleanrl_actor_critic": [True],
        "delta": [0.99],
        "with_rnd": [True],
        "gamma": [0.999],  # extrinsic gamma
        "gae_lambda": [0.95],
        "num_epochs": [4],
        "exploration_loss_coeff": [0.001],
        "repeat_action_probability": [0.25],
        "lr_schedule": ["linear_decay"],  # like in CleanRL
        "lr_adaptive_min": [0.00005],
        "adam_eps": [1e-5],  # like in CleanRL
        "skip_train": [6400],
        "use_shrink_perturb": [False],
        "l2_init_loss_coeff": [0.0],
        "int_gamma": [0.99, 0.999],
        "int_coeff": [1],
        "ext_coeff": [3],
        "log_heatmaps_to_wandb": [True],
        "save_heatmaps_locally": [True],
        "heatmap_save_freq": [5_000_000],
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