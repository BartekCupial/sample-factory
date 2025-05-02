from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "train_for_env_steps": 300_000_000,
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
    "batch_size": 4096,  # like in CleanRL
    "dataset_batch_size": 512,  # this equals bs = 512, 512 * 32 = 16384
    "with_wandb": True,
    "serial_mode": False,
    "use_pretrained_checkpoint": False,
    "kickstarting_loss_coeff": 0.0,
    "load_checkpoint_kind": "best",
}

# params different between exps
games = ["phoenix", "qbert", "breakout", "seaquest", "namethisgame"]
params_grid = []

for atari_game in games:
    params_grid.extend([
        # {
        #     "wandb_tags": ["Control"],
        #     "wandb_group": [f"S+P, {atari_game}"],
        #     "seed": list(range(5)),
        #     "learning_rate": [0.00025],
        #     "async_rl": [True],
        #     "env": [f"atari_{atari_game}"],
        #     "actor_critic_share_weights": [False],
        #     "delta": [0.99],
        #     "num_epochs": [4],
        #     "exploration_loss_coeff": [0.001],
        #     "repeat_action_probability": [0.25],
        #     "lr_schedule": ["constant"],  # like in CleanRL
        #     "adam_eps": [1e-5],  # like in CleanRL
        #     "l2_init_loss_coeff": [0.0],
        # },
        {
            "wandb_tags": ["Sharp RL decay"],
            "wandb_group": [f"S+P, {atari_game}"],
            "seed": list(range(5)),
            "learning_rate": [0.001],
            "async_rl": [True],
            "env": [f"atari_{atari_game}"],
            "actor_critic_share_weights": [False],
            "delta": [0.99],
            "num_epochs": [4],
            "exploration_loss_coeff": [0.001],
            "repeat_action_probability": [0.25],
            "lr_schedule": ["linear_decay"],  # like in CleanRL
            "adam_eps": [1e-5],  # like in CleanRL
            "l2_init_loss_coeff": [0.0],
            "use_shrink_perturb": [True],
            "freq_shrink_perturb": [50_000_000],
            "shrink": [0.4],
            "perturb": [0.6],
        },
        {
            "wandb_tags": ["Epochs"],
            "wandb_group": [f"S+P, {atari_game}"],
            "seed": list(range(5)),
            "learning_rate": [0.00025],
            "async_rl": [True],
            "env": [f"atari_{atari_game}"],
            "actor_critic_share_weights": [False],
            "delta": [0.99],
            "num_epochs": [12],
            "exploration_loss_coeff": [0.001],
            "repeat_action_probability": [0.25],
            "lr_schedule": ["constant"],  # like in CleanRL
            "adam_eps": [1e-5],  # like in CleanRL
            "l2_init_loss_coeff": [0.0],
            "use_shrink_perturb": [True],
            "freq_shrink_perturb": [50_000_000],
            "shrink": [0.4],
            "perturb": [0.6],
        },
    ])

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