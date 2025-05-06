from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "train_for_env_steps": 1_000_000_000,
    "num_workers": 4,
    "num_envs_per_worker": 8,
    "num_batches_per_epoch": 16,
    "batch_size": 256,
    "worker_num_splits": 2,  #??
    "rollout": 128,  #??
    "save_milestones_ith": 10_000_000,
    # Wandb settings
    "wandb_user": "ideas-ncbr",
    "wandb_project": "atari plasticity_ed",
    "wandb_group": "run montezuma, old",
    "wandb_tags": [name],
    "with_wandb": True,
}

atari_games = ["montezuma"]

params_grid = []

for atari_game in atari_games:
    params_grid += [
        {
            "seed": list(range(5)),

            # Check async
            "async_rl": [True],

            # paper's params: env
            "env": [f"atari_{atari_game}"],
            "normalize_input": [False],
            "env_frameskip": [3],

            # paper's params: model
            "actor_critic_share_weights": [True],
            "encoder_conv_mlp_layers": [[512]],
            "nonlinearity": ["relu"],

            # paper's params: training
            "batch_size": [256],
            "learning_rate": [0.00025],
            "lr_schedule": ["constant"],
            "gamma": [0.99],
            "gae_lambda": [0.95],
            "ppo_clip_ratio": [0.1],  # "We use unbiased clip(x, 1+e, 1/(1+e)) instead of clip(x, 1+e, 1-e) in the paper"
            "exploration_loss_coeff": [0.01],
            "value_loss_coeff": [0.5],
            "max_grad_norm": [0.5],
            "optimizer": ["adam"],
            "num_epochs": [1],
            "normalize_returns": [True],
            "repeat_action_probability": [0.25],
            "skip_train": [6400],
            
            # paper's params: plasticity
            "delta": [0.99],

            # RND
            "with_rnd": [True],
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