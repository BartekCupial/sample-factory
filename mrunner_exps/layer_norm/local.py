from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO",
    "train_for_env_steps": 1_000_000,
    "group": "monk-APPO",
    "character": "mon-hum-neu-mal",
    "num_workers": 16,
    "num_envs_per_worker": 2,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 1024,  # this equals bs = 128, 128 * 32 = 4096
    "async_rl": True,
    "use_prev_action": True,
    "restart_behavior": "overwrite",
    "run_teacher_hs": False,
    "use_prev_action": True,
    "model": "ScaledNet",
    "use_resnet": True,
    "rnn_size": 1738,
    "serial_mode": False,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "num_workers": [8],
        "batch_size": [512],
        "use_pretrained_checkpoint": [True],
        "teacher_path": ["/home/bartek/Workspace/data/sf_checkpoints/amzn-AA-BC/pretrained_use_prev_action"],
        "model_path": ["/home/bartek/Workspace/data/sf_checkpoints/amzn-AA-BC/pretrained_use_prev_action"],
        "freeze": [{"actor_encoder": 0, "actor_core": 0, "actor_decoder": 0, "action_parameterization": 0}],
        "actor_critic_share_weights": [False],
        "critic_add_layernorm": [True],
        "critic_mlp_layers": [[512], [512, 512], [512, 512, 512]],
    },
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
