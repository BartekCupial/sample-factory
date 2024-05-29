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
    "serial_mode": False,
    "use_prev_action": True,
    "restart_behavior": "overwrite",
    "dataset_rollout": 32,
    "dataset_batch_size": 1024,  # this equals bs = 256, 256 * 32 = 8192
    "run_teacher_hs": False,
    "use_prev_action": True,
    "model": "ScaledNet",
    "use_resnet": True,
    "rnn_size": 1738,
    "gae_lambda": 0.985,
    "gamma": 1.0,
}

# params different between exps
params_grid = []
expected_batch_size = 256

for rollout in [32]:
    for target_batch_size in [32]:
        batch_size = min(expected_batch_size, min(target_batch_size * rollout, expected_batch_size * 8))
        batches_to_accumulate = max(1, (rollout * target_batch_size) // expected_batch_size)
        optim_step_every_ith = max(1, batches_to_accumulate // 8)
        for actor_learning_rate in [0.00005, 0.00001, 0.000005, 0.000001]:
            params_grid.append(
                {
                    "seed": list(range(1)),
                    "learning_rate": [0.0001],
                    "rollout": [rollout],
                    "batch_size": [batch_size],  # 32 * 512, 64 * 256, 128 * 128
                    "num_batches_per_epoch": [min(8, batches_to_accumulate)],
                    "optim_step_every_ith": [optim_step_every_ith],
                    "target_batch_size": [target_batch_size],
                    "db_path": ["/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db"],
                    "use_pretrained_checkpoint": [True],
                    "teacher_path": [
                        "/home/bartek/Workspace/data/sf_checkpoints/amzn-AA-BC/pretrained_use_prev_action"
                    ],
                    "model_path": ["/home/bartek/Workspace/data/sf_checkpoints/amzn-AA-BC/pretrained_use_prev_action"],
                    "freeze": [{"actor_encoder": 0}],
                    "actor_critic_share_weights": [False],
                    "learning_rate_groups": [{"critic": 0.001, "action_parameterization": 0.1}],
                    "aux_train": [True],
                    "aux_num_epochs": [3, 6, 9],
                    "aux_train_frequency": [8, 16],
                    "aux_kl_loss_coeff": [1.0],
                    "aux_batch_size": [batch_size],
                    "heartbeat_reporting_interval": [180 * 4],
                }
            )


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
