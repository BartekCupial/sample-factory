from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from sf_examples.nethack.utils.paramiko import get_checkpoint_paths

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_script": "sf_examples.nethack.eval_nethack",
    "env": "challenge",
    "exp_tags": [name],
    "character": "mon-hum-neu-mal",
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_nethack",
    "wandb_group": "gmum",
    "with_wandb": True,
    "use_pretrained_checkpoint": False,
    "load_checkpoint_kind": "latest",
    "train_dir": "/home/bartek/Workspace/ideas/sample-factory/train_dir",
    "experiment": "amzn-AA-BC_pretrained",
    "sample_env_episodes": 32,
    "num_workers": 16,
    "num_envs_per_worker": 1,
    "worker_num_splits": 1,
    "restart_behavior": "overwrite",
    "save_videos": True,
    "observation_keys": (
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        # ALSO AVAILABLE (OFF for speed)
        "specials",
        "colors",
        "chars",
        "glyphs",
        "inv_glyphs",
        "inv_strs",
        "inv_letters",
        "inv_oclasses",
    ),
}

csv_folder_name = f"{config['character']}_episodes{config['sample_env_episodes']}"

method_paths = [
    "/net/tscratch/people/plgbartekcupial/mrunner_scratch/sf2_nethack/05_07-12_57-musing_bohr",
]

params_grid = []
for method_path in method_paths:
    checkpoints = get_checkpoint_paths(method_path)
    checkpoints = list(map(Path, checkpoints))
    checkpoints = list(filter(lambda p: p.parent.name in ["default_experiment"], checkpoints))

    for checkpoint in checkpoints:
        train_dir = str(checkpoint.parent.parent)
        experiment = checkpoint.parent.name
        savedir = f"{train_dir}/{experiment}/{csv_folder_name}/nle_data"
        video_folder_names = [f"{train_dir}/{experiment}/videos_{i}" for i in range(5)]
        params_grid.append(
            {
                "csv_folder_name": [csv_folder_name],
                "video_folder_name": video_folder_names,
                "train_dir": [train_dir],
                "experiment": [experiment],
                "savedir": [savedir],
                "save_ttyrec_every": [1],
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
