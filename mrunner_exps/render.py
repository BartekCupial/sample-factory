from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_script": "sf_examples.nethack.eval_nethack",
    "env": "challenge",
    "exp_tags": [name],
    "character": "mon-hum-neu-mal",
    # "character": "@",
    "with_wandb": True,
    "use_pretrained_checkpoint": False,
    "load_checkpoint_kind": "latest",
    "train_dir": "/home/bartek/Workspace/ideas/sample-factory/train_dir",
    "experiment": "amzn-AA-BC_pretrained",
    "sample_env_episodes": 16,
    "num_workers": 4,
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
train_dir = "train_dir"
experiment = "amzn-AA-BC_pretrained"
video_folder_names = [f"{train_dir}/{experiment}/videos_{i}" for i in range(5)]

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "csv_folder_name": [csv_folder_name],
        "save_ttyrec_every": [1],
        "train_dir": [train_dir],
        "experiment": [experiment],
        "savedir": [f"{train_dir}/{experiment}/{csv_folder_name}/nle_data"],
        "video_folder_name": video_folder_names,
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
