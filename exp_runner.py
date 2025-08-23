import os
import sys
import subprocess
import importlib.util
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import json


def load_experiment_config(config_file_path: str) -> List[Dict[str, str]]:
    experiments = []
    
    with open(config_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            try:
                exp = json.loads(line)
                experiments.append(exp)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
    return experiments


def create_sbatch_script(
    command: str,
    script_path: str,
    log_path: str,
    job_name: str,
    venv_path: str,
    partition: str,
    account: str,
    time: str = "24:00:00",
    nodes: int = 1,
    ntasks: int = 1,
    cpus: int = 8,
    mem: str = "32G",
    gpu: int = 1,
) -> None:

    sbatch_content = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --ntasks={ntasks}
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{gpu}

ml ML-bundle/24.06a

export WANDB_API_KEY=...
cd "$(dirname "$0")"

"""
    
    # Add virtual environment activation if specified
    if venv_path:
        sbatch_content += f"source {venv_path}/bin/activate"
    
    # Add the actual command
    sbatch_content += f"""

{command}

echo ""
echo "Job finished at: $(date)"
"""
    
    with open(script_path, 'w') as f:
        f.write(sbatch_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)


def submit_job(sbatch_script_path: str, dry_run: bool = False) -> str:
    if dry_run:
        return
    
    try:
        subprocess.run(
            ["sbatch", sbatch_script_path],
            capture_output=True,
            text=True,
            check=True
        )
        return
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job {sbatch_script_path}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return

def main():
    parser = argparse.ArgumentParser(description="Run experiments with SLURM")
    parser.add_argument("config_file", help="Path to the experiment configs")
    parser.add_argument("base_dir", default="...", help="Storage_dir")
    parser.add_argument("--venv", default=".atari_venv", help="Path to virtual environment to activate")
    parser.add_argument("--account", default="...", help="Account")
    parser.add_argument("--partition", default="...", help="Partition")
    parser.add_argument("--time", default="2880", help="Job time limit")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1)")
    parser.add_argument("--ntasks", type=int, default=1, help="Tasks per node (default: 1)")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per task (default: 8)")
    parser.add_argument("--mem", default="32G", help="Memory per node (default: 32G)")
    parser.add_argument("--gpu", default="1", help="Number of gpus (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually submit jobs, just show what would be done")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config_file):
        print(f"Error: Config file {args.config_file} does not exist")
        sys.exit(1)
    
    # Load experiment configuration
    print(f"Loading experiment configuration from {args.config_file}...")
    try:
        experiments = load_experiment_config(args.config_file)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    project_name = experiments[0].get('project_name')
    unique_name = experiments[0].get('unique_name')

    print(f"Running {len(experiments)} experiments")
    
    # Create directory structure
    base_path = Path(args.base_dir)
    project_path = base_path / project_name
    unique_path = project_path / unique_name
    
    unique_path.mkdir(parents=True, exist_ok=True)
    
    # Generate and submit jobs
    job_ids = []
    
    for i, exp in enumerate(experiments):
        exp_name = f"{exp['name']}_{i}"
        exp_dir = unique_path / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Paths for script and log files
        script_path = exp_dir / f"launch.sbatch"
        log_path = exp_dir / f"log.out"
        
        command = exp.get('command')

        # Create SLURM batch script
        create_sbatch_script(
            command=command,
            script_path=str(script_path),
            log_path=str(log_path),
            job_name=exp_name,
            venv_path=args.venv,
            partition=args.partition,
            time=args.time,
            nodes=args.nodes,
            ntasks=args.ntasks,
            cpus=args.cpus,
            mem=args.mem,
            gpu=args.gpu,
            account=args.account
        )
        
        # Submit job
        submit_job(str(script_path), dry_run=args.dry_run)

    print(f"Experiment files created in: {unique_path}")


if __name__ == "__main__":
    main()