import argparse
import yaml
import itertools
import subprocess
from pathlib import Path
import copy
import sys

def generate_configs(base_config_path: str, sweep_config_path: str, output_dir: str):
    """
    Generate all config files for sweep.
    """
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    with open(sweep_config_path) as f:
        sweep = yaml.safe_load(f)

    # Example sweep config:
    # activation: [identity, relu, softmax, tanh, leaky_relu, softplus]
    # hidden_dim: [16, 32, 64]
    # lambda_tc: [0.1, 1.0, 10.0]
    # seed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]

    configs = []

    for combo in itertools.product(*values):
        config = copy.deepcopy(base_config)
        name_parts = []

        for key, val in zip(keys, combo):
            # Navigate to correct config location
            if key == "activation":
                config["model"]["activation"] = val
            elif key == "hidden_dim":
                config["model"]["hidden_dim"] = val
            elif key == "lambda_tc":
                config["loss"]["lambda_tc"] = val
            elif key == "lambda_wr":
                config["loss"]["lambda_wr"] = val
            elif key == "optimizer":
                config["training"]["optimizer"] = val
            elif key == "seed":
                config["training"]["seed"] = val

            name_parts.append(f"{key}={val}")

        name = "_".join(name_parts)
        config_path = output_dir / f"{name}.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        configs.append({
            "name": name,
            "config_path": str(config_path),
            "output_path": str(output_dir / name),
        })

    return configs

def run_sweep(configs: list, parallel: int = 1):
    """
    Run all experiments.
    """
    total = len(configs)
    print(f"\n{'='*70}")
    print(f"Running Parameter Sweep: {total} experiments")
    print(f"{'='*70}\n")

    import time
    start_time = time.time()
    failed = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total}] Running: {config['name']}")
        print(f"Config: {config['config_path']}")
        print(f"Output: {config['output_path']}")
        print("-" * 70)

        result = subprocess.run([
            "python", "scripts/run_experiment.py",
            "--config", config["config_path"],
            "--output", config["output_path"],
        ])

        if result.returncode != 0:
            print(f"⚠ Experiment failed with return code {result.returncode}")
            failed.append(config['name'])
        else:
            print(f"✓ Experiment {i}/{total} completed successfully")

        # Estimate time remaining
        elapsed = time.time() - start_time
        if i > 0:
            avg_time = elapsed / i
            remaining = avg_time * (total - i)
            print(f"Estimated time remaining: {remaining/60:.1f} minutes")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Sweep completed: {total} experiments finished")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful: {total - len(failed)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed experiments: {', '.join(failed)}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--sweep-config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    print("Generating experiment configurations...")
    configs = generate_configs(args.base_config, args.sweep_config, args.output)
    print(f"✓ Generated {len(configs)} configurations")

    run_sweep(configs, args.parallel)
