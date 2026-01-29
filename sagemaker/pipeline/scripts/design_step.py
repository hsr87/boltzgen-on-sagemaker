#!/usr/bin/env python3
"""
Design Step - SageMaker Processing Script

This script runs inside the SageMaker Processing container to generate
protein designs using the BoltzGen diffusion model.

Input:
    /opt/ml/processing/input/design_specs/ - Design specification YAML files
    /opt/ml/processing/input/config/ - Pipeline configuration

Output:
    /opt/ml/processing/output/designs/ - Generated design structures (CIF/NPZ)
    /opt/ml/processing/output/metadata/ - Step metadata and logs
"""

import argparse
import json
import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


# SageMaker Processing paths
INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")
CACHE_DIR = Path("/opt/ml/processing/cache")

DESIGN_SPECS_DIR = INPUT_DIR / "design_specs"
CONFIG_DIR = INPUT_DIR / "config"
DESIGNS_OUTPUT_DIR = OUTPUT_DIR / "designs"
METADATA_DIR = OUTPUT_DIR / "metadata"


def setup_directories():
    """Create necessary output directories."""
    DESIGNS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load pipeline configuration."""
    config_file = CONFIG_DIR / "step_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def get_design_specs() -> list:
    """Get list of design specification files."""
    specs = []
    if DESIGN_SPECS_DIR.exists():
        for ext in ["*.yaml", "*.yml"]:
            specs.extend(DESIGN_SPECS_DIR.glob(ext))
    return sorted(specs)


def calculate_designs_per_worker(total_designs: int, instance_count: int, worker_id: int) -> tuple:
    """Calculate number of designs for this worker instance.

    Returns:
        (start_index, num_designs) for this worker
    """
    base_designs = total_designs // instance_count
    remainder = total_designs % instance_count

    # First 'remainder' workers get one extra design
    if worker_id < remainder:
        start = worker_id * (base_designs + 1)
        count = base_designs + 1
    else:
        start = remainder * (base_designs + 1) + (worker_id - remainder) * base_designs
        count = base_designs

    return start, count


def run_boltzgen_design(
    design_spec: Path,
    output_dir: Path,
    num_designs: int,
    config: dict,
) -> dict:
    """Run BoltzGen design generation."""

    start_time = time.time()

    # Build command
    cmd = [
        "boltzgen", "run",
        str(design_spec),
        "--output", str(output_dir),
        "--protocol", config.get("protocol", "protein-anything"),
        "--num_designs", str(num_designs),
        "--budget", str(config.get("budget", 100)),
        "--devices", str(config.get("devices", 1)),
        "--steps", "design",  # Only run design step
    ]

    # Add optional arguments
    if config.get("diffusion_batch_size"):
        cmd.extend(["--diffusion_batch_size", str(config["diffusion_batch_size"])])

    if config.get("design_checkpoints"):
        for checkpoint in config["design_checkpoints"]:
            cmd.extend(["--design_checkpoints", checkpoint])

    if config.get("step_scale"):
        cmd.extend(["--step_scale", str(config["step_scale"])])

    if config.get("noise_scale"):
        cmd.extend(["--noise_scale", str(config["noise_scale"])])

    # Add reuse flag for checkpointing
    if config.get("reuse", True):
        cmd.append("--reuse")

    print(f"Running command: {' '.join(cmd)}")

    # Run the command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    elapsed_time = time.time() - start_time

    return {
        "success": result.returncode == 0,
        "return_code": result.returncode,
        "elapsed_time": elapsed_time,
        "stdout": result.stdout[-10000:] if result.stdout else "",  # Last 10KB
        "stderr": result.stderr[-10000:] if result.stderr else "",
        "command": " ".join(cmd),
    }


def save_metadata(metadata: dict):
    """Save step execution metadata."""
    metadata_file = METADATA_DIR / "design_step_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="BoltzGen Design Step")
    parser.add_argument("--num-designs", type=int, default=10000,
                       help="Total number of designs to generate")
    parser.add_argument("--protocol", type=str, default="protein-anything",
                       help="Design protocol")
    parser.add_argument("--budget", type=int, default=100,
                       help="Final design budget")
    parser.add_argument("--instance-count", type=int, default=1,
                       help="Total number of processing instances")
    parser.add_argument("--worker-id", type=int, default=0,
                       help="This worker's ID (0-indexed)")
    parser.add_argument("--devices", type=int, default=1,
                       help="Number of GPU devices")

    args = parser.parse_args()

    print("=" * 60)
    print("BoltzGen Design Step")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Worker ID: {args.worker_id} / {args.instance_count}")
    print(f"Total designs requested: {args.num_designs}")
    print(f"Protocol: {args.protocol}")

    # Setup
    setup_directories()
    config = load_config()

    # Override config with command line args
    config.update({
        "protocol": args.protocol,
        "budget": args.budget,
        "devices": args.devices,
    })

    # Calculate designs for this worker
    start_idx, worker_designs = calculate_designs_per_worker(
        args.num_designs, args.instance_count, args.worker_id
    )
    print(f"This worker will generate designs {start_idx} to {start_idx + worker_designs - 1}")
    print(f"Number of designs for this worker: {worker_designs}")

    # Get design specs
    design_specs = get_design_specs()
    if not design_specs:
        print("ERROR: No design specification files found!")
        sys.exit(1)

    print(f"Found {len(design_specs)} design specification(s)")

    # Process each design spec
    all_results = []
    overall_success = True

    for spec in design_specs:
        print(f"\nProcessing: {spec.name}")

        # Create output directory for this spec
        spec_output_dir = DESIGNS_OUTPUT_DIR / spec.stem
        spec_output_dir.mkdir(parents=True, exist_ok=True)

        # Run design generation
        result = run_boltzgen_design(
            design_spec=spec,
            output_dir=spec_output_dir,
            num_designs=worker_designs,
            config=config,
        )

        result["design_spec"] = spec.name
        result["worker_id"] = args.worker_id
        all_results.append(result)

        if result["success"]:
            print(f"SUCCESS: Generated designs in {result['elapsed_time']:.1f}s")
        else:
            print(f"FAILED: Return code {result['return_code']}")
            print(f"STDERR: {result['stderr'][:1000]}")
            overall_success = False

    # Save metadata
    metadata = {
        "step": "design",
        "timestamp": datetime.now().isoformat(),
        "worker_id": args.worker_id,
        "instance_count": args.instance_count,
        "total_designs": args.num_designs,
        "worker_designs": worker_designs,
        "protocol": args.protocol,
        "results": all_results,
        "overall_success": overall_success,
    }
    save_metadata(metadata)

    print("\n" + "=" * 60)
    if overall_success:
        print("Design step completed successfully!")
    else:
        print("Design step completed with errors!")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
