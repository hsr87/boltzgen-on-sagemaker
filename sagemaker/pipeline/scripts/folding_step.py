#!/usr/bin/env python3
"""
Folding Step - SageMaker Processing Script

This script runs inside the SageMaker Processing container to validate
designed sequences by re-predicting their structures using Boltz-2.

Input:
    /opt/ml/processing/input/inverse_folded/ - Inverse-folded structures
    /opt/ml/processing/input/config/ - Pipeline configuration

Output:
    /opt/ml/processing/output/folded/ - Re-folded structures for validation
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
from typing import List, Tuple


# SageMaker Processing paths
INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")

INVERSE_FOLDED_INPUT_DIR = INPUT_DIR / "inverse_folded"
CONFIG_DIR = INPUT_DIR / "config"
FOLDED_OUTPUT_DIR = OUTPUT_DIR / "folded"
METADATA_DIR = OUTPUT_DIR / "metadata"


def setup_directories():
    """Create necessary output directories."""
    FOLDED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load pipeline configuration."""
    config_file = CONFIG_DIR / "step_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def get_inverse_folded_directories() -> List[Path]:
    """Get list of inverse-folded output directories to process."""
    dirs = []
    if INVERSE_FOLDED_INPUT_DIR.exists():
        for d in INVERSE_FOLDED_INPUT_DIR.iterdir():
            if d.is_dir():
                dirs.append(d)
    return sorted(dirs)


def count_structures(directory: Path) -> int:
    """Count number of structure files in a directory."""
    count = 0
    # Look for inverse-folded structures
    inverse_folded_dir = directory / "intermediate_designs_inverse_folded"
    if inverse_folded_dir.exists():
        count += len(list(inverse_folded_dir.glob("*.cif")))
    else:
        # Fallback to direct CIF files
        count += len(list(directory.glob("**/*.cif")))
    return count


def partition_work(total_items: int, instance_count: int, worker_id: int) -> Tuple[int, int]:
    """Calculate work partition for this worker.

    Returns:
        (start_index, count) for this worker
    """
    base_count = total_items // instance_count
    remainder = total_items % instance_count

    if worker_id < remainder:
        start = worker_id * (base_count + 1)
        count = base_count + 1
    else:
        start = remainder * (base_count + 1) + (worker_id - remainder) * base_count
        count = base_count

    return start, count


def run_boltzgen_folding(
    input_dir: Path,
    output_dir: Path,
    config: dict,
    design_folding: bool = False,
) -> dict:
    """Run BoltzGen structure folding/validation."""

    start_time = time.time()

    # Set environment
    env = os.environ.copy()
    step_name = "design_folding" if design_folding else "folding"
    env["BOLTZGEN_PIPELINE_STEP"] = step_name

    # Build command using boltzgen execute with config
    config_dir = input_dir / "config"

    if config_dir.exists():
        # Use pre-generated config
        config_file = config_dir / f"{step_name}.yaml"
        if config_file.exists():
            cmd = [
                "python", "-m", "boltzgen.resources.main",
                str(config_file),
            ]
        else:
            # Fallback to folding.yaml
            cmd = [
                "python", "-m", "boltzgen.resources.main",
                str(config_dir / "folding.yaml"),
            ]
    else:
        # Build command manually
        cmd = [
            "boltzgen", "run",
            "--output", str(output_dir),
            "--steps", step_name,
            "--devices", str(config.get("devices", 1)),
        ]

        if config.get("reuse", True):
            cmd.append("--reuse")

    print(f"Running command: {' '.join(cmd)}")

    # Run the command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    elapsed_time = time.time() - start_time

    return {
        "success": result.returncode == 0,
        "return_code": result.returncode,
        "elapsed_time": elapsed_time,
        "stdout": result.stdout[-10000:] if result.stdout else "",
        "stderr": result.stderr[-10000:] if result.stderr else "",
        "command": " ".join(cmd),
        "step_type": step_name,
    }


def copy_intermediate_results(src_dir: Path, dst_dir: Path):
    """Copy intermediate results to maintain pipeline state."""
    # Copy inverse-folded designs
    inverse_folded_src = src_dir / "intermediate_designs_inverse_folded"
    if inverse_folded_src.exists():
        inverse_folded_dst = dst_dir / "intermediate_designs_inverse_folded"
        if not inverse_folded_dst.exists():
            print(f"Copying inverse-folded designs to {inverse_folded_dst}")
            shutil.copytree(inverse_folded_src, inverse_folded_dst)

    # Copy original intermediate designs if present
    intermediate_src = src_dir / "intermediate_designs"
    if intermediate_src.exists():
        intermediate_dst = dst_dir / "intermediate_designs"
        if not intermediate_dst.exists():
            print(f"Copying intermediate designs to {intermediate_dst}")
            shutil.copytree(intermediate_src, intermediate_dst)

    # Copy config if present
    config_src = src_dir / "config"
    if config_src.exists():
        config_dst = dst_dir / "config"
        if not config_dst.exists():
            print(f"Copying config to {config_dst}")
            shutil.copytree(config_src, config_dst)


def save_metadata(metadata: dict):
    """Save step execution metadata."""
    metadata_file = METADATA_DIR / "folding_step_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="BoltzGen Folding Step")
    parser.add_argument("--protocol", type=str, default="protein-anything",
                       help="Design protocol")
    parser.add_argument("--instance-count", type=int, default=1,
                       help="Total number of processing instances")
    parser.add_argument("--worker-id", type=int, default=0,
                       help="This worker's ID (0-indexed)")
    parser.add_argument("--devices", type=int, default=1,
                       help="Number of GPU devices")
    parser.add_argument("--skip-design-folding", action="store_true",
                       help="Skip design folding step (for peptides/nanobodies)")

    args = parser.parse_args()

    print("=" * 60)
    print("BoltzGen Folding Step")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Worker ID: {args.worker_id} / {args.instance_count}")
    print(f"Protocol: {args.protocol}")
    print(f"Skip design folding: {args.skip_design_folding}")

    # Setup
    setup_directories()
    config = load_config()

    # Override config with command line args
    config.update({
        "protocol": args.protocol,
        "devices": args.devices,
    })

    # Determine if we need design folding based on protocol
    do_design_folding = args.protocol in ["protein-anything", "protein-small_molecule"]
    if args.skip_design_folding:
        do_design_folding = False

    print(f"Design folding enabled: {do_design_folding}")

    # Get input directories
    input_dirs = get_inverse_folded_directories()
    if not input_dirs:
        print("ERROR: No inverse-folded directories found!")
        sys.exit(1)

    print(f"Found {len(input_dirs)} input directory(ies)")

    # Process each directory
    all_results = []
    overall_success = True

    for input_dir in input_dirs:
        print(f"\nProcessing: {input_dir.name}")

        # Count structures
        num_structures = count_structures(input_dir)
        print(f"  Found {num_structures} structures to fold")

        # Create output directory
        output_dir = FOLDED_OUTPUT_DIR / input_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy intermediate results
        copy_intermediate_results(input_dir, output_dir)

        # Run folding (complex re-folding)
        print("  Running complex folding...")
        result = run_boltzgen_folding(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            design_folding=False,
        )

        result["input_dir"] = input_dir.name
        result["num_structures"] = num_structures
        result["worker_id"] = args.worker_id
        all_results.append(result)

        if result["success"]:
            print(f"  SUCCESS: Complex folding completed in {result['elapsed_time']:.1f}s")
        else:
            print(f"  FAILED: Return code {result['return_code']}")
            if result.get("stderr"):
                print(f"  STDERR: {result['stderr'][:500]}")
            overall_success = False

        # Run design folding if enabled
        if do_design_folding and overall_success:
            print("  Running design folding...")
            design_result = run_boltzgen_folding(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                design_folding=True,
            )

            design_result["input_dir"] = input_dir.name
            design_result["worker_id"] = args.worker_id
            all_results.append(design_result)

            if design_result["success"]:
                print(f"  SUCCESS: Design folding completed in {design_result['elapsed_time']:.1f}s")
            else:
                print(f"  WARNING: Design folding failed (non-critical)")

    # Save metadata
    metadata = {
        "step": "folding",
        "timestamp": datetime.now().isoformat(),
        "worker_id": args.worker_id,
        "instance_count": args.instance_count,
        "protocol": args.protocol,
        "design_folding_enabled": do_design_folding,
        "results": all_results,
        "overall_success": overall_success,
    }
    save_metadata(metadata)

    print("\n" + "=" * 60)
    if overall_success:
        print("Folding step completed successfully!")
    else:
        print("Folding step completed with errors!")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
