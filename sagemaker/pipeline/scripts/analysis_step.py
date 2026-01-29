#!/usr/bin/env python3
"""
Analysis Step - SageMaker Processing Script

This script runs inside the SageMaker Processing container to compute
quality metrics for the designed and folded structures.

This is a CPU-intensive step that calculates:
- pLDDT (prediction confidence)
- RMSD (structural accuracy)
- Interface metrics
- Hydrophobic patches
- And more...

Input:
    /opt/ml/processing/input/folded/ - Folded structures from previous step
    /opt/ml/processing/input/config/ - Pipeline configuration

Output:
    /opt/ml/processing/output/analyzed/ - Analyzed results with metrics CSV
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
from typing import List


# SageMaker Processing paths
INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")

FOLDED_INPUT_DIR = INPUT_DIR / "folded"
CONFIG_DIR = INPUT_DIR / "config"
ANALYZED_OUTPUT_DIR = OUTPUT_DIR / "analyzed"
METADATA_DIR = OUTPUT_DIR / "metadata"


def setup_directories():
    """Create necessary output directories."""
    ANALYZED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load pipeline configuration."""
    config_file = CONFIG_DIR / "step_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def get_folded_directories() -> List[Path]:
    """Get list of folded output directories to process."""
    dirs = []
    if FOLDED_INPUT_DIR.exists():
        for d in FOLDED_INPUT_DIR.iterdir():
            if d.is_dir():
                dirs.append(d)
    return sorted(dirs)


def count_structures(directory: Path) -> int:
    """Count number of structure files in a directory."""
    count = 0
    # Look for refolded structures
    refold_dir = directory / "intermediate_designs_inverse_folded" / "refold_cif"
    if refold_dir.exists():
        count += len(list(refold_dir.glob("*.cif")))
    else:
        # Fallback
        count += len(list(directory.glob("**/*.cif")))
    return count


def run_boltzgen_analysis(
    input_dir: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """Run BoltzGen analysis step."""

    start_time = time.time()

    # Set environment
    env = os.environ.copy()
    env["BOLTZGEN_PIPELINE_STEP"] = "analysis"

    # Try to use pre-generated config
    config_dir = input_dir / "config"
    analysis_config = config_dir / "analysis.yaml"

    if analysis_config.exists():
        cmd = [
            "python", "-m", "boltzgen.resources.main",
            str(analysis_config),
        ]
    else:
        # Build command for analysis
        # Analysis step reads from the design directory
        design_dir = input_dir / "intermediate_designs_inverse_folded"
        if not design_dir.exists():
            design_dir = input_dir

        cmd = [
            "boltzgen", "run",
            "--output", str(output_dir),
            "--steps", "analysis",
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
    }


def copy_all_intermediate_results(src_dir: Path, dst_dir: Path):
    """Copy all intermediate results to maintain pipeline state."""
    directories_to_copy = [
        "intermediate_designs",
        "intermediate_designs_inverse_folded",
        "config",
    ]

    for dir_name in directories_to_copy:
        src = src_dir / dir_name
        dst = dst_dir / dir_name
        if src.exists() and not dst.exists():
            print(f"Copying {dir_name} to output...")
            shutil.copytree(src, dst)


def aggregate_metrics(output_dir: Path) -> dict:
    """Aggregate metrics from analysis output."""
    metrics_summary = {
        "total_designs": 0,
        "metrics_computed": False,
    }

    # Look for metrics CSV files
    for metrics_file in output_dir.glob("**/all_designs_metrics.csv"):
        metrics_summary["metrics_computed"] = True
        try:
            import pandas as pd
            df = pd.read_csv(metrics_file)
            metrics_summary["total_designs"] = len(df)

            # Add summary statistics
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols[:10]:  # First 10 numeric columns
                metrics_summary[f"{col}_mean"] = float(df[col].mean())
                metrics_summary[f"{col}_std"] = float(df[col].std())
        except Exception as e:
            print(f"Warning: Could not parse metrics file: {e}")

    return metrics_summary


def save_metadata(metadata: dict):
    """Save step execution metadata."""
    metadata_file = METADATA_DIR / "analysis_step_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="BoltzGen Analysis Step")
    parser.add_argument("--protocol", type=str, default="protein-anything",
                       help="Design protocol")
    parser.add_argument("--instance-count", type=int, default=1,
                       help="Total number of processing instances")
    parser.add_argument("--worker-id", type=int, default=0,
                       help="This worker's ID (0-indexed)")
    parser.add_argument("--largest-hydrophobic", action="store_true",
                       help="Compute largest hydrophobic patch metrics")
    parser.add_argument("--affinity-metrics", action="store_true",
                       help="Compute affinity metrics (for small molecule designs)")

    args = parser.parse_args()

    print("=" * 60)
    print("BoltzGen Analysis Step")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Worker ID: {args.worker_id} / {args.instance_count}")
    print(f"Protocol: {args.protocol}")

    # Setup
    setup_directories()
    config = load_config()

    # Override config with command line args
    config.update({
        "protocol": args.protocol,
        "largest_hydrophobic": args.largest_hydrophobic,
        "affinity_metrics": args.affinity_metrics,
    })

    # Determine metrics based on protocol
    if args.protocol in ["peptide-anything", "nanobody-anything"]:
        config["largest_hydrophobic"] = False
        config["largest_hydrophobic_refolded"] = False

    if args.protocol == "protein-small_molecule":
        config["affinity_metrics"] = True

    # Get input directories
    input_dirs = get_folded_directories()
    if not input_dirs:
        print("ERROR: No folded directories found!")
        sys.exit(1)

    print(f"Found {len(input_dirs)} input directory(ies)")

    # Process each directory
    all_results = []
    overall_success = True

    for input_dir in input_dirs:
        print(f"\nProcessing: {input_dir.name}")

        # Count structures
        num_structures = count_structures(input_dir)
        print(f"  Found {num_structures} structures to analyze")

        # Create output directory
        output_dir = ANALYZED_OUTPUT_DIR / input_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy intermediate results
        copy_all_intermediate_results(input_dir, output_dir)

        # Run analysis
        print("  Running analysis...")
        result = run_boltzgen_analysis(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
        )

        result["input_dir"] = input_dir.name
        result["num_structures"] = num_structures
        result["worker_id"] = args.worker_id
        all_results.append(result)

        if result["success"]:
            print(f"  SUCCESS: Analysis completed in {result['elapsed_time']:.1f}s")

            # Aggregate metrics
            metrics_summary = aggregate_metrics(output_dir)
            result["metrics_summary"] = metrics_summary
            print(f"  Metrics computed: {metrics_summary.get('metrics_computed', False)}")
            print(f"  Total designs analyzed: {metrics_summary.get('total_designs', 0)}")
        else:
            print(f"  FAILED: Return code {result['return_code']}")
            if result.get("stderr"):
                print(f"  STDERR: {result['stderr'][:500]}")
            overall_success = False

    # Save metadata
    metadata = {
        "step": "analysis",
        "timestamp": datetime.now().isoformat(),
        "worker_id": args.worker_id,
        "instance_count": args.instance_count,
        "protocol": args.protocol,
        "results": all_results,
        "overall_success": overall_success,
    }
    save_metadata(metadata)

    print("\n" + "=" * 60)
    if overall_success:
        print("Analysis step completed successfully!")
    else:
        print("Analysis step completed with errors!")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
