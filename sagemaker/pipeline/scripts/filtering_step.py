#!/usr/bin/env python3
"""
Filtering Step - SageMaker Processing Script

This script runs inside the SageMaker Processing container to rank designs
and select the final set based on quality and diversity metrics.

This is a fast CPU step that:
- Ranks designs by quality metrics
- Selects diverse subset using quality/diversity trade-off
- Generates final output files and visualizations

Input:
    /opt/ml/processing/input/analyzed/ - Analyzed results from previous step
    /opt/ml/processing/input/config/ - Pipeline configuration

Output:
    /opt/ml/processing/output/final/ - Final ranked designs and reports
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
from typing import List, Optional


# SageMaker Processing paths
INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")

ANALYZED_INPUT_DIR = INPUT_DIR / "analyzed"
CONFIG_DIR = INPUT_DIR / "config"
FINAL_OUTPUT_DIR = OUTPUT_DIR / "final"
METADATA_DIR = OUTPUT_DIR / "metadata"


def setup_directories():
    """Create necessary output directories."""
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load pipeline configuration."""
    config_file = CONFIG_DIR / "step_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def get_analyzed_directories() -> List[Path]:
    """Get list of analyzed output directories to process."""
    dirs = []
    if ANALYZED_INPUT_DIR.exists():
        for d in ANALYZED_INPUT_DIR.iterdir():
            if d.is_dir():
                dirs.append(d)
    return sorted(dirs)


def run_boltzgen_filtering(
    input_dir: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """Run BoltzGen filtering step."""

    start_time = time.time()

    # Set environment
    env = os.environ.copy()
    env["BOLTZGEN_PIPELINE_STEP"] = "filtering"

    # Try to use pre-generated config
    config_dir = input_dir / "config"
    filtering_config = config_dir / "filtering.yaml"

    if filtering_config.exists():
        cmd = [
            "python", "-m", "boltzgen.resources.main",
            str(filtering_config),
        ]
    else:
        # Build command for filtering
        cmd = [
            "boltzgen", "run",
            "--output", str(output_dir),
            "--steps", "filtering",
            "--budget", str(config.get("budget", 100)),
        ]

        # Add optional filtering parameters
        if config.get("alpha") is not None:
            cmd.extend(["--alpha", str(config["alpha"])])

        if config.get("filter_biased") is not None:
            cmd.extend(["--filter_biased", str(config["filter_biased"])])

        if config.get("refolding_rmsd_threshold") is not None:
            cmd.extend(["--refolding_rmsd_threshold", str(config["refolding_rmsd_threshold"])])

        if config.get("metrics_override"):
            for metric, weight in config["metrics_override"].items():
                cmd.extend(["--metrics_override", f"{metric}={weight}"])

        if config.get("additional_filters"):
            for filter_spec in config["additional_filters"]:
                cmd.extend(["--additional_filters", filter_spec])

        if config.get("size_buckets"):
            for bucket in config["size_buckets"]:
                cmd.extend(["--size_buckets", bucket])

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


def copy_all_results(src_dir: Path, dst_dir: Path):
    """Copy all intermediate results to final output."""
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


def summarize_final_results(output_dir: Path) -> dict:
    """Summarize the final filtering results."""
    summary = {
        "final_designs_count": 0,
        "ranked_designs_count": 0,
        "has_overview_pdf": False,
        "has_metrics_csv": False,
    }

    # Look for final designs directory
    final_ranked_dir = output_dir / "final_ranked_designs"
    if final_ranked_dir.exists():
        # Count final designs
        for budget_dir in final_ranked_dir.glob("final_*_designs"):
            cif_files = list(budget_dir.glob("*.cif"))
            summary["final_designs_count"] = len(cif_files)
            break

        # Count ranked designs
        for ranked_dir in final_ranked_dir.glob("intermediate_ranked_*_designs"):
            cif_files = list(ranked_dir.glob("*.cif"))
            summary["ranked_designs_count"] = len(cif_files)
            break

        # Check for PDF report
        pdf_files = list(final_ranked_dir.glob("*.pdf"))
        summary["has_overview_pdf"] = len(pdf_files) > 0

        # Check for metrics CSV
        csv_files = list(final_ranked_dir.glob("*_metrics*.csv"))
        summary["has_metrics_csv"] = len(csv_files) > 0

    return summary


def create_pipeline_summary(output_dir: Path, all_results: List[dict]) -> dict:
    """Create a summary of the entire pipeline execution."""
    summary = {
        "pipeline_status": "completed",
        "steps_executed": ["design", "inverse_folding", "folding", "analysis", "filtering"],
        "total_elapsed_time": sum(r.get("elapsed_time", 0) for r in all_results),
        "all_steps_successful": all(r.get("success", False) for r in all_results),
    }

    # Add final results summary
    final_summary = summarize_final_results(output_dir)
    summary.update(final_summary)

    return summary


def save_metadata(metadata: dict):
    """Save step execution metadata."""
    metadata_file = METADATA_DIR / "filtering_step_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to {metadata_file}")


def save_pipeline_summary(output_dir: Path, summary: dict):
    """Save overall pipeline summary."""
    summary_file = output_dir / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Pipeline summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="BoltzGen Filtering Step")
    parser.add_argument("--protocol", type=str, default="protein-anything",
                       help="Design protocol")
    parser.add_argument("--budget", type=int, default=100,
                       help="Final design budget")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Quality/diversity trade-off (0.0=quality, 1.0=diversity)")
    parser.add_argument("--filter-biased", type=str, default="true",
                       choices=["true", "false"],
                       help="Remove amino-acid composition outliers")
    parser.add_argument("--refolding-rmsd-threshold", type=float, default=None,
                       help="RMSD threshold for filtering")
    parser.add_argument("--filter-cysteine", action="store_true",
                       help="Filter out designs with cysteine")

    args = parser.parse_args()

    print("=" * 60)
    print("BoltzGen Filtering Step")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Protocol: {args.protocol}")
    print(f"Budget: {args.budget}")
    print(f"Alpha: {args.alpha}")

    # Setup
    setup_directories()
    config = load_config()

    # Override config with command line args
    config.update({
        "protocol": args.protocol,
        "budget": args.budget,
        "filter_biased": args.filter_biased,
    })

    if args.alpha is not None:
        config["alpha"] = args.alpha

    if args.refolding_rmsd_threshold is not None:
        config["refolding_rmsd_threshold"] = args.refolding_rmsd_threshold

    # Protocol-specific defaults
    if args.protocol in ["peptide-anything", "nanobody-anything"]:
        if args.alpha is None:
            config["alpha"] = 0.01
        if args.refolding_rmsd_threshold is None:
            config["refolding_rmsd_threshold"] = 2.0
        if args.filter_cysteine:
            config["filter_cysteine"] = True

    # Get input directories
    input_dirs = get_analyzed_directories()
    if not input_dirs:
        print("ERROR: No analyzed directories found!")
        sys.exit(1)

    print(f"Found {len(input_dirs)} input directory(ies)")

    # Process each directory
    all_results = []
    overall_success = True

    for input_dir in input_dirs:
        print(f"\nProcessing: {input_dir.name}")

        # Create output directory
        output_dir = FINAL_OUTPUT_DIR / input_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy all intermediate results to final output
        copy_all_results(input_dir, output_dir)

        # Run filtering
        print("  Running filtering...")
        result = run_boltzgen_filtering(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
        )

        result["input_dir"] = input_dir.name
        all_results.append(result)

        if result["success"]:
            print(f"  SUCCESS: Filtering completed in {result['elapsed_time']:.1f}s")

            # Summarize results
            final_summary = summarize_final_results(output_dir)
            result["final_summary"] = final_summary
            print(f"  Final designs: {final_summary.get('final_designs_count', 0)}")
            print(f"  Ranked designs: {final_summary.get('ranked_designs_count', 0)}")
            print(f"  PDF report: {final_summary.get('has_overview_pdf', False)}")
        else:
            print(f"  FAILED: Return code {result['return_code']}")
            if result.get("stderr"):
                print(f"  STDERR: {result['stderr'][:500]}")
            overall_success = False

    # Create and save pipeline summary
    pipeline_summary = create_pipeline_summary(FINAL_OUTPUT_DIR, all_results)
    save_pipeline_summary(FINAL_OUTPUT_DIR, pipeline_summary)

    # Save step metadata
    metadata = {
        "step": "filtering",
        "timestamp": datetime.now().isoformat(),
        "protocol": args.protocol,
        "budget": args.budget,
        "alpha": config.get("alpha"),
        "results": all_results,
        "overall_success": overall_success,
        "pipeline_summary": pipeline_summary,
    }
    save_metadata(metadata)

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Final designs selected: {pipeline_summary.get('final_designs_count', 0)}")
    print(f"Total ranked designs: {pipeline_summary.get('ranked_designs_count', 0)}")
    print(f"PDF report generated: {pipeline_summary.get('has_overview_pdf', False)}")
    print(f"Metrics CSV available: {pipeline_summary.get('has_metrics_csv', False)}")
    print("=" * 60)

    if overall_success:
        print("\nFiltering step completed successfully!")
        print("Pipeline execution complete!")
    else:
        print("\nFiltering step completed with errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()
