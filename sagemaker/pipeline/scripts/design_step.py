#!/usr/bin/env python3
"""
Design Step - SageMaker Processing Script

This script runs inside the SageMaker Processing container to generate
protein designs using the BoltzGen diffusion model.

Supports:
- Multi-GPU parallel processing within a single instance
- Multi-instance scaling across the pipeline
- Multiple design specs processing

Input:
    /opt/ml/processing/input/design_specs/ - Design specification YAML files

Output:
    /opt/ml/processing/output/designs/ - Generated design structures (CIF/NPZ)
    /opt/ml/processing/output/metadata/ - Step metadata and logs
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional


# ============================================================================
# Inline Utilities (SageMaker ProcessingStep only uploads single script)
# ============================================================================

class ProgressTracker:
    """Thread-safe progress counter for parallel processing."""

    def __init__(self, total: int = 0):
        self._lock = threading.Lock()
        self._completed = 0
        self._total = total

    def increment(self) -> int:
        """Increment counter and return current value."""
        with self._lock:
            self._completed += 1
            return self._completed

    @property
    def completed(self) -> int:
        with self._lock:
            return self._completed

    @property
    def total(self) -> int:
        return self._total

    @total.setter
    def total(self, value: int):
        self._total = value


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return 1
        lines = [line for line in result.stdout.strip().split("\n") if line.startswith("GPU")]
        return max(len(lines), 1)
    except subprocess.TimeoutExpired:
        print("WARNING: nvidia-smi timed out, assuming 1 GPU")
        return 1
    except FileNotFoundError:
        print("WARNING: nvidia-smi not found, assuming 1 GPU")
        return 1
    except Exception as e:
        print(f"WARNING: GPU detection failed ({e}), assuming 1 GPU")
        return 1


def load_step_config(config_dir: Path) -> Dict:
    """Load pipeline step configuration from JSON file."""
    config_file = config_dir / "step_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to parse config file: {e}")
            return {}
    return {}


def save_step_metadata(metadata: Dict, metadata_dir: Path, step_name: str) -> None:
    """Save step execution metadata to JSON file."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / f"{step_name}_step_metadata.json"

    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to {metadata_file}")
    except IOError as e:
        print(f"WARNING: Failed to save metadata: {e}")


def run_command_with_timeout(
    cmd: list,
    timeout: int,
    env: Optional[Dict] = None,
    capture_output: bool = True,
) -> Dict:
    """Run a command with timeout and structured error handling."""
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start_time

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout[-10000:] if result.stdout else "",
            "stderr": result.stderr[-10000:] if result.stderr else "",
            "elapsed_time": elapsed,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "elapsed_time": elapsed,
            "timed_out": True,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "elapsed_time": elapsed,
            "timed_out": False,
            "exception": type(e).__name__,
        }


# ============================================================================
# SageMaker Processing paths
# ============================================================================

INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")

DESIGN_SPECS_DIR = INPUT_DIR / "design_specs"
CONFIG_DIR = INPUT_DIR / "config"
DESIGNS_OUTPUT_DIR = OUTPUT_DIR / "designs"
METADATA_DIR = OUTPUT_DIR / "metadata"

# Progress tracker (thread-safe)
progress = ProgressTracker()


def setup_directories():
    """Create necessary output directories."""
    DESIGNS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def get_design_specs() -> List[Path]:
    """Get list of design specification files."""
    specs = []
    if DESIGN_SPECS_DIR.exists():
        for ext in ["*.yaml", "*.yml"]:
            specs.extend(DESIGN_SPECS_DIR.glob(ext))
    return sorted(specs)


def process_single_spec(args: Tuple) -> Dict:
    """Process a single design spec on a specific GPU.

    This function runs in a separate process for GPU parallelization.
    """
    spec_file, gpu_id, output_dir, config, sample_idx, total = args

    # Set GPU for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    spec_name = spec_file.stem
    spec_output = output_dir / spec_name
    spec_output.mkdir(parents=True, exist_ok=True)

    # Build boltzgen command
    cmd = [
        "boltzgen", "run",
        str(spec_file),
        "--output", str(spec_output),
        "--protocol", config.get("protocol", "protein-anything"),
        "--num_designs", str(config.get("num_designs", 100)),
        "--budget", str(config.get("budget", 10)),
        "--devices", "1",
    ]

    # Add optional arguments
    if config.get("diffusion_batch_size"):
        cmd.extend(["--diffusion_batch_size", str(config["diffusion_batch_size"])])

    if config.get("step_scale"):
        cmd.extend(["--step_scale", str(config["step_scale"])])

    if config.get("noise_scale"):
        cmd.extend(["--noise_scale", str(config["noise_scale"])])

    if config.get("reuse", True):
        cmd.append("--reuse")

    # Run command
    timeout = config.get("timeout", 7200)
    result = run_command_with_timeout(cmd, timeout, env)

    # Update progress
    current = progress.increment()

    # Log result
    if result["success"]:
        print(f"[{current}/{total}] GPU{gpu_id} SUCCESS: {spec_name} ({result['elapsed_time']:.0f}s)")
    elif result.get("timed_out"):
        print(f"[{current}/{total}] GPU{gpu_id} TIMEOUT: {spec_name}")
    else:
        print(f"[{current}/{total}] GPU{gpu_id} FAILED: {spec_name}")
        if result.get("stderr"):
            print(f"  Error: {result['stderr'][:200]}")

    return {
        "spec": spec_name,
        "status": "success" if result["success"] else ("timeout" if result.get("timed_out") else "failed"),
        "elapsed": result["elapsed_time"],
        "gpu": gpu_id,
        "error": result.get("stderr", "")[:500] if not result["success"] else None,
    }


def get_worker_info() -> Tuple[int, int]:
    """Get worker ID and instance count from SageMaker environment.

    SageMaker provides:
    - SM_HOSTS: JSON list of hostnames (e.g., '["algo-1", "algo-2"]')
    - SM_CURRENT_HOST: Current hostname (e.g., 'algo-1')

    Returns:
        (worker_id, instance_count) tuple
    """
    hosts_str = os.environ.get("SM_HOSTS", "")
    current_host = os.environ.get("SM_CURRENT_HOST", "")

    if hosts_str and current_host:
        try:
            hosts = json.loads(hosts_str)
            instance_count = len(hosts)
            worker_id = hosts.index(current_host)
            return worker_id, instance_count
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse SM_HOSTS: {e}")

    return 0, 1  # Default single instance


def main():
    parser = argparse.ArgumentParser(description="BoltzGen Design Step")
    parser.add_argument("--num-designs", type=int, default=100,
                       help="Number of designs to generate per spec")
    parser.add_argument("--protocol", type=str, default="protein-anything",
                       help="Design protocol")
    parser.add_argument("--budget", type=int, default=10,
                       help="Final design budget")
    parser.add_argument("--instance-count", type=int, default=None,
                       help="Total number of processing instances (auto-detected if not set)")
    parser.add_argument("--worker-id", type=int, default=None,
                       help="This worker's ID (auto-detected if not set)")
    parser.add_argument("--devices", type=int, default=1,
                       help="Number of GPU devices to use")
    parser.add_argument("--timeout", type=int, default=7200,
                       help="Timeout per sample in seconds")

    args = parser.parse_args()

    # Auto-detect worker info from SageMaker environment
    auto_worker_id, auto_instance_count = get_worker_info()
    worker_id = args.worker_id if args.worker_id is not None else auto_worker_id
    instance_count = args.instance_count if args.instance_count is not None else auto_instance_count

    print("=" * 70)
    print("BoltzGen Design Step - Multi-GPU Parallel Processing")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Worker ID: {worker_id} / {instance_count}")
    print(f"Designs per spec: {args.num_designs}")
    print(f"Protocol: {args.protocol}")
    print(f"Budget: {args.budget}")

    # Setup
    setup_directories()
    config = load_step_config(CONFIG_DIR)

    # Get GPU count
    gpu_count = get_gpu_count()
    print(f"GPUs available: {gpu_count}")

    # Override config with command line args
    config.update({
        "protocol": args.protocol,
        "budget": args.budget,
        "num_designs": args.num_designs,
        "devices": 1,
        "timeout": args.timeout,
    })

    # Get design specs
    all_specs = get_design_specs()
    if not all_specs:
        print("ERROR: No design specification files found!")
        print(f"Looked in: {DESIGN_SPECS_DIR}")
        if DESIGN_SPECS_DIR.exists():
            print("Directory contents:")
            for f in DESIGN_SPECS_DIR.iterdir():
                print(f"  {f}")
        else:
            print("Directory does not exist!")
        sys.exit(1)

    print(f"Total design specs found: {len(all_specs)}")

    # Partition specs across instances (for multi-instance scaling)
    if instance_count > 1:
        specs_per_instance = len(all_specs) // instance_count
        remainder = len(all_specs) % instance_count

        if worker_id < remainder:
            start = worker_id * (specs_per_instance + 1)
            end = start + specs_per_instance + 1
        else:
            start = remainder * (specs_per_instance + 1) + (worker_id - remainder) * specs_per_instance
            end = start + specs_per_instance

        specs = all_specs[start:end]
        print(f"This instance processes specs {start} to {end-1} ({len(specs)} specs)")
    else:
        specs = all_specs

    if not specs:
        print("No specs assigned to this worker.")
        save_step_metadata({
            "step": "design",
            "timestamp": datetime.now().isoformat(),
            "worker_id": worker_id,
            "instance_count": instance_count,
            "total_specs": 0,
            "results": [],
            "overall_success": True,
        }, METADATA_DIR, "design")
        return

    # Set progress total
    progress.total = len(specs)

    # Create tasks - round-robin distribution across GPUs
    tasks = []
    for i, spec in enumerate(specs):
        gpu_id = i % gpu_count
        tasks.append((spec, gpu_id, DESIGNS_OUTPUT_DIR, config, i, len(specs)))

    print(f"\nProcessing {len(specs)} specs using {gpu_count} GPUs in parallel...")
    print("=" * 70)
    sys.stdout.flush()

    # Execute in parallel
    results = []
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=gpu_count) as executor:
        futures = [executor.submit(process_single_spec, t) for t in tasks]

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"ERROR: Task failed with exception: {e}")
                results.append({
                    "spec": "unknown",
                    "status": "error",
                    "error": str(e),
                })
            sys.stdout.flush()

    # Calculate statistics
    elapsed_total = (datetime.now() - start_time).total_seconds()
    success_count = len([r for r in results if r["status"] == "success"])
    failed_count = len(results) - success_count

    # Per-GPU statistics
    gpu_stats = {}
    for r in results:
        gpu = r.get("gpu", 0)
        if gpu not in gpu_stats:
            gpu_stats[gpu] = {"success": 0, "failed": 0, "total_time": 0}
        if r["status"] == "success":
            gpu_stats[gpu]["success"] += 1
            gpu_stats[gpu]["total_time"] += r.get("elapsed", 0)
        else:
            gpu_stats[gpu]["failed"] += 1

    # Save metadata
    metadata = {
        "step": "design",
        "timestamp": datetime.now().isoformat(),
        "worker_id": worker_id,
        "instance_count": instance_count,
        "gpu_count": gpu_count,
        "total_specs": len(specs),
        "num_designs_per_spec": args.num_designs,
        "protocol": args.protocol,
        "budget": args.budget,
        "total_elapsed_seconds": elapsed_total,
        "total_elapsed_hours": elapsed_total / 3600,
        "success_count": success_count,
        "failed_count": failed_count,
        "gpu_stats": gpu_stats,
        "results": results,
        "overall_success": failed_count == 0,
    }
    save_step_metadata(metadata, METADATA_DIR, "design")

    # Print summary
    print()
    print("=" * 70)
    print("DESIGN STEP COMPLETE")
    print("=" * 70)
    print(f"Total specs: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {elapsed_total/3600:.2f} hours")
    print()
    print("GPU Statistics:")
    for gpu, stats in sorted(gpu_stats.items()):
        avg_time = stats["total_time"] / max(stats["success"], 1)
        print(f"  GPU {gpu}: {stats['success']} success, {stats['failed']} failed, avg {avg_time:.0f}s/spec")
    print("=" * 70)

    if failed_count > 0:
        print(f"\nWARNING: {failed_count} specs failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
