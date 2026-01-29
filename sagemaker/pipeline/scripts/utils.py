#!/usr/bin/env python3
"""
Shared Utilities for BoltzGen Pipeline Steps

Common functions used across multiple pipeline step scripts.
"""

import json
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional


# Thread-safe progress tracking
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
    """Get number of available GPUs.

    Returns:
        Number of GPUs detected, defaults to 1 if detection fails.
    """
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
    """Load pipeline step configuration from JSON file.

    Args:
        config_dir: Directory containing step_config.json

    Returns:
        Configuration dictionary, empty dict if file not found.
    """
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
    """Save step execution metadata to JSON file.

    Args:
        metadata: Metadata dictionary to save
        metadata_dir: Directory to save metadata file
        step_name: Name of the step (used in filename)
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / f"{step_name}_step_metadata.json"

    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to {metadata_file}")
    except IOError as e:
        print(f"WARNING: Failed to save metadata: {e}")


def validate_directory(path: Path, create: bool = False) -> bool:
    """Validate directory exists or create it.

    Args:
        path: Directory path to validate
        create: If True, create directory if it doesn't exist

    Returns:
        True if directory exists (or was created), False otherwise.
    """
    if path.exists():
        if not path.is_dir():
            print(f"ERROR: {path} exists but is not a directory")
            return False
        return True

    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            print(f"ERROR: Failed to create directory {path}: {e}")
            return False

    return False


def run_command_with_timeout(
    cmd: list,
    timeout: int,
    env: Optional[Dict] = None,
    capture_output: bool = True,
) -> Dict:
    """Run a command with timeout and structured error handling.

    Args:
        cmd: Command and arguments as list
        timeout: Timeout in seconds
        env: Environment variables (optional)
        capture_output: Whether to capture stdout/stderr

    Returns:
        Dictionary with success, return_code, stdout, stderr, and elapsed_time
    """
    import time
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
