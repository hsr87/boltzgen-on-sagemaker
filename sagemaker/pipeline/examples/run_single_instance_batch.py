#!/usr/bin/env python3
"""
Single Instance Large Batch Execution

Processes large batches using a single multi-GPU instance.
Each GPU independently processes samples in parallel.

Usage:
    python run_single_instance_batch.py --config single_instance_config.yaml --dry-run
    python run_single_instance_batch.py --config single_instance_config.yaml
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import boto3
import yaml


GPU_COUNT = {
    "ml.g5.xlarge": 1,
    "ml.g5.2xlarge": 1,
    "ml.g5.12xlarge": 4,
    "ml.g5.24xlarge": 4,
    "ml.g5.48xlarge": 8,
    "ml.g4dn.xlarge": 1,
    "ml.g4dn.12xlarge": 4,
}

HOURLY_COST = {
    "ml.g5.xlarge": 1.41,
    "ml.g5.2xlarge": 1.69,
    "ml.g5.12xlarge": 7.09,
    "ml.g5.24xlarge": 10.18,
    "ml.g5.48xlarge": 20.36,
    "ml.g4dn.xlarge": 0.74,
    "ml.g4dn.12xlarge": 5.67,
}


def load_config(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_specs(config: Dict) -> List[Path]:
    specs_dir = Path(config["design"]["specs_dir"])
    pattern = config["design"]["file_pattern"]
    specs = sorted(specs_dir.glob(pattern))

    limit = config["batch"].get("total_samples", 0)
    if limit > 0:
        specs = specs[:limit]

    return specs


def upload_all_samples(specs: List[Path], s3_bucket: str, batch_prefix: str) -> str:
    """Upload all samples to S3."""
    s3 = boto3.client("s3")
    uploaded = set()

    prefix = f"{batch_prefix}/input"

    print(f"Uploading {len(specs)} samples to S3...")

    for i, spec in enumerate(specs):
        # YAML file
        s3_key = f"{prefix}/{spec.name}"
        if s3_key not in uploaded:
            s3.upload_file(str(spec), s3_bucket, s3_key)
            uploaded.add(s3_key)

        # Related files
        for ext in ["*.cif", "*.pdb"]:
            for ref_file in spec.parent.glob(ext):
                ref_key = f"{prefix}/{ref_file.name}"
                if ref_key not in uploaded:
                    s3.upload_file(str(ref_file), s3_bucket, ref_key)
                    uploaded.add(ref_key)

        if (i + 1) % 100 == 0:
            print(f"  Uploaded {i + 1}/{len(specs)}")

    print(f"  Done! {len(uploaded)} files uploaded.")
    return f"s3://{s3_bucket}/{prefix}"


def create_processing_script(num_designs: int, budget: int, protocol: str) -> str:
    """Create multi-GPU parallel processing script."""
    return f'''#!/usr/bin/env python3
"""
Multi-GPU Parallel Batch Processing
Processes samples in parallel using all available GPUs.
"""

import os
import sys
import json
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue

INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")

# Progress tracking
progress_lock = threading.Lock()
completed_count = 0
total_count = 0

def get_gpu_count():
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        lines = [l for l in result.stdout.strip().split("\\n") if l.startswith("GPU")]
        return max(len(lines), 1)
    except:
        return 4

def process_sample(args):
    global completed_count
    spec_file, gpu_id, output_dir, sample_idx, total = args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    spec_name = spec_file.stem
    spec_output = output_dir / spec_name
    spec_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "boltzgen", "run", str(spec_file),
        "--output", str(spec_output),
        "--protocol", "{protocol}",
        "--num_designs", "{num_designs}",
        "--budget", "{budget}",
        "--devices", "1",
    ]

    start = datetime.now()

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=7200
        )
        elapsed = (datetime.now() - start).total_seconds()

        with progress_lock:
            completed_count += 1
            progress = completed_count

        if result.returncode == 0:
            print(f"[{{progress}}/{{total}}] GPU{{gpu_id}} SUCCESS: {{spec_name}} ({{elapsed:.0f}}s)")
            return {{"spec": spec_name, "status": "success", "elapsed": elapsed, "gpu": gpu_id}}
        else:
            print(f"[{{progress}}/{{total}}] GPU{{gpu_id}} FAILED: {{spec_name}}")
            return {{"spec": spec_name, "status": "failed", "error": result.stderr[:300], "gpu": gpu_id}}

    except subprocess.TimeoutExpired:
        with progress_lock:
            completed_count += 1
        print(f"[{{completed_count}}/{{total}}] GPU{{gpu_id}} TIMEOUT: {{spec_name}}")
        return {{"spec": spec_name, "status": "timeout", "gpu": gpu_id}}
    except Exception as e:
        with progress_lock:
            completed_count += 1
        print(f"[{{completed_count}}/{{total}}] GPU{{gpu_id}} ERROR: {{spec_name}} - {{e}}")
        return {{"spec": spec_name, "status": "error", "error": str(e), "gpu": gpu_id}}

def main():
    global total_count

    print("=" * 70)
    print("Large-Scale Multi-GPU Batch Processing")
    print(f"Start: {{datetime.now().isoformat()}}")
    print("=" * 70)

    gpu_count = get_gpu_count()
    print(f"GPUs available: {{gpu_count}}")

    specs = sorted(INPUT_DIR.glob("*.yaml"))
    total_count = len(specs)
    print(f"Samples to process: {{total_count}}")

    if not specs:
        print("ERROR: No specs found!")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate estimated time
    rounds = (total_count + gpu_count - 1) // gpu_count
    est_hours = rounds * 1.5
    print(f"Estimated time: ~{{est_hours:.1f}} hours ({{est_hours/24:.1f}} days)")
    print("=" * 70)
    sys.stdout.flush()

    # Create tasks - round-robin distribution across GPUs
    tasks = []
    for i, spec in enumerate(specs):
        gpu_id = i % gpu_count
        tasks.append((spec, gpu_id, OUTPUT_DIR, i, total_count))

    # Parallel execution
    results = []
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=gpu_count) as executor:
        futures = [executor.submit(process_sample, t) for t in tasks]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            sys.stdout.flush()

    # Results summary
    elapsed_total = (datetime.now() - start_time).total_seconds()
    success = len([r for r in results if r["status"] == "success"])
    failed = len(results) - success

    # Per-GPU statistics
    gpu_stats = {{}}
    for r in results:
        gpu = r.get("gpu", 0)
        if gpu not in gpu_stats:
            gpu_stats[gpu] = {{"success": 0, "failed": 0, "total_time": 0}}
        if r["status"] == "success":
            gpu_stats[gpu]["success"] += 1
            gpu_stats[gpu]["total_time"] += r.get("elapsed", 0)
        else:
            gpu_stats[gpu]["failed"] += 1

    summary = {{
        "total": len(results),
        "success": success,
        "failed": failed,
        "gpu_count": gpu_count,
        "total_elapsed_seconds": elapsed_total,
        "total_elapsed_hours": elapsed_total / 3600,
        "gpu_stats": gpu_stats,
        "results": results,
        "started_at": start_time.isoformat(),
        "completed_at": datetime.now().isoformat(),
    }}

    with open(OUTPUT_DIR / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"Total samples: {{len(results)}}")
    print(f"Success: {{success}}")
    print(f"Failed: {{failed}}")
    print(f"Total time: {{elapsed_total/3600:.2f}} hours")
    print()
    print("GPU Statistics:")
    for gpu, stats in sorted(gpu_stats.items()):
        avg_time = stats["total_time"] / max(stats["success"], 1)
        print(f"  GPU {{gpu}}: {{stats['success']}} success, {{stats['failed']}} failed, avg {{avg_time:.0f}}s/sample")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


def upload_script(s3_bucket: str, batch_prefix: str, config: Dict) -> str:
    """Upload processing script to S3."""
    s3 = boto3.client("s3")

    script = create_processing_script(
        config["design"]["num_designs"],
        config["design"]["budget"],
        config["design"]["protocol"],
    )

    key = f"{batch_prefix}/scripts/batch_processor.py"
    s3.put_object(Bucket=s3_bucket, Key=key, Body=script)

    return f"s3://{s3_bucket}/{batch_prefix}/scripts"


def create_job(
    job_name: str,
    config: Dict,
    script_uri: str,
    input_uri: str,
    output_uri: str,
) -> Dict:
    """Create SageMaker Processing Job."""
    sm = boto3.client("sagemaker", region_name=config["aws"]["region"])

    try:
        sm.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": config["instances"]["type"],
                    "VolumeSizeInGB": config["instances"]["volume_size"],
                }
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": config["instances"]["max_runtime"],
            },
            AppSpecification={
                "ImageUri": config["aws"]["image_uri"],
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/scripts/batch_processor.py"],
            },
            RoleArn=config["aws"]["role_arn"],
            ProcessingInputs=[
                {
                    "InputName": "input",
                    "S3Input": {
                        "S3Uri": input_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                },
                {
                    "InputName": "scripts",
                    "S3Input": {
                        "S3Uri": script_uri,
                        "LocalPath": "/opt/ml/processing/scripts",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                },
            ],
            ProcessingOutputConfig={
                "Outputs": [{
                    "OutputName": "output",
                    "S3Output": {
                        "S3Uri": output_uri,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }],
            },
        )
        return {"status": "created", "job_name": job_name}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def monitor_job(job_name: str, region: str):
    """Monitor job status."""
    sm = boto3.client("sagemaker", region_name=region)
    print(f"\nMonitoring job: {job_name}")

    while True:
        try:
            resp = sm.describe_processing_job(ProcessingJobName=job_name)
            status = resp["ProcessingJobStatus"]
            print(f"  Status: {status}", end="\r")

            if status in ["Completed", "Failed", "Stopped"]:
                print(f"\n\nJob {status}!")
                if status == "Failed":
                    print(f"Reason: {resp.get('FailureReason', 'Unknown')}")
                return status
        except Exception as e:
            print(f"  Error checking status: {e}")

        time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    batch_id = f"{config['batch']['name_prefix']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    batch_prefix = f"{config['output']['s3_prefix']}/{batch_id}"

    instance_type = config["instances"]["type"]
    gpus = GPU_COUNT.get(instance_type, 4)
    cost_per_hour = HOURLY_COST.get(instance_type, 7.09)

    print("=" * 70)
    print("Single Instance Large Batch Execution")
    print("=" * 70)
    print(f"Batch ID: {batch_id}")
    print(f"Instance: {instance_type} ({gpus} GPUs)")
    print()

    # Load samples
    specs = get_specs(config)
    print(f"Total samples: {len(specs)}")

    if not specs:
        print("ERROR: No specs found!")
        sys.exit(1)

    # Calculate estimates
    rounds = math.ceil(len(specs) / gpus)
    total_hours = rounds * 1.5
    total_cost = total_hours * cost_per_hour

    print()
    print("Estimation:")
    print(f"  Parallel capacity: {gpus} samples (GPUs)")
    print(f"  Rounds: {rounds}")
    print(f"  Time: ~{total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print(f"  Cost: ~${total_cost:.2f}")
    print()

    if args.dry_run:
        print("[DRY RUN] No job created.")
        return

    # Upload to S3
    input_uri = upload_all_samples(specs, config["aws"]["s3_bucket"], batch_prefix)
    script_uri = upload_script(config["aws"]["s3_bucket"], batch_prefix, config)
    output_uri = f"s3://{config['aws']['s3_bucket']}/{batch_prefix}/output"

    print()
    print(f"Input: {input_uri}")
    print(f"Output: {output_uri}")
    print()

    # Create job
    print("Creating Processing Job...")
    result = create_job(batch_id, config, script_uri, input_uri, output_uri)

    if result["status"] == "created":
        print(f"  [OK] Job created: {batch_id}")
    else:
        print(f"  [FAIL] {result.get('error', 'Unknown')}")
        sys.exit(1)

    # Save batch info
    batch_info = {
        "batch_id": batch_id,
        "total_samples": len(specs),
        "instance_type": instance_type,
        "gpus": gpus,
        "estimate": {
            "rounds": rounds,
            "hours": total_hours,
            "days": total_hours / 24,
            "cost": total_cost,
        },
        "s3_prefix": f"s3://{config['aws']['s3_bucket']}/{batch_prefix}",
        "started_at": datetime.now().isoformat(),
    }

    info_file = Path(f"batch_{batch_id}.json")
    with open(info_file, "w") as f:
        json.dump(batch_info, f, indent=2)

    print()
    print("=" * 70)
    print(f"Batch: {batch_id}")
    print(f"Samples: {len(specs)}")
    print(f"Est. Time: ~{total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print(f"Est. Cost: ~${total_cost:.2f}")
    print("=" * 70)
    print(f"\nBatch info: {info_file}")
    print(f"\nMonitor:")
    print(f"  aws sagemaker describe-processing-job --processing-job-name {batch_id}")
    print(f"\nLogs:")
    print(f"  aws logs tail /aws/sagemaker/ProcessingJobs --follow --log-stream-name-prefix {batch_id}")
    print(f"\nResults:")
    print(f"  aws s3 sync {output_uri} ./results_{batch_id}")

    if args.wait:
        monitor_job(batch_id, config["aws"]["region"])


if __name__ == "__main__":
    main()
