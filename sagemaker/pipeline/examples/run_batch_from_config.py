#!/usr/bin/env python3
"""
Config-based Batch Execution

Each instance processes multiple samples.
Example: 2 instances processing 1000 samples -> 500 samples per instance

Usage:
    python run_batch_from_config.py --config batch_config.yaml
    python run_batch_from_config.py --config batch_config.yaml --dry-run
    python run_batch_from_config.py --config batch_config.yaml --wait
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import boto3
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_design_specs(config: Dict) -> List[Path]:
    """Get design spec files based on config."""
    specs_dir = Path(config["design"]["specs_dir"])
    pattern = config["design"]["file_pattern"]

    specs = sorted(specs_dir.glob(pattern))

    # Limit total samples
    total = config["batch"].get("total_samples", 0)
    if total > 0:
        specs = specs[:total]

    return specs


def distribute_samples(specs: List[Path], num_instances: int, method: str = "chunk") -> List[List[Path]]:
    """Distribute samples across instances.

    Args:
        specs: Full sample list
        num_instances: Number of instances
        method: "chunk" (consecutive allocation) or "round_robin" (cyclic allocation)

    Returns:
        Sample list per instance
    """
    if method == "chunk":
        # Split into consecutive chunks
        chunk_size = math.ceil(len(specs) / num_instances)
        return [specs[i:i + chunk_size] for i in range(0, len(specs), chunk_size)]
    else:
        # Round-robin distribution
        distribution = [[] for _ in range(num_instances)]
        for i, spec in enumerate(specs):
            distribution[i % num_instances].append(spec)
        return distribution


def upload_samples_for_instance(
    specs: List[Path],
    s3_bucket: str,
    batch_prefix: str,
    instance_id: int,
) -> str:
    """Upload samples for an instance to S3."""
    s3 = boto3.client("s3")
    uploaded = set()

    instance_prefix = f"{batch_prefix}/input/instance_{instance_id:02d}"

    for spec in specs:
        # Upload YAML file
        s3_key = f"{instance_prefix}/{spec.name}"
        if s3_key not in uploaded:
            s3.upload_file(str(spec), s3_bucket, s3_key)
            uploaded.add(s3_key)

        # Upload related CIF/PDB files
        for ext in ["*.cif", "*.pdb"]:
            for ref_file in spec.parent.glob(ext):
                ref_key = f"{instance_prefix}/{ref_file.name}"
                if ref_key not in uploaded:
                    s3.upload_file(str(ref_file), s3_bucket, ref_key)
                    uploaded.add(ref_key)

    return f"s3://{s3_bucket}/{instance_prefix}"


def create_multi_sample_script(num_designs: int, budget: int, protocol: str) -> str:
    """Create script for sequential multi-sample processing."""
    return f'''#!/usr/bin/env python3
"""
Multi-Sample Processing Script
Processes multiple design specs sequentially on a single instance.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")

def main():
    print("=" * 60)
    print("Multi-Sample Batch Processing")
    print(f"Start time: {{datetime.now().isoformat()}}")
    print("=" * 60)

    # Find design specs
    specs = sorted(INPUT_DIR.glob("*.yaml"))
    print(f"Found {{len(specs)}} design specs to process")

    if not specs:
        print("ERROR: No design specs found!")
        for f in INPUT_DIR.iterdir():
            print(f"  Found: {{f}}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    success_count = 0
    fail_count = 0

    for i, spec in enumerate(specs):
        spec_name = spec.stem
        spec_output = OUTPUT_DIR / spec_name
        spec_output.mkdir(parents=True, exist_ok=True)

        print(f"\\n[{{i+1}}/{{len(specs)}}] Processing {{spec_name}}...")
        sys.stdout.flush()

        cmd = [
            "boltzgen", "run",
            str(spec),
            "--output", str(spec_output),
            "--protocol", "{protocol}",
            "--num_designs", "{num_designs}",
            "--budget", "{budget}",
            "--devices", "1",
        ]

        start_time = datetime.now()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                print(f"  SUCCESS: {{spec_name}} ({{elapsed:.1f}}s)")
                results.append({{"spec": spec_name, "status": "success", "elapsed": elapsed}})
                success_count += 1
            else:
                print(f"  FAILED: {{spec_name}}")
                print(f"  STDERR: {{result.stderr[:500]}}")
                results.append({{"spec": spec_name, "status": "failed", "error": result.stderr[:1000]}})
                fail_count += 1

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {{spec_name}}")
            results.append({{"spec": spec_name, "status": "timeout"}})
            fail_count += 1

        except Exception as e:
            print(f"  ERROR: {{spec_name}} - {{e}}")
            results.append({{"spec": spec_name, "status": "error", "error": str(e)}})
            fail_count += 1

        sys.stdout.flush()

    # Save results
    summary = {{
        "total": len(specs),
        "success": success_count,
        "failed": fail_count,
        "results": results,
        "completed_at": datetime.now().isoformat(),
    }}

    with open(OUTPUT_DIR / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Total: {{len(specs)}}")
    print(f"Success: {{success_count}}")
    print(f"Failed: {{fail_count}}")
    print("=" * 60)

    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


def upload_processing_script(s3_bucket: str, batch_prefix: str, config: Dict) -> str:
    """Upload processing script to S3."""
    s3 = boto3.client("s3")

    script_content = create_multi_sample_script(
        num_designs=config["design"]["num_designs"],
        budget=config["design"]["budget"],
        protocol=config["design"]["protocol"],
    )

    script_key = f"{batch_prefix}/scripts/multi_sample_processor.py"
    s3.put_object(Bucket=s3_bucket, Key=script_key, Body=script_content)

    return f"s3://{s3_bucket}/{batch_prefix}/scripts"


def create_processing_job(
    job_name: str,
    config: Dict,
    script_s3_uri: str,
    input_s3_uri: str,
    output_s3_uri: str,
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
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/scripts/multi_sample_processor.py"],
            },
            RoleArn=config["aws"]["role_arn"],
            ProcessingInputs=[
                {
                    "InputName": "input",
                    "S3Input": {
                        "S3Uri": input_s3_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                },
                {
                    "InputName": "scripts",
                    "S3Input": {
                        "S3Uri": script_s3_uri,
                        "LocalPath": "/opt/ml/processing/scripts",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                },
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "output",
                        "S3Output": {
                            "S3Uri": output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    },
                ],
            },
        )

        return {"job_name": job_name, "status": "created", "output_s3_uri": output_s3_uri}

    except Exception as e:
        return {"job_name": job_name, "status": "failed", "error": str(e)}


def monitor_jobs(job_names: List[str], region: str, check_interval: int = 30) -> Dict:
    """Monitor status of all jobs."""
    sm = boto3.client("sagemaker", region_name=region)

    print(f"\nMonitoring {len(job_names)} jobs...")

    while True:
        statuses = {"Completed": 0, "Failed": 0, "InProgress": 0, "Other": 0}

        for job_name in job_names:
            try:
                resp = sm.describe_processing_job(ProcessingJobName=job_name)
                status = resp["ProcessingJobStatus"]
                if status in statuses:
                    statuses[status] += 1
                else:
                    statuses["Other"] += 1
            except Exception:
                statuses["Other"] += 1

        print(f"  InProgress: {statuses['InProgress']} | Completed: {statuses['Completed']} | Failed: {statuses['Failed']}", end="\r")

        if statuses["InProgress"] == 0:
            print(f"\n\nAll jobs finished!")
            print(f"Completed: {statuses['Completed']}, Failed: {statuses['Failed']}")
            return statuses

        time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description="Run batch from YAML config")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Generate batch ID
    batch_id = f"{config['batch']['name_prefix']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    batch_prefix = f"{config['output']['s3_prefix']}/{batch_id}"

    print("=" * 70)
    print(f"Batch Execution from Config: {args.config}")
    print("=" * 70)
    print(f"Batch ID: {batch_id}")
    print(f"Instances: {config['instances']['count']} x {config['instances']['type']}")
    print(f"Designs/sample: {config['design']['num_designs']}")
    print(f"Budget/sample: {config['design']['budget']}")
    print()

    # Get design specs
    specs = get_design_specs(config)
    print(f"Total samples: {len(specs)}")

    if not specs:
        print("ERROR: No design specs found!")
        sys.exit(1)

    # Distribute samples to instances
    num_instances = min(config["instances"]["count"], len(specs))
    distribution = distribute_samples(
        specs,
        num_instances,
        config["batch"]["distribution"]
    )

    print(f"\nSample distribution ({config['batch']['distribution']}):")
    for i, instance_specs in enumerate(distribution):
        print(f"  Instance {i}: {len(instance_specs)} samples")
        if len(instance_specs) <= 5:
            for s in instance_specs:
                print(f"    - {s.name}")
        else:
            for s in instance_specs[:3]:
                print(f"    - {s.name}")
            print(f"    ... and {len(instance_specs) - 3} more")

    # Exit on dry run
    if args.dry_run:
        print("\n[DRY RUN] No jobs created.")
        return

    print()

    # Upload processing script
    print("Uploading processing script...")
    script_s3_uri = upload_processing_script(
        config["aws"]["s3_bucket"],
        batch_prefix,
        config
    )

    # Create jobs for each instance
    print(f"\nCreating {num_instances} jobs...")
    jobs = []

    for i, instance_specs in enumerate(distribution):
        if not instance_specs:
            continue

        job_name = f"{batch_id}-inst{i:02d}"

        # Upload samples
        input_s3_uri = upload_samples_for_instance(
            instance_specs,
            config["aws"]["s3_bucket"],
            batch_prefix,
            i
        )

        output_s3_uri = f"s3://{config['aws']['s3_bucket']}/{batch_prefix}/output/instance_{i:02d}"

        # Create job
        result = create_processing_job(
            job_name=job_name,
            config=config,
            script_s3_uri=script_s3_uri,
            input_s3_uri=input_s3_uri,
            output_s3_uri=output_s3_uri,
        )

        result["instance_id"] = i
        result["num_samples"] = len(instance_specs)
        jobs.append(result)

        if result["status"] == "created":
            print(f"  [OK] Instance {i}: {job_name} ({len(instance_specs)} samples)")
        else:
            print(f"  [FAIL] Instance {i}: {result.get('error', 'Unknown')}")

        # Rate limiting
        time.sleep(config["batch"].get("start_delay", 2))

    # Save results
    job_names = [j["job_name"] for j in jobs if j["status"] == "created"]

    batch_info = {
        "batch_id": batch_id,
        "config_file": str(args.config),
        "total_samples": len(specs),
        "num_instances": num_instances,
        "jobs": jobs,
        "s3_prefix": f"s3://{config['aws']['s3_bucket']}/{batch_prefix}",
        "started_at": datetime.now().isoformat(),
    }

    info_file = Path(f"batch_{batch_id}.json")
    with open(info_file, "w") as f:
        json.dump(batch_info, f, indent=2)

    print()
    print("=" * 70)
    print(f"Batch: {batch_id}")
    print(f"Jobs created: {len(job_names)}/{num_instances}")
    print("=" * 70)
    print(f"\nBatch info: {info_file}")
    print(f"\nMonitor:")
    print(f"  aws sagemaker list-processing-jobs --name-contains {batch_id}")
    print(f"\nLogs:")
    print(f"  aws logs tail /aws/sagemaker/ProcessingJobs --follow --log-stream-name-prefix {batch_id}")
    print(f"\nResults:")
    print(f"  aws s3 sync s3://{config['aws']['s3_bucket']}/{batch_prefix}/output {config['output']['local_dir']}/{batch_id}")

    # Wait for completion
    if args.wait and job_names:
        monitor_jobs(job_names, config["aws"]["region"])


if __name__ == "__main__":
    main()
