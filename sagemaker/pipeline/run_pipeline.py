#!/usr/bin/env python3
"""
BoltzGen SageMaker Pipeline Runner

This script provides a CLI interface to:
1. Create/update the SageMaker Pipeline
2. Upload design specifications to S3
3. Start pipeline executions
4. Monitor execution status
5. Download results

Supports YAML configuration with environment variable substitution.

Usage:
    # Using YAML config
    python run_pipeline.py --config pipeline_config.yaml create
    python run_pipeline.py --config pipeline_config.yaml run --design-spec example/protein.yaml

    # Using command line arguments
    python run_pipeline.py create --s3-bucket my-bucket --role-arn arn:aws:iam::...
    python run_pipeline.py run --design-spec example/protein.yaml --num-designs 10000

    # Check status
    python run_pipeline.py status --execution-arn arn:aws:sagemaker:...

    # Download results
    python run_pipeline.py download --execution-arn arn:aws:sagemaker:... --output ./results
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, estimate_cost, get_scaling_preset_for_designs, SCALING_PRESETS
from pipeline import BoltzGenPipeline, create_default_pipeline


def load_env_file(env_path: Path = None):
    """Load environment variables from .env file."""
    if env_path is None:
        # Look for .env in current directory and parent directories
        for search_dir in [Path.cwd(), Path(__file__).parent, Path(__file__).parent.parent]:
            env_file = search_dir / ".env"
            if env_file.exists():
                env_path = env_file
                break

    if env_path and env_path.exists():
        print(f"Loading environment from: {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key not in os.environ:  # Don't override existing env vars
                        os.environ[key] = value


def substitute_env_vars(value: str) -> str:
    """Substitute environment variables in a string.

    Supports formats:
    - ${VAR_NAME}
    - ${VAR_NAME:default_value}
    """
    if not isinstance(value, str):
        return value

    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default if default is not None else match.group(0))

    return re.sub(pattern, replacer, value)


def process_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively process config values to substitute environment variables."""
    if isinstance(config, dict):
        return {k: process_config_values(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [process_config_values(v) for v in config]
    elif isinstance(config, str):
        return substitute_env_vars(config)
    return config


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and process YAML configuration file."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Process environment variable substitutions
    config = process_config_values(config)

    return config


def get_aws_config_from_yaml(yaml_config: Dict) -> Dict[str, str]:
    """Extract AWS configuration from YAML config."""
    aws = yaml_config.get("aws", {})
    return {
        "region": aws.get("region", "us-east-1"),
        "s3_bucket": aws.get("s3_bucket", ""),
        "role_arn": aws.get("role_arn", ""),
        "image_uri": aws.get("image_uri", ""),
    }


def get_default_role_arn(region: str) -> Optional[str]:
    """Try to get the default SageMaker execution role."""
    try:
        from sagemaker import get_execution_role
        return get_execution_role()
    except Exception:
        # Try to find a SageMaker role
        iam = boto3.client('iam', region_name=region)
        try:
            roles = iam.list_roles()['Roles']
            for role in roles:
                if 'SageMaker' in role['RoleName'] or 'sagemaker' in role['RoleName']:
                    return role['Arn']
        except Exception:
            pass
    return None


def get_default_bucket(region: str) -> str:
    """Get default SageMaker bucket."""
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()['Account']
    return f"sagemaker-{region}-{account_id}"


def upload_design_specs(
    design_specs: list,
    s3_bucket: str,
    s3_prefix: str,
    region: str,
    max_file_size_mb: int = 100,
) -> str:
    """Upload design specification files to S3.

    Args:
        design_specs: List of design spec file paths
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix for uploads
        region: AWS region
        max_file_size_mb: Maximum file size in MB (default: 100)

    Returns:
        S3 URI of uploaded files

    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If spec file doesn't exist
    """
    # Input validation
    if not design_specs:
        raise ValueError("No design specs provided")
    if not s3_bucket or not isinstance(s3_bucket, str):
        raise ValueError("Invalid S3 bucket name")
    if not s3_prefix or not isinstance(s3_prefix, str):
        raise ValueError("Invalid S3 prefix")

    s3 = boto3.client('s3', region_name=region)

    input_prefix = f"{s3_prefix}/input"
    uploaded = set()
    max_size_bytes = max_file_size_mb * 1024 * 1024

    for spec_path in design_specs:
        spec_path = Path(spec_path).resolve()  # Resolve to absolute path
        if not spec_path.exists():
            raise FileNotFoundError(f"Design spec not found: {spec_path}")

        # Check file size
        if spec_path.stat().st_size > max_size_bytes:
            raise ValueError(f"File too large (>{max_file_size_mb}MB): {spec_path}")

        # Upload the spec file
        s3_key = f"{input_prefix}/{spec_path.name}"
        if s3_key not in uploaded:
            print(f"Uploading {spec_path} to s3://{s3_bucket}/{s3_key}")
            s3.upload_file(str(spec_path), s3_bucket, s3_key)
            uploaded.add(s3_key)

        # Upload any referenced .cif or .pdb files in the same directory
        spec_dir = spec_path.parent
        for ext in ['*.cif', '*.pdb']:
            for file in spec_dir.glob(ext):
                # Skip files that are too large
                if file.stat().st_size > max_size_bytes:
                    print(f"Warning: Skipping large file (>{max_file_size_mb}MB): {file}")
                    continue
                s3_key = f"{input_prefix}/{file.name}"
                if s3_key not in uploaded:
                    print(f"Uploading {file} to s3://{s3_bucket}/{s3_key}")
                    s3.upload_file(str(file), s3_bucket, s3_key)
                    uploaded.add(s3_key)

    print(f"Uploaded {len(uploaded)} files to S3")
    return f"s3://{s3_bucket}/{input_prefix}"


def create_pipeline_command(args):
    """Create or update the SageMaker Pipeline."""
    print("=" * 60)
    print("Creating/Updating BoltzGen Pipeline")
    print("=" * 60)

    # Load config from YAML if provided
    yaml_config = {}
    if args.config:
        yaml_config = load_yaml_config(Path(args.config))
        aws_config = get_aws_config_from_yaml(yaml_config)
    else:
        aws_config = {}

    # Get configuration (command line > yaml > defaults)
    region = args.region or aws_config.get("region") or "us-east-1"
    s3_bucket = args.s3_bucket or aws_config.get("s3_bucket") or get_default_bucket(region)
    role_arn = args.role_arn or aws_config.get("role_arn") or get_default_role_arn(region)
    image_uri = aws_config.get("image_uri", "")

    if not role_arn:
        print("ERROR: Could not determine IAM role. Please specify --role-arn or set AWS_ROLE_ARN")
        sys.exit(1)

    # Get pipeline name from config or args
    pipeline_name = args.pipeline_name
    if yaml_config and "pipeline" in yaml_config:
        pipeline_name = yaml_config["pipeline"].get("name", pipeline_name)

    print(f"Region: {region}")
    print(f"S3 Bucket: {s3_bucket}")
    print(f"Role ARN: {role_arn}")
    print(f"Image URI: {image_uri or '(auto)'}")
    print(f"Pipeline Name: {pipeline_name}")

    # Build instance overrides from YAML config
    instance_overrides = {}
    if yaml_config and "instances" in yaml_config:
        for step_name, step_config in yaml_config["instances"].items():
            instance_overrides[step_name] = {
                "instance_type": step_config.get("type"),
                "volume_size_gb": step_config.get("volume_size"),
                "max_runtime_seconds": step_config.get("max_runtime"),
                "use_spot": step_config.get("use_spot"),
            }
            # Remove None values
            instance_overrides[step_name] = {k: v for k, v in instance_overrides[step_name].items() if v is not None}

    # Create pipeline config
    config_kwargs = {
        "s3_bucket": s3_bucket,
        "role_arn": role_arn,
        "region": region,
        "pipeline_name": pipeline_name,
    }
    if image_uri:
        config_kwargs["image_uri"] = image_uri
    if instance_overrides:
        config_kwargs["instance_overrides"] = instance_overrides

    config = PipelineConfig(**config_kwargs)
    pipeline = BoltzGenPipeline(config)
    result = pipeline.upsert_pipeline()

    print("\n" + "=" * 60)
    print("Pipeline created successfully!")
    print("=" * 60)
    print(f"Pipeline ARN: {result['pipeline_arn']}")
    print(f"Pipeline Name: {result['pipeline_name']}")

    # Save pipeline info
    info_file = Path("pipeline_info.json")
    with open(info_file, 'w') as f:
        json.dump({
            "pipeline_arn": result['pipeline_arn'],
            "pipeline_name": result['pipeline_name'],
            "s3_bucket": s3_bucket,
            "role_arn": role_arn,
            "region": region,
            "config_file": str(args.config) if args.config else None,
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nPipeline info saved to: {info_file}")


def run_pipeline_command(args):
    """Start a pipeline execution."""
    print("=" * 60)
    print("Starting BoltzGen Pipeline Execution")
    print("=" * 60)

    # Load config from YAML if provided
    yaml_config = {}
    if args.config:
        yaml_config = load_yaml_config(Path(args.config))
        aws_config = get_aws_config_from_yaml(yaml_config)
    else:
        aws_config = {}

    # Get configuration
    region = args.region or aws_config.get("region") or "us-east-1"
    s3_bucket = args.s3_bucket or aws_config.get("s3_bucket") or get_default_bucket(region)
    role_arn = args.role_arn or aws_config.get("role_arn") or get_default_role_arn(region)
    image_uri = aws_config.get("image_uri", "")

    if not role_arn:
        print("ERROR: Could not determine IAM role. Please specify --role-arn or set AWS_ROLE_ARN")
        sys.exit(1)

    # Get design parameters from YAML or args
    design_config = yaml_config.get("design", {})
    num_designs = args.num_designs or design_config.get("num_designs", 10000)
    budget = args.budget or design_config.get("budget", 100)
    protocol = args.protocol or design_config.get("protocol", "protein-anything")

    # Auto-determine scaling
    scaling_config = yaml_config.get("scaling", {})
    scaling_preset = scaling_config.get("preset", "auto")
    if scaling_preset == "auto":
        scaling_preset = get_scaling_preset_for_designs(num_designs)
    print(f"Scaling preset: {scaling_preset}")

    # Build instance overrides
    instance_overrides = {}
    design_instance_count = None
    folding_instance_count = None
    if yaml_config and "instances" in yaml_config:
        for step_name, step_config in yaml_config["instances"].items():
            instance_overrides[step_name] = {
                "instance_type": step_config.get("type"),
                "instance_count": step_config.get("instance_count"),
                "volume_size_gb": step_config.get("volume_size"),
                "max_runtime_seconds": step_config.get("max_runtime"),
                "use_spot": step_config.get("use_spot"),
            }
            instance_overrides[step_name] = {k: v for k, v in instance_overrides[step_name].items() if v is not None}
            # Extract instance counts for pipeline parameters
            if step_name == "design" and step_config.get("instance_count"):
                design_instance_count = step_config["instance_count"]
            if step_name == "folding" and step_config.get("instance_count"):
                folding_instance_count = step_config["instance_count"]

    # Get scaling step overrides
    if scaling_config.get("steps"):
        for step_name, step_scaling in scaling_config["steps"].items():
            if step_name not in instance_overrides:
                instance_overrides[step_name] = {}
            instance_overrides[step_name].update({
                k: v for k, v in step_scaling.items() if v is not None
            })

    # Create pipeline config
    config_kwargs = {
        "s3_bucket": s3_bucket,
        "role_arn": role_arn,
        "region": region,
        "num_designs": num_designs,
        "budget": budget,
        "protocol": protocol,
        "scaling_preset": scaling_preset,
    }
    if image_uri:
        config_kwargs["image_uri"] = image_uri
    if instance_overrides:
        config_kwargs["instance_overrides"] = instance_overrides

    config = PipelineConfig(**config_kwargs)

    # Estimate cost
    cost_estimate = estimate_cost(config)
    print(f"\nEstimated cost: ${cost_estimate['total']:.2f}")
    print("Cost breakdown:")
    for step, cost in cost_estimate['breakdown'].items():
        print(f"  {step}: ${cost:.2f}")

    # Upload design specs
    design_specs = args.design_spec
    if not design_specs and design_config.get("specs_dir"):
        # Load from config directory
        specs_dir = Path(design_config["specs_dir"])
        pattern = design_config.get("file_pattern", "*.yaml")
        if specs_dir.exists():
            design_specs = list(specs_dir.glob(pattern))
            print(f"\nFound {len(design_specs)} design specs in {specs_dir}")

    if design_specs:
        print(f"\nUploading design specifications...")
        input_s3_uri = upload_design_specs(
            design_specs,
            s3_bucket,
            config.s3_prefix,
            region,
        )
        print(f"Input S3 URI: {input_s3_uri}")
    else:
        input_s3_uri = args.input_s3_uri or config.input_s3_uri
        if not input_s3_uri or "your-" in input_s3_uri:
            print("ERROR: No design specs provided. Use --design-spec or configure specs_dir in YAML")
            sys.exit(1)

    # Create timestamp for this execution
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_s3_uri = f"s3://{s3_bucket}/{config.s3_prefix}/output/{timestamp}"
    execution_name = f"boltzgen-{timestamp}"

    print(f"\nOutput S3 URI: {output_s3_uri}")
    print(f"Execution name: {execution_name}")

    # Start execution
    pipeline = BoltzGenPipeline(config)

    # Ensure pipeline exists
    try:
        pipeline.upsert_pipeline()
    except Exception as e:
        print(f"Warning: Could not verify pipeline: {e}")

    result = pipeline.start_execution(
        num_designs=num_designs,
        budget=budget,
        protocol=protocol,
        input_s3_uri=input_s3_uri,
        output_s3_uri=output_s3_uri,
        design_instance_count=design_instance_count,
        folding_instance_count=folding_instance_count,
        execution_name=execution_name,
    )

    print("\n" + "=" * 60)
    print("Pipeline execution started!")
    print("=" * 60)
    print(f"Execution ARN: {result['execution_arn']}")
    print(f"Parameters: {json.dumps(result['parameters'], indent=2)}")

    # Save execution info
    info_file = Path(f"execution_info_{timestamp}.json")
    with open(info_file, 'w') as f:
        json.dump({
            "execution_arn": result['execution_arn'],
            "execution_name": execution_name,
            "input_s3_uri": input_s3_uri,
            "output_s3_uri": output_s3_uri,
            "parameters": result['parameters'],
            "cost_estimate": cost_estimate,
            "config_file": str(args.config) if args.config else None,
            "started_at": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nExecution info saved to: {info_file}")

    print("\nTo monitor execution:")
    print(f"  python run_pipeline.py status --execution-arn {result['execution_arn']}")

    print("\nTo download results when complete:")
    print(f"  aws s3 sync {output_s3_uri} ./results")

    # Wait if requested
    if args.wait:
        print("\nWaiting for execution to complete...")
        wait_for_execution(result['execution_arn'], region)


def status_command(args):
    """Check pipeline execution status."""
    sm = boto3.client('sagemaker', region_name=args.region)

    try:
        response = sm.describe_pipeline_execution(
            PipelineExecutionArn=args.execution_arn
        )

        print("=" * 60)
        print("Pipeline Execution Status")
        print("=" * 60)
        print(f"Status: {response['PipelineExecutionStatus']}")
        print(f"Started: {response.get('CreationTime', 'N/A')}")

        if 'LastModifiedTime' in response:
            print(f"Last Modified: {response['LastModifiedTime']}")

        if response['PipelineExecutionStatus'] == 'Failed':
            print(f"Failure Reason: {response.get('FailureReason', 'Unknown')}")

        # Get step details
        steps_response = sm.list_pipeline_execution_steps(
            PipelineExecutionArn=args.execution_arn
        )

        print("\nStep Status:")
        for step in steps_response.get('PipelineExecutionSteps', []):
            status = step.get('StepStatus', 'Unknown')
            name = step.get('StepName', 'Unknown')
            start_time = step.get('StartTime', '')
            end_time = step.get('EndTime', '')

            status_icon = {
                'Succeeded': '[OK]',
                'Failed': '[FAIL]',
                'Executing': '[...]',
                'Pending': '[ ]',
            }.get(status, '[?]')

            print(f"  {status_icon} {name}: {status}")

    except ClientError as e:
        print(f"Error: {e}")
        sys.exit(1)


def wait_for_execution(execution_arn: str, region: str):
    """Wait for pipeline execution to complete."""
    sm = boto3.client('sagemaker', region_name=region)

    while True:
        response = sm.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
        status = response['PipelineExecutionStatus']

        if status in ['Succeeded', 'Failed', 'Stopped']:
            print(f"\nExecution {status.lower()}!")
            if status == 'Failed':
                print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
                sys.exit(1)
            break

        # Get current step
        steps_response = sm.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn
        )

        current_step = "Unknown"
        for step in steps_response.get('PipelineExecutionSteps', []):
            if step.get('StepStatus') == 'Executing':
                current_step = step.get('StepName', 'Unknown')
                break

        print(f"  Status: {status} | Current step: {current_step}", end='\r')
        time.sleep(30)


def download_command(args):
    """Download pipeline results."""
    import subprocess

    print("=" * 60)
    print("Downloading Pipeline Results")
    print("=" * 60)

    # Get output location from execution
    sm = boto3.client('sagemaker', region_name=args.region)

    try:
        response = sm.describe_pipeline_execution(
            PipelineExecutionArn=args.execution_arn
        )

        if response['PipelineExecutionStatus'] != 'Succeeded':
            print(f"Warning: Execution status is {response['PipelineExecutionStatus']}")

        # For now, we need to construct the output path
        # In a real implementation, we'd extract this from the execution parameters
        print("Please use the output S3 URI from execution_info_*.json")
        print("Example: aws s3 sync s3://bucket/boltzgen-pipeline/output/TIMESTAMP ./results")

    except ClientError as e:
        print(f"Error: {e}")


def list_executions_command(args):
    """List recent pipeline executions."""
    sm = boto3.client('sagemaker', region_name=args.region)

    try:
        response = sm.list_pipeline_executions(
            PipelineName=args.pipeline_name,
            MaxResults=args.limit,
            SortBy='CreationTime',
            SortOrder='Descending',
        )

        print("=" * 60)
        print(f"Recent Executions for {args.pipeline_name}")
        print("=" * 60)

        for execution in response.get('PipelineExecutionSummaries', []):
            status = execution.get('PipelineExecutionStatus', 'Unknown')
            created = execution.get('StartTime', 'Unknown')
            name = execution.get('PipelineExecutionDisplayName', 'N/A')

            status_icon = {
                'Succeeded': '[OK]',
                'Failed': '[FAIL]',
                'Executing': '[...]',
                'Stopping': '[X]',
                'Stopped': '[X]',
            }.get(status, '[?]')

            print(f"{status_icon} {name}")
            print(f"    Status: {status}")
            print(f"    Created: {created}")
            print(f"    ARN: {execution.get('PipelineExecutionArn', 'N/A')}")
            print()

    except ClientError as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="BoltzGen SageMaker Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument("--config", type=str, help="YAML configuration file")
    parser.add_argument("--env", type=str, help="Environment file path (.env)")
    parser.add_argument("--region", default=None, help="AWS region")
    parser.add_argument("--s3-bucket", help="S3 bucket for pipeline data")
    parser.add_argument("--role-arn", help="IAM role ARN for SageMaker")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create/update pipeline")
    create_parser.add_argument(
        "--pipeline-name",
        default="BoltzGen-Protein-Design",
        help="Pipeline name"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline")
    run_parser.add_argument(
        "--design-spec",
        nargs="+",
        help="Design specification YAML file(s)"
    )
    run_parser.add_argument(
        "--input-s3-uri",
        help="S3 URI with pre-uploaded design specs"
    )
    run_parser.add_argument(
        "--num-designs",
        type=int,
        default=None,
        help="Number of designs to generate"
    )
    run_parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Final design budget"
    )
    run_parser.add_argument(
        "--protocol",
        default=None,
        choices=["protein-anything", "peptide-anything", "protein-small_molecule", "nanobody-anything"],
        help="Design protocol"
    )
    run_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for execution to complete"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check execution status")
    status_parser.add_argument(
        "--execution-arn",
        required=True,
        help="Pipeline execution ARN"
    )

    # Download command
    download_parser = subparsers.add_parser("download", help="Download results")
    download_parser.add_argument(
        "--execution-arn",
        required=True,
        help="Pipeline execution ARN"
    )
    download_parser.add_argument(
        "--output",
        default="./results",
        help="Output directory"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List recent executions")
    list_parser.add_argument(
        "--pipeline-name",
        default="BoltzGen-Protein-Design",
        help="Pipeline name"
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of executions to list"
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    env_path = Path(args.env) if args.env else None
    load_env_file(env_path)

    # Route to appropriate command
    if args.command == "create":
        create_pipeline_command(args)
    elif args.command == "run":
        run_pipeline_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "download":
        download_command(args)
    elif args.command == "list":
        list_executions_command(args)


if __name__ == "__main__":
    main()
