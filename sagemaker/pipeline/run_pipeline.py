#!/usr/bin/env python3
"""
BoltzGen SageMaker Pipeline Runner

This script provides a CLI interface to:
1. Create/update the SageMaker Pipeline
2. Upload design specifications to S3
3. Start pipeline executions
4. Monitor execution status
5. Download results

Usage:
    # Create pipeline
    python run_pipeline.py create --s3-bucket my-bucket --role-arn arn:aws:iam::...

    # Run pipeline with design spec
    python run_pipeline.py run --design-spec example/protein.yaml --num-designs 10000

    # Check status
    python run_pipeline.py status --execution-arn arn:aws:sagemaker:...

    # Download results
    python run_pipeline.py download --execution-arn arn:aws:sagemaker:... --output ./results
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, estimate_cost, get_scaling_preset_for_designs
from pipeline import BoltzGenPipeline, create_default_pipeline


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
) -> str:
    """Upload design specification files to S3.

    Returns:
        S3 URI of uploaded files
    """
    s3 = boto3.client('s3', region_name=region)

    input_prefix = f"{s3_prefix}/input"

    for spec_path in design_specs:
        spec_path = Path(spec_path)
        if not spec_path.exists():
            raise FileNotFoundError(f"Design spec not found: {spec_path}")

        # Upload the spec file
        s3_key = f"{input_prefix}/{spec_path.name}"
        print(f"Uploading {spec_path} to s3://{s3_bucket}/{s3_key}")
        s3.upload_file(str(spec_path), s3_bucket, s3_key)

        # Upload any referenced .cif or .pdb files in the same directory
        spec_dir = spec_path.parent
        for ext in ['*.cif', '*.pdb']:
            for file in spec_dir.glob(ext):
                s3_key = f"{input_prefix}/{file.name}"
                print(f"Uploading {file} to s3://{s3_bucket}/{s3_key}")
                s3.upload_file(str(file), s3_bucket, s3_key)

    return f"s3://{s3_bucket}/{input_prefix}"


def create_pipeline_command(args):
    """Create or update the SageMaker Pipeline."""
    print("=" * 60)
    print("Creating/Updating BoltzGen Pipeline")
    print("=" * 60)

    # Get defaults
    region = args.region
    s3_bucket = args.s3_bucket or get_default_bucket(region)
    role_arn = args.role_arn or get_default_role_arn(region)

    if not role_arn:
        print("ERROR: Could not determine IAM role. Please specify --role-arn")
        sys.exit(1)

    print(f"Region: {region}")
    print(f"S3 Bucket: {s3_bucket}")
    print(f"Role ARN: {role_arn}")

    # Create pipeline
    config = PipelineConfig(
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
        pipeline_name=args.pipeline_name,
    )

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
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nPipeline info saved to: {info_file}")


def run_pipeline_command(args):
    """Start a pipeline execution."""
    print("=" * 60)
    print("Starting BoltzGen Pipeline Execution")
    print("=" * 60)

    # Get configuration
    region = args.region
    s3_bucket = args.s3_bucket or get_default_bucket(region)
    role_arn = args.role_arn or get_default_role_arn(region)

    if not role_arn:
        print("ERROR: Could not determine IAM role. Please specify --role-arn")
        sys.exit(1)

    # Auto-determine scaling
    scaling_preset = get_scaling_preset_for_designs(args.num_designs)
    print(f"Auto-selected scaling preset: {scaling_preset}")

    # Estimate cost
    config = PipelineConfig(
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
        num_designs=args.num_designs,
        budget=args.budget,
        protocol=args.protocol,
        scaling_preset=scaling_preset,
    )

    cost_estimate = estimate_cost(config)
    print(f"\nEstimated cost: ${cost_estimate['total']:.2f}")
    print("Cost breakdown:")
    for step, cost in cost_estimate['breakdown'].items():
        print(f"  {step}: ${cost:.2f}")

    # Upload design specs
    if args.design_spec:
        print(f"\nUploading design specifications...")
        input_s3_uri = upload_design_specs(
            args.design_spec,
            s3_bucket,
            config.s3_prefix,
            region,
        )
        print(f"Input S3 URI: {input_s3_uri}")
    else:
        input_s3_uri = args.input_s3_uri or config.input_s3_uri

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
        num_designs=args.num_designs,
        budget=args.budget,
        protocol=args.protocol,
        input_s3_uri=input_s3_uri,
        output_s3_uri=output_s3_uri,
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
    parser.add_argument("--region", default="us-east-1", help="AWS region")
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
        default=10000,
        help="Number of designs to generate"
    )
    run_parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Final design budget"
    )
    run_parser.add_argument(
        "--protocol",
        default="protein-anything",
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
