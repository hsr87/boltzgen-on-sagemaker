#!/usr/bin/env python3
"""
Launch BoltzGen as a SageMaker Processing Job
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import boto3
from sagemaker import get_execution_role
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput


def get_image_uri(region='us-east-1'):
    """Get the ECR image URI"""
    # Try to read from saved file first
    image_uri_file = Path(__file__).parent / 'image_uri.txt'
    if image_uri_file.exists():
        with open(image_uri_file) as f:
            return f.read().strip()

    # Otherwise, construct it
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/boltzgen-sagemaker:latest"


def upload_design_spec_to_s3(local_path, s3_bucket, s3_prefix):
    """Upload design specification and related files to S3"""
    s3 = boto3.client('s3')

    local_path = Path(local_path)
    design_dir = local_path.parent

    # Upload the design spec
    s3_key = f"{s3_prefix}/input/{local_path.name}"
    print(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
    s3.upload_file(str(local_path), s3_bucket, s3_key)

    # Upload any referenced .cif or .pdb files
    for ext in ['*.cif', '*.pdb']:
        for file in design_dir.glob(ext):
            s3_key = f"{s3_prefix}/input/{file.name}"
            print(f"Uploading {file} to s3://{s3_bucket}/{s3_key}")
            s3.upload_file(str(file), s3_bucket, s3_key)

    return f"s3://{s3_bucket}/{s3_prefix}/input"


def main():
    parser = argparse.ArgumentParser(description='Run BoltzGen on SageMaker Processing')
    parser.add_argument('--design-spec', type=str, required=True,
                        help='Path to local design specification YAML file')
    parser.add_argument('--s3-bucket', type=str, required=True,
                        help='S3 bucket for input/output data')
    parser.add_argument('--s3-prefix', type=str, default='boltzgen',
                        help='S3 prefix for organizing data')
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                        help='SageMaker instance type (default: ml.g4dn.xlarge)')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--volume-size', type=int, default=50,
                        help='EBS volume size in GB (default: 50)')
    parser.add_argument('--max-runtime', type=int, default=86400,
                        help='Max runtime in seconds (default: 86400 = 24 hours)')
    parser.add_argument('--protocol', type=str, default='protein-anything',
                        choices=['protein-anything', 'peptide-anything',
                                'protein-small_molecule', 'nanobody-anything'],
                        help='BoltzGen protocol')
    parser.add_argument('--num-designs', type=int, default=10,
                        help='Number of designs to generate')
    parser.add_argument('--budget', type=int, default=2,
                        help='Final design budget')
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS region')
    parser.add_argument('--role', type=str, default=None,
                        help='IAM role ARN (defaults to SageMaker execution role)')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for job to complete')

    args = parser.parse_args()

    # Get IAM role
    try:
        role = args.role or get_execution_role()
    except Exception:
        print("Error: Could not get execution role. Please specify --role with an IAM role ARN")
        print("The role needs SageMaker, S3, and ECR permissions")
        return

    # Get image URI
    image_uri = get_image_uri(args.region)
    print(f"Using image: {image_uri}")

    # Upload design spec to S3
    print("\nUploading design specification to S3...")
    input_s3_uri = upload_design_spec_to_s3(
        args.design_spec,
        args.s3_bucket,
        args.s3_prefix
    )

    # Create job name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"boltzgen-{timestamp}"

    # Setup output paths
    output_s3_uri = f"s3://{args.s3_bucket}/{args.s3_prefix}/output/{job_name}"

    # Create processor
    print(f"\nCreating SageMaker Processor...")
    print(f"Instance type: {args.instance_type}")
    print(f"Instance count: {args.instance_count}")
    print(f"Volume size: {args.volume_size} GB")

    processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size_in_gb=args.volume_size,
        max_runtime_in_seconds=args.max_runtime,
        base_job_name='boltzgen',
        command=['python3'],
        sagemaker_session=None  # Use default session
    )

    # Prepare processing script arguments
    design_spec_filename = Path(args.design_spec).name

    # Run processing job
    print(f"\n{'='*60}")
    print(f"Launching SageMaker Processing Job: {job_name}")
    print(f"{'='*60}")
    print(f"Input S3: {input_s3_uri}")
    print(f"Output S3: {output_s3_uri}")
    print(f"Protocol: {args.protocol}")
    print(f"Num designs: {args.num_designs}")
    print(f"Budget: {args.budget}")
    print(f"{'='*60}\n")

    try:
        processor.run(
            code=str(Path(__file__).parent / 'processing_script.py'),
            inputs=[
                ProcessingInput(
                    source=input_s3_uri,
                    destination='/opt/ml/processing/input',
                    input_name='design-spec'
                )
            ],
            outputs=[
                ProcessingOutput(
                    source='/opt/ml/processing/output',
                    destination=output_s3_uri,
                    output_name='results'
                )
            ],
            arguments=[
                '--design-spec', design_spec_filename,
                '--protocol', args.protocol,
                '--num-designs', str(args.num_designs),
                '--budget', str(args.budget),
                '--devices', '1'
            ],
            wait=args.wait,
            logs=True,
            job_name=job_name
        )

        print(f"\n{'='*60}")
        if args.wait:
            print("✓ Processing job completed successfully!")
        else:
            print("✓ Processing job launched successfully!")
        print(f"{'='*60}")
        print(f"Job name: {job_name}")
        print(f"Output location: {output_s3_uri}")
        print(f"\nTo monitor the job:")
        print(f"  aws sagemaker describe-processing-job --processing-job-name {job_name}")
        print(f"\nTo view logs:")
        print(f"  aws logs tail /aws/sagemaker/ProcessingJobs --follow --log-stream-name-prefix {job_name}")
        print(f"\nTo download results:")
        print(f"  aws s3 sync {output_s3_uri} ./results")
        print(f"{'='*60}\n")

        # Save job info
        job_info = {
            'job_name': job_name,
            'input_s3_uri': input_s3_uri,
            'output_s3_uri': output_s3_uri,
            'instance_type': args.instance_type,
            'protocol': args.protocol,
            'num_designs': args.num_designs,
            'budget': args.budget,
            'timestamp': timestamp
        }

        job_info_file = Path(f'job_info_{timestamp}.json')
        with open(job_info_file, 'w') as f:
            json.dump(job_info, f, indent=2)

        print(f"Job info saved to: {job_info_file}")

    except Exception as e:
        print(f"\n✗ Error launching processing job: {e}")
        raise


if __name__ == '__main__':
    main()
