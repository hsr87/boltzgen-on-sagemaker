# BoltzGen Batch Processing Examples

This directory contains scripts and configurations for running large-scale BoltzGen batch processing jobs on AWS SageMaker.

## Overview

Two main execution modes are supported:

| Mode | Script | Use Case |
|------|--------|----------|
| **Single Instance** | `run_single_instance_batch.py` | One multi-GPU instance processing all samples |
| **Multi Instance** | `run_batch_from_config.py` | Multiple instances processing samples in parallel |

## Prerequisites

### 1. AWS Credentials Setup

Copy the example environment file and fill in your AWS credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:
```bash
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-sagemaker-bucket
AWS_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole
AWS_IMAGE_URI=123456789012.dkr.ecr.us-east-1.amazonaws.com/boltzgen-sagemaker:latest
```

Then update `batch_config.yaml` or `prot_only_config.yaml` with these values.

### 2. Prepare Example Data

Copy example specs from the main example directory:

```bash
# Copy protein-only specs (self-contained, no external dependencies)
mkdir -p prot_only_specs
cp ../../../example/hard_targets/*prot.yaml prot_only_specs/
cp ../../../example/hard_targets/*.cif prot_only_specs/
```

## Quick Start

### 1. Estimate Time and Cost

```bash
# Estimate for 1000 samples
python estimate_batch.py --samples 1000

# Estimate with specific configuration
python estimate_batch.py --samples 1000 --instance-type ml.g5.12xlarge --instances 10
```

### 2. Configure Your Batch

Edit `batch_config.yaml` with your settings:

```yaml
aws:
  region: us-east-1
  s3_bucket: your-bucket-name
  role_arn: arn:aws:iam::ACCOUNT:role/SageMakerRole
  image_uri: ACCOUNT.dkr.ecr.REGION.amazonaws.com/boltzgen-sagemaker:latest

instances:
  type: ml.g5.12xlarge  # 4 GPUs
  count: 2              # Number of instances (2x = 8 GPUs total)
  volume_size: 100
  max_runtime: 432000   # 5 days max

batch:
  per_sample_timeout: 10800  # 3 hours per sample (for complex targets)

design:
  specs_dir: /path/to/your/specs
  file_pattern: "*.yaml"
  num_designs: 100
  budget: 10
```

### 3. Run Batch Processing

**Single Instance Mode** (recommended for < 500 samples):
```bash
# Dry run to verify configuration
python run_single_instance_batch.py --config batch_config.yaml --dry-run

# Execute
python run_single_instance_batch.py --config batch_config.yaml

# Execute and wait for completion
python run_single_instance_batch.py --config batch_config.yaml --wait
```

**Multi Instance Mode** (recommended for > 500 samples):
```bash
# Dry run
python run_batch_from_config.py --config batch_config.yaml --dry-run

# Execute
python run_batch_from_config.py --config batch_config.yaml
```

### 4. Monitor Jobs

```bash
# List running jobs
aws sagemaker list-processing-jobs --status-equals InProgress

# Check specific job
aws sagemaker describe-processing-job --processing-job-name YOUR_JOB_NAME

# View logs
aws logs tail /aws/sagemaker/ProcessingJobs --follow --log-stream-name-prefix YOUR_JOB_NAME
```

### 5. Download Results

```bash
aws s3 sync s3://your-bucket/boltzgen-batch/YOUR_BATCH_ID/output ./results
```

## Instance Pricing (US East - N. Virginia)

| Instance Type | GPUs | $/hour | 100 samples | 1000 samples |
|--------------|------|--------|-------------|--------------|
| ml.g4dn.xlarge | 1 | $0.74 | 150h / $111 | 1500h / $1,110 |
| ml.g5.xlarge | 1 | $1.41 | 150h / $212 | 1500h / $2,115 |
| ml.g5.12xlarge | 4 | $7.09 | 38h / $269 | 375h / $2,659 |
| ml.g5.48xlarge | 8 | $20.36 | 19h / $387 | 188h / $3,828 |

*Pricing source: [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)*
*Estimates based on ~1.5 hours per sample with 100 designs*

## Why More Instances = Faster

Each GPU processes one sample at a time. Scaling instances directly reduces total time:

| 1000 Samples | Instances | Total GPUs | Time | Cost |
|--------------|-----------|------------|------|------|
| Baseline | 1x ml.g5.12xlarge | 4 | 375h | $2,659 |
| 5x faster | 5x ml.g5.12xlarge | 20 | 75h | $2,659 |
| 10x faster | 10x ml.g5.12xlarge | 40 | 38h | $2,695 |

**Note**: Cost remains similar because total GPU-hours is constant (1000 samples × 1.5h = 1500 GPU-hours).

## Scaling Strategies

### Strategy 1: Single Multi-GPU Instance
Best for: Simplicity, consistent workloads

```yaml
instances:
  type: ml.g5.12xlarge  # 4 GPUs = 4 parallel samples
  count: 1
```

### Strategy 2: Multiple Instances
Best for: Large batches, faster completion

```yaml
instances:
  type: ml.g5.xlarge    # 1 GPU each
  count: 10             # 10 parallel samples
```

### Strategy 3: Hybrid (Maximum Throughput)
Best for: Very large batches (1000+ samples)

Use `run_max_quota_batch.py` to utilize all available quota across different instance types.

## Quota Management

Check your current SageMaker quotas:
```bash
aws service-quotas list-service-quotas \
  --service-code sagemaker \
  --query 'Quotas[?contains(QuotaName, `g5`) && contains(QuotaName, `processing`)]'
```

Request quota increase:
```bash
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-B013C051 \
  --desired-value 10
```

See [QUOTA_INCREASE_GUIDE.md](./QUOTA_INCREASE_GUIDE.md) for detailed instructions.

## Files

| File | Description |
|------|-------------|
| `run_single_instance_batch.py` | Single multi-GPU instance batch processing |
| `run_batch_from_config.py` | Multi-instance batch processing |
| `estimate_batch.py` | Time and cost estimator |
| `batch_config.yaml` | General configuration template |
| `prot_only_config.yaml` | Config for protein-only examples (10 samples) |
| `.env.example` | AWS credentials template |
| `QUOTA_INCREASE_GUIDE.md` | AWS quota increase instructions |

## Troubleshooting

### ResourceLimitExceeded
Your account quota is insufficient. Request a quota increase or use a different instance type.

### Job Timeout
Increase `max_runtime` in config (max: 432000 seconds = 5 days) or split into multiple jobs.

### AlgorithmError
Check CloudWatch logs for detailed error messages:
```bash
aws logs get-log-events \
  --log-group-name /aws/sagemaker/ProcessingJobs \
  --log-stream-name YOUR_JOB_NAME/algo-1-XXXXX
```
