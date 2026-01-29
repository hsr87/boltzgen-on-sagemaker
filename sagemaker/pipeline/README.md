# BoltzGen SageMaker Pipeline

Scalable protein design pipeline using BoltzGen on AWS SageMaker.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BoltzGen on SageMaker                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Design     │    │   Inverse    │    │   Folding    │       │
│  │   Specs      │───▶│   Folding    │───▶│  Validation  │       │
│  │   (S3)       │    │   (GPU)      │    │   (GPU)      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                             │                    │               │
│                             ▼                    ▼               │
│                      ┌──────────────┐    ┌──────────────┐       │
│                      │   Analysis   │───▶│   Results    │       │
│                      │   & Ranking  │    │   (S3)       │       │
│                      └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Execution Modes:                                                │
│  • Single Instance: 1x ml.g5.12xlarge (4 GPUs parallel)         │
│  • Multi Instance:  Nx instances for massive parallelization    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. AWS account with SageMaker access
2. Docker image pushed to ECR (see `/sagemaker/docker/`)
3. IAM role with SageMaker permissions
4. Python 3.8+ with boto3, pyyaml

```bash
pip install boto3 pyyaml python-dotenv
```

### Run Batch Processing

```bash
cd sagemaker/pipeline/examples

# 1. Setup AWS credentials
cp .env.example .env
# Edit .env with your AWS settings

# 2. Prepare example data
mkdir -p prot_only_specs
cp ../../../example/hard_targets/*prot.yaml prot_only_specs/
cp ../../../example/hard_targets/*.cif prot_only_specs/

# 3. Estimate cost and time
python estimate_batch.py --samples 100

# 4. Edit configuration
vim prot_only_config.yaml  # Update with your .env values

# 5. Run batch (dry-run first)
python run_single_instance_batch.py --config prot_only_config.yaml --dry-run

# 6. Execute
python run_single_instance_batch.py --config prot_only_config.yaml
```

## Directory Structure

```
sagemaker/pipeline/
├── README.md                 # This file
├── config.py                 # Pipeline configuration classes
├── pipeline.py               # SageMaker Pipeline definition
├── run_pipeline.py           # Pipeline execution CLI
├── scripts/                  # Processing step scripts
│   ├── design_step.py
│   ├── inverse_folding_step.py
│   ├── folding_step.py
│   ├── analysis_step.py
│   └── filtering_step.py
└── examples/                 # Batch processing examples
    ├── README.md
    ├── batch_config.yaml
    ├── run_single_instance_batch.py
    ├── run_batch_from_config.py
    ├── estimate_batch.py
    └── QUOTA_INCREASE_GUIDE.md
```

## Execution Modes

### 1. Single Instance Mode
One multi-GPU instance processes all samples sequentially with GPU parallelism.

```bash
python examples/run_single_instance_batch.py --config examples/batch_config.yaml
```

**Best for:**
- Small to medium batches (< 500 samples)
- Simple setup
- Consistent performance

### 2. Multi Instance Mode
Multiple instances process samples in parallel.

```bash
python examples/run_batch_from_config.py --config examples/batch_config.yaml
```

**Best for:**
- Large batches (500+ samples)
- Faster completion time
- When quota allows multiple instances

## Scaling: More Instances = Faster Processing

The key to faster batch processing is parallelization. Each GPU processes one sample at a time, so adding more instances directly reduces total processing time.

### Scaling Example: 1000 Samples

| Configuration | Total GPUs | Time | Cost |
|--------------|------------|------|------|
| 1x ml.g5.12xlarge | 4 | ~375h (15.6 days) | ~$2,659 |
| 5x ml.g5.12xlarge | 20 | ~75h (3.1 days) | ~$2,659 |
| 10x ml.g5.12xlarge | 40 | ~38h (1.6 days) | ~$2,836 |

**Key insight**: Scaling from 1 to 10 instances reduces time by 10x with similar cost.

### Instance Pricing (US East - N. Virginia)

| Instance Type | GPUs | Price/hour | Parallel Samples |
|--------------|------|------------|------------------|
| ml.g4dn.xlarge | 1 | $0.74 | 1 |
| ml.g5.xlarge | 1 | $1.41 | 1 |
| ml.g5.12xlarge | 4 | $7.09 | 4 |
| ml.g5.48xlarge | 8 | $20.36 | 8 |

*Pricing source: [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)*

### Performance Estimates

| Samples | Configuration | Time | Cost |
|---------|--------------|------|------|
| 100 | 1x ml.g5.12xlarge (4 GPU) | ~38h | ~$269 |
| 100 | 4x ml.g5.xlarge (4 GPU) | ~38h | ~$214 |
| 100 | 10x ml.g5.xlarge (10 GPU) | ~15h | ~$212 |
| 1000 | 1x ml.g5.12xlarge (4 GPU) | ~375h | ~$2,659 |
| 1000 | 10x ml.g5.12xlarge (40 GPU) | ~38h | ~$2,695 |

*Based on ~1.5 hours per sample with 100 designs*

## Configuration

See `examples/batch_config.yaml` for full configuration options:

```yaml
aws:
  region: us-east-1
  s3_bucket: your-bucket
  role_arn: arn:aws:iam::ACCOUNT:role/SageMakerRole
  image_uri: ACCOUNT.dkr.ecr.REGION.amazonaws.com/boltzgen-sagemaker:latest

instances:
  type: ml.g5.12xlarge
  count: 2                # Multiple instances for parallel processing
  volume_size: 100
  max_runtime: 432000     # 5 days max

design:
  specs_dir: /path/to/specs
  num_designs: 100
  budget: 10

batch:
  per_sample_timeout: 10800  # 3 hours per sample
```

## Monitoring

```bash
# List jobs
aws sagemaker list-processing-jobs --name-contains boltzgen

# View job details
aws sagemaker describe-processing-job --processing-job-name JOB_NAME

# Stream logs
aws logs tail /aws/sagemaker/ProcessingJobs --follow --log-stream-name-prefix JOB_NAME

# Download results
aws s3 sync s3://bucket/boltzgen-batch/BATCH_ID/output ./results
```

## Quota Management

Default SageMaker quotas may be insufficient for large batches. See `examples/QUOTA_INCREASE_GUIDE.md` for instructions on requesting quota increases.

```bash
# Check current quotas
aws service-quotas list-service-quotas --service-code sagemaker \
  --query 'Quotas[?contains(QuotaName, `g5`)]'
```

## Related Documentation

- [BoltzGen Documentation](https://github.com/jwohlwend/boltzgen)
- [SageMaker Processing Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [SageMaker Quotas](https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html)
