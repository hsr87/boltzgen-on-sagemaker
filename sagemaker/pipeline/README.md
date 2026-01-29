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
│  │   (GPU)      │───▶│   Folding    │───▶│  Validation  │       │
│  │              │    │   (GPU)      │    │   (GPU)      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                    │               │
│         ▼                   ▼                    ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Analysis   │───▶│   Filtering  │───▶│   Results    │       │
│  │   (CPU)      │    │   (CPU)      │    │   (S3)       │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Two Execution Modes:                                           │
│  • SageMaker Pipeline: 5-step workflow with caching & scaling   │
│  • Processing Job: Direct batch execution for simple cases      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. AWS account with SageMaker access
2. Docker image pushed to ECR (see `/sagemaker/docker/`)
3. IAM role with SageMaker permissions
4. Python 3.8+ with required packages

```bash
pip install boto3 pyyaml sagemaker
```

### Setup AWS Credentials

```bash
# Copy environment template
cp .env.example .env

# Edit with your AWS settings
vim .env
```

Example `.env`:
```bash
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-sagemaker-bucket
AWS_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole
AWS_IMAGE_URI=123456789012.dkr.ecr.us-east-1.amazonaws.com/boltzgen-sagemaker:latest
```

## Execution Modes

### Mode 1: SageMaker Pipeline (Recommended)

Full 5-step workflow with caching, scaling, and monitoring.

```bash
# 1. Create/update pipeline
python run_pipeline.py --config local_config.yaml create

# 2. Run pipeline with design specs
python run_pipeline.py --config local_config.yaml run

# 3. Check status
python run_pipeline.py --region us-east-1 status --execution-arn <ARN>

# 4. List recent executions
python run_pipeline.py --region us-east-1 list
```

**Configuration via YAML (`pipeline_config.yaml`):**

```yaml
aws:
  region: ${AWS_REGION}          # From .env
  s3_bucket: ${AWS_S3_BUCKET}
  role_arn: ${AWS_ROLE_ARN}
  image_uri: ${AWS_IMAGE_URI}

pipeline:
  name: BoltzGen-Protein-Design
  s3_prefix: boltzgen-pipeline

design:
  num_designs: 10000
  budget: 100
  protocol: protein-anything
  specs_dir: ./specs
  file_pattern: "*.yaml"

scaling:
  preset: auto    # auto, small, medium, large, xlarge

instances:
  design:
    type: ml.g5.12xlarge   # 4 GPUs for parallel processing
    use_spot: true
```

### Mode 2: Processing Job (Simple Batch)

Direct batch execution for simple use cases.

```bash
cd examples

# Single instance with multi-GPU
python run_single_instance_batch.py --config prot_only_config.yaml

# Multiple instances in parallel
python run_batch_from_config.py --config batch_config.yaml
```

## Directory Structure

```
sagemaker/pipeline/
├── README.md                 # This file
├── .env.example              # AWS credentials template
├── .env                      # AWS credentials (gitignored)
├── pipeline_config.yaml      # Pipeline configuration template
├── local_config.yaml         # Local testing configuration
├── config.py                 # Pipeline configuration classes
├── pipeline.py               # SageMaker Pipeline definition
├── run_pipeline.py           # Pipeline execution CLI
├── scripts/                  # Processing step scripts
│   ├── utils.py              # Shared utilities (GPU detection, progress tracking)
│   ├── design_step.py        # Multi-GPU parallel design generation
│   ├── inverse_folding_step.py
│   ├── folding_step.py
│   ├── analysis_step.py
│   └── filtering_step.py
└── examples/                 # Batch processing examples
    ├── README.md
    ├── .env.example
    ├── batch_config.yaml
    ├── prot_only_config.yaml
    ├── run_single_instance_batch.py
    ├── run_batch_from_config.py
    ├── estimate_batch.py
    └── QUOTA_INCREASE_GUIDE.md
```

## Multi-GPU Parallel Processing

The design step supports multi-GPU parallel processing within a single instance:

- Design specs are distributed across GPUs using round-robin scheduling
- Each GPU processes specs independently via `ProcessPoolExecutor`
- Progress tracking is thread-safe using `ProgressTracker` class
- Automatic GPU detection via `nvidia-smi`

**Example with ml.g5.12xlarge (4 GPUs):**
```
10 design specs → GPU0: specs 0,4,8 | GPU1: specs 1,5,9 | GPU2: specs 2,6 | GPU3: specs 3,7
```

## Data Flow Between Steps

All intermediate results are stored in S3 and passed between pipeline steps:

```
S3 Intermediate Storage:
├── designs/              ← Design step output
│   └── spec_name/
│       └── intermediate_designs/*.cif
├── inverse_folded/       ← Inverse folding output
├── folded/               ← Folding validation output
├── analyzed/             ← Analysis metrics output
└── metadata/             ← Step execution metadata
    ├── design/
    ├── inverse_folding/
    ├── folding/
    ├── analysis/
    └── filtering/
```

## Scaling

### Auto-Scaling Presets

The pipeline automatically selects scaling based on `num_designs`:

| Preset | Designs | Instance Count |
|--------|---------|----------------|
| small | ≤10K | 1 per step |
| medium | 10K-100K | 5 per step |
| large | 100K-1M | 10 per step |
| xlarge | >1M | 50 design, 25 others |

### Manual Scaling

Override in config:
```yaml
scaling:
  preset: auto
  steps:
    design:
      instance_count: 10
      instance_type: ml.g5.12xlarge
    folding:
      instance_count: 5
```

### Performance Estimates

| Samples | Configuration | Time | Cost |
|---------|--------------|------|------|
| 100 | 1x ml.g5.12xlarge (4 GPU) | ~38h | ~$269 |
| 1000 | 1x ml.g5.12xlarge (4 GPU) | ~375h | ~$2,659 |
| 1000 | 10x ml.g5.12xlarge (40 GPU) | ~38h | ~$2,695 |

*Based on ~1.5 hours per sample with 100 designs*

## Instance Pricing (US East)

| Instance Type | GPUs | Price/hour | Parallel Samples |
|--------------|------|------------|------------------|
| ml.g4dn.xlarge | 1 | $0.74 | 1 |
| ml.g5.xlarge | 1 | $1.41 | 1 |
| ml.g5.12xlarge | 4 | $7.09 | 4 |
| ml.g5.48xlarge | 8 | $20.36 | 8 |

## Monitoring

```bash
# Pipeline status
python run_pipeline.py --region us-east-1 status --execution-arn <ARN>

# List jobs
aws sagemaker list-processing-jobs --name-contains boltzgen

# Stream logs
aws logs tail /aws/sagemaker/ProcessingJobs --follow --log-stream-name-prefix <JOB_NAME>

# Download results
aws s3 sync s3://bucket/boltzgen-pipeline/output/<TIMESTAMP> ./results
```

## Security

### Credential Management
- AWS credentials are managed via `.env` files (gitignored)
- Use `.env.example` as template for credential setup
- Never commit sensitive values to the repository
- IAM roles should follow least-privilege principle

### Environment Variable Substitution

YAML config files support environment variable substitution:

```yaml
aws:
  region: ${AWS_REGION}           # Required - from .env
  s3_bucket: ${AWS_S3_BUCKET}     # Required - from .env
  role_arn: ${AWS_ROLE_ARN}       # Required - from .env
  image_uri: ${AWS_IMAGE_URI:default-image}  # With default value
```

Syntax:
- `${VAR_NAME}` - Required variable
- `${VAR_NAME:default}` - Variable with default value

The `.env` file is automatically loaded when running `run_pipeline.py`.

## Quota Management

Default SageMaker quotas may be insufficient. See `examples/QUOTA_INCREASE_GUIDE.md`.

```bash
# Check current quotas
aws service-quotas list-service-quotas --service-code sagemaker \
  --query 'Quotas[?contains(QuotaName, `g5`)]'
```

## Shared Utilities (`scripts/utils.py`)

Common utilities used across pipeline step scripts:

| Function | Description |
|----------|-------------|
| `ProgressTracker` | Thread-safe progress counter for parallel processing |
| `get_gpu_count()` | Detect available GPUs via nvidia-smi |
| `load_step_config()` | Load pipeline step configuration |
| `save_step_metadata()` | Save step execution metadata |
| `run_command_with_timeout()` | Execute commands with timeout handling |
| `validate_directory()` | Validate/create directories |

**Note**: SageMaker ProcessingStep only uploads a single script file per step. For this reason,
`design_step.py` includes inline copies of the utility functions rather than importing from `utils.py`.
The `utils.py` file serves as the reference implementation and is used by example scripts.

## Security Considerations

### Input Validation
- File size limits enforced on uploads (default: 100MB)
- S3 bucket and prefix names validated before operations
- Design spec paths resolved to absolute paths to prevent traversal

### Safe Subprocess Execution
- All subprocess calls use argument lists (not shell strings) to prevent injection
- Timeout handling for long-running commands
- Environment variables isolated per process

### Error Handling
- Specific exception types caught (not bare `except:`)
- Graceful fallbacks with informative error messages
- Step validation at pipeline initialization time

## Troubleshooting

### Common Issues

1. **ResourceLimitExceeded**: Quota limit reached
   - Stop running jobs or request quota increase
   - See `examples/QUOTA_INCREASE_GUIDE.md`

2. **ImportError for sagemaker**: SDK version incompatibility
   ```bash
   pip install 'sagemaker>=2.0,<3.0'
   ```

3. **No design specs found**: Input directory structure issue
   - Ensure YAML specs are in the configured `specs_dir`
   - Check file pattern matches your files

## Related Documentation

- [BoltzGen Documentation](https://github.com/jwohlwend/boltzgen)
- [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Processing Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
