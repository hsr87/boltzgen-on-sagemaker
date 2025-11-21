# BoltzGen on AWS SageMaker Processing

This guide explains how to run BoltzGen as an AWS SageMaker Processing Job.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building and Pushing Docker Image to ECR](#building-and-pushing-docker-image-to-ecr)
3. [Running SageMaker Processing Job](#running-sagemaker-processing-job)
4. [Checking Results](#checking-results)
5. [Cost Optimization](#cost-optimization)

## Prerequisites

### 1. AWS CLI Setup

```bash
# Check AWS CLI installation
aws --version

# Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g., us-east-1)
```

### 2. Install Required Python Packages

```bash
pip install boto3 sagemaker
```

### 3. Prepare IAM Role

You need an IAM role for SageMaker with the following permissions:

- `AmazonSageMakerFullAccess`
- S3 bucket read/write permissions
- ECR image read permissions

**Create IAM Role (Optional)**:

```bash
# Create SageMaker execution role (can also be done via AWS Console)
aws iam create-role \
  --role-name BoltzGenSageMakerRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach required policy
aws iam attach-role-policy \
  --role-name BoltzGenSageMakerRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

### 4. Create S3 Bucket

```bash
# Create S3 bucket (bucket name must be unique)
aws s3 mb s3://your-boltzgen-bucket --region us-east-1
```

## Building and Pushing Docker Image to ECR

### Automated Build (Recommended)

```bash
# Run from project root
cd /path/to/boltzgen

# Execute build and push script
./sagemaker/build_and_push.sh

# Or specify a specific region
AWS_REGION=us-west-2 ./sagemaker/build_and_push.sh
```

This script automatically:
1. Creates ECR repository (`boltzgen-sagemaker`)
2. Builds Docker image for AMD64 architecture
3. Pushes image to ECR
4. Saves image URI to `sagemaker/image_uri.txt`

### Manual Build (Optional)

```bash
# 1. Create ECR repository
aws ecr create-repository \
  --repository-name boltzgen-sagemaker \
  --region us-east-1

# 2. Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# 3. Build Docker image (AMD64 for SageMaker compatibility)
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.sagemaker \
  -t boltzgen-sagemaker:latest \
  .

# 4. Tag image
docker tag boltzgen-sagemaker:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/boltzgen-sagemaker:latest

# 5. Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/boltzgen-sagemaker:latest
```

## Running SageMaker Processing Job

### Basic Usage

```bash
python sagemaker/run_processing_job.py \
  --design-spec example/vanilla_protein/1g13prot.yaml \
  --s3-bucket your-boltzgen-bucket \
  --instance-type ml.g4dn.xlarge \
  --num-designs 10 \
  --budget 2
```

### All Options

```bash
python sagemaker/run_processing_job.py \
  --design-spec <local-yaml-file> \
  --s3-bucket <your-s3-bucket> \
  --s3-prefix boltzgen \
  --instance-type ml.g4dn.xlarge \
  --instance-count 1 \
  --volume-size 50 \
  --max-runtime 86400 \
  --protocol protein-anything \
  --num-designs 100 \
  --budget 10 \
  --region us-east-1 \
  --role <iam-role-arn> \
  --wait
```

**Key Parameters**:
- `--design-spec`: Local design specification YAML file
- `--s3-bucket`: S3 bucket to store results
- `--instance-type`: SageMaker instance type (GPU instance recommended)
- `--num-designs`: Number of designs to generate
- `--budget`: Number of final designs to select
- `--wait`: Wait for job completion (optional)

### Available Instance Types

| Instance Type | GPU | vCPU | Memory | On-Demand Price/hr |
|--------------|-----|------|--------|-------------------|
| ml.g4dn.xlarge | T4 x1 | 4 | 16GB | ~$0.74 |
| ml.g4dn.2xlarge | T4 x1 | 8 | 32GB | ~$1.00 |
| ml.g5.xlarge | A10G x1 | 4 | 16GB | ~$1.41 |
| ml.g5.2xlarge | A10G x1 | 8 | 32GB | ~$1.69 |

## Checking Results

### 1. Monitor Job Status

```bash
# Check job status
aws sagemaker describe-processing-job \
  --processing-job-name boltzgen-<timestamp>

# Check logs (real-time)
aws logs tail /aws/sagemaker/ProcessingJobs \
  --follow \
  --log-stream-name-prefix boltzgen-<timestamp>
```

### 2. Download Results

```bash
# Download results from S3
aws s3 sync s3://your-boltzgen-bucket/boltzgen/output/boltzgen-<timestamp> ./results

# Result structure
./results/
├── results/                          # BoltzGen output
│   ├── final_ranked_designs/         # Final designs
│   ├── intermediate_designs/         # Intermediate designs
│   └── ...
└── job_metadata.json                 # Job metadata
```

### 3. Check in SageMaker Console

AWS Console → SageMaker → Processing Jobs to view job status and logs

## Cost Optimization

### 1. Choose Appropriate Instance

- **Small-scale testing**: ml.g4dn.xlarge (~$0.74/hour)
- **Production**: ml.g5.xlarge (~$1.41/hour)

### 2. Optimize Job Duration

```bash
# First, test with small number
python sagemaker/run_processing_job.py \
  --design-spec example/vanilla_protein/1g13prot.yaml \
  --s3-bucket your-bucket \
  --num-designs 10 \
  --budget 2

# After confirming success, run large batch
python sagemaker/run_processing_job.py \
  --design-spec example/vanilla_protein/1g13prot.yaml \
  --s3-bucket your-bucket \
  --num-designs 10000 \
  --budget 100
```

### 3. Leverage Model Caching

Model weights (~6GB) are downloaded only on first run; subsequent runs use cached weights.

### 4. Estimated Cost Calculation

**Example**: Generate 1000 designs (using ml.g4dn.xlarge)

- Estimated runtime: ~3 hours
- Cost: 3 hours × $0.74 = **~$2.22**
- S3 storage: ~$0.10/month

## Troubleshooting

### ECR Login Failure

```bash
# Update AWS CLI
pip install --upgrade awscli

# Verify credentials
aws sts get-caller-identity
```

### IAM Permission Errors

Verify IAM role has the following permissions:
- `sagemaker:CreateProcessingJob`
- `s3:GetObject`, `s3:PutObject`
- `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`

### GPU Memory Insufficient

Use larger instance type:
```bash
--instance-type ml.g4dn.2xlarge  # or ml.g5.2xlarge
```

### Job Timeout

Increase maximum runtime:
```bash
--max-runtime 172800  # 48 hours
```

## Next Steps

1. **Batch Processing**: Process multiple design specifications sequentially
2. **Step Functions Integration**: Automate complex workflows
3. **Result Analysis**: Analyze results in SageMaker Notebook

## References

- [AWS SageMaker Processing Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [BoltzGen Official Documentation](../README.md)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
