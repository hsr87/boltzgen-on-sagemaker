# BoltzGen SageMaker Processing - Quick Start

## 1-Minute Summary

A quick guide to run BoltzGen using AWS SageMaker Processing.

## Complete Workflow (3 Steps)

```bash
# 1. Set environment variables
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET=your-bucket-name
export ECR_IMAGE_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/boltzgen-sagemaker:latest
export IAM_ROLE=arn:aws:iam::${AWS_ACCOUNT_ID}:role/BoltzGenSageMakerRole

# 2. Automated setup and test run
./sagemaker/setup_and_test.sh

# 3. Download results (after job completion)
aws s3 sync s3://$S3_BUCKET/boltzgen/output/ ./results
```

## Individual Step Execution

### Step 1: Build and Push Docker Image to ECR

```bash
./sagemaker/build_and_push.sh
```

### Step 2: Run Processing Job

```bash
# Set IAM_ROLE first
export IAM_ROLE=arn:aws:iam::${AWS_ACCOUNT_ID}:role/BoltzGenSageMakerRole

python sagemaker/run_processing_job.py \
  --design-spec example/hard_targets/1g13prot.yaml \
  --s3-bucket $S3_BUCKET \
  --instance-type ml.g4dn.xlarge \
  --region $AWS_REGION \
  --role $IAM_ROLE \
  --num-designs 10 \
  --budget 2
```

### Step 3: Monitor Job

```bash
# Check job status
aws sagemaker list-processing-jobs --max-results 1

# View logs
aws logs tail /aws/sagemaker/ProcessingJobs --follow
```

## Key Files

| File | Description |
|------|-------------|
| `Dockerfile.sagemaker` | Docker image for SageMaker |
| `build_and_push.sh` | ECR build/push script |
| `run_processing_job.py` | Processing job execution script |
| `processing_script.py` | Script executed in container |
| `setup_and_test.sh` | Complete automation script |

## Cost Examples

| Designs | Instance | Est. Time | Est. Cost |
|---------|----------|-----------|-----------|
| 10 | ml.g4dn.xlarge | ~10 min | ~$0.12 |
| 100 | ml.g4dn.xlarge | ~1 hour | ~$0.74 |
| 1000 | ml.g4dn.xlarge | ~3 hours | ~$2.22 |
| 10000 | ml.g5.xlarge | ~5 hours | ~$7.05 |

## Troubleshooting

**"No module named 'sagemaker'"**
```bash
pip install boto3 sagemaker
```

**"AWS credentials not configured"**
```bash
aws configure
```

**"S3 bucket does not exist"**
```bash
aws s3 mb s3://your-bucket-name --region us-east-1
```

**"Could not assume role" error**
```bash
# This error occurs when using SSO roles or incorrect IAM roles.
# Solution: Always specify the BoltzGenSageMakerRole explicitly:
export IAM_ROLE=arn:aws:iam::${AWS_ACCOUNT_ID}:role/BoltzGenSageMakerRole

# Then use --role parameter in your command:
python sagemaker/run_processing_job.py \
  --design-spec example/hard_targets/1g13prot.yaml \
  --s3-bucket $S3_BUCKET \
  --instance-type ml.g4dn.xlarge \
  --region $AWS_REGION \
  --role $IAM_ROLE \
  --num-designs 10 \
  --budget 2
```

## More Information

For detailed information, see [README.md](./README.md).
