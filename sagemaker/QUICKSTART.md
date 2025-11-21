# BoltzGen SageMaker Processing - Quick Start

## 1-Minute Summary

A quick guide to run BoltzGen using AWS SageMaker Processing.

## Complete Workflow (3 Steps)

```bash
# 1. Set environment variables
export S3_BUCKET=your-bucket-name
export AWS_REGION=us-east-1
export IAM_ROLE=arn:aws:iam::123456789012:role/YourSageMakerRole  # Optional

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
python sagemaker/run_processing_job.py \
  --design-spec example/hard_targets/1g13prot.yaml \
  --s3-bucket your-bucket-name \
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

## More Information

For detailed information, see [README.md](./README.md).
