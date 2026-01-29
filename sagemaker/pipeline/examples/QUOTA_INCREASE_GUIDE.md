# AWS SageMaker Quota Increase Guide

## Check Current Quota

```bash
aws service-quotas list-service-quotas \
  --service-code sagemaker \
  --region us-east-1 \
  --query 'Quotas[?contains(QuotaName, `g5`) && contains(QuotaName, `processing`)].{Name: QuotaName, Value: Value}' \
  --output table
```

## How to Request Quota Increase

### Method 1: AWS Console
1. Go to [AWS Service Quotas Console](https://console.aws.amazon.com/servicequotas/home)
2. Select "Amazon SageMaker" from the service list
3. Search for "processing job usage"
4. Select the desired instance type (e.g., `ml.g5.12xlarge for processing job usage`)
5. Click "Request quota increase"
6. Enter the desired quantity (e.g., 10 or 20)

### Method 2: AWS CLI

```bash
# Request ml.g5.12xlarge quota increase (4 GPU instance)
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-B013C051 \
  --desired-value 10 \
  --region us-east-1

# Request ml.g5.xlarge quota increase (1 GPU instance)
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-BE792A7A \
  --desired-value 50 \
  --region us-east-1

# Request ml.g5.48xlarge quota increase (8 GPU instance)
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-339123BF \
  --desired-value 10 \
  --region us-east-1
```

## Quota Codes by Instance Type

| Instance Type | GPUs | Quota Code | Cost/hour |
|--------------|------|------------|-----------|
| ml.g5.xlarge | 1 | L-BE792A7A | $1.41 |
| ml.g5.2xlarge | 1 | L-23412DF7 | $1.69 |
| ml.g5.12xlarge | 4 | L-B013C051 | $7.09 |
| ml.g5.24xlarge | 4 | L-BC6B3288 | $10.18 |
| ml.g5.48xlarge | 8 | L-339123BF | $20.36 |

## Recommended Quota Settings

### For 1000 Samples

| Target Completion | Instance Type | Quantity | Est. Cost |
|-------------------|--------------|----------|-----------|
| 1 day | ml.g5.12xlarge | 20 | ~$2,800 |
| 2 days | ml.g5.12xlarge | 10 | ~$2,700 |
| 3 days | ml.g5.xlarge | 20 | ~$2,000 |
| 1 week | ml.g5.xlarge | 5 | ~$1,700 |

## Quota Request Tips

1. **Justification**: "Running large-scale protein design batch jobs for research"
2. **Processing time**: Usually takes 24-48 hours. For urgent requests, create an AWS Support ticket
3. **Initial request**: Start with 10, request more if needed
4. **Region selection**: Some regions may have limited capacity. us-east-1, us-west-2 recommended

## Check Quota Request Status

```bash
aws service-quotas list-requested-service-quota-change-history \
  --service-code sagemaker \
  --region us-east-1 \
  --query 'RequestedQuotas[*].{QuotaName: QuotaName, Status: Status, DesiredValue: DesiredValue}' \
  --output table
```
