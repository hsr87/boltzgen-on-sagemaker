#!/bin/bash
# Build and push BoltzGen Docker image to ECR for SageMaker Processing

set -e

# Configuration
REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOSITORY_NAME="boltzgen-sagemaker"
IMAGE_TAG=${IMAGE_TAG:-latest}

echo "=========================================="
echo "BoltzGen ECR Build and Push Script"
echo "=========================================="
echo "Region: $REGION"
echo "Account ID: $ACCOUNT_ID"
echo "Repository: $REPOSITORY_NAME"
echo "Image Tag: $IMAGE_TAG"
echo "=========================================="

# Step 1: Create ECR repository if it doesn't exist
echo "Step 1: Creating ECR repository..."
aws ecr describe-repositories --repository-names $REPOSITORY_NAME --region $REGION 2>/dev/null || \
    aws ecr create-repository \
        --repository-name $REPOSITORY_NAME \
        --region $REGION \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256

echo "✓ ECR repository ready"

# Step 2: Get ECR login credentials
echo ""
echo "Step 2: Logging in to ECR..."
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

echo "✓ Logged in to ECR"

# Step 3: Build Docker image
echo ""
echo "Step 3: Building Docker image..."
FULL_IMAGE_NAME="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG"

# Navigate to project root (parent of sagemaker directory)
cd "$(dirname "$0")/.."

docker build \
    -f Dockerfile.sagemaker \
    -t $REPOSITORY_NAME:$IMAGE_TAG \
    -t $FULL_IMAGE_NAME \
    .

echo "✓ Docker image built"

# Step 4: Push to ECR
echo ""
echo "Step 4: Pushing image to ECR..."
docker push $FULL_IMAGE_NAME

echo "✓ Image pushed to ECR"

# Step 5: Display summary
echo ""
echo "=========================================="
echo "✓ Build and Push Complete!"
echo "=========================================="
echo "Image URI: $FULL_IMAGE_NAME"
echo ""
echo "You can now use this image URI in your SageMaker Processing jobs."
echo ""
echo "Next steps:"
echo "1. Update run_processing_job.py with this image URI"
echo "2. Run: python sagemaker/run_processing_job.py"
echo "=========================================="

# Save image URI to file for later use
echo $FULL_IMAGE_NAME > sagemaker/image_uri.txt
echo "Image URI saved to sagemaker/image_uri.txt"
