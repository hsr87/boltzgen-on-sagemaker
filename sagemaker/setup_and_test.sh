#!/bin/bash
# Complete setup and test script for BoltzGen on SageMaker Processing

set -e

echo "=========================================="
echo "BoltzGen SageMaker Setup and Test"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - EDIT THESE
S3_BUCKET="${S3_BUCKET:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
IAM_ROLE="${IAM_ROLE:-}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Step 0: Check prerequisites
echo ""
echo "Step 0: Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI not found. Please install it first."
    exit 1
fi
print_status "AWS CLI installed"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install it first."
    exit 1
fi
print_status "Docker installed"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured. Run 'aws configure'"
    exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
print_status "AWS credentials configured (Account: $ACCOUNT_ID)"

# Check Python and required packages
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found"
    exit 1
fi
print_status "Python3 installed"

# Check S3 bucket
if [ -z "$S3_BUCKET" ]; then
    print_error "S3_BUCKET not set. Please export S3_BUCKET=your-bucket-name"
    echo "Example: export S3_BUCKET=my-boltzgen-bucket"
    exit 1
fi

# Check if bucket exists
if aws s3 ls "s3://$S3_BUCKET" 2>&1 | grep -q 'NoSuchBucket'; then
    print_warning "S3 bucket $S3_BUCKET does not exist. Creating..."
    aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION"
    print_status "Created S3 bucket: $S3_BUCKET"
else
    print_status "S3 bucket exists: $S3_BUCKET"
fi

echo ""
echo "=========================================="
echo "Step 1: Installing Python dependencies..."
echo "=========================================="

cd "$(dirname "$0")"
pip install -q -r requirements.txt
print_status "Python dependencies installed"

echo ""
echo "=========================================="
echo "Step 2: Building and pushing Docker image..."
echo "=========================================="

cd ..
AWS_REGION="$AWS_REGION" ./sagemaker/build_and_push.sh

IMAGE_URI=$(cat sagemaker/image_uri.txt)
print_status "Docker image ready: $IMAGE_URI"

echo ""
echo "=========================================="
echo "Step 3: Preparing test design spec..."
echo "=========================================="

# Check if example files exist
if [ ! -f "example/7rpz.cif" ]; then
    print_error "Example files not found. Are you in the boltzgen directory?"
    exit 1
fi

# Use a simple example
TEST_DESIGN_SPEC="example/vanilla_protein/1g13prot.yaml"
if [ ! -f "$TEST_DESIGN_SPEC" ]; then
    # Try alternative
    TEST_DESIGN_SPEC="example/hard_targets/1g13prot.yaml"
    if [ ! -f "$TEST_DESIGN_SPEC" ]; then
        print_error "Could not find test design spec"
        exit 1
    fi
fi

print_status "Using design spec: $TEST_DESIGN_SPEC"

echo ""
echo "=========================================="
echo "Step 4: Launching SageMaker Processing Job..."
echo "=========================================="

# Build the command
CMD="python sagemaker/run_processing_job.py \
  --design-spec $TEST_DESIGN_SPEC \
  --s3-bucket $S3_BUCKET \
  --instance-type ml.g4dn.xlarge \
  --num-designs 2 \
  --budget 1 \
  --region $AWS_REGION"

# Add role if specified
if [ -n "$IAM_ROLE" ]; then
    CMD="$CMD --role $IAM_ROLE"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Execute the command
eval $CMD

print_status "Processing job launched!"

echo ""
echo "=========================================="
echo "Setup and Test Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Monitor your job in the SageMaker console"
echo "2. Check CloudWatch logs for progress"
echo "3. Download results when complete:"
echo "   aws s3 sync s3://$S3_BUCKET/boltzgen/output/ ./results"
echo ""
echo "For more information, see sagemaker/README.md"
echo "=========================================="
