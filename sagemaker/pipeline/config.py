"""
BoltzGen SageMaker Pipeline Configuration

This module contains all configuration parameters for the protein design pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import boto3
from sagemaker import get_execution_role


@dataclass
class InstanceConfig:
    """Instance configuration for each pipeline step."""
    instance_type: str
    instance_count: int = 1
    volume_size_gb: int = 50
    max_runtime_seconds: int = 86400  # 24 hours
    use_spot: bool = True
    max_wait_seconds: int = 86400  # For spot instances


# Default instance configurations for each step
INSTANCE_CONFIGS = {
    "design": InstanceConfig(
        instance_type="ml.g5.xlarge",
        instance_count=1,
        volume_size_gb=100,
        use_spot=True,
    ),
    "inverse_folding": InstanceConfig(
        instance_type="ml.g5.xlarge",
        instance_count=1,
        volume_size_gb=50,
        use_spot=True,
    ),
    "folding": InstanceConfig(
        instance_type="ml.g5.xlarge",
        instance_count=1,
        volume_size_gb=50,
        use_spot=True,
    ),
    "analysis": InstanceConfig(
        instance_type="ml.m5.xlarge",
        instance_count=1,
        volume_size_gb=50,
        use_spot=True,
    ),
    "filtering": InstanceConfig(
        instance_type="ml.m5.large",
        instance_count=1,
        volume_size_gb=30,
        use_spot=False,  # Fast step, no need for spot
    ),
}


# Scaling presets based on number of designs
SCALING_PRESETS = {
    "small": {  # Up to 10K designs
        "design": {"instance_count": 1},
        "inverse_folding": {"instance_count": 1},
        "folding": {"instance_count": 1},
    },
    "medium": {  # 10K - 100K designs
        "design": {"instance_count": 5},
        "inverse_folding": {"instance_count": 5},
        "folding": {"instance_count": 5},
    },
    "large": {  # 100K - 1M designs
        "design": {"instance_count": 10},
        "inverse_folding": {"instance_count": 10},
        "folding": {"instance_count": 10},
    },
    "xlarge": {  # 1M+ designs
        "design": {"instance_count": 50},
        "inverse_folding": {"instance_count": 25},
        "folding": {"instance_count": 25},
    },
}


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Pipeline identification
    pipeline_name: str = "BoltzGen-Protein-Design"
    pipeline_description: str = "AI-powered protein binder design pipeline"

    # S3 configuration
    s3_bucket: str = ""
    s3_prefix: str = "boltzgen-pipeline"

    # Design parameters
    num_designs: int = 10000
    budget: int = 100
    protocol: str = "protein-anything"

    # Scaling
    scaling_preset: str = "small"

    # Docker image
    image_uri: str = ""

    # IAM role
    role_arn: str = ""

    # Region
    region: str = "us-east-1"

    # Step-specific overrides
    instance_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived values."""
        if not self.s3_bucket:
            # Try to get default bucket
            sts = boto3.client('sts', region_name=self.region)
            account_id = sts.get_caller_identity()['Account']
            self.s3_bucket = f"sagemaker-{self.region}-{account_id}"

        if not self.role_arn:
            try:
                self.role_arn = get_execution_role()
            except Exception:
                pass  # Will need to be set manually

        if not self.image_uri:
            self._set_default_image_uri()

    def _set_default_image_uri(self):
        """Set default ECR image URI."""
        sts = boto3.client('sts', region_name=self.region)
        account_id = sts.get_caller_identity()['Account']
        self.image_uri = f"{account_id}.dkr.ecr.{self.region}.amazonaws.com/boltzgen-sagemaker:latest"

    def get_instance_config(self, step_name: str) -> InstanceConfig:
        """Get instance configuration for a specific step with scaling applied."""
        base_config = INSTANCE_CONFIGS[step_name]

        # Apply scaling preset
        scaling = SCALING_PRESETS.get(self.scaling_preset, {})
        step_scaling = scaling.get(step_name, {})

        # Apply manual overrides
        overrides = self.instance_overrides.get(step_name, {})

        # Merge configurations
        return InstanceConfig(
            instance_type=overrides.get("instance_type", base_config.instance_type),
            instance_count=overrides.get("instance_count", step_scaling.get("instance_count", base_config.instance_count)),
            volume_size_gb=overrides.get("volume_size_gb", base_config.volume_size_gb),
            max_runtime_seconds=overrides.get("max_runtime_seconds", base_config.max_runtime_seconds),
            use_spot=overrides.get("use_spot", base_config.use_spot),
            max_wait_seconds=overrides.get("max_wait_seconds", base_config.max_wait_seconds),
        )

    @property
    def input_s3_uri(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/input"

    @property
    def output_s3_uri(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/output"

    @property
    def intermediate_s3_uri(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/intermediate"

    @property
    def cache_s3_uri(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/cache"


# Protocol-specific configurations
PROTOCOL_CONFIGS = {
    "protein-anything": {
        "skip_design_folding": False,
        "skip_affinity": True,
        "inverse_fold_avoid": "",
    },
    "peptide-anything": {
        "skip_design_folding": True,
        "skip_affinity": True,
        "inverse_fold_avoid": "C",
    },
    "protein-small_molecule": {
        "skip_design_folding": False,
        "skip_affinity": False,
        "inverse_fold_avoid": "",
    },
    "nanobody-anything": {
        "skip_design_folding": True,
        "skip_affinity": True,
        "inverse_fold_avoid": "C",
    },
}


def get_scaling_preset_for_designs(num_designs: int) -> str:
    """Automatically determine scaling preset based on number of designs."""
    if num_designs <= 10000:
        return "small"
    elif num_designs <= 100000:
        return "medium"
    elif num_designs <= 1000000:
        return "large"
    else:
        return "xlarge"


def estimate_cost(config: PipelineConfig) -> dict:
    """Estimate pipeline execution cost."""
    # Approximate hourly costs (USD) for different instance types
    HOURLY_COSTS = {
        "ml.g5.xlarge": 1.41,
        "ml.g5.2xlarge": 1.69,
        "ml.g4dn.xlarge": 0.74,
        "ml.m5.xlarge": 0.23,
        "ml.m5.large": 0.115,
    }

    SPOT_DISCOUNT = 0.3  # ~70% discount for spot

    # Rough time estimates per 1000 designs (hours)
    TIME_PER_1K = {
        "design": 0.5,
        "inverse_folding": 0.2,
        "folding": 0.3,
        "analysis": 0.1,
        "filtering": 0.01,
    }

    total_cost = 0.0
    cost_breakdown = {}

    num_k_designs = config.num_designs / 1000

    for step_name in ["design", "inverse_folding", "folding", "analysis", "filtering"]:
        inst_config = config.get_instance_config(step_name)
        hourly_cost = HOURLY_COSTS.get(inst_config.instance_type, 1.0)

        if inst_config.use_spot:
            hourly_cost *= SPOT_DISCOUNT

        # Time = base_time * num_designs / instance_count
        step_time = TIME_PER_1K.get(step_name, 0.1) * num_k_designs / inst_config.instance_count
        step_cost = hourly_cost * inst_config.instance_count * step_time

        cost_breakdown[step_name] = round(step_cost, 2)
        total_cost += step_cost

    return {
        "total": round(total_cost, 2),
        "breakdown": cost_breakdown,
        "currency": "USD",
    }
