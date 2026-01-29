#!/usr/bin/env python3
"""
BoltzGen SageMaker Pipeline Definition

This module defines the SageMaker Pipeline for protein design with the following steps:
1. Design - Generate protein structures using diffusion model (GPU)
2. Inverse Folding - Design amino acid sequences (GPU)
3. Folding - Validate structures by re-prediction (GPU)
4. Analysis - Compute quality metrics (CPU)
5. Filtering - Rank and select final designs (CPU)

The pipeline supports horizontal scaling through instance_count parameter
and cost optimization through Spot instances.
"""

import json
from pathlib import Path
from typing import Optional, List

import boto3
from sagemaker import Session
from sagemaker.processing import (
    ScriptProcessor,
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.pipeline_context import PipelineSession

from config import PipelineConfig, PROTOCOL_CONFIGS, get_scaling_preset_for_designs


# Script paths
SCRIPTS_DIR = Path(__file__).parent / "scripts"


class BoltzGenPipeline:
    """SageMaker Pipeline for BoltzGen protein design."""

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.session = PipelineSession()

        # Define pipeline parameters
        self._define_parameters()

    def _define_parameters(self):
        """Define pipeline parameters that can be overridden at runtime."""
        self.param_num_designs = ParameterInteger(
            name="NumDesigns",
            default_value=self.config.num_designs,
        )
        self.param_budget = ParameterInteger(
            name="Budget",
            default_value=self.config.budget,
        )
        self.param_protocol = ParameterString(
            name="Protocol",
            default_value=self.config.protocol,
        )
        self.param_input_s3_uri = ParameterString(
            name="InputS3Uri",
            default_value=self.config.input_s3_uri,
        )
        self.param_output_s3_uri = ParameterString(
            name="OutputS3Uri",
            default_value=self.config.output_s3_uri,
        )
        # Scaling parameter
        self.param_design_instance_count = ParameterInteger(
            name="DesignInstanceCount",
            default_value=self.config.get_instance_config("design").instance_count,
        )
        self.param_folding_instance_count = ParameterInteger(
            name="FoldingInstanceCount",
            default_value=self.config.get_instance_config("folding").instance_count,
        )

    def _create_processor(
        self,
        step_name: str,
        instance_count_param: Optional[ParameterInteger] = None,
    ) -> ScriptProcessor:
        """Create a ScriptProcessor for a pipeline step.

        Args:
            step_name: Name of the step (design, inverse_folding, etc.)
            instance_count_param: Optional parameter for instance count override

        Returns:
            Configured ScriptProcessor
        """
        inst_config = self.config.get_instance_config(step_name)

        processor_kwargs = {
            "role": self.config.role_arn,
            "image_uri": self.config.image_uri,
            "instance_type": inst_config.instance_type,
            "instance_count": (
                instance_count_param
                if instance_count_param
                else inst_config.instance_count
            ),
            "volume_size_in_gb": inst_config.volume_size_gb,
            "max_runtime_in_seconds": inst_config.max_runtime_seconds,
            "base_job_name": f"boltzgen-{step_name}",
            "command": ["python3"],
            "sagemaker_session": self.session,
        }

        # Add Spot instance configuration if enabled
        if inst_config.use_spot:
            processor_kwargs["env"] = {
                "AWS_DEFAULT_REGION": self.config.region,
            }

        return ScriptProcessor(**processor_kwargs)

    def _create_design_step(self) -> ProcessingStep:
        """Create the Design processing step."""
        processor = self._create_processor(
            "design",
            instance_count_param=self.param_design_instance_count,
        )

        step = ProcessingStep(
            name="Design",
            processor=processor,
            code=str(SCRIPTS_DIR / "design_step.py"),
            inputs=[
                ProcessingInput(
                    source=self.param_input_s3_uri,
                    destination="/opt/ml/processing/input/design_specs",
                    input_name="design_specs",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output/designs",
                    destination=f"{self.config.intermediate_s3_uri}/designs",
                    output_name="designs",
                ),
                ProcessingOutput(
                    source="/opt/ml/processing/output/metadata",
                    destination=f"{self.config.intermediate_s3_uri}/metadata/design",
                    output_name="metadata",
                ),
            ],
            job_arguments=[
                "--num-designs", self.param_num_designs.to_string(),
                "--protocol", self.param_protocol,
                "--budget", self.param_budget.to_string(),
                "--instance-count", self.param_design_instance_count.to_string(),
                "--devices", "1",
            ],
            cache_config=CacheConfig(
                enable_caching=True,
                expire_after="7d",
            ),
        )

        return step

    def _create_inverse_folding_step(self, design_step: ProcessingStep) -> ProcessingStep:
        """Create the Inverse Folding processing step."""
        processor = self._create_processor("inverse_folding")

        step = ProcessingStep(
            name="InverseFolding",
            processor=processor,
            code=str(SCRIPTS_DIR / "inverse_folding_step.py"),
            inputs=[
                ProcessingInput(
                    source=design_step.properties.ProcessingOutputConfig.Outputs[
                        "designs"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/designs",
                    input_name="designs",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output/inverse_folded",
                    destination=f"{self.config.intermediate_s3_uri}/inverse_folded",
                    output_name="inverse_folded",
                ),
                ProcessingOutput(
                    source="/opt/ml/processing/output/metadata",
                    destination=f"{self.config.intermediate_s3_uri}/metadata/inverse_folding",
                    output_name="metadata",
                ),
            ],
            job_arguments=[
                "--protocol", self.param_protocol,
                "--num-sequences", "1",
                "--devices", "1",
            ],
            cache_config=CacheConfig(
                enable_caching=True,
                expire_after="7d",
            ),
        )

        return step

    def _create_folding_step(self, inverse_folding_step: ProcessingStep) -> ProcessingStep:
        """Create the Folding processing step."""
        processor = self._create_processor(
            "folding",
            instance_count_param=self.param_folding_instance_count,
        )

        step = ProcessingStep(
            name="Folding",
            processor=processor,
            code=str(SCRIPTS_DIR / "folding_step.py"),
            inputs=[
                ProcessingInput(
                    source=inverse_folding_step.properties.ProcessingOutputConfig.Outputs[
                        "inverse_folded"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/inverse_folded",
                    input_name="inverse_folded",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output/folded",
                    destination=f"{self.config.intermediate_s3_uri}/folded",
                    output_name="folded",
                ),
                ProcessingOutput(
                    source="/opt/ml/processing/output/metadata",
                    destination=f"{self.config.intermediate_s3_uri}/metadata/folding",
                    output_name="metadata",
                ),
            ],
            job_arguments=[
                "--protocol", self.param_protocol,
                "--instance-count", self.param_folding_instance_count.to_string(),
                "--devices", "1",
            ],
            cache_config=CacheConfig(
                enable_caching=True,
                expire_after="7d",
            ),
        )

        return step

    def _create_analysis_step(self, folding_step: ProcessingStep) -> ProcessingStep:
        """Create the Analysis processing step (CPU)."""
        processor = self._create_processor("analysis")

        step = ProcessingStep(
            name="Analysis",
            processor=processor,
            code=str(SCRIPTS_DIR / "analysis_step.py"),
            inputs=[
                ProcessingInput(
                    source=folding_step.properties.ProcessingOutputConfig.Outputs[
                        "folded"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/folded",
                    input_name="folded",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output/analyzed",
                    destination=f"{self.config.intermediate_s3_uri}/analyzed",
                    output_name="analyzed",
                ),
                ProcessingOutput(
                    source="/opt/ml/processing/output/metadata",
                    destination=f"{self.config.intermediate_s3_uri}/metadata/analysis",
                    output_name="metadata",
                ),
            ],
            job_arguments=[
                "--protocol", self.param_protocol,
            ],
            cache_config=CacheConfig(
                enable_caching=True,
                expire_after="7d",
            ),
        )

        return step

    def _create_filtering_step(self, analysis_step: ProcessingStep) -> ProcessingStep:
        """Create the Filtering processing step (CPU)."""
        processor = self._create_processor("filtering")

        step = ProcessingStep(
            name="Filtering",
            processor=processor,
            code=str(SCRIPTS_DIR / "filtering_step.py"),
            inputs=[
                ProcessingInput(
                    source=analysis_step.properties.ProcessingOutputConfig.Outputs[
                        "analyzed"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/analyzed",
                    input_name="analyzed",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output/final",
                    destination=self.param_output_s3_uri,
                    output_name="final",
                ),
                ProcessingOutput(
                    source="/opt/ml/processing/output/metadata",
                    destination=f"{self.config.intermediate_s3_uri}/metadata/filtering",
                    output_name="metadata",
                ),
            ],
            job_arguments=[
                "--protocol", self.param_protocol,
                "--budget", self.param_budget.to_string(),
            ],
            cache_config=CacheConfig(
                enable_caching=False,  # Always run filtering to get fresh results
            ),
        )

        return step

    def create_pipeline(self) -> Pipeline:
        """Create the complete SageMaker Pipeline.

        Returns:
            Configured SageMaker Pipeline object
        """
        # Create steps in order
        design_step = self._create_design_step()
        inverse_folding_step = self._create_inverse_folding_step(design_step)
        folding_step = self._create_folding_step(inverse_folding_step)
        analysis_step = self._create_analysis_step(folding_step)
        filtering_step = self._create_filtering_step(analysis_step)

        # Define pipeline
        pipeline = Pipeline(
            name=self.config.pipeline_name,
            parameters=[
                self.param_num_designs,
                self.param_budget,
                self.param_protocol,
                self.param_input_s3_uri,
                self.param_output_s3_uri,
                self.param_design_instance_count,
                self.param_folding_instance_count,
            ],
            steps=[
                design_step,
                inverse_folding_step,
                folding_step,
                analysis_step,
                filtering_step,
            ],
            sagemaker_session=self.session,
        )

        return pipeline

    def upsert_pipeline(self) -> dict:
        """Create or update the pipeline in SageMaker.

        Returns:
            Pipeline ARN and other metadata
        """
        pipeline = self.create_pipeline()

        response = pipeline.upsert(
            role_arn=self.config.role_arn,
            description=self.config.pipeline_description,
        )

        return {
            "pipeline_arn": response["PipelineArn"],
            "pipeline_name": self.config.pipeline_name,
        }

    def start_execution(
        self,
        num_designs: Optional[int] = None,
        budget: Optional[int] = None,
        protocol: Optional[str] = None,
        input_s3_uri: Optional[str] = None,
        output_s3_uri: Optional[str] = None,
        design_instance_count: Optional[int] = None,
        folding_instance_count: Optional[int] = None,
        execution_name: Optional[str] = None,
    ) -> dict:
        """Start a pipeline execution.

        Args:
            num_designs: Override number of designs
            budget: Override final budget
            protocol: Override design protocol
            input_s3_uri: Override input S3 URI
            output_s3_uri: Override output S3 URI
            design_instance_count: Override design step instance count
            folding_instance_count: Override folding step instance count
            execution_name: Custom execution name

        Returns:
            Execution ARN and other metadata
        """
        pipeline = self.create_pipeline()

        # Build execution parameters
        execution_params = {}

        if num_designs is not None:
            execution_params["NumDesigns"] = num_designs
            # Auto-scale if not explicitly set
            if design_instance_count is None:
                preset = get_scaling_preset_for_designs(num_designs)
                from config import SCALING_PRESETS
                execution_params["DesignInstanceCount"] = SCALING_PRESETS[preset]["design"]["instance_count"]
                execution_params["FoldingInstanceCount"] = SCALING_PRESETS[preset]["folding"]["instance_count"]

        if budget is not None:
            execution_params["Budget"] = budget

        if protocol is not None:
            execution_params["Protocol"] = protocol

        if input_s3_uri is not None:
            execution_params["InputS3Uri"] = input_s3_uri

        if output_s3_uri is not None:
            execution_params["OutputS3Uri"] = output_s3_uri

        if design_instance_count is not None:
            execution_params["DesignInstanceCount"] = design_instance_count

        if folding_instance_count is not None:
            execution_params["FoldingInstanceCount"] = folding_instance_count

        # Start execution
        execution = pipeline.start(
            parameters=execution_params,
            execution_display_name=execution_name,
        )

        return {
            "execution_arn": execution.arn,
            "execution_name": execution_name,
            "parameters": execution_params,
        }


def create_default_pipeline(
    s3_bucket: str,
    role_arn: str,
    region: str = "us-east-1",
    **kwargs,
) -> BoltzGenPipeline:
    """Create a pipeline with default configuration.

    Args:
        s3_bucket: S3 bucket for pipeline data
        role_arn: IAM role ARN for SageMaker
        region: AWS region
        **kwargs: Additional configuration overrides

    Returns:
        Configured BoltzGenPipeline object
    """
    config = PipelineConfig(
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        region=region,
        **kwargs,
    )

    return BoltzGenPipeline(config)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="BoltzGen SageMaker Pipeline")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket for pipeline data")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--create", action="store_true", help="Create/update pipeline")
    parser.add_argument("--start", action="store_true", help="Start pipeline execution")
    parser.add_argument("--num-designs", type=int, default=10000, help="Number of designs")
    parser.add_argument("--budget", type=int, default=100, help="Final budget")

    args = parser.parse_args()

    pipeline = create_default_pipeline(
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
        region=args.region,
        num_designs=args.num_designs,
        budget=args.budget,
    )

    if args.create:
        result = pipeline.upsert_pipeline()
        print(f"Pipeline created/updated: {result['pipeline_arn']}")

    if args.start:
        result = pipeline.start_execution(
            num_designs=args.num_designs,
            budget=args.budget,
        )
        print(f"Execution started: {result['execution_arn']}")
