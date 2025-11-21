#!/usr/bin/env python3
"""
SageMaker Processing Script for BoltzGen
This script runs inside the SageMaker Processing container
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run BoltzGen in SageMaker Processing')
    parser.add_argument('--design-spec', type=str, required=True,
                        help='Path to design specification YAML file')
    parser.add_argument('--protocol', type=str, default='protein-anything',
                        choices=['protein-anything', 'peptide-anything',
                                'protein-small_molecule', 'nanobody-anything'],
                        help='Protocol to use for the design')
    parser.add_argument('--num-designs', type=int, default=10,
                        help='Number of designs to generate')
    parser.add_argument('--budget', type=int, default=2,
                        help='Number of designs in final diversity optimized set')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPU devices to use')

    args = parser.parse_args()

    # SageMaker Processing paths
    input_path = Path('/opt/ml/processing/input')
    output_path = Path('/opt/ml/processing/output')
    cache_path = Path('/opt/ml/processing/cache')

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Cache path: {cache_path}")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Find design spec file
    design_spec_path = input_path / args.design_spec
    if not design_spec_path.exists():
        # Try to find any .yaml file in input
        yaml_files = list(input_path.rglob('*.yaml'))
        if yaml_files:
            design_spec_path = yaml_files[0]
            logger.info(f"Using found design spec: {design_spec_path}")
        else:
            logger.error(f"Design spec not found: {design_spec_path}")
            sys.exit(1)

    logger.info(f"Design specification: {design_spec_path}")

    # Build boltzgen command
    cmd_parts = [
        'boltzgen', 'run',
        str(design_spec_path),
        '--output', str(output_path / 'results'),
        '--protocol', args.protocol,
        '--num_designs', str(args.num_designs),
        '--budget', str(args.budget),
        '--cache', str(cache_path),
        '--devices', str(args.devices)
    ]

    cmd = ' '.join(cmd_parts)
    logger.info(f"Running command: {cmd}")

    # Execute boltzgen
    import subprocess
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        logger.info("BoltzGen output:")
        logger.info(result.stdout)

        # Save job metadata
        metadata = {
            'design_spec': str(design_spec_path),
            'protocol': args.protocol,
            'num_designs': args.num_designs,
            'budget': args.budget,
            'status': 'completed'
        }

        with open(output_path / 'job_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Processing job completed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"BoltzGen failed with error: {e}")
        logger.error(f"Output: {e.stdout}")

        # Save error metadata
        metadata = {
            'design_spec': str(design_spec_path),
            'protocol': args.protocol,
            'num_designs': args.num_designs,
            'budget': args.budget,
            'status': 'failed',
            'error': str(e)
        }

        with open(output_path / 'job_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        sys.exit(1)


if __name__ == '__main__':
    main()
