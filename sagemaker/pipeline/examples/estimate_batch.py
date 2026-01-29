#!/usr/bin/env python3
"""
Batch Execution Estimator

Estimates time and cost before running large-scale batch jobs.

Usage:
    python estimate_batch.py --samples 1000
    python estimate_batch.py --samples 1000 --instance-type ml.g5.12xlarge --instances 10
"""

import argparse
import math


# Instance specifications
INSTANCE_INFO = {
    "ml.g5.xlarge": {"gpus": 1, "cost": 1.41, "memory": 16},
    "ml.g5.2xlarge": {"gpus": 1, "cost": 1.69, "memory": 32},
    "ml.g5.4xlarge": {"gpus": 1, "cost": 2.27, "memory": 64},
    "ml.g5.8xlarge": {"gpus": 1, "cost": 3.40, "memory": 128},
    "ml.g5.12xlarge": {"gpus": 4, "cost": 7.09, "memory": 192},
    "ml.g5.16xlarge": {"gpus": 1, "cost": 5.44, "memory": 256},
    "ml.g5.24xlarge": {"gpus": 4, "cost": 10.18, "memory": 384},
    "ml.g5.48xlarge": {"gpus": 8, "cost": 20.36, "memory": 768},
    "ml.g4dn.xlarge": {"gpus": 1, "cost": 0.74, "memory": 16},
    "ml.g4dn.12xlarge": {"gpus": 4, "cost": 5.67, "memory": 192},
}


def estimate(
    num_samples: int,
    instance_type: str,
    num_instances: int,
    time_per_sample: float = 1.5,
):
    """Estimate batch execution time and cost."""
    info = INSTANCE_INFO.get(instance_type)
    if not info:
        print(f"Unknown instance type: {instance_type}")
        return None

    gpus = info["gpus"]
    cost_per_hour = info["cost"]

    # Parallel capacity
    parallel_capacity = num_instances * gpus

    # Number of rounds
    rounds = math.ceil(num_samples / parallel_capacity)

    # Total time
    total_hours = rounds * time_per_sample

    # Instance-hours
    instance_hours = total_hours * num_instances

    # Total cost
    total_cost = instance_hours * cost_per_hour

    return {
        "instance_type": instance_type,
        "num_instances": num_instances,
        "gpus_per_instance": gpus,
        "parallel_capacity": parallel_capacity,
        "rounds": rounds,
        "total_hours": total_hours,
        "total_days": total_hours / 24,
        "instance_hours": instance_hours,
        "cost_per_hour": cost_per_hour,
        "total_cost": total_cost,
    }


def find_optimal_config(num_samples: int, target_hours: float = 24, budget: float = 5000):
    """Find optimal configuration within target time and budget."""
    configs = []

    for instance_type, info in INSTANCE_INFO.items():
        for num_instances in [1, 2, 5, 10, 20, 50]:
            result = estimate(num_samples, instance_type, num_instances)
            if result and result["total_hours"] <= target_hours and result["total_cost"] <= budget:
                configs.append(result)

    # Sort by cost
    configs.sort(key=lambda x: x["total_cost"])

    return configs[:5]


def main():
    parser = argparse.ArgumentParser(description="Batch Execution Estimator")
    parser.add_argument("--samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--instance-type", default=None, help="Instance type")
    parser.add_argument("--instances", type=int, default=None, help="Number of instances")
    parser.add_argument("--time-per-sample", type=float, default=1.5, help="Hours per sample")
    parser.add_argument("--target-hours", type=float, default=48, help="Target completion time")
    parser.add_argument("--budget", type=float, default=10000, help="Max budget ($)")

    args = parser.parse_args()

    print("=" * 70)
    print(f"Batch Execution Estimator")
    print("=" * 70)
    print(f"Samples: {args.samples}")
    print(f"Time per sample: {args.time_per_sample} hours")
    print()

    if args.instance_type and args.instances:
        # Estimate with specific configuration
        result = estimate(
            args.samples,
            args.instance_type,
            args.instances,
            args.time_per_sample,
        )

        if result:
            print("Estimation:")
            print(f"  Instance: {result['num_instances']} x {result['instance_type']}")
            print(f"  GPUs per instance: {result['gpus_per_instance']}")
            print(f"  Parallel capacity: {result['parallel_capacity']} samples")
            print(f"  Rounds needed: {result['rounds']}")
            print(f"  Total time: {result['total_hours']:.1f} hours ({result['total_days']:.2f} days)")
            print(f"  Instance-hours: {result['instance_hours']:.1f}")
            print(f"  Estimated cost: ${result['total_cost']:.2f}")
    else:
        # Find optimal configuration
        print(f"Finding optimal configs (target: {args.target_hours}h, budget: ${args.budget})...")
        print()

        configs = find_optimal_config(args.samples, args.target_hours, args.budget)

        if configs:
            print("Top 5 Cost-Effective Configurations:")
            print("-" * 70)
            print(f"{'Config':<30} {'Parallel':<10} {'Time':<12} {'Cost':<10}")
            print("-" * 70)

            for i, c in enumerate(configs, 1):
                config_str = f"{c['num_instances']}x {c['instance_type']}"
                time_str = f"{c['total_hours']:.1f}h ({c['total_days']:.1f}d)"
                print(f"{config_str:<30} {c['parallel_capacity']:<10} {time_str:<12} ${c['total_cost']:.0f}")
        else:
            print("No configuration found within constraints.")
            print("Try increasing budget or target hours.")

    print()
    print("=" * 70)

    # Show all options
    print("\nAll instance options:")
    print("-" * 70)
    print(f"{'Instance Type':<20} {'GPUs':<6} {'$/hr':<8} {'1000 samples (10 inst)':<20}")
    print("-" * 70)

    for inst_type, info in sorted(INSTANCE_INFO.items(), key=lambda x: x[1]["cost"]):
        result = estimate(1000, inst_type, 10, args.time_per_sample)
        if result:
            time_str = f"{result['total_hours']:.0f}h / ${result['total_cost']:.0f}"
            print(f"{inst_type:<20} {info['gpus']:<6} ${info['cost']:<7.2f} {time_str:<20}")


if __name__ == "__main__":
    main()
