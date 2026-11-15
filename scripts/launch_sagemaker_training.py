#!/usr/bin/env python3
"""
scripts/launch_sagemaker_training.py
====================================
Submits and monitors an MPM Training Job on AWS SageMaker.

Usage:
  python3 scripts/launch_sagemaker_training.py \
      --instance-type ml.m5.xlarge \
      --rounds 1 \
      --trees-per-round 50 \
      --bags-per-round 3 \
      --dae-epochs 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sagemaker_launcher")

# Default AWS project parameters
DEFAULT_REGION = "ap-southeast-2"
DEFAULT_ROLE_ARN = "arn:aws:iam::355103369883:role/service-role/AmazonSageMaker-ExecutionRole-20261027T202509"
DEFAULT_BUCKET = "sagemaker-ap-southeast-2-355103369883"
DEFAULT_IMAGE = "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker-v1.7"
GPU_IMAGE = "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker-v1.26"


def parse_args():
    parser = argparse.ArgumentParser(description="Submit SageMaker Training Job")
    parser.add_argument("--job-name", type=str, default=None, help="Custom job name")
    parser.add_argument("--instance-type", type=str, default="ml.m5.xlarge", help="SageMaker instance type (e.g. ml.m5.xlarge, ml.g4dn.xlarge)")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--volume-size-gb", type=int, default=50, help="EBS volume size in GB")
    parser.add_argument("--rounds", type=int, default=1, help="Incremental rounds to execute")
    parser.add_argument("--trees-per-round", type=int, default=50, help="Trees added per round")
    parser.add_argument("--bags-per-round", type=int, default=3, help="Bags per round")
    parser.add_argument("--dae-epochs", type=int, default=2, help="DAE UNet fine-tuning epochs per round")
    parser.add_argument("--region-target", type=str, default="EAST_KALIMANTAN", help="Target exploration region")
    parser.add_argument("--max-runtime-hours", type=int, default=4, help="Max job runtime in hours")
    parser.add_argument("--wait", action="store_true", help="Wait and stream status until job completes")
    return parser.parse_args()


def run_aws_cli(cmd: list[str]) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("AWS_PROFILE", "aws-project")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def launch_training_job(args: argparse.Namespace) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"mpm-train-{args.region_target.lower().replace('_', '-')}-{timestamp}"

    # Pick image based on instance type
    if "g4dn" in args.instance_type or "p3" in args.instance_type:
        image = GPU_IMAGE
    else:
        image = DEFAULT_IMAGE

    s3_code = f"s3://{DEFAULT_BUCKET}/geospatials/code/sourcedir.tar.gz"
    s3_data = f"s3://{DEFAULT_BUCKET}/geospatials/data/"
    s3_models = f"s3://{DEFAULT_BUCKET}/geospatials/models/latest/"
    s3_output = f"s3://{DEFAULT_BUCKET}/geospatials/outputs/"

    hyperparameters = {
        "sagemaker_program": "scripts/sagemaker_entry.py",
        "sagemaker_submit_directory": s3_code,
        "region": args.region_target,
        "rounds": str(args.rounds),
        "trees-per-round": str(args.trees_per_round),
        "bags-per-round": str(args.bags_per_round),
        "dae-epochs": str(args.dae_epochs),
    }

    input_data_config = [
        {
            "ChannelName": "data",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_data,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "InputMode": "File",
        },
        {
            "ChannelName": "models",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_models,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "InputMode": "File",
        },
    ]

    algorithm_spec = {
        "TrainingImage": image,
        "TrainingInputMode": "File",
    }

    resource_config = {
        "InstanceType": args.instance_type,
        "InstanceCount": args.instance_count,
        "VolumeSizeInGB": args.volume_size_gb,
    }

    stopping_condition = {
        "MaxRuntimeInSeconds": args.max_runtime_hours * 3600,
    }

    output_data_config = {
        "S3OutputPath": s3_output,
    }

    logger.info("Submitting SageMaker Training Job...")
    logger.info("  Job Name:       %s", job_name)
    logger.info("  Instance:       %s (Count: %d)", args.instance_type, args.instance_count)
    logger.info("  Container:      %s", image)
    logger.info("  Code Archive:   %s", s3_code)
    logger.info("  Data Channel:   %s", s3_data)
    logger.info("  Models Channel: %s", s3_models)
    logger.info("  Output S3:      %s", s3_output)

    cmd = [
        "aws", "sagemaker", "create-training-job",
        "--region", DEFAULT_REGION,
        "--training-job-name", job_name,
        "--hyper-parameters", json.dumps(hyperparameters),
        "--algorithm-specification", json.dumps(algorithm_spec),
        "--role-arn", DEFAULT_ROLE_ARN,
        "--input-data-config", json.dumps(input_data_config),
        "--output-data-config", json.dumps(output_data_config),
        "--resource-config", json.dumps(resource_config),
        "--stopping-condition", json.dumps(stopping_condition),
    ]

    ret, stdout, stderr = run_aws_cli(cmd)
    if ret != 0:
        logger.error("Failed to create training job: %s", stderr)
        sys.exit(1)

    logger.info("Successfully initiated SageMaker Training Job: %s", job_name)
    return job_name


def check_status(job_name: str) -> dict:
    cmd = [
        "aws", "sagemaker", "describe-training-job",
        "--region", DEFAULT_REGION,
        "--training-job-name", job_name,
    ]
    ret, stdout, stderr = run_aws_cli(cmd)
    if ret != 0:
        logger.error("Failed to describe training job: %s", stderr)
        return {}
    return json.loads(stdout)


def main():
    args = parse_args()
    job_name = launch_training_job(args)

    if args.wait:
        logger.info("Waiting for training job %s to progress...", job_name)
        prev_status = None
        while True:
            desc = check_status(job_name)
            status = desc.get("TrainingJobStatus")
            secondary_status = desc.get("SecondaryStatus")

            if status != prev_status or secondary_status != prev_status:
                logger.info("Status: %s | Secondary: %s", status, secondary_status)
                prev_status = (status, secondary_status)

            if status in ("Completed", "Failed", "Stopped"):
                logger.info("Training job finished with status: %s", status)
                if status == "Failed":
                    logger.error("Failure reason: %s", desc.get("FailureReason"))
                    sys.exit(1)
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
