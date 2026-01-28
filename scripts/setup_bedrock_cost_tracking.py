#!/usr/bin/env python3
"""
AWS Bedrock Cost Tracking Setup

Creates Application Inference Profiles with cost allocation tags for each model.
This enables tracking costs by experiment/model in AWS Cost Explorer.

Usage:
    # List existing inference profiles
    python scripts/setup_bedrock_cost_tracking.py --list

    # Create inference profiles for all models
    python scripts/setup_bedrock_cost_tracking.py --create

    # Create profile for specific model
    python scripts/setup_bedrock_cost_tracking.py --create --model claude-3-haiku

After creation:
1. Go to AWS Billing > Cost Allocation Tags
2. Activate the tags (project, experiment, model)
3. Wait 24 hours for tags to appear in Cost Explorer

References:
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-create.html
- https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/create_inference_profile.html
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("boto3 required: pip install boto3")
    sys.exit(1)


# Model configurations - map friendly names to Bedrock model ARNs
# Using us-east-2 region ARNs
MODELS = {
    "claude-3-haiku": {
        "arn": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
        "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "description": "WattBot-Claude3-Haiku",
    },
    "claude-3-5-haiku": {
        "arn": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0",
        "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "description": "WattBot-Claude3.5-Haiku",
    },
    "claude-3-5-sonnet": {
        "arn": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "description": "WattBot-Claude3.5-Sonnet",
    },
    "claude-3-7-sonnet": {
        "arn": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0",
        "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "description": "WattBot-Claude3.7-Sonnet",
    },
    "nova-pro": {
        "arn": "arn:aws:bedrock:us-east-2::foundation-model/amazon.nova-pro-v1:0",
        "model_id": "us.amazon.nova-pro-v1:0",
        "description": "WattBot-Amazon-NovaPro",
    },
    "llama-3-70b": {
        "arn": "arn:aws:bedrock:us-east-2::foundation-model/meta.llama3-70b-instruct-v1:0",
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "description": "WattBot-Llama3-70B",
    },
    "llama-4-maverick": {
        "arn": "arn:aws:bedrock:us-east-2::foundation-model/meta.llama4-maverick-17b-instruct-v1:0",
        "model_id": "meta.llama4-maverick-17b-instruct-v1:0",
        "description": "WattBot-Llama4-Maverick",
    },
    "deepseek-r1": {
        "arn": "arn:aws:bedrock:us-east-2::foundation-model/deepseek.deepseek-r1-distill-llama-70b-v1:0",
        "model_id": "deepseek.deepseek-r1-distill-llama-70b-v1:0",
        "description": "WattBot-DeepSeek-R1",
    },
}

# Default tags for cost allocation
DEFAULT_TAGS = [
    {"key": "project", "value": "wattbot"},
    {"key": "team", "value": "kohaku-rag"},
]


def get_bedrock_client(profile_name: str, region: str):
    """Create Bedrock client (not bedrock-runtime)."""
    session = boto3.Session(profile_name=profile_name, region_name=region)
    return session.client("bedrock")


def list_inference_profiles(client) -> list:
    """List all existing inference profiles."""
    try:
        response = client.list_inference_profiles()
        profiles = response.get("inferenceProfileSummaries", [])
        return profiles
    except ClientError as e:
        print(f"Error listing profiles: {e}")
        return []


def create_inference_profile(
    client,
    profile_name: str,
    model_arn: str,
    description: str,
    tags: list,
) -> dict | None:
    """Create an application inference profile with tags."""
    try:
        params = {
            "inferenceProfileName": profile_name,
            "modelSource": {"copyFrom": model_arn},
            "description": description,
            "tags": tags,
        }

        response = client.create_inference_profile(**params)

        return {
            "arn": response["inferenceProfileArn"],
            "status": response.get("status"),
            "name": profile_name,
        }

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"]

        if error_code == "ConflictException":
            print(f"  Profile '{profile_name}' already exists")
            return None
        elif error_code == "ValidationException":
            print(f"  Validation error: {error_msg}")
            return None
        else:
            print(f"  Error ({error_code}): {error_msg}")
            return None


def delete_inference_profile(client, profile_id: str) -> bool:
    """Delete an inference profile."""
    try:
        client.delete_inference_profile(inferenceProfileIdentifier=profile_id)
        return True
    except ClientError as e:
        print(f"Error deleting profile: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup AWS Bedrock cost tracking with inference profiles"
    )
    parser.add_argument(
        "--profile",
        default="bedrock_nils",
        help="AWS profile name",
    )
    parser.add_argument(
        "--region",
        default="us-east-2",
        help="AWS region",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing inference profiles",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create inference profiles for models",
    )
    parser.add_argument(
        "--model",
        help="Specific model to create profile for (default: all)",
    )
    parser.add_argument(
        "--experiment",
        default="wattbot-eval",
        help="Experiment name for tagging",
    )
    parser.add_argument(
        "--delete",
        help="Delete a specific inference profile by ID/ARN",
    )
    parser.add_argument(
        "--output",
        help="Output file for profile ARN mapping (JSON)",
    )

    args = parser.parse_args()

    print(f"Using AWS profile: {args.profile}")
    print(f"Region: {args.region}")
    print()

    client = get_bedrock_client(args.profile, args.region)

    if args.list:
        print("=== Existing Inference Profiles ===")
        profiles = list_inference_profiles(client)
        if profiles:
            for p in profiles:
                print(f"  Name: {p.get('inferenceProfileName')}")
                print(f"  ARN:  {p.get('inferenceProfileArn')}")
                print(f"  Type: {p.get('type')}")
                print(f"  Status: {p.get('status')}")
                print()
        else:
            print("  No inference profiles found")
        return

    if args.delete:
        print(f"Deleting profile: {args.delete}")
        if delete_inference_profile(client, args.delete):
            print("  Deleted successfully")
        return

    if args.create:
        print("=== Creating Inference Profiles ===")
        print(f"Experiment tag: {args.experiment}")
        print()

        # Determine which models to create profiles for
        if args.model:
            if args.model not in MODELS:
                print(f"Unknown model: {args.model}")
                print(f"Available: {', '.join(MODELS.keys())}")
                return
            models_to_create = {args.model: MODELS[args.model]}
        else:
            models_to_create = MODELS

        created_profiles = {}

        for model_name, model_info in models_to_create.items():
            profile_name = f"wattbot-{model_name}"
            print(f"Creating profile: {profile_name}")

            # Build tags
            tags = DEFAULT_TAGS.copy()
            tags.append({"key": "experiment", "value": args.experiment})
            tags.append({"key": "model", "value": model_name})

            result = create_inference_profile(
                client=client,
                profile_name=profile_name,
                model_arn=model_info["arn"],
                description=f"WattBot evaluation - {model_info['description']}",
                tags=tags,
            )

            if result:
                print(f"  Created: {result['arn']}")
                created_profiles[model_name] = {
                    "profile_arn": result["arn"],
                    "original_model_id": model_info["model_id"],
                }
            print()

        # Output mapping file
        if args.output and created_profiles:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(created_profiles, f, indent=2)
            print(f"Saved profile mapping to: {output_path}")

        print()
        print("=== Next Steps ===")
        print("1. Go to AWS Billing Console > Cost Allocation Tags")
        print("2. Find and activate these tags: project, team, experiment, model")
        print("3. Wait up to 24 hours for tags to appear in Cost Explorer")
        print()
        print("To use profiles in code, replace model_id with the profile ARN:")
        print("  model_id = 'arn:aws:bedrock:us-east-2:ACCOUNT:inference-profile/...'")


if __name__ == "__main__":
    main()
