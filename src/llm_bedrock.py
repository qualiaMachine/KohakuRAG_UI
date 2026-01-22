"""
AWS Bedrock Integration for KohakuRAG

This module is the bridge between our RAG pipeline and AWS Bedrock. It lets us swap out
local models or OpenRouter for enterprise-grade Claude 3 models hosted on AWS.

Key features:
- Handles all the AWS auth complexity (SSO profiles, regions)
- Smart retry logic so we don't get crushed by rate limits
- Implements the standard `ChatModel` protocol so it drops right into existing pipelines

Usage:
    from src.llm_bedrock import BedrockChatModel
    
    # Initialize with your local AWS profile (setup via `aws sso login`)
    model = BedrockChatModel(
        profile_name="bedrock_nils",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0"  # Haiku is fast & cheap
    )
    
    # Fire off a query
    response = await model.complete("What's the energy cost of training GPT-3?")
"""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Protocol


# ChatModel protocol - matches KohakuRAG's pipeline.py definition
class ChatModel(Protocol):
    """Protocol for chat backends."""

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:  # pragma: no cover
        raise NotImplementedError


def _load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load environment variables from a .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    env_vars: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars


class BedrockChatModel:
    """
    A unified wrapper for AWS Bedrock's Chat API.
    
    This class handles the messy parts of working with Bedrock:
    1. Authentication via SSO profiles (so we don't need hardcoded keys)
    2. Rate limiting (exponential backoff when AWS says "slow down")
    3. Threading (boto3 is synchronous, so we offload it to keep the UI snappy)
    """

    def __init__(
        self,
        *,
        model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0",
        profile_name: str | None = None,
        region_name: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
        base_retry_delay: float = 3.0,
        max_concurrent: int = 10,
    ) -> None:
        """
        Set up the Bedrock connection.

        Args:
            model_id: Which model to use. Defaults to Claude 3 Haiku (best balance of speed/cost).
            profile_name: Your AWS SSO profile (e.g., 'bedrock_nils'). Falls back to AWS_PROFILE env var.
            region_name: The AWS region where the model is enabled. Defaults to us-east-2 (Ohio).
            system_prompt: The persona the model should adopt.
            max_tokens: Hard limit on response length. 4096 is usually plenty.
            max_retries: How many times to try again if AWS throttles us.
            base_retry_delay: Initial wait time between retries (increases exponentially).
            max_concurrent: How many parallel requests we allow before queuing. Critical for bulk processing.
        """
        # Lazy import boto3 to allow using the rest of the app without AWS dependencies
        try:
            import boto3
            self._boto3 = boto3
        except ImportError as e:
            raise ImportError(
                "Missing the 'boto3' library. You need it to talk to AWS. Run: pip install boto3"
            ) from e

        # Load environment variables
        dotenv_vars = _load_dotenv()

        # Resolve profile name
        self._profile_name = (
            profile_name
            or os.environ.get("AWS_PROFILE")
            or dotenv_vars.get("AWS_PROFILE")
        )

        # Resolve region
        self._region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or dotenv_vars.get("AWS_REGION")
            or "us-east-2"
        )

        self._model_id = model_id
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

        # Rate limiting semaphore
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

        # Create boto3 session and client
        self._session = self._boto3.Session(
            profile_name=self._profile_name,
            region_name=self._region_name,
        )
        self._client = self._session.client("bedrock-runtime")

    def _make_request_sync(self, prompt: str, system_prompt: str) -> str:
        """
        The actual API call to Bedrock.
        
        Note: This is a synchronous (blocking) function because boto3 doesn't support async yet.
        We run this inside `asyncio.to_thread` elsewhere to prevent freezing the application.
        """
        # Create the standard Claude 3 request body
        # Docs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """
        Send a message to Bedrock and get the response text.

        This method includes a robust retry loop to handle AWS "ThrottlingException" errors.
        It uses exponential backoff: if we get rated limited, we wait 3s, then 6s, then 12s...
        It also uses a semaphore to ensure we never have more than `max_concurrent` requests in flight.

        Args:
            prompt: The text you want Claude to process.
            system_prompt: (Optional) Override the default system persona for this specific call.

        Returns:
            The raw text string from the model.
        """
        system = system_prompt or self._system_prompt

        for attempt in range(self._max_retries + 1):
            try:
                # Use semaphore for rate limiting if enabled
                if self._semaphore is not None:
                    async with self._semaphore:
                        response = await asyncio.to_thread(
                            self._make_request_sync, prompt, system
                        )
                else:
                    response = await asyncio.to_thread(
                        self._make_request_sync, prompt, system
                    )
                return response

            except Exception as e:
                error_str = str(e).lower()

                # Check for retryable errors
                is_throttling = (
                    "throttling" in error_str
                    or "rate" in error_str
                    or "too many requests" in error_str
                    or "429" in error_str
                )
                is_server_error = any(
                    code in error_str for code in ["500", "502", "503", "504"]
                )
                is_retryable = is_throttling or is_server_error

                if not is_retryable or attempt >= self._max_retries:
                    raise  # Not retryable or exhausted retries

                # Exponential backoff with jitter
                wait_time = self._base_retry_delay * (2 ** attempt)
                jitter_factor = random.random() * 0.5 + 0.75  # 75-125%
                wait_time = wait_time * jitter_factor

                error_type = "Throttling" if is_throttling else "Server error"
                print(
                    f"Bedrock {error_type} (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected end of retry loop")

    def __repr__(self) -> str:
        return (
            f"BedrockChatModel(model_id={self._model_id!r}, "
            f"region_name={self._region_name!r}, "
            f"profile_name={self._profile_name!r})"
        )
