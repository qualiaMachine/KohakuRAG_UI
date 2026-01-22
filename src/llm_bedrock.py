"""AWS Bedrock Chat Model Integration for KohakuRAG.

This module provides a BedrockChatModel class that implements the ChatModel
protocol from KohakuRAG, enabling the use of AWS Bedrock foundation models
(e.g., Claude) for query planning and answer generation.

Usage:
    from src.llm_bedrock import BedrockChatModel
    
    model = BedrockChatModel(
        profile_name="bedrock_nils",
        region_name="us-east-2",
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0"
    )
    
    response = await model.complete("What is the carbon footprint of LLMs?")
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
    """Chat backend powered by AWS Bedrock with automatic rate limit handling.
    
    Implements the ChatModel protocol for integration with KohakuRAG pipelines.
    Uses boto3 to call Bedrock Runtime API with Anthropic Claude models.
    
    Attributes:
        profile_name: AWS SSO profile name for authentication
        region_name: AWS region (e.g., "us-east-2")
        model_id: Bedrock model identifier
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
        """Initialize Bedrock chat model with automatic rate limit retry.

        Args:
            model_id: Bedrock model identifier (e.g., "us.anthropic.claude-3-haiku-20240307-v1:0")
            profile_name: AWS SSO profile name (reads from AWS_PROFILE env if not provided)
            region_name: AWS region (reads from AWS_REGION env if not provided, defaults to us-east-2)
            system_prompt: Default system message for all completions
            max_tokens: Maximum tokens to generate (default: 4096)
            max_retries: Maximum retry attempts on rate limit errors
            base_retry_delay: Base delay for exponential backoff (seconds)
            max_concurrent: Maximum number of concurrent API requests (0 = unlimited)
        """
        # Lazy import boto3 to avoid import errors if not installed
        try:
            import boto3
            self._boto3 = boto3
        except ImportError as e:
            raise ImportError(
                "boto3 is required for BedrockChatModel. Install with: pip install boto3"
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
        """Make synchronous Bedrock API request.
        
        This is called via asyncio.to_thread() to avoid blocking the event loop.
        """
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
        """Execute chat completion with automatic rate limit retry.

        Uses intelligent retry strategy:
        1. Semaphore limits concurrent requests (if enabled)
        2. Exponential backoff with jitter for rate limits
        3. Handles Bedrock-specific throttling errors

        Args:
            prompt: User message/question
            system_prompt: Optional system message override

        Returns:
            Model's text response

        Raises:
            RuntimeError: If all retries are exhausted
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
