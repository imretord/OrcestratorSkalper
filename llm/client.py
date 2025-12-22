"""
LLM client for AI Trading System V3.
Supports Groq, OpenRouter, and Ollama APIs.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import httpx

from core.logger import get_logger

log = get_logger("llm")


class LLMClient:
    """
    LLM client with multiple provider support.

    Supported providers:
    - groq: Groq API (llama-3.1-70b-versatile)
    - openrouter: OpenRouter API (deepseek, llama, etc.)
    - ollama: Local Ollama instance

    Features:
    - Rate limiting (20 requests/minute for Groq)
    - Automatic retry with exponential backoff
    - JSON response parsing
    - Fallback to alternative providers
    """

    # Rate limit: 20 requests per minute
    RATE_LIMIT = 20
    RATE_WINDOW = 60  # seconds

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        fallback_provider: str = "ollama",
        fallback_model: str = "llama3.1:8b",
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> None:
        """
        Initialize LLM client.

        Args:
            provider: Primary provider ("groq", "openrouter", or "ollama")
            model: Model name for primary provider
            api_key: API key (auto-detected from env if not provided)
            fallback_provider: Fallback provider
            fallback_model: Fallback model name
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        # Auto-detect provider and API key from environment
        if provider is None or api_key is None:
            provider, model, api_key = self._auto_detect_provider()

        self.provider = provider
        self.model = model or "llama-3.1-70b-versatile"
        self.api_key = api_key
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Rate limiting
        self._request_times: deque[float] = deque(maxlen=self.RATE_LIMIT)

        # Stats
        self.stats = {
            'requests': 0,
            'successful': 0,
            'failed': 0,
            'fallback_used': 0,
            'total_tokens': 0,
            'total_time_ms': 0,
        }

        # API endpoints
        self._groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self._openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self._ollama_url = "http://localhost:11434/api/generate"

        log.info(f"[LLM] Initialized: provider={self.provider}, model={self.model}")

    def _auto_detect_provider(self) -> tuple[str, str, str]:
        """
        Auto-detect LLM provider from environment variables.

        Returns:
            Tuple of (provider, model, api_key)
        """
        # Check for OpenRouter first (from existing .env)
        openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY", "")
        if openrouter_key and len(openrouter_key) > 10:
            model = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
            log.info("[LLM] Auto-detected OpenRouter provider")
            return "openrouter", model, openrouter_key

        # Check for Groq
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key and len(groq_key) > 10:
            log.info("[LLM] Auto-detected Groq provider")
            return "groq", "llama-3.1-70b-versatile", groq_key

        # Default to groq with empty key (will fail but gracefully)
        log.warning("[LLM] No API key found, defaulting to Groq (will fail)")
        return "groq", "llama-3.1-70b-versatile", ""

    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is reached."""
        if self.provider != "groq":
            return

        now = time.time()

        # Remove old timestamps outside the window
        while self._request_times and (now - self._request_times[0]) > self.GROQ_RATE_WINDOW:
            self._request_times.popleft()

        # If at limit, wait
        if len(self._request_times) >= self.GROQ_RATE_LIMIT:
            oldest = self._request_times[0]
            wait_time = self.GROQ_RATE_WINDOW - (now - oldest) + 0.1
            if wait_time > 0:
                log.warning(f"[LLM] Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

        self._request_times.append(time.time())

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated text response

        Raises:
            Exception: If all providers fail
        """
        start_time = time.time()
        self.stats['requests'] += 1

        # Try primary provider
        try:
            response = self._call_provider(
                provider=self.provider,
                model=self.model,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['successful'] += 1
            self.stats['total_time_ms'] += elapsed_ms

            log.info(f"[LLM] Response in {elapsed_ms:.0f}ms: {response[:100]}...")
            return response

        except Exception as e:
            log.warning(f"[LLM] Primary provider failed: {e}")

            # Try fallback
            if self.fallback_provider:
                try:
                    log.info(f"[LLM] Trying fallback: {self.fallback_provider}/{self.fallback_model}")

                    response = self._call_provider(
                        provider=self.fallback_provider,
                        model=self.fallback_model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens or self.max_tokens,
                        temperature=temperature or self.temperature,
                    )

                    elapsed_ms = (time.time() - start_time) * 1000
                    self.stats['successful'] += 1
                    self.stats['fallback_used'] += 1
                    self.stats['total_time_ms'] += elapsed_ms

                    log.info(f"[LLM] Fallback response in {elapsed_ms:.0f}ms")
                    return response

                except Exception as fallback_error:
                    log.error(f"[LLM] Fallback also failed: {fallback_error}")

            self.stats['failed'] += 1
            raise Exception(f"All LLM providers failed: {e}")

    def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate completion and parse as JSON.

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: System prompt

        Returns:
            Parsed JSON dict

        Raises:
            Exception: If response is not valid JSON
        """
        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\n\nRespond with valid JSON only. No markdown, no explanation."

        response = self.complete(prompt, system_prompt=json_system)

        # Try to extract JSON from response
        json_str = self._extract_json(response)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log.error(f"[LLM] JSON parse error: {e}\nResponse: {response[:500]}")
            raise Exception(f"Invalid JSON response: {e}")

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that might contain markdown or extra content.

        Args:
            text: Raw response text

        Returns:
            Extracted JSON string
        """
        # Try to find JSON in code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)

        # Try to find raw JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()

        # Return as-is and let JSON parser handle it
        return text.strip()

    def _call_provider(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Call a specific LLM provider.

        Args:
            provider: Provider name
            model: Model name
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            Response text
        """
        if provider == "groq":
            return self._call_groq(prompt, system_prompt, model, max_tokens, temperature)
        elif provider == "openrouter":
            return self._call_openrouter(prompt, system_prompt, model, max_tokens, temperature)
        elif provider == "ollama":
            return self._call_ollama(prompt, system_prompt, model, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _call_groq(
        self,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call Groq API."""
        self._wait_for_rate_limit()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 3

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        self._groq_url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                # Extract response text
                content = data["choices"][0]["message"]["content"]

                # Track tokens
                usage = data.get("usage", {})
                self.stats['total_tokens'] += usage.get("total_tokens", 0)

                return content

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(e.response.headers.get("Retry-After", 10))
                    log.warning(f"[LLM] Groq rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                elif attempt < max_retries - 1:
                    delay = 2 ** attempt
                    log.warning(f"[LLM] Groq error {e.response.status_code}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    log.warning(f"[LLM] Groq error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

        raise Exception("Groq API failed after all retries")

    def _call_openrouter(
        self,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call OpenRouter API."""
        self._wait_for_rate_limit()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://trading-bot-v3.local",
            "X-Title": "AI Trading System V3",
        }

        max_retries = 3

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        self._openrouter_url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                # Extract response text
                content = data["choices"][0]["message"]["content"]

                # Track tokens
                usage = data.get("usage", {})
                self.stats['total_tokens'] += usage.get("total_tokens", 0)

                return content

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 10))
                    log.warning(f"[LLM] OpenRouter rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                elif attempt < max_retries - 1:
                    delay = 2 ** attempt
                    log.warning(f"[LLM] OpenRouter error {e.response.status_code}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    log.warning(f"[LLM] OpenRouter error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

        raise Exception("OpenRouter API failed after all retries")

    def _call_ollama(
        self,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call Ollama API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self._ollama_url, json=payload)
                response.raise_for_status()
                data = response.json()

            return data.get("response", "")

        except Exception as e:
            raise Exception(f"Ollama API error: {e}")

    def test_connection(self) -> tuple[bool, str]:
        """
        Test connection to LLM provider.

        Returns:
            Tuple of (success, message)
        """
        try:
            start_time = time.time()
            response = self.complete("What is 2+2? Reply with just the number.")
            elapsed_ms = (time.time() - start_time) * 1000

            if "4" in response:
                return True, f"Connected to {self.provider}/{self.model}. Response time: {elapsed_ms:.0f}ms"
            else:
                return False, f"Unexpected response: {response}"

        except Exception as e:
            return False, f"Connection failed: {e}"

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        avg_time = 0
        if self.stats['successful'] > 0:
            avg_time = self.stats['total_time_ms'] / self.stats['successful']

        return {
            'requests': self.stats['requests'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'success_rate': self.stats['successful'] / max(1, self.stats['requests']) * 100,
            'fallback_used': self.stats['fallback_used'],
            'total_tokens': self.stats['total_tokens'],
            'avg_response_ms': avg_time,
        }
