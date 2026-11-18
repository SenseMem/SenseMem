"""Explicit ownership of one SDK client, shared by independent request adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sensemem.exceptions import ModelError

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIClient:
    def __init__(self, client: OpenAI | None, *, owns_client: bool = False) -> None:
        self._sdk = client
        self._owns_client = owns_client
        self._client_closed = False

    def _client(self, operation: str) -> OpenAI:
        if self._sdk is None:
            raise ModelError(
                f"{operation} SDK client is not configured", reason="backend_not_configured"
            )
        return self._sdk

    def close(self) -> None:
        """Close only clients explicitly transferred to this adapter."""
        if not self._client_closed:
            self._client_closed = True
            if self._owns_client and self._sdk is not None:
                self._sdk.close()
