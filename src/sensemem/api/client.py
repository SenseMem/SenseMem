"""Synchronous JSON transport to a running SenseMem owner, independent of CLI policy."""

from __future__ import annotations

import json
from typing import TypeAlias
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sensemem.exceptions import StorageError

JsonDocument: TypeAlias = dict[str, object]


class RemoteFailure(Exception):
    """An error envelope a running owner already produced; forwarded, never re-derived."""

    def __init__(self, envelope: JsonDocument) -> None:
        super().__init__(str(envelope.get("message", "")))
        self.envelope = envelope


def request_json(
    url: str, method: str, path: str, body: JsonDocument | None, timeout: float
) -> JsonDocument:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = Request(f"{url.rstrip('/')}{path}", data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return _response_document(response.read())
    except HTTPError as error:
        # An error response is still an open response: HTTPError wraps the body file, so leaving
        # it unclosed leaks the socket and raises in the deallocator.
        try:
            envelope = _response_document(error.read())
        finally:
            error.close()
        raise RemoteFailure(envelope) from error
    except (TimeoutError, URLError) as error:
        cause = error.reason if isinstance(error, URLError) else error
        if isinstance(cause, TimeoutError):
            raise StorageError(
                f"request to the SenseMem owner at {url} timed out after {timeout:g} seconds",
                reason="timeout",
                stage="request",
            ) from error
        raise StorageError(
            f"cannot reach the SenseMem owner at {url}",
            reason="connection_failed",
            stage="request",
            subject=str(cause),
        ) from error


def _response_document(raw: bytes) -> JsonDocument:
    try:
        document = json.loads(raw or b"{}")
    except ValueError as error:
        raise StorageError(
            "the SenseMem owner returned a body that is not JSON",
            reason="response_invalid",
            stage="request",
        ) from error
    if not isinstance(document, dict):
        raise StorageError(
            "the SenseMem owner returned a body that is not a JSON object",
            reason="response_invalid",
            stage="request",
        )
    return document
