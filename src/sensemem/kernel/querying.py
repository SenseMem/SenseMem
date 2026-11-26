"""Execute query-policy proposals against leased inputs and declared model capabilities."""

import logging
from dataclasses import dataclass

from opentelemetry.trace import Tracer

from sensemem.capabilities import MemoryCapabilities
from sensemem.exceptions import ModelError, SenseMemError, ValidationError
from sensemem.kernel.embedding import Embedding
from sensemem.kernel.ranking import lexical_query_terms
from sensemem.kernel.tracing import Traced
from sensemem.kernel.validation import MAX_TEXT_CHARACTERS
from sensemem.models.base import EmbedTask, ModelInput
from sensemem.query_policy import QueryPolicy, QueryRepresentationPlan, QueryRequest
from sensemem.types import Modality, RetrievalMode

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryEncoding:
    text: str
    vectors: tuple[tuple[float, ...], ...]
    image_queries: tuple[tuple[str, tuple[float, ...]], ...]
    lexical_queries: tuple[str, ...]


class Querying(Traced):
    """Own query-policy output admission and actual embedding work."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        policy: QueryPolicy,
        embedding: Embedding,
        capabilities: MemoryCapabilities,
    ) -> None:
        super().__init__(tracer)
        self._policy = policy
        self._embedding = embedding
        self._capabilities = capabilities

    def encode(self, request: QueryRequest, *, capture_visual_matches: bool) -> QueryEncoding:
        with self._trace("sensemem.query.plan", kind="stage"):
            try:
                plan = self._policy.plan(request)
            except ValidationError:
                raise ModelError(
                    "query policy returned an invalid plan",
                    reason="response_invalid",
                    stage="query.plan",
                ) from None
            except SenseMemError:
                raise
            except Exception:
                raise ModelError(
                    "query policy failed", reason="model_failed", stage="query.plan"
                ) from None
            if not isinstance(plan, QueryRepresentationPlan):
                raise ModelError(
                    "query policy returned an invalid plan",
                    reason="response_invalid",
                    stage="query.plan",
                )
            self._validate(plan, request)

        encoded: list[tuple[ModelInput, tuple[float, ...]]] = []
        if plan.embedding_inputs:
            vectors = self._embedding.embed(plan.embedding_inputs, task=EmbedTask.QUERY)
            encoded.extend(zip(plan.embedding_inputs, vectors, strict=True))
        if plan.optional_embedding_inputs:
            try:
                vectors = self._embedding.embed(
                    plan.optional_embedding_inputs, task=EmbedTask.QUERY
                )
            except ModelError as error:
                LOGGER.warning(f"optional query embedding failed: {error.reason}")
            else:
                encoded.extend(zip(plan.optional_embedding_inputs, vectors, strict=True))
        return QueryEncoding(
            text=plan.text,
            vectors=tuple(dict.fromkeys(vector for _input, vector in encoded)),
            image_queries=tuple(
                (value.assets[0].id, vector)
                for value, vector in encoded
                if capture_visual_matches
                and not value.text
                and len(value.assets) == 1
                and value.assets[0].modality is Modality.IMAGE
            ),
            lexical_queries=tuple(
                dict.fromkeys(
                    value.strip() for value in plan.lexical_queries if lexical_query_terms(value)
                )
            ),
        )

    def _validate(self, plan: QueryRepresentationPlan, request: QueryRequest) -> None:
        inputs = (*plan.embedding_inputs, *plan.optional_embedding_inputs)
        if (request.mode is RetrievalMode.LEXICAL and inputs) or (
            request.mode is RetrievalMode.DENSE and plan.lexical_queries
        ):
            raise ModelError(
                "query policy routes conflict with the retrieval mode",
                reason="response_invalid",
                stage="query.plan",
            )
        if any(
            len(text) > MAX_TEXT_CHARACTERS
            for text in (plan.text, *plan.lexical_queries, *(value.text for value in inputs))
        ):
            raise ModelError(
                "query policy text exceeds the content limit",
                reason="response_invalid",
                stage="query.plan",
            )
        allowed_assets = {asset.id: asset for asset in request.content.assets}
        for value in inputs:
            if any(allowed_assets.get(asset.id) != asset for asset in value.assets):
                raise ModelError(
                    "query policy assets must belong to the leased query",
                    reason="response_invalid",
                    stage="query.plan",
                )
            if not any(
                route.supports(value.modalities, assets_per_modality=value.assets_per_modality)
                for route in self._capabilities.embedding_routes
            ):
                raise ModelError(
                    "query policy selected an unsupported embedding input",
                    reason="unsupported_modality",
                    stage="query.plan",
                )
