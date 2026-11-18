"""Apply typed ingestion proposals while preserving source identity and durable recipes."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

from opentelemetry.trace import Tracer

from sensemem.exceptions import ModelError, SenseMemError, ValidationError
from sensemem.kernel.content import PreparedContent, PreparedMemory
from sensemem.kernel.hydration import Hydrator
from sensemem.kernel.lifecycle import OperationAssets
from sensemem.kernel.tracing import Traced
from sensemem.kernel.validation import MAX_TEXT_CHARACTERS
from sensemem.processing import IngestionStage, StageResult


class IngestionStages(Traced):
    """Run computations outside storage transactions and append labelled proposals only."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        stages: Sequence[IngestionStage],
        hydrator: Hydrator,
    ) -> None:
        super().__init__(tracer)
        self._stages = tuple((stage.name, stage.space, stage) for stage in stages)
        self._headers = {
            (name, space): (
                f"[processing interpretation {json.dumps({'name': name, 'space': space}, ensure_ascii=False)}"
                ": not original evidence]\n"
            )
            for name, space, _stage in self._stages
        }
        self._hydrator = hydrator

    @property
    def enabled(self) -> bool:
        return bool(self._stages)

    def process_memory(self, memory: PreparedMemory, operation: OperationAssets) -> PreparedMemory:
        """Append ordered stage interpretations, reusing this operation's completed results."""
        if not self._stages:
            return memory
        memory = replace(memory, content=self._without_previous_interpretations(memory.content))
        for name, space, stage in self._stages:
            # Retries share results only for the same effective input. A refreshed transcript
            # or stream synopsis may keep its observation ID while changing what a stage sees.
            fingerprint = hashlib.sha256(
                json.dumps(
                    (memory.content.text, memory.content.canonical_parts),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            key = (memory.memory_id, name, f"{space}:{fingerprint}")
            result = operation.stage_results.get(key)
            if result is None:
                with self._trace("sensemem.ingestion.process", kind="stage") as span:
                    span.set_attribute("sensemem.processing.stage", name)
                    span.set_attribute("sensemem.processing.space", space)
                    value = self._hydrator.model_input(memory.content)
                    try:
                        result = stage.process(value)
                    except ValidationError:
                        raise ModelError(
                            "ingestion stage returned an invalid result",
                            reason="response_invalid",
                            stage="ingestion.process",
                            subject=name,
                        ) from None
                    except SenseMemError as error:
                        raise ModelError(
                            "ingestion stage failed",
                            reason=error.reason if error.retryable else "model_failed",
                            stage="ingestion.process",
                            subject=name,
                        ) from None
                    except Exception:
                        raise ModelError(
                            "ingestion stage failed",
                            reason="model_failed",
                            stage="ingestion.process",
                            subject=name,
                        ) from None
                    if not isinstance(result, StageResult):
                        raise ModelError(
                            "ingestion stage returned an invalid result",
                            reason="response_invalid",
                            stage="ingestion.process",
                            subject=name,
                        )
                    operation.stage_results[key] = result
            if not result.interpretation:
                continue
            section = self._headers[name, space] + result.interpretation
            content = memory.content
            text = "\n\n".join(value for value in (content.text, section) if value)
            if len(text) > MAX_TEXT_CHARACTERS:
                raise ModelError(
                    "ingestion stage interpretation exceeds the content limit",
                    reason="response_invalid",
                    stage="ingestion.process",
                    subject=name,
                )
            memory = replace(
                memory,
                content=replace(
                    content,
                    text=text,
                    canonical_parts=(*content.canonical_parts, ("model_interpretation", section)),
                ),
            )
        return memory

    def _without_previous_interpretations(self, content: PreparedContent) -> PreparedContent:
        """Refresh this pipeline's derivations without editing original or stream evidence."""
        headers = tuple(self._headers.values())
        previous = tuple(
            part
            for part in content.canonical_parts
            if part[0] == "model_interpretation" and part[1].startswith(headers)
        )
        if not previous:
            return content
        original = "\n\n".join(value for kind, value in content.canonical_parts if kind == "text")
        suffix = content.text[len(original) :] if original else f"\n\n{content.text}"
        for _kind, value in previous:
            suffix = suffix.replace(f"\n\n{value}", "", 1)
        return replace(
            content,
            text="\n\n".join(value for value in (original, suffix.strip()) if value),
            canonical_parts=tuple(part for part in content.canonical_parts if part not in previous),
        )
