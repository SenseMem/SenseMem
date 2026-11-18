# Optional tool-using agents

`AgentRunner` lets an application combine a VLM with the same typed tools it can call from a
fixed pipeline. The application selects the model, algorithms and knowledge sources explicitly.
SenseMem does not require an agent loop to use `Memory`.

The runner sends tool definitions to an `AgentBackend`, executes proposed calls through a
`ToolRegistry`, and returns the results to the model until it produces final text. Unknown tools,
invalid arguments and handler failures become typed error results, so the model can correct a
call. Tool schemas validate each invocation and output.

```python
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from sensemem import AgentRunner, ModelInput, OpenAIAgent, Tool, ToolRegistry


class LookupInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    term: str


class LookupOutput(BaseModel):
    definition: str


def lookup(value: LookupInput) -> LookupOutput:
    return LookupOutput(definition={"CAS": "content-addressed storage"}.get(value.term, "unknown"))


with OpenAI() as client:
    tools = ToolRegistry(
        (Tool("lookup", "Look up a local term.", LookupInput, LookupOutput, lookup),)
    )
    backend = OpenAIAgent(client, model="your-explicit-model-revision")
    runner = AgentRunner(backend, tools, max_steps=6, max_tool_calls=8)
    result = runner.run(ModelInput(text="Use lookup to explain CAS."))
    print(result.content)
```

`ModelInput` accepts ordered text and resolved image, audio and video assets. Set the adapter's
`capabilities` to the endpoint's supported atomic modalities; it preserves native attachments
across model steps. Tool arguments can refer to host-approved asset IDs. The registry's handlers
decide how to resolve those IDs; model-supplied local paths do not grant file access.

A core model without audio input receives a reference to audio available through a tool. Keep
the original recording on the tool side, rather than adding unsupported native audio to the model
request. For example, register `transcription_tool(memory, asr)` once, capture the original with
`memory.capture(audio)`, and pass its ID to the agent:

```python
from sensemem import AgentRunner, ContextSupplement, ModelInput


async def recording_context(runner: AgentRunner, memory_id: str) -> ContextSupplement:
    result = await runner.run_async(
        ModelInput(
            text=f"Who is visiting? The recording has memory_id={memory_id}. "
            "Use transcribe to inspect its speech when needed."
        )
    )
    return ContextSupplement(result.content, source=f"agent:recording:{memory_id}")
```

The registered ASR resolves that ID to leased originals. A fixed pipeline can also call
`memory.call_tool("transcribe", {"memory_id": record.id})`. The host can register the same ASR
as the ordinary `transcriber` for later settlement. Capture preserves the source without requiring
ASR first; the tool's returned transcript and the agent's supplement remain temporary.

## Public protocol

```text
AgentBackend.step(
    input: ModelInput,
    *,
    tools: Sequence[ToolDefinition],
    history: Sequence[AgentTurn],
) -> AgentStep

AgentStep(content=None, calls=())
AgentTurn(step, results)
AgentResult(content, turns=())
AgentRunner(backend, tools, *, max_steps=8, max_tool_calls=16)
AgentRunner.run(input: ModelInput) -> AgentResult
await AgentRunner.run_async(input: ModelInput) -> AgentResult
```

`AgentStep` contains either nonempty final content or a nonempty tuple of `ToolCall` values.
`AgentTurn` pairs every call with one matching `ToolResult`. `AgentResult.steps` counts model
steps, and `tool_calls` counts executed tool calls. Results and transcript remain temporary;
neither the runner nor a model result writes memory or authorizes a durable action.

The runner stops with `ModelError(reason="agent_limit_exceeded", stage="agent")` before executing
a batch that exceeds the tool-call budget or leaves no model step to consume its results. Call IDs
must remain unique for the entire run. It never retries model calls itself.

`OpenAIAgent` uses Chat Completions and the caller's synchronous SDK client. It translates refusal,
truncation and provider errors without logging response content. `max_tokens_parameter` explicitly
selects `max_completion_tokens` (default) or `max_tokens`. It adds no sampling controls. Set
`strict_tools=True` only when the endpoint and registered schemas support strict function schemas;
there is no silent protocol downgrade. `audio_data_url=True` explicitly selects compatible
endpoints that require data URLs instead of raw base64 audio.

## Ownership and context composition

The runner borrows its backend and registry and closes neither. `OpenAIAgent.close()` leaves its
caller-owned SDK client open. Close the registry's owned resources once at their application
boundary; the same registry can be reused by a fixed pipeline or a `Memory` composition.

An async host uses `await runner.run_async(input)`. It runs the synchronous loop off-thread and waits
for started work to finish before re-raising cancellation. An `AsyncContextSource` can therefore
cancel and drain its provider before the host closes the borrowed backend or tool resources.
Synchronous providers and tool handlers still need timeouts; cancellation cannot interrupt their
running thread. The host's input preparation can resolve either text or media before the agent:

```python
from collections.abc import Callable, Sequence

from sensemem import AgentRunner, AsyncContextSource, ContentInput, ContextSupplement, ModelInput


def agent_context_source(
    runner: AgentRunner,
    prepare: Callable[[ContentInput], ModelInput],
    *,
    model_revision: str,
) -> AsyncContextSource:
    async def provide(query: ContentInput) -> Sequence[ContextSupplement]:
        result = await runner.run_async(prepare(query))
        return (ContextSupplement(result.content, source=f"agent:{model_revision}"),)

    return AsyncContextSource(provide)
```

This explicitly converts the final result into a context supplement. That supplement is request
context, not a durable fact or authority to change policy. Tool outputs, stored evidence and media
remain untrusted data. Only host instructions authorize tool effects.

## Validation and model quality

Local SDK-transport tests cover native tool messages, error results, media transport, refusal,
truncation and execution bounds. They do not establish model quality or prompt-injection
resistance. The opt-in `tests/integration/test_agent_tool_quality.py` fixtures require real tool
calls, recovery from invalid arguments and rejection of instructions in tool output, using a
pinned model revision. Real-service quality has not been verified by the local transport tests.
