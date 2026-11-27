# Optional tool-using agents

`AgentRunner` lets an application combine a VLM with the same typed tools it can call from a fixed pipeline. The application selects the model, algorithms, and knowledge sources explicitly; SenseMem does not require an agent loop to use `Memory`.

The runner sends tool definitions to an `AgentBackend`, executes proposed calls through a `ToolRegistry`, and returns the results to the model until it produces final text. Unknown tools, invalid arguments, and handler failures become typed error results so the model can correct a call. Tool schemas validate each invocation and output.

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

`ModelInput` accepts ordered text and resolved image, audio, and video assets. Set the adapter's `capabilities` to the endpoint's supported atomic modalities to preserve native attachments across model steps. Tool arguments can refer to host-approved asset or memory IDs; model-supplied local paths never grant file access.

A model without native audio input can inspect recordings through a registered tool such as `transcription_tool(memory, asr)`:

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

The registered ASR tool resolves that ID to leased originals. A fixed pipeline can also invoke `memory.call_tool("transcribe", {"memory_id": record.id})`. The tool's transcript and the agent's supplement remain temporary unless explicitly stored by the host.

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

`AgentStep` contains either nonempty final content or a nonempty tuple of `ToolCall` values. `AgentTurn` pairs every call with one matching `ToolResult`. `AgentResult.steps` counts model steps and `tool_calls` counts executed tool calls. Neither the runner nor a model result writes memory or authorizes durable actions.

The runner raises `ModelError(reason="agent_limit_exceeded", stage="agent")` before executing a batch that would exceed `max_tool_calls` or leave no model step to consume its results. Call IDs must be unique across the run, and the runner does not retry model calls itself.

`OpenAIAgent` calls Chat Completions through a caller-owned synchronous OpenAI client with `store=False`, translating refusals, truncation, and provider errors without logging response bodies. Set `max_tokens_parameter` (`"max_completion_tokens"` or `"max_tokens"`), `strict_tools=True`, or `audio_data_url=True` explicitly as required by your endpoint.

## Ownership and context composition

The runner borrows its backend and registry and closes neither. `OpenAIAgent.close()` leaves its caller-owned SDK client open; close the registry's owned resources once at the application boundary.

An async host uses `await runner.run_async(input)`, which runs the synchronous loop off-thread and drains started work before propagating cancellation. Combine it with `AsyncContextSource` to supply temporary supplements to `compile()`:

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

Supplements are request-local context, not durable facts or policy authority. See the [model quality gate](../model-quality.md#check-agent-tool-reasoning) for the opt-in agent-tool integration check.
