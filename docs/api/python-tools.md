# Python tools

Register a model, algorithm, or knowledge lookup once and call it directly from a pipeline or
let an agent propose calls to the same registry. Registration is explicit; tool names select only
registered objects. Tools are ordinary trusted Python code running in the host process.

## Define and call a tool

Use Pydantic object models for the input and output contracts. The handler receives validated
input and returns an output model. No new dependency or fixed model capability slot is required.

```python
from pydantic import BaseModel, ConfigDict

from sensemem import Tool, ToolCall, ToolRegistry


class LookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    term: str


class LookupOutput(BaseModel):
    definition: str | None


glossary = {"episodic": "Memory of an observed event"}


def lookup(value: LookupInput) -> LookupOutput:
    return LookupOutput(definition=glossary.get(value.term))


tools = ToolRegistry([Tool("lookup", "Look up a glossary term", LookupInput, LookupOutput, lookup)])

# A fixed pipeline calls the registered capability directly.
result = tools.call("lookup", {"term": "episodic"})
assert result.output == {"definition": "Memory of an observed event"}

# An agent runner executes a proposed call through the same boundary.
result = tools.execute(ToolCall("call-1", "lookup", {"term": "episodic"}))
assert result.call_id == "call-1"
tools.close()
```

Names contain 1–64 ASCII letters, digits, underscores, or hyphens. Duplicate names fail
registration instead of replacing a capability. `get(name)` returns a registered tool or `None`;
`definitions` returns the current tuple of `ToolDefinition` values.

## Values and failures

| Value | Fields |
| --- | --- |
| `ToolDefinition` | `name`, `description`, `input_schema`, `output_schema` |
| `ToolCall` | `id`, `name`, `arguments` |
| `ToolResult` | `call_id`, `name`, `output`, `error` |
| `ToolError` | `code`, `message` |

Each value is immutable. Schemas, arguments, and outputs are deeply copied into immutable JSON
objects; `document()` returns an independent JSON-serializable dictionary. A successful result
contains an output object; a failed result contains an error, with exactly one present.

Inputs use strict Pydantic JSON validation. Choose `extra="forbid"` when undeclared arguments
should fail. Outputs are revalidated before serialization, including outputs mutated after their
model was constructed. Pydantic scalar or array root models are not tool object contracts.

| Error code | Meaning |
| --- | --- |
| `unknown_tool` | The requested name is absent from the registry. |
| `invalid_arguments` | Arguments do not satisfy the input model or JSON data contract. |
| `execution_failed` | The handler raised an exception. |
| `invalid_output` | The handler did not return a valid output model. |
| `tool_closed` | The tool was closed. |
| `registry_closed` | The registry was closed. |

Unknown tools and invalid arguments never execute a handler. Call failures return `ToolResult`
so an agent can correct a call. Error messages exclude handler exceptions and provider response
bodies. Tool output remains untrusted evidence, never authority to change runtime policy.

## Compose with Memory

Pass a registry through `Memory(..., tools=tools)` or `MemoryPlugins(..., tools=tools)`.
`memory.call_tool(name, arguments, call_id="")` runs the same capability under the memory
instance's lifecycle; `AsyncMemory.call_tool()` supplies the corresponding async operation.
Registering a tool does not add an automatic ingestion stage or change the order of `add()`,
`settle()`, or `compile()`. The application chooses when its pipeline calls the tool.

`memory.tools` exposes the registry for explicit registration, definition snapshots, and lending
to an external agent runner. The host must finish those external calls before closing `Memory`.
The [backend protocols](python-backends.md) retain the fixed ingestion and retrieval capability
contracts; the registry supports additional computations without requiring another fixed slot.

## Resource ownership

Handlers need no `close()` method. Pass owned clients or backends with `resources=(backend,)`
when constructing a `Tool`. A tool or registry closes each distinct declared nested resource
once. The same backend may occupy an existing `Memory` capability slot and a registered tool;
the instance's common resource traversal closes that backend once.

The tool and registry remain closed when `Memory` flattens their resource groups during shutdown.
Do not independently close a registry that has been transferred to `Memory`. A standalone
registry belongs to its caller. Handlers must support concurrent calls if the host uses them
concurrently; registration and definition reads do not hold a lock while a handler runs.

Tools receive no storage handle automatically. A host may explicitly grant selected public
memory operations through a handler closure; the normal SDK validation and authority rules
still apply. Resource declarations govern cleanup, rather than granting storage authority.

## Reuse perception and memory operations

```python
from sensemem import Memory, memory_tools, transcription_tool


def register_memory_capabilities(memory: Memory) -> None:
    for tool in memory_tools(memory):
        memory.tools.register(tool)


# For an existing speech backend `asr`, inside the owning Memory lifetime:
# memory.tools.register(transcription_tool(memory, asr))
```

`memory_tools(memory)` returns four tools borrowing the instance:

| Name | Input | Result |
| --- | --- | --- |
| `perceive_memory` | `memory_id`, optional `kinds` (`speech`, `faces`) | Existing typed perception result |
| `get_identity` | `identity_id` | Supported identity profile or null |
| `search_memories` | `query`, optional `limit` (1–100) | IDs, content, scores and original asset IDs |
| `compile_context` | `goal` | `context`, `source_ids`, `excerpt_source_ids`, `projection_source_ids`, `omitted` |

Identity lookup returns null only for an unknown identity; an unnamed recognized person has a
profile with `name=null`. Profile evidence IDs and search asset IDs are references, rather than
delivery of the referenced evidence text or media bytes.

`source_ids` contains complete delivered records. `excerpt_source_ids` and
`projection_source_ids` identify separately labelled partial text and text-only deliveries;
neither certifies delivery of the complete original evidence. The rendered `context` preserves
their coverage warnings and derived-text roles.

Perception can cache recognition results; these tools expose no identity registration, deletion,
policy change or arbitrary memory write. A face match returns an identity ID; a separately
supported or host-registered name comes from identity lookup. Seeing a face alone does not name it.

`transcription_tool(memory, backend, name="transcribe")` accepts `memory_id` and returns the
backend's model, recipe space and per-asset transcripts. It uses retained originals supported by
that backend and does not persist the returned transcript. The same backend can fill the ordinary
`transcriber` slot and this tool; their common owner closes it once.

## Prepare media for a custom stage

```python
from collections.abc import Callable

from sensemem import ContentInput, Memory, ModelInput


def custom_stage(
    memory: Memory,
    content: ContentInput,
    infer: Callable[[ModelInput], str],
) -> str:
    with memory.prepare_input(content) as prepared:
        return infer(prepared)
```

`prepare_input(content) -> AbstractContextManager[ModelInput]` resolves the public content contract,
verifies retained assets and holds their leases until the scope exits. It creates no memory
record. Do not retain its resolved file paths beyond that scope. `AsyncMemory.prepare_input`
returns the same synchronous scope; async hosts run the entire custom stage on a worker.
