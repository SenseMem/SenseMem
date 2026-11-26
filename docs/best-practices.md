# Best practices for multimodal memory

Track changing facts across sessions and trace every recalled claim back to its timed original evidence. This guide shows practical patterns for cross-session capture, time-bounded retrieval, source inspection, and continuous media replay.

## Track changes across sessions

Start the [live audio/video demo](live-demo.md#run-locally) with a fresh `data_dir` and reuse that directory across three separate calls:

| Call | Action | Verification |
| --- | --- | --- |
| First observation | Show a key and place it in a blue box while keeping both in view. | Open the experience view and replay the stored image or clip showing the placement. |
| State change | Start a new call and move the key from the blue box to a red pouch. | Confirm both placements carry distinct timestamps and separate original media. |
| Later recall | Point the camera away from both containers, start a third call, and ask where the key was first placed and where it was last placed. | Verify the recalled evidence orders the blue box first and the red pouch last, with original media attached. |

To verify abstention, ask about a detail kept out of view (such as a serial number on the underside of the pouch) and confirm the system reports insufficient evidence. To test reversible memory changes, cognitively forget an eligible record in the demo, inspect the logged operation in history, and roll it back to restore recall eligibility.

## Replay timed recordings into events

Open the [best practices notebook](../examples/best_practices.ipynb) from `examples/` so its `media/` paths resolve. Use a Jupyter kernel with the [repository environment](../CONTRIBUTING.md#setup), `ffmpeg` on `PATH`, and a multimodal generation endpoint configured via environment variables so credentials stay out of the notebook file.

The notebook streams 14 prepared windows of the Sintel trailer while event segmentation and retrieval run concurrently. After inspecting the table mapping detected events to source windows, filter by occurrence bounds and play back the retrieved original video and audio through `read_asset()`. See [included media](../examples/media/README.md) for source and license details.

## Match the question to the evidence

| Scenario | Example question | Evidence to capture and verify |
| --- | --- | --- |
| Find an object | "Where did I put the spare key during the morning workshop?" | Store a timed image, video, or observation as an episode; query the workshop interval and show the latest supported placement. |
| Recall a conversation | "Which deadline did we agree on in the handoff?" | Store completed speech with its transcript under the same source provenance; replay the audio to verify exact wording. |
| Recap a time period | "What happened during the afternoon setup?" | Apply explicit `occurred_from` / `occurred_until` bounds and present matching episodes in source-time order. |
| Recall a routine | "What do I usually check before rehearsal?" | Corroborate recurring observations across independent captures, or store a host-confirmed checklist as procedural memory. |
| Understand the current scene | "What is on the workbench right now?" | Supply a fresh camera observation with an expiry window and label historical recall with its original timestamp. |
| Handle a change | "Where did I move the spare key after lunch?" | Capture the new placement with its own timestamp; review or correct derived claims through [memory management](memory-management.md#review-measure-and-undo). |

## Verify cross-session recall in Python

Install `sensemem[local]` as shown in the [quick start](quickstart.md#1-install-sensemem). The example below captures a morning observation, reopens the same store in a second session, settles the capture, adds a later move, and verifies time-bounded recall:

```python
from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory

from sensemem import Memory, MemoryCategory

morning = datetime(2026, 8, 20, 9, tzinfo=timezone.utc)
morning_end = morning + timedelta(minutes=5)
afternoon = morning + timedelta(hours=5)
query = "Where did Maya put the spare key?"
episode = MemoryCategory.EPISODIC

with TemporaryDirectory() as data_dir:
    config = {
        "data_dir": data_dir,
        "embedding": {
            "provider": "sentence-transformers",
            "model": "google/embeddinggemma-2",
            "modalities": ["text"],
        },
    }
    with Memory.from_config(config) as memory:
        captured = memory.capture(
            "Maya put the spare key in the blue toolbox.",
            memory_type=episode,
            occurred_at=morning,
            occurred_end=morning_end,
        )

    with Memory.from_config(config) as memory:
        assert memory.get(captured.id).content == captured.content
        assert not memory.search(query, memory_type=episode).hits
        memory.settle(memory_ids=[captured.id])
        moved = memory.add(
            "Maya moved the spare key to the cabinet.",
            memory_type=episode,
            occurred_at=afternoon,
        )
        for start, end, expected in (
            (morning, morning_end, captured),
            (afternoon, afternoon + timedelta(minutes=5), moved),
        ):
            hits = memory.search(
                query, memory_type=episode, occurred_from=start, occurred_until=end
            ).hits
            assert [hit.id for hit in hits] == [expected.id]
            print(hits[0].content)

        gap = memory.search(
            query, memory_type=episode, occurred_from=morning_end, occurred_until=afternoon
        )
        assert not gap.hits
```

The same query retrieves the blue toolbox for the morning window and the cabinet for the afternoon window. The gap query excludes both because occurrence intervals are half-open; see [event time and strict filters](memory-types-time-and-decay.md#event-time-and-strict-filters). Use a persistent `data_dir` with one live owner and compatible embedding settings to retain memory across launches.

## Connect capture, recall, and review

1. **Preserve source time and finality.** Record timezone-aware occurrence times and keep original media attached. Use `add()` when records must be searchable on return, `capture()` plus host-scheduled `settle()` when durable acceptance comes first, and checkpointed `AsyncEventStream` for continuous audio/video; see [automatic event segmentation](omni-streaming-and-interaction-memory.md#automatic-event-segmentation).
2. **Choose the right output surface.** Use `search()` for ranked evidence and strict occurrence filters, `compile()` for budgeted agent context, and `ask()` for grounded answers with explicit `AnswerResult.abstained` handling; see [Python memory operations](api/python-memory.md).
3. **Keep evidence reviewable.** Display record IDs and source times beside claims, use `read_asset()` to play back originals, and keep transcripts and event synopses clearly labelled as derived interpretations.
4. **Verify positive, out-of-window, and missing-evidence cases.** Test known evidence, queries outside the event window, and unanswerable questions after reopening the store. See the [model quality gate](model-quality.md) for repeatable real-model checks.
