# Continuous audio and video demo

The demo follows **microphone + camera → SenseMem perception and early recall →
[speech-to-speech](https://github.com/huggingface/speech-to-speech) response**. SenseMem listens to
the audio, observes timed frames and retrieves relevant history while input continues. The speech
service reads this attributed text context before answering. Camera images are not sent to the
speech service. Validated visual windows and completed user speech enter memory independently of
the longer audiovisual recording. Durable memory formation runs in the background; a reply does
not wait for the whole video event to finish.

The speech service uses an STT → LLM → TTS pipeline with the OpenAI Realtime protocol. The demo
keeps collecting input during output, clears playback when interrupted, and can request a response
from a validated visual, audio or memory event. This is application-level duplex interaction;
it does not turn the pipeline into one model with native incremental audiovisual understanding.

## Run locally

Start your existing speech-to-speech installation in a separate terminal. Its LLM adapter reads
your gateway credential from `OPENAI_API_KEY`; keep the value in the process environment,
not a command-line argument. Replace `https://api.example.com/v1` below with your compatible
gateway. For a CUDA, Paraformer and Qwen TTS setup:

```bash
speech-to-speech serve \
  --device cuda --stt paraformer \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_speaker Aiden --qwen3_tts_language auto \
  --qwen3_tts_non_streaming_mode True \
  --llm_backend chat-completions \
  --responses_api_base_url https://api.example.com/v1 \
  --model_name qwen3.8-omni-flash \
  --enable_live_transcription --responses_api_stream \
  --responses_api_reasoning_effort none \
  --host 127.0.0.1 --port 8765 --thresh 0.35 --language auto
```

This command preserves the existing TTS synthesis mode. It does not claim that each TTS segment
is synthesized token by token. See the speech server's
[LLM documentation](https://github.com/huggingface/speech-to-speech/blob/main/src/speech_to_speech/LLM/README.md)
for backend configuration and its
[Realtime protocol](https://github.com/huggingface/speech-to-speech/blob/main/src/speech_to_speech/api/openai_realtime/README.md)
for supported events.

Install and start the SenseMem demo from this repository:

```bash
uv sync --locked --default-index https://pypi.org/simple \
  --all-groups --extra openai --extra server
export SENSEMEM_GENERATION_BASE_URL="https://api.example.com/v1"
uv run --frozen python -m examples.live \
  --config examples/live/config.example.json --data-dir .sensemem/live-demo
```

Set `SENSEMEM_GENERATION_API_KEY` in the demo server's environment for its memory models.
An existing environment file can be loaded with
`uv run --env-file /path/to/your.env --frozen python -m examples.live ...`.
Credentials are not returned to the browser.

Open `http://localhost:8787`, allow microphone and camera access, then start a call. For remote
development, use an SSH forward such as `ssh -L 8787:127.0.0.1:8787 your-host`. Audio, camera frames
and memory traffic use the same local page and forward. Headphones reduce acoustic feedback.

The upstream default is `ws://127.0.0.1:8765/v1/realtime`. Override it with `--realtime-url` or
`SENSEMEM_REALTIME_URL`. The demo listener defaults to `127.0.0.1:8787`, and its data directory
defaults to `.sensemem/live-demo`. Start the speech service before placing a call.

The [example memory config](../examples/live/config.example.json) uses the existing
`google/embeddinggemma-2` service at `http://127.0.0.1:8000/v1`, with 768-dimensional multimodal
`messages` embeddings. It configures `qwen3.8-omni-flash` for event segmentation, typed memory
formation, consolidation and original-media review. These asynchronous calls remain separate from
the speech pipeline. An explicitly configured memory credential takes precedence over the demo's
generation environment variable. See [provider configuration](configuration/providers.md) and
[EmbeddingGemma 2 setup](configuration/models.md#embeddinggemma-2).

One physical data directory has one running SenseMem owner. Keep the directory to retain committed
memories, or choose another for an independent experiment. Startup rejects an incompatible stored
configuration or an existing owner. The demo does not delete local data or automatically resume
historical unfinished event checkpoints.

## Try the memory experience

### Remember what you showed and said

Show your keys clearly, then put them in a distinctive container. Say something you want remembered,
such as “我们的排练约在周四下午三点。” Initial and meaningfully changed visual windows are saved
while the activity continues. Background settlement makes those sources searchable and runs native
event and fact formation. In **经历**, expand a resulting record and replay its original media.
A model synopsis is an interpretation;
the original is the evidence. For a multi-part event, a shared synopsis does not mean every detail
is verified in each clip.

After processing, hide the keys and ask where you placed them. SenseMem can retrieve and review
the retained source, then update the current conversation's memory context. Compare the actual
answer with the original in the panel. Repeat in a new call using the same directory: the model's
earlier conversation and a still-visible object do not establish persistent memory retrieval.

### Let its understanding change with you

Say “我最近更喜欢喝茶。” Open **了解你** after processing to inspect the interpretation and sources.
Use **纠正** on an eligible interpretation when its wording is wrong. The native operation retains
the correction source and lineage. **确认回应** records explicit feedback. **暂时遗忘** removes an
eligible record from recall while retaining its media. In **历史**, inspect or undo an operation
when the native operation remains reversible; a newer dependent change can prevent reversal.

A temporary emotion need not become a stable personality claim. Inferred traits need at least
two independent evidence groups before active recall. Several codec groups from one call share
one source and do not become independent personality witnesses.

### Recall while the conversation continues

Arriving audio, ASR hypotheses and current images can update temporary recall. **正在回忆** means
retrieval is pending. A separate source review produces bounded `memory_feedback` facts with IDs,
occurrence times and uncertainty. Original-media review is marked `source_review`; a record-text
excerpt is explicitly unverified. These temporary results do not create formal memory records.

Eligible facts update the same speech session through its native text-context mechanism. The
demo no longer draws historical text into the camera image. A service acknowledgement means the
context was accepted; it does not prove the model used the facts or answered correctly. Check the
answer against its retained sources. Replacement and withdrawal remove obsolete application
context, but cannot erase facts already mentioned in a conversation or a person's hearing.

The page displays actual user transcription events from the speech service. Interim hypotheses
may change. A completed native VAD interval with final ASR and all corresponding original PCM is
captured through `AsyncCaptureStream(capture=True)` as a user-attributed WAV and transcript. This
does not wait for a visual event to conclude. The transcript remains ASR output, with unverified
word alignment, rather than proof of the user's intended meaning. Partial or empty ASR and an
unfinished interval do not create a completed user speech record. Generated Agent captions remain
distinct from both user statements and actual played output.

### Share a changing scene and interrupt

Ask “你现在看到了什么？” and show an unnamed object. Timed JPEG frames enter SenseMem perception
and multimodal prefetch. Its observations and relevant historical evidence enter the speech session
as attributed text. Perception continues while the Agent speaks, so a validated scene change can
update pending context or request a correction. The camera preview
contains only video or its honest off state, without status overlays or a recall button.

Speak during a reply: microphone capture continues, queued playback clears, generation is
cancelled, and the played position is reported to the speech server. A useful memory or scene
event can request a short response at an appropriate pause. User speech has priority; quiet time,
expiry, revision checks, deduplication and cooldown prevent every frame or record from speaking.

## Implementation

### Speech session and live context

The browser connects to same-origin `WebSocket /api/realtime`; FastAPI relays to the local speech
service. `session.created` is followed by `session.update` with instructions, tools, audio format
and server VAD. Input begins after the effective `session.updated` acknowledgement. Server VAD
uses `create_response=false`: final ASR schedules a reply after the matching final SenseMem recall
has been acknowledged through native session instructions. Partial hypotheses still initiate
recall earlier. A new user turn supersedes an older waiting reply; microphone capture and output
interruption remain independent of this scheduling boundary.

With the camera enabled, an answer also waits for an acknowledged, unexpired SenseMem visual
observation, then reuses that observation within its stated lifetime. Retrieval and first visual
context have a 15-second waiting bound; a context acknowledgement has a 5-second bound. If these
fail, the page reports the missing context and the reply must not claim recalled facts or current
vision that were not supplied. These are failure deadlines, not claimed normal response latency.

The AudioWorklet continuously emits 20 ms mono PCM16 input at 24 kHz through
`input_audio_buffer.append`. Output uses 24 kHz mono PCM16
`response.output_audio.delta`; transcript deltas and `response.done` have separate lifecycles.
No one-second Float32 aggregation or ModelBest-specific handshake remains. The native service
provides user speech-start/stop and transcription events, explicit response creation, cancellation,
conversation-item acknowledgement and function calls.

No `input_image` conversation items are created. SenseMem owns timed-frame interpretation, early
retrieval and original audiovisual storage. The speech service receives user audio for ASR and
speech interaction, plus SenseMem's replaceable text context. This demonstrates a memory-augmented
STT → LLM → TTS pipeline, not native incremental video understanding inside that LLM request.

`POST /api/session` returns the call's `source_id` and current `memory_revision`, without an
arbitrary startup history snapshot. Live recall rehydrates current native record IDs and reviews
original evidence asynchronously. Its bounded text result reaches the ongoing speech session;
tool requests use native `function_call_output`, followed by one eligible `response.create` after
the originating response completes. User speech delays proactive and tool follow-up responses.
Stored evidence remains untrusted data and cannot replace the fixed runtime policy.

Context acceptance, response delivery and correct grounding are separate measurements. An event
received after an LLM request started cannot retroactively change that request. Ordinary relevant
updates apply to a subsequent answer; an urgent validated correction can cancel the current
response, clear local audio and request a replacement using the new evidence. A context revision
does not itself prove that the model followed it.

### Independent perception and temporary recall

`WebSocket /api/perception` accepts timed, sequenced JPEGs and interaction state independently of
speech input and playback. Its bounded worker reviews actual observed windows. The live result
updates temporary recall immediately. Camera-off clears the current view; a frame acknowledgement
confirms receipt rather than understanding.

The first validated window and meaningful changes also enter public
`AsyncCaptureStream(capture=True)` through an independent background writer. Each source contains
the actual JPEGs, frame sequences, and capture timestamps; the perception summary stays separate
in `interpretation`. The capture boundary is that finite observed window, not the end of the
ongoing activity. At most two reviewed windows wait behind an active write: the first accepted
window and the latest change. Intermediate reviews may be coalesced; the continuous recording
remains the complete audiovisual source.

Capture commits the source first. Background `settle()` then performs ordinary enrichment,
indexing, and native formation, which can produce an event or supported facts. These sources can
enter recall before the longer activity finishes. A temporary perception summary alone is never
submitted as original evidence, and observation does not force the separate audiovisual event to
complete. Capture and formation progress remain distinct from live perception.

Original audio and the latest image feed public `AsyncOmniPrefetch` separately. Results preserve
revisions, source times and returned hits. New ASR hypotheses can update textual prefetch without
being saved as final records. One request drains while a newer snapshot replaces pending work.
Sensory retrieval preserves the input's original timestamps even when a model takes longer than
the camera freshness window. Historical recall and source review receive a delivery lifetime after
completion; that does not renew an old picture's claim to describe the present. Already expired
queued review input is discarded. The feedback worker rehydrates IDs before and after source review, rejecting pending, forgotten,
hidden or retired records. Agent-only output from this call is excluded from new external-fact
feedback to avoid self-reinforcing response loops.

Source, revision and expiry checks withdraw stale feedback and queued initiatives. Candidate
publication, native context acknowledgement and a spoken reply remain different states. The model
and configured pipeline still determine answer quality and end-to-end latency.

### Original continuous recording

One browser `MediaRecorder` sends continuous MP4 bytes through `WebSocket /api/media`. Its 250 ms
flush is transport, not a standalone clip or semantic event. A one-second keyframe hint helps
assemble decodable codec groups without restarting the recorder. The server preserves encoded
packets, timestamps and audio/video offsets while remuxing independent fragmented MP4 observations.

Each observation enters public `AsyncEventStream.append()` for durable acceptance before model
work. Background `process()` assesses the native window; `capture=True` commits completed events.
Background `settle()` makes them searchable and runs formation, then bounded consolidation. Codec
boundaries and EOF do not force semantic completion. An unfinished event stays retained.

The final media `accepted` message means originals through EOF reached durable acceptance, not
that analysis or indexing finished. Stop releases devices and lets accepted work drain. Upload and
codec buffers are bounded at 8 MiB; pending staging retains a 64 MiB bound. With a checkpoint,
unfinished and ready originals remain content-addressed disk references. The 64 MiB active budget
applies to hydrated analysis windows and individual storage parts, rather than the accumulated
length of an unfinished activity. The demo configures `max_chunks=4`, bounding both each native
assessment and each completed storage part. A long semantic event retains its shared ID and
whole-event summary across parts, without combining dozens of videos in one enrichment request.
A memory-side capacity failure preserves accepted
originals and allows speech to continue. It does not raise limits or end voice to hide backlog.

Checkpoints under `--data-dir/live-events/` are v5 manifests with sibling `.json.media/` directories.
The manifest pins `max_chunks` and `max_buffer_bytes` to preserve storage-part boundaries; explicit
SDK resumption requires the same values, model, and recipe. Older v1–v4 checkpoint formats are
rejected without migration.
New calls create new streams. The demo does not reopen historical checkpoints automatically.
Committed records remain readable after restart. Explicit reopening is a separate
[public SDK operation](omni-streaming-and-interaction-memory.md#resume-continuous-capture).

The Qwen memory adapter may make an AAC transport copy for MP4 with Opus audio, preserving video
packets, timestamps and audio offset. Original bytes, digests, intervals and replay remain intact.
This adaptation does not alter live speech input. See
[automatic event segmentation](omni-streaming-and-interaction-memory.md#automatic-event-segmentation).

### Agent reply memory

Agent replies use separate attributed dialogue records. The client archives only PCM reported
rendered by the Web Audio clock as WAV, with completed, interrupted or stopped status. Interrupted
replies retain their rendered prefix and exclude unplayed trailing audio. Generated text remains
separate: it is not a word-exact transcript of the played prefix or proof that a person heard it.

`POST /api/dialogue` retains assistant attribution, rendered sample counts, playback status and
matching feedback receipts. A receipt establishes accepted context and its provenance, not verified
model adoption. The user's original recording and Agent rendered WAV remain distinct sources.
Completed user speech also has its own native capture record, attributed to the same physical call
source. It can be recalled while the original audiovisual event is still unfinished; Agent output
cannot stand in for that user evidence.
Failed dialogue storage displays a safe notice while speech continues.

The memory view presents dialogue as “你 / SenseMem”, including existing records with structured
dialogue metadata. Playback status and transcription source sit in the collapsed “记录信息” section.
Cards include a `dialogue` projection alongside the unchanged source `content` and `parts`;
playback provenance remains in stored metadata rather than repeated in conversation text.
This projection applies only to original observations. Derived memories show their own content;
source dialogue and media remain available under a collapsed source row. The version history
omits the selected record because its content already appears above.

### Local application surface

These routes belong to the demo, separately from SenseMem's supported `/v1` product API.

| Route | Purpose |
| --- | --- |
| `GET /api/config` | Local speech configuration and capabilities, without credentials. |
| `POST /api/session`, `DELETE /api/session` | Start with source/revision, or stop the call's capture. |
| `WebSocket /api/realtime` | Local speech-to-speech Realtime relay. |
| `WebSocket /api/media` | Original continuous MP4 acceptance into the native stream. |
| `WebSocket /api/perception` | Independent timed-frame perception and sensory input. |
| `WebSocket /api/recall` | Background recall, source-reviewed feedback and candidate subscription. |
| `GET /api/memories`, `GET /api/memories/{id}` | Actual records, sources, states and history. |
| `GET /api/memories/{id}/assets/{asset_id}` | Integrity-checked originals belonging to that record. |
| `GET /api/events` | Memory-view and processing notifications. |
| `POST /api/dialogue` | Attributed Agent text, rendered WAV/prefix and delivery receipts. |
| `GET /api/operations` | Native operation history. |
| `POST /api/memories/{id}/corrections` | Correct and retire an eligible interpretation. |
| `POST /api/memories/{id}/confirmations` | Explicit evidence-backed response feedback. |
| `POST /api/memories/{id}/forget` | Native cognitive forgetting. |
| `POST /api/operations/{id}/rollback` | Reverse an eligible operation and report the result. |

The panel stays subscribed after hangup so settlement can update records. Eligibility belongs to
the native kernel. There is no physical-delete control. One call source groups original media and
attributed Agent output without pretending they are independent witnesses.

## Evaluation

Use the following acceptance checks with the configured speech and memory services. Allocate a
fresh physical directory per case, except for the persistence case. Record the SenseMem and speech
repository revisions, configured and returned models, available immutable revisions, browser and
device conditions, input fixtures, prompt/config digests, pass criteria and actual outcomes. Keep
keys and provider bodies out of ordinary logs. See the [model quality gate](model-quality.md).

| Case | Procedure and pass criteria |
| --- | --- |
| Current scene | Show a distinctive object and ask a spoken color or location question. Confirm final ASR and a matching, unexpired SenseMem visual-context acknowledgement precede `response.create`, then compare the spoken answer with the observed frames. Confirm the speech service receives no camera images. |
| Memory before call end | Show a meaningful change while the call continues. Inspect the captured window's original JPEGs, timestamps and derived event. Confirm the completed source becomes searchable without ending the call or forcing the longer audiovisual event complete. |
| Persistence across calls | End the first call, allow accepted work to drain, and close every owner. Reopen the same directory for a new call with the camera disabled and ask about a distinctive earlier event or statement. Trace the selected evidence to the prior user's original media, verify it remains readable, and compare the answer with that evidence. |
| Context updates | Supply a relevant new fact, then correct or withdraw it. Capture the context revision and acknowledgement used when each response is requested. Check that a subsequent answer follows current evidence, an urgent replacement clears obsolete playback, and withdrawn evidence no longer supports new claims. |
| Proactive response | Create an eligible visual or memory event during a suitable pause. Check that it produces useful speech, respects user speech and cooldowns, and does not repeat after its source expires or is superseded. |
| Interruption and archives | Interrupt an audible reply with actual microphone input. Check input continues, playback clears and generation is cancelled. Replay the retained Agent WAV and confirm it contains only the rendered prefix; inspect the separately attributed original user WAV and audiovisual source. |
| Shutdown and presentation | Stop the call and check devices and owners release, accepted originals remain readable, and pending work drains or reports its retained checkpoint. Check desktop and narrow mobile layouts for overflow and inspect page, HTTP and protocol errors. |

Measure first audible output, physical interruption, durable acceptance, event completion and
searchability separately, with explicit clock boundaries. Report failed cases and unfinished
events as such. A context acknowledgement establishes delivery, and a retrospective recall replay
establishes what the store can retrieve at that replay; neither recovers the exact evidence used
by an earlier model request. Inspect request-time provenance when claiming that a particular source
influenced an actual answer. Conversation history alone cannot establish persistent-memory recall.

These checks define acceptance conditions rather than report a new qualification run. Successful
sockets, mocked memory dependencies and deterministic tool selection establish only the exercised
wiring. Synthetic media can exercise the real services but cannot qualify physical microphones,
speaker echo or human interruption. Bounded scene and recall cases do not establish general model
accuracy or proactive, correction and withdrawal quality. Physical-device behavior, general
proactive/correction/withdrawal quality and end-to-end latency improvement remain unqualified.
The configured model aliases provide no immutable revision guarantee; record that limitation
whenever the provider cannot return one.
