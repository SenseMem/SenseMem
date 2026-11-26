# Continuous audio and video demo

The live demo connects a browser microphone and camera to SenseMem perception, speculative recall, and a [speech-to-speech](https://github.com/huggingface/speech-to-speech) Realtime server. SenseMem listens to user speech, observes timed camera frames, retrieves relevant history while input continues, and injects attributed text context into the speech session before it replies. Camera images stay inside SenseMem; the speech server receives only audio and SenseMem's text context.

## Run locally

Start your speech-to-speech server in a separate terminal. Provide your gateway key in `OPENAI_API_KEY` and replace the base URL and model name with your configured endpoint:

```bash
speech-to-speech serve \
  --device cuda --stt paraformer \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_speaker Aiden --qwen3_tts_language auto \
  --qwen3_tts_non_streaming_mode True \
  --llm_backend chat-completions \
  --responses_api_base_url https://api.example.com/v1 \
  --model_name your-chat-model \
  --enable_live_transcription --responses_api_stream \
  --responses_api_reasoning_effort none \
  --host 127.0.0.1 --port 8765 --thresh 0.35 --language auto
```

See the speech server's [LLM documentation](https://github.com/huggingface/speech-to-speech/blob/main/src/speech_to_speech/LLM/README.md) and [Realtime protocol](https://github.com/huggingface/speech-to-speech/blob/main/src/speech_to_speech/api/openai_realtime/README.md) for backend and event options.

Install the required extras and start the SenseMem demo from the repository root:

```bash
uv sync --locked --default-index https://pypi.org/simple \
  --all-groups --extra openai --extra server
export SENSEMEM_GENERATION_BASE_URL="https://api.example.com/v1"
export SENSEMEM_GENERATION_API_KEY="replace-with-your-key"
uv run --frozen python -m examples.live \
  --config examples/live/config.example.json --data-dir .sensemem/live-demo
```

To load credentials from a local environment file, use `uv run --env-file /path/to/your.env --frozen python -m examples.live ...`. Credentials are never sent to the browser.

Open `http://localhost:8787`, grant microphone and camera permissions, and start a call. For remote development, forward port 8787 over SSH (`ssh -L 8787:127.0.0.1:8787 your-host`). Use headphones to reduce acoustic feedback.

The demo defaults to listening on `127.0.0.1:8787`, connecting to `ws://127.0.0.1:8765/v1/realtime` (override with `--realtime-url` or `SENSEMEM_REALTIME_URL`), and storing memory in `.sensemem/live-demo`. Start the speech server before placing a call.

Update [the example config](../examples/live/config.example.json) with your multimodal embedding endpoint and multimodal generation model for event segmentation, formation, consolidation, and original-media review. See [provider fields](configuration/providers.md) and [EmbeddingGemma 2 setup](configuration/models.md#embeddinggemma-2).

One physical `data_dir` has one live SenseMem owner. Reuse the directory to test persistence across calls, or pass a fresh `--data-dir` for an isolated session.

## Walkthrough

1. **Capture observations.** Show an object (for example, placing keys into a container) and speak a fact you want remembered. Initial and changed visual windows commit through `AsyncCaptureStream(capture=True)` and settle in the background while the continuous recording streams into `AsyncEventStream`. Open any record in the experience panel to replay its original media.
2. **Recall across calls.** Hide the object, start a new call on the same `data_dir`, and ask where you placed it. SenseMem retrieves matching records, reviews original media asynchronously, injects bounded `memory_feedback` context into the speech session, and shows supporting originals in the recall panel.
3. **Inspect, correct, forget, and undo.** Open the profile view to inspect derived preferences and their supporting sources. Use the correction, confirmation, and cognitive-forgetting controls on eligible records, then inspect or roll back reversible operations in the history view.
4. **Share a changing scene and interrupt.** Ask what is currently visible while holding up an object; timed JPEG frames feed `AsyncOmniPrefetch` and live visual context. Speak during an agent reply to interrupt playback, cancel generation, and archive only the rendered audio prefix.

## Implementation

### Speech session and live context

The browser connects to same-origin `WebSocket /api/realtime`, which FastAPI relays to the speech server. Server VAD uses `create_response=false` so final ASR waits for the matching SenseMem recall (and an unexpired visual observation when the camera is enabled) to be acknowledged via session instructions before requesting a reply. Retrieval and initial visual context use a 15-second wait bound and context acknowledgement uses a 5-second bound; on timeout, the reply proceeds without claiming missing memory or vision.

Audio streams continuously as 24 kHz mono PCM16 via an AudioWorklet (`input_audio_buffer.append` / `response.output_audio.delta`). Completed user speech intervals with final ASR commit through `AsyncCaptureStream(capture=True)` as user-attributed WAV plus transcript, independently of longer audiovisual events.

### Perception and speculative recall

`WebSocket /api/perception` receives timed JPEG frames independently of audio. Meaningful visual changes commit through `AsyncCaptureStream(capture=True)` with the raw JPEGs as source media and the live summary separated in `interpretation`. Original audio, ASR hypotheses, and current frames feed `AsyncOmniPrefetch`; a separate worker rehydrates active IDs and reviews original media before pushing bounded `memory_feedback` into the speech session. Agent-only replies from the current call are excluded from external-fact feedback to prevent self-reinforcing loops.

### Continuous recording and dialogue archiving

The browser `MediaRecorder` streams continuous MP4 bytes over `WebSocket /api/media`. The server remuxes independent fragmented MP4 codec groups and feeds them into `AsyncEventStream.append()` with a checkpoint under `--data-dir/live-events/`, while background `process()` runs semantic event segmentation (`max_chunks=4`, 64 MiB active and pending staging bounds) and background `settle()` indexes and forms derived memories. Upload and codec buffers are bounded at 8 MiB. Committed records persist across restarts; resuming unfinished checkpoints explicitly uses the [SDK recovery path](omni-streaming-and-interaction-memory.md#resume-continuous-capture).

Agent replies archive only the PCM prefix actually rendered by Web Audio via `POST /api/dialogue`, keeping assistant output, rendered audio status, and feedback receipts distinct from user observations.

### Local application surface

These routes belong to the demo application, separate from SenseMem's supported `/v1` product API:

| Route | Purpose |
| --- | --- |
| `GET /api/config` | Local speech configuration and capabilities, without credentials. |
| `POST /api/session`, `DELETE /api/session` | Start a call with source/revision, or stop the call's capture. |
| `WebSocket /api/realtime` | Local speech-to-speech Realtime relay. |
| `WebSocket /api/media` | Continuous MP4 acceptance into `AsyncEventStream`. |
| `WebSocket /api/perception` | Independent timed-frame perception and sensory input. |
| `WebSocket /api/recall` | Background recall, source-reviewed feedback, and candidate subscription. |
| `GET /api/memories`, `GET /api/memories/{id}` | Records, sources, states, and version history. |
| `GET /api/memories/{id}/assets/{asset_id}` | Integrity-checked original media belonging to a record. |
| `GET /api/events` | Memory-view and processing notifications. |
| `POST /api/dialogue` | Attributed agent text, rendered WAV prefix, and delivery receipts. |
| `GET /api/operations` | Native operation history. |
| `POST /api/memories/{id}/corrections` | Correct and retire an eligible derived record. |
| `POST /api/memories/{id}/confirmations` | Record explicit evidence-backed response feedback. |
| `POST /api/memories/{id}/forget` | Native cognitive forgetting. |
| `POST /api/operations/{id}/rollback` | Roll back a reversible operation. |
