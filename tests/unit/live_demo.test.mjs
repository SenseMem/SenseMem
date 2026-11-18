import assert from "node:assert/strict";
import test from "node:test";
import { CallSession } from "../../examples/live/static/session.js";
import { MemoryLink } from "../../examples/live/static/memory-link.js";
import { PcmAudio } from "../../examples/live/static/pcm-audio.js";
import { VisualCapture } from "../../examples/live/static/visual-capture.js";

globalThis.WebSocket = class {
  static OPEN = 1; static CONNECTING = 0;
  constructor() { this.readyState = 1; }
  send() {}
  close() { this.readyState = 3; }
};
globalThis.location = { href: "http://localhost:8787/" };

function fixture(t) {
  const sent = [], events = [], writes = [], recall = [];
  const tracks = ["audio", "video"].map(kind => ({ kind, enabled: true, stop() {} }));
  const call = new CallSession({ video: {}, audio: {}, config: { realtime_ready: true,
    session: { type: "realtime", instructions: "base policy" },
    recall_unavailable_instructions: "No recalled evidence. Diagnostic data:\n",
    context_unavailable_instructions: "Do not use previous recalled evidence.",
    visual_unavailable_instructions: "No current visual evidence is available.",
  } });
  Object.assign(call, { active: true, epoch: 1, phase: "listening", mediaStarted: true,
    sourceId: "live:" + "a".repeat(32), memoryRevision: 0, sensoryRevision: -1, asrRecallRevision: -1,
    voiceMs: 0, quietMs: 0, userSpeaking: false, userAwaitingFinal: false,
    inputTurns: new Map(), toolCalls: new Set(), toolResults: new Map(), toolFollowups: new Set(),
    completedResponses: new Set(), answeredTurns: new Set(),
    context: { currentTime: 0 },
    stream: { getAudioTracks: () => [tracks[0]], getVideoTracks: () => [tracks[1]], getTracks: () => tracks },
    clock: { start: performance.now() - 1500, sequence: 0,
      next() { return { sequence: ++this.sequence, timestamp_ms: 1500 }; } },
    channel: { readyState: 1, bufferedAmount: 0, send: raw => sent.push(JSON.parse(raw)), close() { this.readyState = 3; } },
    recallSocket: { readyState: 1, send: raw => recall.push(JSON.parse(raw)), close() {} },
    pcm: { pending: false, nextTime: 0, append(audio) { this.lastAudio = audio; this.pending = true; this.nextTime += 1; },
      playedFor: () => 120, clear() { this.pending = false; this.cleared = true; return 120; }, takePlayback() { return null; } },
  });
  for (const name of ["caption", "error", "memorychange", "sensoryrecall", "perception"]) {
    call.addEventListener(name, ({ detail }) => events.push({ name, detail }));
  }
  call.checked = async (url, options) => { writes.push({ url, options });
    return { json: async () => ({ context: "source reviewed", hits: [], memory_revision: 0 }) }; };
  t.after(() => { clearTimeout(call.outputWatch); call.clearAnswer(); });
  return { call, sent, events, writes, recall, tracks };
}

function memoryLink(call) {
  call.memoryLink = new MemoryLink({ sourceId: call.sourceId, current: () => call.current(1),
    instructions: call.config.session.instructions, send: value => call.send(value),
    emit: (name, detail) => call.emit(name, detail), interrupt: () => call.interrupt(),
    getState: () => ({ userSpeaking: call.userSpeaking, playing: call.pcm.pending,
      generating: Boolean(call.responseId || call.cancellingId || call.responseRequested),
      awaitingAnswer: call.userAwaitingFinal || Boolean(call.pendingAnswer) }) });
  call.openRecall(1);
  const deliver = event => call.recallSocket.onmessage({ data: JSON.stringify(event) });
  const result = (extra = {}) => ({ type: "result", source_id: call.sourceId, memory_revision: call.memoryRevision,
    turn_id: call.userTurnId, revision: 2, query: call.userText, context: "钥匙在蓝色盒子里。", hits: [],
    final: true, input_final: true, ...extra });
  const ack = async () => call.handle({ type: "session.updated", session: { instructions: call.memoryLink.pending.instructions } }, 1);
  return { deliver, result, ack };
}

async function answer(call, id = "r1", text = "回答") {
  await call.handle({ type: "response.created", response: { id } }, 1);
  await call.handle({ type: "response.output_audio_transcript.delta", response_id: id, item_id: "item-" + id, delta: text }, 1);
  await call.handle({ type: "response.output_audio.delta", response_id: id, item_id: "item-" + id, delta: "PCM" }, 1);
}

test("GA handshake applies policy once and starts media only after session.updated", async t => {
  const { call, sent } = fixture(t);
  call.mediaStarted = false;
  let started = 0;
  call.startMedia = async () => { started++; call.mediaStarted = true; };
  await call.handle({ type: "session.created" }, 1);
  await call.handle({ type: "session.created" }, 1);
  assert.equal(started, 0);
  assert.deepEqual(sent, [{ type: "session.update", session: { type: "realtime", instructions: "base policy" } }]);
  await call.handle({ type: "session.updated", session: {} }, 1);
  await call.handle({ type: "session.updated", session: {} }, 1);
  assert.equal(started, 1);
});

test("audio remains continuous during reply generation without image/OCR aggregation", t => {
  const { call, sent } = fixture(t);
  call.responseId = "answer";
  call.sendInput({ type: "input_audio_buffer.append", audio: "packet1" }, 1);
  call.sendInput({ type: "input_audio_buffer.append", audio: "packet2" }, 1);
  assert.deepEqual(sent, [
    { type: "input_audio_buffer.append", audio: "packet1", event_id: "av_1_1500" },
    { type: "input_audio_buffer.append", audio: "packet2", event_id: "av_2_1500" },
  ]);
});

test("speculative snapshots replace guesses and final ASR feeds authoritative multimodal recall", async t => {
  const { call, events, recall } = fixture(t);
  call.latestFrame = { sequence: 1, timestamp_ms: 1400, image: "data:image/jpeg;base64,frame" };
  await call.handle({ type: "speech_to_speech.input_audio_transcription.snapshot", item_id: "u1", transcript: "绿色" }, 1);
  await call.handle({ type: "conversation.item.input_audio_transcription.delta", item_id: "u1", delta: "绿" }, 1);
  await call.handle({ type: "speech_to_speech.input_audio_transcription.snapshot", item_id: "u1", transcript: "蓝色盒子" }, 1);
  await call.handle({ type: "conversation.item.input_audio_transcription.completed", item_id: "u1", transcript: "钥匙在蓝色盒子" }, 1);
  assert.deepEqual(events.filter(x => x.name === "caption").map(x => x.detail.text), ["绿色", "绿色", "蓝色盒子", "钥匙在蓝色盒子"]);
  assert.equal(recall.at(-1).type, "final");
  assert.equal(recall.at(-1).text, "钥匙在蓝色盒子");
  assert.equal(recall.at(-1).image, call.latestFrame.image);
  assert.deepEqual(call.ensureResponse("r1").user, { item_id: "u1", text: "钥匙在蓝色盒子" });
});

test("ASR recall results feed current native context only for their active turn, source and revision", t => {
  const { call, events } = fixture(t);
  const feedback = [];
  call.userTurnId = "u1";
  call.memoryLink = { handle: event => feedback.push(event) };
  call.openRecall(1);
  const deliver = event => call.recallSocket.onmessage({ data: JSON.stringify(event) });
  const result = { type: "result", source_id: call.sourceId, memory_revision: 0,
    turn_id: "u1", revision: 2, query: "钥匙", context: "临时检索结果", hits: [], final: false };
  deliver({ ...result, turn_id: "old" });
  deliver({ ...result, source_id: "other" });
  deliver({ ...result, memory_revision: -1 });
  assert.equal(feedback.length, 0);
  deliver(result);
  assert.equal(feedback[0].type, "asr_recall");
  assert.ok(Date.parse(feedback[0].expires_at) > Date.now());
  deliver({ ...result, revision: 1 });
  assert.equal(feedback.length, 1);
  deliver({ type: "pending", turn_id: "u1", revision: 3, query: "新问题" });
  assert.equal(feedback[1].type, "asr_recall_clear");
  assert.ok(events.some(event => event.name === "sensoryrecall" && event.detail.status === "pending"));
});

test("committed visual memory updates the current call without accepting another source", t => {
  const { call, events } = fixture(t);
  const visual = [];
  call.addEventListener("visualmemory", ({ detail }) => visual.push(detail));
  call.openRecall(1);
  const deliver = event => call.recallSocket.onmessage({ data: JSON.stringify(event) });
  const observation = { type: "visual_memory", source_id: call.sourceId,
    perception_id: "observed-window", memory_id: "original-visual-memory", status: "saved" };
  deliver({ ...observation, source_id: "another-call" });
  assert.deepEqual(visual, []);
  assert.ok(!events.some(event => event.name === "memorychange"));
  deliver(observation);
  assert.deepEqual(visual, [observation]);
  assert.equal(events.filter(event => event.name === "memorychange").length, 1);
});

test("a user response waits for its exact final recall context acknowledgement while audio continues", async t => {
  const { call, sent, tracks } = fixture(t);
  tracks[1].enabled = false;
  const f = memoryLink(call);
  await call.handle({ type: "conversation.item.input_audio_transcription.completed", item_id: "u1", transcript: "钥匙在哪" }, 1);
  assert.equal(sent.length, 0);
  call.sendInput({ type: "input_audio_buffer.append", audio: "continuing" }, 1);
  f.deliver(f.result({ final: false }));
  await f.ack();
  assert.ok(!sent.some(event => event.type === "response.create"));
  f.deliver(f.result());
  assert.ok(!sent.some(event => event.type === "response.create"));
  await call.handle({ type: "session.updated", session: { instructions: "old context" } }, 1);
  assert.ok(!sent.some(event => event.type === "response.create"));
  await f.ack();
  assert.deepEqual(sent.at(-1), { type: "response.create", response: { metadata: { sensemem_user_turn_id: "u1" } } });
  call.flushAnswer();
  assert.equal(sent.filter(event => event.type === "response.create").length, 1);
});

test("camera-enabled first response also waits for an acknowledged timed SenseMem visual observation", async t => {
  const { call, sent } = fixture(t);
  const f = memoryLink(call);
  await call.handle({ type: "conversation.item.input_audio_transcription.completed", item_id: "u1", transcript: "你看到什么" }, 1);
  f.deliver(f.result());
  await f.ack();
  assert.ok(!sent.some(event => event.type === "response.create"));
  f.deliver({ type: "visual_context", source_id: call.sourceId, memory_revision: 0,
    perception_id: "blue-square", occurred_at: new Date(Date.now() - 300).toISOString(),
    occurred_end: new Date(Date.now() - 100).toISOString(), expires_at: new Date(Date.now() + 5000).toISOString(),
    context: "SenseMem observed a blue square at the stated past capture time." });
  assert.ok(sent.at(-1).session.instructions.includes("blue square"));
  assert.ok(!sent.some(event => event.type === "response.create"));
  await f.ack();
  assert.equal(sent.at(-1).type, "response.create");
  assert.ok(!sent.some(event => event.item?.content?.some(part => part.type === "input_image")));
});

test("a native memory write during final recall invalidates old ACK and accepts the refreshed result", async t => {
  const { call, sent, tracks } = fixture(t);
  tracks[1].enabled = false;
  const f = memoryLink(call);
  await call.handle({ type: "conversation.item.input_audio_transcription.completed", item_id: "u1", transcript: "以前放哪" }, 1);
  f.deliver(f.result());
  const old = call.memoryLink.pending.instructions;
  f.deliver({ type: "invalidated", memory_revision: 1 });
  f.deliver({ type: "memory_changed", memory_revision: 1 });
  await call.handle({ type: "session.updated", session: { instructions: old } }, 1);
  assert.ok(!sent.some(event => event.type === "response.create"));
  await f.ack();
  f.deliver(f.result({ revision: 3 }));
  await f.ack();
  assert.equal(sent.filter(event => event.type === "response.create").length, 1);
  assert.equal(call.memoryRevision, 1);
});

test("new speech withdraws a waiting answer and late old-turn evidence cannot trigger generation", async t => {
  const { call, sent, tracks } = fixture(t);
  tracks[1].enabled = false;
  const f = memoryLink(call);
  await call.handle({ type: "conversation.item.input_audio_transcription.completed", item_id: "u1", transcript: "旧问题" }, 1);
  f.deliver(f.result());
  const old = call.memoryLink.pending.instructions;
  await call.handle({ type: "input_audio_buffer.speech_started", item_id: "u2" }, 1);
  await call.handle({ type: "session.updated", session: { instructions: old } }, 1);
  f.deliver(f.result({ turn_id: "u1" }));
  assert.equal(call.pendingAnswer, null);
  assert.ok(!sent.some(event => event.type === "response.create"));
});

test("recall failure replaces prior evidence with an explicit unavailable context before answering", async t => {
  const { call, sent, tracks, events } = fixture(t);
  tracks[1].enabled = false;
  const f = memoryLink(call);
  await call.handle({ type: "conversation.item.input_audio_transcription.completed", item_id: "u1", transcript: "记得什么" }, 1);
  f.deliver({ type: "error", code: "recall_failed", turn_id: "u1", final: true });
  assert.equal(sent.at(-1).type, "session.update");
  assert.ok(sent.at(-1).session.instructions.includes("No recalled evidence"));
  assert.ok(!sent.some(event => event.type === "response.create"));
  await f.ack();
  assert.equal(sent.at(-1).type, "response.create");
  assert.ok(events.some(event => event.name === "error" && event.detail.includes("不会")));
});

test("missing recall and missing context acknowledgement have a bounded safe fallback", async t => {
  const { call, sent, tracks } = fixture(t);
  tracks[1].enabled = false;
  const timers = [];
  const previous = globalThis.setTimeout;
  globalThis.setTimeout = (callback, delay) => { timers.push({ callback, delay }); return undefined; };
  t.after(() => { globalThis.setTimeout = previous; });
  memoryLink(call);
  await call.handle({ type: "conversation.item.input_audio_transcription.completed", item_id: "u1", transcript: "等待太久" }, 1);
  timers.find(timer => timer.delay === 15000).callback();
  assert.ok(!sent.some(event => event.type === "response.create"));
  timers.find(timer => timer.delay === 5000).callback();
  assert.equal(sent.at(-1).type, "response.create");
  assert.ok(sent.at(-1).response.instructions.includes("Do not use previous recalled evidence"));
});

test("camera frames go to SenseMem while the native speech channel receives audio only", async t => {
  const { call, sent } = fixture(t);
  const oldStart = PcmAudio.prototype.start, oldVisualStart = VisualCapture.prototype.start;
  const oldDocument = globalThis.document;
  PcmAudio.prototype.start = async function() { this.stream = {}; };
  VisualCapture.prototype.start = function() {};
  globalThis.document = { createElement: () => ({}) };
  t.after(() => { PcmAudio.prototype.start = oldStart; VisualCapture.prototype.start = oldVisualStart; globalThis.document = oldDocument; });
  call.meter = () => null;
  call.audio.play = async () => {};
  call.record = () => {};
  await call.startMedia(1);
  call.visualCapture.onFrame({ sequence: 1, timestamp_ms: 1400, image: "data:image/jpeg;base64,frame" });
  call.sendInput({ type: "input_audio_buffer.append", audio: "PCM" }, 1);
  assert.equal(call.latestFrame.image, "data:image/jpeg;base64,frame");
  assert.deepEqual(sent.map(event => event.type), ["input_audio_buffer.append"]);
});

test("barge-in clears immediately, cancels then truncates once and archives only rendered audio", async t => {
  const { call, sent, writes } = fixture(t);
  call.pcm.takePlayback = () => ({ sample_rate: 24000, rendered_samples: 2880, pcm_s16le: "prefix" });
  await answer(call);
  for (let i = 0; i < 5; i++) call.updateVoice(.06, 1);
  await call.handle({ type: "input_audio_buffer.speech_started" }, 1);
  await call.dialogueWrites.get("r1");
  assert.deepEqual(sent, [
    { type: "response.cancel", response_id: "r1" },
    { type: "conversation.item.truncate", item_id: "item-r1", content_index: 0, audio_end_ms: 120 },
  ]);
  assert.equal(call.pcm.cleared, true);
  assert.equal(call.cancellingId, "r1");
  await call.handle({ type: "response.output_audio.delta", response_id: "r1", delta: "late" }, 1);
  assert.equal(call.pcm.lastAudio, "PCM");
  assert.equal(JSON.parse(writes[0].options.body).playback.status, "interrupted");
  await call.handle({ type: "response.done", response: { id: "r1", status: "cancelled" } }, 1);
  assert.equal(call.cancellingId, null);
});

test("tool results wait for item ACK, completed owner and user final before follow-up", async t => {
  const { call, sent, writes } = fixture(t);
  call.userAwaitingFinal = true;
  call.responseId = "tools";
  await call.executeTool({ type: "response.function_call_arguments.done", response_id: "tools", call_id: "c1",
    name: "search_memory", arguments: '{"query":"钥匙"}' }, 1);
  assert.equal(writes[0].url, "/api/search");
  assert.deepEqual(JSON.parse(writes[0].options.body), { query: "钥匙" });
  assert.equal(sent.length, 1);
  await call.handle({ type: "response.done", response: { id: "tools", status: "completed" } }, 1);
  assert.equal(sent.length, 1);
  await call.handle({ type: "conversation.item.created", item: { type: "function_call_output", call_id: "c1" } }, 1);
  assert.equal(sent.length, 1);
  call.userAwaitingFinal = false;
  call.flushTools();
  assert.equal(sent.length, 2);
  assert.deepEqual(sent[1], { type: "response.create", response: { metadata: { s2s_tool_followup_call_ids: '["c1"]' } } });
  call.flushTools(); assert.equal(sent.length, 2);
});

test("cancelled tool owner never causes a spurious follow-up", async t => {
  const { call, sent } = fixture(t);
  await call.executeTool({ response_id: "old", call_id: "c1", name: "search_memory", arguments: '{"query":"钥匙"}' }, 1);
  await call.handle({ type: "conversation.item.created", item: { type: "function_call_output", call_id: "c1" } }, 1);
  await call.handle({ type: "response.done", response: { id: "old", status: "cancelled" } }, 1);
  assert.equal(sent.length, 1);
});

test("completed audio-only output and evidence receipts are archived after playback", async t => {
  const { call, writes } = fixture(t);
  call.memoryLink = { responseStarted: () => ({ input_id: "memory_0_1", feedback_id: "f1",
    memory_revision: 0, memory_ids: ["record"], action: "quiet", event_id: "memory_0_1", operation: "replace" }) };
  call.pcm.takePlayback = () => ({ sample_rate: 24000, rendered_samples: 48000, pcm_s16le: "A".repeat(128000) });
  await call.handle({ type: "response.output_audio.delta", response_id: "audio-only", item_id: "i1", delta: "PCM" }, 1);
  await call.handle({ type: "response.done", response: { id: "audio-only", status: "completed" } }, 1);
  assert.equal(writes.length, 0);
  call.context.currentTime = 2; call.pcm.pending = false;
  await call.saveDialogue(call.responses.get("audio-only"), 1);
  const body = JSON.parse(writes[0].options.body);
  assert.equal(body.assistant.text, "");
  assert.equal(body.playback.status, "completed");
  assert.equal(body.feedback_receipts[0].input_id, "memory_0_1");
  assert.equal(writes[0].options.keepalive, false);
});

test("memory write failures leave audio input active and show a recoverable message", async t => {
  const { call, sent, events } = fixture(t);
  call.checked = async () => { throw new Error("private body"); };
  await call.handle({ type: "response.output_audio_transcript.delta", response_id: "r1", delta: "完整" }, 1);
  await call.handle({ type: "response.done", response: { id: "r1", status: "completed" } }, 1);
  await call.dialogueWrites.get("r1");
  call.sendInput({ type: "input_audio_buffer.append", audio: "PCM" }, 1);
  assert.equal(call.active, true);
  assert.equal(sent.at(-1).type, "input_audio_buffer.append");
  assert.ok(events.some(x => x.name === "error" && !x.detail.includes("private")));
});

test("stale session events and network backpressure cannot silently change the current call", async t => {
  const { call, sent, events } = fixture(t);
  await call.handle({ type: "session.created" }, 0);
  assert.equal(sent.length, 0);
  call.channel.bufferedAmount = 300_000;
  let stopped = 0; call.stop = async () => { stopped++; };
  assert.equal(call.send({ type: "input_audio_buffer.append", audio: "PCM" }), false);
  assert.equal(stopped, 1);
  assert.ok(events.some(x => x.name === "error"));
});

test("stop archives a rendered stopped prefix and releases all local resources", async t => {
  const { call, writes } = fixture(t);
  const closed = [];
  call.audio = { pause() { closed.push("audio"); }, srcObject: {} };
  call.video = { srcObject: {} };
  call.context.close = async () => { closed.push("context"); };
  call.pcm.close = () => { closed.push("pcm"); };
  call.pcm.takePlayback = () => ({ sample_rate: 24000, rendered_samples: 100, pcm_s16le: "prefix" });
  call.visualCapture = { stop() { closed.push("vision"); }, sendState() {} };
  call.media = { finish: async () => { closed.push("media"); } };
  await answer(call);
  await Promise.all([call.stop(), call.stop()]);
  assert.equal(JSON.parse(writes.find(x => x.url === "/api/dialogue").options.body).playback.status, "stopped");
  assert.equal(writes.filter(x => x.options.method === "DELETE").length, 1);
  assert.deepEqual(closed.sort(), ["audio", "context", "media", "pcm", "vision"]);
  assert.equal(call.active, false);
});

test("recording still requires original MP4 consumed by the native memory stream", t => {
  const { call, events } = fixture(t);
  const old = globalThis.MediaRecorder;
  globalThis.MediaRecorder = class { static isTypeSupported(type) { assert.equal(type, "video/mp4"); return false; } };
  t.after(() => { globalThis.MediaRecorder = old; });
  call.record(1);
  assert.ok(events.some(x => x.name === "error"));
  assert.equal(call.active, true);
});
