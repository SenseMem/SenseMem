import { MediaCapture } from "./media-capture.js";
import { PcmAudio } from "./pcm-audio.js";
import { VisualCapture } from "./visual-capture.js";
import { MemoryLink } from "./memory-link.js";

const VOICE_THRESHOLD = 0.025;
const VOICE_START_MS = 100;
const VOICE_END_MS = 600;
const MAX_BUFFERED_BYTES = 256 * 1024;
const RECALL_WAIT_MS = 15_000;
const CONTEXT_ACK_MS = 5000;

export class CallSession extends EventTarget {
  constructor({ video, audio, config }) {
    super();
    Object.assign(this, { video, audio, config });
    this.active = false;
    this.phase = "idle";
    this.epoch = 0;
    this.dialogueWrites = new Map();
    this.responses = new Map();
    this.ignoredResponses = new Set();
  }

  emit(name, detail) {
    this.dispatchEvent(new CustomEvent(name, { detail }));
  }

  state(phase) {
    if (this.phase === phase) return;
    this.phase = phase;
    this.emit("state", phase);
  }

  current(epoch) {
    return this.active && this.phase !== "stopping" && epoch === this.epoch;
  }

  async checked(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error("本地记忆服务未能完成操作，请检查服务后重试。");
    return response;
  }

  async start() {
    if (this.active) return;
    this.active = true;
    const epoch = ++this.epoch;
    const start = performance.now();
    this.clock = {
      start, sequence: 0, startedAt: new Date(performance.timeOrigin + start).toISOString(),
      next(at = performance.now()) {
        const timestamp_ms = Math.round(at - this.start);
        return { sequence: ++this.sequence, timestamp_ms,
          captured_at: new Date(Date.parse(this.startedAt) + timestamp_ms).toISOString() };
      },
    };
    this.dialogueWrites.clear();
    this.responses.clear();
    this.ignoredResponses.clear();
    this.responseId = this.sourceId = null;
    this.latestFrame = null;
    this.userSpeaking = false;
    this.nativeTurnId = null;
    this.userText = "";
    this.inputTurns = new Map();
    this.toolCalls = new Set();
    this.toolFollowups = new Set();
    this.toolResults = new Map();
    this.completedResponses = new Set();
    this.cancellingId = null;
    this.responseRequested = false;
    this.userAwaitingFinal = false;
    this.pendingAnswer = null;
    this.answeredTurns = new Set();
    this.recallUnavailable = this.perceptionUnavailable = false;
    this.voiceMs = this.quietMs = 0;
    this.initialized = this.mediaStarted = false;
    this.memoryRevision = -1;
    this.sensoryRevision = -1;
    this.asrRecallRevision = -1;
    this.lastUserAt = this.lastOutputAt = Date.now();
    this.state("connecting");
    try {
      if (!this.config.realtime_ready) {
        throw new Error("请先启动本地 speech-to-speech 服务，再刷新页面。");
      }
      this.context = new AudioContext({ sampleRate: 24000 });
      await this.context.resume();
      if (!this.current(epoch)) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (!this.current(epoch)) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      this.stream = stream;
      this.video.srcObject = stream;
      await this.video.play();
      if (!this.current(epoch)) return;
      this.inputMeter = this.meter(stream);
      const response = await this.checked("/api/session", { method: "POST" });
      const session = await response.json();
      if (!this.current(epoch)) return;
      this.sourceId = session.source_id;
      this.memoryRevision = session.memory_revision;
      this.memoryLink = new MemoryLink({
        sourceId: this.sourceId, current: () => this.current(epoch),
        getState: () => ({ userSpeaking: this.userSpeaking, playing: Boolean(this.pcm?.pending),
          generating: Boolean(this.responseId || this.cancellingId || this.responseRequested), responseId: this.responseId || this.cancellingId || this.pcm?.responseId,
          awaitingAnswer: this.userAwaitingFinal || Boolean(this.pendingAnswer),
          lastUserAt: this.lastUserAt, lastOutputAt: this.lastOutputAt }),
        interrupt: () => this.interrupt(), emit: (name, detail) => this.emit(name, detail),
        send: event => this.mediaStarted && this.send(event), instructions: this.config.session.instructions,
      });
      this.memoryLink.handle({ type: "memory_changed", memory_revision: this.memoryRevision });
      this.openRecall(epoch);
      const url = new URL("/api/realtime", location.href);
      url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
      const channel = this.channel = new WebSocket(url);
      this.handshakeTimeout = setTimeout(() => {
        if (this.current(epoch) && !this.mediaStarted) {
          this.emit("error", "实时模型连接等待过久，请稍后重新开始。");
          void this.stop();
        }
      }, 120_000);
      channel.onmessage = async ({ data }) => {
        if (!this.current(epoch)) return;
        try {
          await this.handle(JSON.parse(data), epoch);
        } catch {
          if (!this.current(epoch)) return;
          this.emit("error", "实时音视频消息无法处理，请重新连接。");
          void this.stop();
        }
      };
      channel.onerror = channel.onclose = () => {
        if (!this.current(epoch)) return;
        this.emit("error", "实时模型连接已断开，请重新开始通话。");
        void this.stop();
      };
    } catch (error) {
      if (!this.current(epoch)) return;
      const messages = {
        NotAllowedError: "请允许浏览器使用麦克风和摄像头，然后重试。",
        NotFoundError: "未找到可用的麦克风或摄像头。",
        NotReadableError: "无法打开设备，请检查是否被其他应用占用。",
      };
      this.emit("error", messages[error.name] || error.message || "连接失败，请重试。");
      await this.stop();
    }
  }

  send(event) {
    if (this.channel?.readyState !== WebSocket.OPEN) return false;
    if (this.channel.bufferedAmount > MAX_BUFFERED_BYTES) {
      this.emit("error", "实时音视频连接积压，请检查网络后重新开始。");
      void this.stop();
      return false;
    }
    this.channel.send(JSON.stringify(event));
    if (event.type === "response.create") this.responseRequested = true;
    return true;
  }

  async handle(event, epoch) {
    if (!this.current(epoch)) return;
    if (event.type === "session.created") {
      if (this.initialized) return;
      this.initialized = true;
      this.send({ type: "session.update", session: this.config.session });
    } else if (event.type === "session.updated") {
      this.ackAnswer(this.memoryLink?.ack(event));
      if (this.initialized && !this.mediaStarted) {
        clearTimeout(this.handshakeTimeout);
        await this.startMedia(epoch);
        this.memoryLink?.flush();
      }
    } else if (event.type === "input_audio_buffer.speech_started") {
      this.clearAnswer();
      this.nativeTurnId = event.item_id;
      this.userSpeaking = true;
      this.userFinal = false;
      this.userAwaitingFinal = true;
      this.lastUserAt = Date.now();
      this.interrupt();
    } else if (event.type === "input_audio_buffer.speech_stopped") {
      this.userSpeaking = false;
      this.lastUserAt = Date.now();
      this.syncPhase();
    } else if (["speech_to_speech.input_audio_transcription.snapshot",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed"].includes(event.type)) {
      const id = event.item_id;
      if (typeof id !== "string" || !id) return;
      if (this.nativeTurnId && this.nativeTurnId !== id) return;
      const final = event.type.endsWith(".completed");
      const previous = this.inputTurns?.get(id) || { stable: "", display: "", snapshot: false };
      const snapshot = event.type.endsWith(".snapshot");
      const stable = final ? event.transcript : snapshot ? previous.stable : previous.stable + event.delta;
      const text = final || snapshot ? event.transcript : previous.snapshot ? previous.display : stable;
      if (typeof text !== "string") return;
      this.inputTurns ||= new Map();
      this.inputTurns.set(id, { stable, display: text, snapshot: snapshot || previous.snapshot });
      if (this.inputTurns.size > 8) this.inputTurns.delete(this.inputTurns.keys().next().value);
      this.userText = text;
      if (this.userTurnId !== id) this.asrRecallRevision = -1;
      this.userTurnId = id;
      this.userFinal = final;
      this.emit("caption", { role: "user", text, final, itemId: id });
      if (final && text.trim()) this.queueAnswer(id, text, epoch);
      else if (final) this.clearAnswer();
      this.submitRecall(id, text, final);
      this.publishVisualState();
      if (final) { this.userAwaitingFinal = false; this.flushAnswer(); this.flushTools(); this.syncPhase(); }
    } else if (event.type === "conversation.item.input_audio_transcription.failed") {
      if (event.item_id !== this.nativeTurnId && event.item_id !== this.userTurnId) return;
      this.userAwaitingFinal = false;
      this.clearAnswer();
      this.emit("error", "这句话未能识别，请再说一次。收音仍在继续。");
      this.syncPhase();
    } else if (event.type === "conversation.item.created") {
      if (event.item?.type === "function_call_output") {
        const tool = this.toolResults?.get(event.item.call_id);
        if (tool) { tool.acknowledged = true; this.flushTools(); }
      }
    } else if (event.type === "response.function_call_arguments.done") {
      void this.executeTool(event, epoch);
    } else if (event.type === "response.created") {
      this.responseRequested = false;
      const followupIds = event.response?.metadata?.s2s_tool_followup_call_ids;
      if (typeof followupIds === "string") {
        const ids = JSON.parse(followupIds);
        if (Array.isArray(ids)) for (const id of ids) { this.toolFollowups?.delete(id); this.toolResults?.delete(id); }
      }
      const id = event.response?.id;
      if (typeof id === "string" && !this.ignoredResponses.has(id)) {
        this.ensureResponse(id, event.response.metadata || {});
        this.responseId = id;
        this.syncPhase();
      }
    } else if (["response.output_audio_transcript.delta", "response.output_audio_transcript.done",
        "response.output_text.delta", "response.output_text.done", "response.output_audio.delta"].includes(event.type)) {
      if (!this.mediaStarted) return;
      const id = event.response_id;
      if (typeof id !== "string" || !id || id.length > 128) throw new Error("Missing native response identity");
      if (this.userSpeaking) { this.ignoredResponses.add(id); return; }
      if (this.ignoredResponses.has(id)) return;
      const response = this.ensureResponse(id);
      if (response.interrupted || response.completedAt) return;
      this.responseId = id;
      if (event.type === "response.output_audio.delta") {
        if (typeof event.delta !== "string") throw new Error("Invalid audio delta");
        response.itemId = event.item_id;
        this.pcm.append(event.delta, id);
        this.lastOutputAt = Date.now();
        response.audioExpected = true;
        response.playbackEnd = this.pcm.nextTime;
        this.watchOutput(epoch);
      } else {
        const text = event.type.endsWith(".done") ? event.transcript ?? event.text : event.delta;
        if (typeof text !== "string") throw new Error("Invalid text delta");
        response.text = event.type.endsWith(".done") ? text : response.text + text;
        if ([...response.text].length > 8000) throw new Error("Response exceeds capture limit");
        this.emit("caption", { role: "assistant", text: response.text, responseId: id });
      }
      this.syncPhase();
    } else if (event.type === "response.done") {
      const id = event.response?.id;
      if (event.response?.status === "completed") this.completedResponses?.add(id);
      const consumed = event.response?.metadata?.s2s_tool_input_call_ids;
      if (event.response?.status === "completed" && typeof consumed === "string") {
        const ids = JSON.parse(consumed);
        if (Array.isArray(ids)) for (const id of ids) { this.toolFollowups?.delete(id); this.toolResults?.delete(id); }
      }
      if (this.cancellingId === id) this.cancellingId = null;
      const response = this.responses.get(id);
      if (response && !response.completedAt) {
        response.completedAt = new Date().toISOString();
        if (event.response.status !== "completed") {
          response.interrupted = true;
          response.playbackStatus = "interrupted";
          this.pcm?.clear();
        }
        this.watchOutput(epoch);
        void this.saveDialogue(response, epoch);
      }
      if (this.responseId === id) this.responseId = null;
      this.flushAnswer();
      this.flushTools();
      this.syncPhase();
    } else if (event.type === "error") {
      const code = event.error?.code || event.error?.type || event.code;
      if (["response_cancel_not_active", "conversation_already_has_active_response",
          "tool_followup_already_answered", "tool_followup_user_turn_open"].includes(code)) {
        this.responseRequested = false;
        if (code === "tool_followup_user_turn_open") this.userAwaitingFinal = true;
        if (code === "tool_followup_already_answered") {
          for (const id of this.requestedToolIds || []) { this.toolFollowups?.delete(id); this.toolResults?.delete(id); }
        }
        return;
      }
      this.emit("error", code === "realtime_not_configured"
        ? "请先启动本地 speech-to-speech 服务，再刷新页面。"
        : "本地语音服务无法完成请求，请检查服务日志后重试。");
      void this.stop();
    }
  }

  ensureResponse(id, metadata = {}) {
    let response = this.responses.get(id);
    if (!response) {
      const receipt = this.memoryLink?.responseStarted(metadata);
      response = { id, sourceId: this.sourceId, startedAt: new Date().toISOString(),
        text: "", audioExpected: false, interrupted: false, playbackEnd: 0,
        user: !metadata.sensemem_candidate_id && this.userFinal && this.userText?.trim()
          ? { item_id: this.userTurnId, text: this.userText } : null,
        feedback: receipt ? [receipt] : [] };
      this.responses.set(id, response);
    }
    return response;
  }

  async startMedia(epoch) {
    this.mediaStarted = true;
    const pcm = this.pcm = new PcmAudio({
      context: this.context, stream: this.stream,
      onLevel: level => this.updateVoice(level, epoch),
      send: event => this.sendInput(event, epoch),
    });
    await pcm.start();
    if (!this.current(epoch)) return;
    this.audio.muted = false;
    this.audio.srcObject = pcm.stream;
    this.outputMeter = this.meter(pcm.stream);
    await this.audio.play();
    if (!this.current(epoch)) return;
    this.visualCapture = new VisualCapture({
      video: this.video, clock: this.clock,
      enabled: this.stream.getVideoTracks()[0].enabled,
      current: () => this.current(epoch),
      onFrame: frame => { this.latestFrame = frame; },
      emit: (name, detail) => {
        if (name === "perceptionstate" && detail.state === "paused") {
          this.perceptionUnavailable = true;
          this.flushAnswer();
        }
        this.emit(name, detail);
      },
    });
    this.visualCapture.start(Math.round(performance.now() - this.clock.start));
    this.record(epoch);
    this.state("listening");
    this.publishVisualState();
  }

  sendInput(event, epoch) {
    if (!this.current(epoch) || !this.mediaStarted) return;
    const stamp = this.clock.next();
    const inputId = "av_" + stamp.sequence + "_" + stamp.timestamp_ms;
    this.send({ ...event, event_id: inputId });
    this.memoryLink?.tick();
    this.flushAnswer();
    this.flushTools();
  }

  submitRecall(id, text, final) {
    if (this.recallUnavailable || this.recallSocket?.readyState !== WebSocket.OPEN) {
      if (final) this.fallbackAnswer("recall_unavailable");
      return;
    }
    const frame = this.latestFrame;
    const fresh = frame && this.stream?.getVideoTracks()[0]?.enabled &&
      Math.round(performance.now() - this.clock.start) - frame.timestamp_ms <= 2000;
    this.recallSocket.send(JSON.stringify({ type: final ? "final" : "update", turn_id: id,
      text: [...text].slice(0, 2000).join(""), ...(fresh ? { image: frame.image } : {}) }));
  }

  clearAnswer() {
    clearTimeout(this.answerTimeout);
    clearTimeout(this.answerAckTimeout);
    this.answerTimeout = this.answerAckTimeout = null;
    this.pendingAnswer = null;
  }

  queueAnswer(id, text, epoch) {
    if (this.answeredTurns?.has(id)) return;
    const query = [...text].slice(0, 2000).join("");
    if (this.pendingAnswer?.turnId === id && this.pendingAnswer.query === query) return;
    this.clearAnswer();
    const pending = this.pendingAnswer = { turnId: id, query, epoch, sourceId: this.sourceId,
      memoryRevision: this.memoryRevision, hypothesis: null, receipt: null, unavailable: false,
      visualUnavailable: false };
    this.answerTimeout = setTimeout(() => {
      if (this.pendingAnswer !== pending || !this.current(epoch)) return;
      if (pending.receipt) {
        pending.visualUnavailable = true;
        this.emit("error", "本轮画面分析尚未完成，将依据已检索的记忆回答。");
        this.flushAnswer();
      } else this.fallbackAnswer("recall_timeout");
    }, RECALL_WAIT_MS);
    this.syncPhase();
  }

  ackAnswer(receipt) {
    const pending = this.pendingAnswer;
    if (!pending || !receipt || receipt.source_id !== pending.sourceId ||
        receipt.memory_revision !== this.memoryRevision || this.memoryRevision !== pending.memoryRevision) return;
    const recall = receipt.asr_recall;
    if (!recall || recall.turn_id !== pending.turnId || recall.query !== pending.query ||
        recall.final !== true || recall.input_final !== true ||
        recall.revision !== pending.hypothesis || Date.parse(recall.expires_at) <= Date.now() ||
        (pending.unavailable ? recall.status !== "unavailable" : recall.status !== "ready")) {
      pending.receipt = null;
      return;
    }
    pending.receipt = receipt;
    clearTimeout(this.answerAckTimeout);
    this.flushAnswer();
  }

  fallbackAnswer(reason) {
    const pending = this.pendingAnswer;
    if (!pending || !this.current(pending.epoch) || pending.unavailable) return;
    pending.unavailable = true;
    pending.receipt = null;
    pending.hypothesis ??= this.asrRecallRevision;
    this.emit("error", "本轮记忆检索暂不可用，将先回答；不会把未检索到的内容当作记忆。");
    this.memoryLink?.handle({ type: "asr_recall", source_id: pending.sourceId,
      memory_revision: this.memoryRevision, turn_id: pending.turnId, revision: pending.hypothesis,
      query: pending.query, input_final: true, final: true, status: "unavailable",
      context: this.config.recall_unavailable_instructions + JSON.stringify({ reason }),
      expires_at: new Date(Date.now() + 20_000).toISOString() });
    clearTimeout(this.answerTimeout);
    clearTimeout(this.answerAckTimeout);
    this.answerAckTimeout = setTimeout(() => {
      if (this.pendingAnswer !== pending || !this.current(pending.epoch)) return;
      pending.directFallback = true;
      pending.visualUnavailable = true;
      this.flushAnswer();
    }, CONTEXT_ACK_MS);
  }

  flushAnswer() {
    const pending = this.pendingAnswer;
    if (!pending || !this.current(pending.epoch) || pending.sourceId !== this.sourceId ||
        pending.turnId !== this.userTurnId || !this.userFinal || this.userSpeaking || this.userAwaitingFinal ||
        this.responseId || this.cancellingId || this.responseRequested || this.pcm?.pending ||
        pending.memoryRevision !== this.memoryRevision || (!pending.receipt && !pending.directFallback)) return;
    if (pending.receipt && Date.parse(pending.receipt.asr_recall.expires_at) <= Date.now()) {
      this.fallbackAnswer("recall_expired");
      return;
    }
    const visual = pending.receipt?.perception;
    const hasVisual = visual?.source_id === this.sourceId &&
      Number.isFinite(Date.parse(visual.occurred_end)) && Date.parse(visual.occurred_end) <= Date.now() + 1000 &&
      Date.parse(visual.expires_at) > Date.now();
    const cameraEnabled = this.stream?.getVideoTracks()[0]?.enabled;
    if (cameraEnabled && !hasVisual && !this.perceptionUnavailable && !pending.visualUnavailable) return;
    const response = { metadata: { sensemem_user_turn_id: pending.turnId } };
    if (pending.directFallback) {
      response.instructions = this.config.session.instructions + "\n" + this.config.context_unavailable_instructions;
    } else if (cameraEnabled && !hasVisual) {
      response.instructions = (this.memoryLink?.contextInstructions || this.config.session.instructions) +
        "\n" + this.config.visual_unavailable_instructions;
    }
    if (this.send({ type: "response.create", response })) {
      this.answeredTurns ||= new Set();
      this.answeredTurns.add(pending.turnId);
      this.clearAnswer();
      this.syncPhase();
    }
  }

  async executeTool(event, epoch) {
    const callId = event.call_id;
    if (typeof callId !== "string" || typeof event.response_id !== "string" || this.toolCalls.has(callId)) return;
    this.toolCalls.add(callId);
    let output;
    try {
      const args = JSON.parse(event.arguments);
      if (event.name !== "search_memory" || typeof args.query !== "string" || !args.query.trim() ||
          args.query.length > 2000 || Object.keys(args).some(key => key !== "query")) throw new Error("Invalid memory tool");
      const result = await this.checked("/api/search", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: args.query }) });
      output = await result.json();
      if (!this.current(epoch)) return;
      if (output.memory_revision !== this.memoryRevision) output = { error: "Memory changed; search again before answering." };
    } catch {
      output = { error: "Memory retrieval unavailable. Do not invent recalled evidence." };
    }
    if (!this.current(epoch)) return;
    this.toolResults.set(callId, { responseId: event.response_id, acknowledged: false });
    if (this.send({ type: "conversation.item.create", item: { type: "function_call_output", call_id: callId,
      output: JSON.stringify(output) } })) {
      this.toolFollowups.add(callId);
      this.flushTools();
    }
  }

  flushTools() {
    if (!this.toolFollowups?.size || this.userSpeaking || this.userAwaitingFinal || this.pendingAnswer ||
        this.responseId || this.cancellingId || this.responseRequested || this.pcm?.pending) return;
    const ids = [...this.toolFollowups].filter(id => {
      const result = this.toolResults.get(id);
      return result?.acknowledged && this.completedResponses.has(result.responseId);
    });
    if (!ids.length) return;
    const encoded = JSON.stringify(ids);
    if (encoded.length > 512) return;
    if (this.send({ type: "response.create", response: { metadata: { s2s_tool_followup_call_ids: encoded } } })) {
      this.requestedToolIds = ids;
    }
  }

  updateVoice(level, epoch) {
    if (!this.current(epoch)) return;
    const voiced = this.stream.getAudioTracks()[0].enabled && level >= VOICE_THRESHOLD;
    if (voiced) this.lastUserAt = Date.now();
    this.voiceMs = voiced ? this.voiceMs + 20 : 0;
    this.quietMs = voiced ? 0 : this.quietMs + 20;
    if (!this.userSpeaking && this.voiceMs >= VOICE_START_MS) {
      this.clearAnswer();
      this.userSpeaking = true;
      this.userFinal = false;
      this.userAwaitingFinal = true;
      this.interrupt();
    } else if (this.userSpeaking && this.quietMs >= VOICE_END_MS) {
      this.userSpeaking = false;
      this.syncPhase();
      this.publishVisualState();
    }
  }

  interrupt(status = "interrupted") {
    const response = this.responses.get(this.responseId || this.pcm?.responseId);
    const played = response ? this.pcm?.playedFor?.(response.id) ?? this.pcm?.playedMs ?? 0 : 0;
    if (this.responseId && !response?.interrupted) {
      this.cancellingId = this.responseId;
      this.send({ type: "response.cancel", response_id: this.responseId });
    }
    if (response?.itemId && !response.interrupted) this.send({ type: "conversation.item.truncate", item_id: response.itemId,
      content_index: 0, audio_end_ms: played });
    this.pcm?.clear();
    for (const response of this.responses.values()) {
      if (!response.completedAt || response.playbackEnd > this.context.currentTime) {
        response.interrupted = true;
        response.playbackStatus = status;
        response.completedAt ||= new Date().toISOString();
        this.ignoredResponses.add(response.id);
        void this.saveDialogue(response, this.epoch);
      }
    }
    if (response) this.emit("caption", { role: "assistant", text: response.text, interrupted: true });
    this.responseId = null;
    this.state("listening");
    this.publishVisualState();
  }

  syncPhase() {
    if (!this.active || ["connecting", "stopping"].includes(this.phase)) return;
    this.state(this.userSpeaking ? "listening" : this.responseId || this.pcm?.pending ? "speaking"
      : this.pendingAnswer || this.responseRequested ? "thinking" : "listening");
    this.publishVisualState();
  }

  meter(stream) {
    const source = this.context.createMediaStreamSource(stream);
    const analyser = this.context.createAnalyser();
    analyser.fftSize = 256;
    const silent = this.context.createGain();
    silent.gain.value = 0;
    source.connect(analyser).connect(silent).connect(this.context.destination);
    return { analyser, data: new Uint8Array(analyser.fftSize) };
  }

  getLevel() {
    let level = 0;
    for (const meter of [this.inputMeter, this.outputMeter]) {
      if (!meter) continue;
      meter.analyser.getByteTimeDomainData(meter.data);
      const energy = meter.data.reduce((sum, value) => sum + ((value - 128) / 128) ** 2, 0);
      level = Math.max(level, Math.sqrt(energy / meter.data.length));
    }
    return Math.min(1, level * 3);
  }

  record(epoch) {
    const mimeType = "video/mp4";
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      this.emit("error", "浏览器不支持保存音视频原片，通话可继续。");
      return;
    }
    this.media = new MediaCapture({
      stream: this.stream, mimeType, current: () => this.current(epoch),
      emit: (name, detail) => this.emit(name, detail),
      onPaused: () => {},
    });
  }

  openRecall(epoch) {
    const url = new URL("/api/recall", location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const socket = this.recallSocket = new WebSocket(url);
    socket.onmessage = ({ data }) => {
      if (!this.current(epoch)) return;
      let event;
      try { event = JSON.parse(data); } catch { return; }
      if (event.type === "error" && event.code === "recall_failed" && event.final &&
          event.turn_id === this.pendingAnswer?.turnId) {
        this.fallbackAnswer("recall_failed");
        return;
      }
      if (["pending", "cleared", "result"].includes(event.type)) {
        if (event.turn_id !== this.userTurnId) return;
        if (event.type === "result" && (event.source_id !== this.sourceId ||
            event.memory_revision !== this.memoryRevision || !Number.isSafeInteger(event.revision) ||
            event.revision < this.asrRecallRevision)) return;
        if (Number.isSafeInteger(event.revision)) this.asrRecallRevision = event.revision;
        const pending = this.pendingAnswer;
        if (pending?.turnId === event.turn_id && !pending.unavailable) {
          if (Number.isSafeInteger(event.revision)) pending.hypothesis = event.revision;
          pending.receipt = null;
          if (event.type === "result" && event.final && event.input_final && event.query === pending.query) {
            clearTimeout(this.answerAckTimeout);
            this.answerAckTimeout = setTimeout(() => {
              if (this.pendingAnswer === pending) this.fallbackAnswer("context_ack_timeout");
            }, CONTEXT_ACK_MS);
          }
        }
        if (pending?.unavailable) return;
        const bound = { ...event, source_id: this.sourceId, memory_revision: this.memoryRevision,
          expires_at: new Date(Date.now() + 20_000).toISOString() };
        this.memoryLink?.handle({ ...bound, type: event.type === "result" ? "asr_recall" : "asr_recall_clear" });
        this.emit("sensoryrecall", { ...event, status: event.type === "result" ? "ready" : event.type === "pending" ? "pending" : "idle",
          hits: event.hits || [], final: Boolean(event.final) });
        return;
      }
      this.memoryLink?.handle(event);
      if (event.type === "memory_changed" || event.type === "memory_event") {
        this.memoryRevision = Math.max(this.memoryRevision, event.memory_revision ?? -1);
        this.emit("memorychange");
      } else if (event.type === "invalidated") {
        this.memoryRevision = Math.max(this.memoryRevision, event.memory_revision ?? -1);
        this.sensoryRevision = -1;
        this.emit("sensoryrecall", { status: "idle", query: "", hits: [], revision: -1 });
      } else if (event.source_id === this.sourceId) {
        if (event.type === "perception_reset") {
          this.memoryLink?.handle({ type: "perception_clear", source_id: this.sourceId,
            memory_revision: this.memoryRevision, expires_at: new Date(Date.now() + 20_000).toISOString() });
          this.emit("perceptionreset", event);
        } else if (event.type === "perception_error") {
          this.perceptionUnavailable = true;
          this.flushAnswer();
        } else if (event.type === "visual_context") {
          if (event.memory_revision === this.memoryRevision && Date.parse(event.expires_at) > Date.now()) {
            this.perceptionUnavailable = false;
          }
        } else if (event.type === "perception" && typeof event.summary === "string") {
          this.emit("perception", event);
        } else if (event.type === "visual_memory") {
          this.emit("visualmemory", event);
          this.emit("memorychange");
        } else if (event.type === "sensory_recall") {
          const end = Date.parse(event.occurred_end), expires = Date.parse(event.expires_at);
          if (!Number.isSafeInteger(event.revision) || event.revision < this.sensoryRevision ||
              event.memory_revision < this.memoryRevision || !Number.isFinite(end) ||
              end > Date.now() + 1000 || !Number.isFinite(expires) || expires <= Date.now()) return;
          this.sensoryRevision = event.revision;
          this.emit("sensoryrecall", { ...event, final: false });
        }
      }
      if (this.pendingAnswer && this.pendingAnswer.memoryRevision !== this.memoryRevision) {
        this.pendingAnswer.memoryRevision = this.memoryRevision;
        this.pendingAnswer.hypothesis = null;
        this.pendingAnswer.receipt = null;
        if (this.pendingAnswer.unavailable) {
          this.pendingAnswer.unavailable = false;
          this.fallbackAnswer("memory_changed");
        }
      }
    };
    socket.onerror = socket.onclose = () => {
      if (this.current(epoch)) {
        this.recallUnavailable = true;
        this.emit("error", "后台记忆连接已断开，实时音视频通话可继续。");
        this.fallbackAnswer("recall_disconnected");
      }
    };
  }

  publishVisualState() {
    const id = this.responseId || this.pcm?.responseId;
    const response = this.responses.get(id);
    this.visualCapture?.sendState({
      user_speaking: this.userSpeaking, generating: Boolean(this.responseId),
      playing: Boolean(this.pcm?.pending), response_id: id || null,
      assistant_text: response?.text || "", user_text: this.userText || "",
    });
  }

  watchOutput(epoch) {
    if (this.outputWatch) return;
    const tick = () => {
      this.outputWatch = null;
      if (!this.current(epoch)) return;
      if (this.pcm?.pending) this.lastOutputAt = Date.now();
      for (const response of this.responses.values()) {
        if (response.completedAt && !response.interrupted && response.playbackEnd <= this.context.currentTime) {
          void this.saveDialogue(response, epoch);
        }
      }
      this.syncPhase();
      this.memoryLink?.tick();
      this.flushAnswer();
      this.flushTools();
      if (this.pcm?.pending || [...this.responses.values()].some(response => !response.completedAt && !response.interrupted)) {
        this.outputWatch = setTimeout(tick, 80);
      }
    };
    this.outputWatch = setTimeout(tick, 80);
  }

  saveDialogue(response, epoch) {
    if (!this.current(epoch) || !response.completedAt ||
        (!response.interrupted && response.audioExpected && response.playbackEnd > this.context.currentTime)) return;
    if (this.dialogueWrites.has(response.id)) return this.dialogueWrites.get(response.id);
    response.playback ||= this.pcm?.takePlayback(response.id);
    if ((!response.text.trim() && !response.playback) || (response.interrupted && !response.playback)) return;
    const body = JSON.stringify({
      source_id: response.sourceId, response_id: response.id, origin: "realtime",
      started_at: response.startedAt, completed_at: response.completedAt,
      assistant: { item_ids: [response.id], text: response.text },
      ...(response.user ? { user: response.user } : {}),
      feedback_receipts: (response.feedback || []).slice(-8).map(receipt => ({
        input_id: receipt.input_id, feedback_id: receipt.feedback_id,
        memory_revision: receipt.memory_revision, memory_ids: receipt.memory_ids,
        candidate_id: receipt.candidate_id || null, action: receipt.action,
        event_id: receipt.event_id, operation: receipt.operation,
      })),
      ...(response.playback ? { playback: { ...response.playback, status: response.playbackStatus || "completed" } } : {}),
    });
    const write = Promise.resolve().then(async () => {
      await this.checked("/api/dialogue", {
        method: "POST", headers: { "Content-Type": "application/json" }, body,
        // Fetch keepalive has a 64 KiB total body budget; audio archives use a normal request.
        keepalive: new TextEncoder().encode(body).length < 60_000,
      });
      this.emit("memorychange");
    }).catch(() => this.emit("error", "这段 Agent 回复未能写入记忆，原始音视频会继续保存。"));
    this.dialogueWrites.set(response.id, write);
    return write;
  }

  toggleMic() {
    const track = this.stream?.getAudioTracks()[0];
    if (!track) return false;
    track.enabled = !track.enabled;
    if (!track.enabled) {
      this.userSpeaking = false;
      this.voiceMs = this.quietMs = 0;
      this.publishVisualState();
    }
    return track.enabled;
  }

  toggleCamera() {
    const track = this.stream?.getVideoTracks()[0];
    if (!track) return false;
    track.enabled = !track.enabled;
    this.latestFrame = null;
    this.visualCapture?.setEnabled(track.enabled);
    if (!track.enabled) this.emit("perceptionreset");
    this.flushAnswer();
    return track.enabled;
  }

  async stop({ unloading = false } = {}) {
    if (unloading && this.active && !this.unloadHangup) {
      this.unloadHangup = this.checked("/api/session", { method: "DELETE", keepalive: true })
        .catch(() => this.emit("error", "本地连接清理未完成，请检查服务后重试。"));
    }
    if (this.stopping) return this.stopping;
    if (!this.active) return;
    this.interrupt("stopped");
    for (const response of this.responses.values()) void this.saveDialogue(response, this.epoch);
    this.state("stopping");
    ++this.epoch;
    clearTimeout(this.handshakeTimeout);
    clearTimeout(this.outputWatch);
    this.clearAnswer();
    this.outputWatch = null;
    this.visualCapture?.stop();
    this.memoryLink?.stop();
    this.recallSocket?.close();
    this.recallSocket = null;
    const finalMedia = this.media?.finish() || Promise.resolve();
    this.channel?.close();
    this.pcm?.close();
    this.stream?.getTracks().forEach(track => track.stop());
    this.audio.pause();
    this.audio.srcObject = null;
    this.video.srcObject = null;
    const contextClosed = this.context?.close().catch(() => {});
    this.inputMeter = this.outputMeter = null;
    this.stopping = (async () => {
      await Promise.all([finalMedia, ...this.dialogueWrites.values()]);
      try {
        await (this.unloadHangup || this.checked("/api/session", { method: "DELETE" }));
      } catch {
        this.emit("error", "本地连接清理未完成，请检查服务后重试。");
      }
      await contextClosed;
      this.stream = this.channel = this.context = this.pcm = this.media = this.visualCapture = null;
      this.active = false;
      this.unloadHangup = null;
      this.stopping = null;
      this.state("idle");
    })();
    return this.stopping;
  }
}
