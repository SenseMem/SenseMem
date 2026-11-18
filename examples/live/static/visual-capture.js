const MAX_FRAME_BYTES = 256 * 1024;

export class VisualCapture {
  constructor({ video, clock, current, emit, onFrame, enabled = true }) {
    Object.assign(this, { video, clock, current, emit, onFrame, enabled });
    this.latestCanvas = document.createElement("canvas");
    this.encodingCanvas = document.createElement("canvas");
    this.inFlight = new Set();
    this.closed = this.ready = this.encoding = false;
    this.perceptionClosed = false;
    this.maxFps = 6;
    this.maxFrameBytes = MAX_FRAME_BYTES;
    this.maxInFlight = 2;
    this.generation = 0;
    this.lastEncodedAt = -Infinity;
    this.lastMediaTime = null;
    this.recall = false;
    this.latest = this.pendingPerception = this.stateSnapshot = null;
    this.sentState = null;
    this.useVideoCallback = typeof video.requestVideoFrameCallback === "function";
  }

  start(audioStartedMs) {
    const url = new URL("/api/perception", location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const socket = this.socket = new WebSocket(url);
    this.emit("perceptionstate", { state: "connecting" });
    socket.onopen = () => {
      if (!this.live()) { socket.close(); return; }
      socket.send(JSON.stringify({ type: "start", started_at: this.clock.startedAt,
        audio_started_ms: audioStartedMs }));
    };
    socket.onmessage = ({ data }) => {
      if (!this.live() || this.perceptionClosed) return;
      let event;
      try { event = JSON.parse(data); }
      catch { this.pause("后台画面分析已暂停：无法读取服务消息，通话可继续。"); return; }
      if (event.type === "ready" && !this.ready) {
        if (!Number.isFinite(event.max_fps) || event.max_fps <= 0 || event.max_fps > 60 ||
            !Number.isInteger(event.max_frame_bytes) || event.max_frame_bytes <= 0 || event.max_frame_bytes > MAX_FRAME_BYTES ||
            !Number.isInteger(event.max_in_flight) || event.max_in_flight < 1 || event.max_in_flight > 4) {
          this.pause("后台画面分析已暂停：服务配置无效，通话可继续。");
          return;
        }
        this.maxFps = Math.min(event.max_fps, 6);
        this.maxFrameBytes = event.max_frame_bytes;
        this.maxInFlight = event.max_in_flight;
        this.ready = true;
        this.control("camera", this.enabled);
        this.control("recall", this.recall);
        this.flushState();
        this.emit("perceptionstate", { state: this.enabled ? "live" : "idle" });
        this.flushPerception();
      } else if (event.type === "ack" && this.inFlight.delete(event.sequence)) {
        this.emit("perceptionstate", { state: this.enabled ? "live" : "idle", sequence: event.sequence });
        this.flushPerception();
      } else if (event.type === "error") {
        this.pause("后台画面分析暂不可用，通话继续接收画面。");
      }
    };
    socket.onerror = socket.onclose = () => {
      if (this.live() && !this.perceptionClosed) this.pause("后台画面分析连接已断开，通话继续接收画面。");
    };
    this.schedule();
  }

  live() {
    return !this.closed && this.current();
  }

  schedule() {
    if (!this.live()) return;
    if (this.useVideoCallback) {
      this.callback = this.video.requestVideoFrameCallback((now, metadata) => this.frame(now, metadata));
    } else {
      this.callback = requestAnimationFrame(now => this.frame(now, { mediaTime: this.video.currentTime }));
    }
  }

  frame(now, metadata) {
    if (!this.live()) return;
    this.schedule();
    if (!this.enabled || this.video.readyState < 2 || !this.video.videoWidth || !this.video.videoHeight ||
        metadata.mediaTime === this.lastMediaTime) return;
    this.lastMediaTime = metadata.mediaTime;
    const captured = Number.isFinite(metadata.captureTime) ? metadata.captureTime
      : Number.isFinite(metadata.presentationTime) ? metadata.presentationTime : now;
    if (captured < this.clock.start) return;
    const canvas = this.latestCanvas;
    const width = Math.min(this.video.videoWidth, 640);
    const height = Math.round(this.video.videoHeight * width / this.video.videoWidth);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    canvas.getContext("2d").drawImage(this.video, 0, 0, width, height);
    this.latest = this.clock.next(captured);
    this.pump();
  }

  pump() {
    if (!this.live() || !this.enabled) return;
    this.flushState();
    if (this.encoding || !this.latest ||
        performance.now() - this.lastEncodedAt < 1000 / this.maxFps) return;
    const frame = this.latest;
    this.latest = null;
    const canvas = this.encodingCanvas;
    canvas.width = this.latestCanvas.width;
    canvas.height = this.latestCanvas.height;
    canvas.getContext("2d").drawImage(this.latestCanvas, 0, 0);
    this.encoding = true;
    this.lastEncodedAt = performance.now();
    const generation = this.generation;
    try {
      canvas.toBlob(blob => this.finishEncoding(blob, frame, generation), "image/jpeg", 0.65);
    } catch {
      this.encoding = false;
      this.pause("画面输入已暂停：无法开启画面编码，语音通话可继续。", true);
    }
  }

  async finishEncoding(blob, frame, generation) {
    try {
      if (!this.live() || !this.enabled || generation !== this.generation) return;
      if (!blob || blob.type !== "image/jpeg" || blob.size * 4 / 3 > this.maxFrameBytes) {
        this.pause("画面输入已暂停：无法生成兼容大小的画面，语音通话可继续。", true);
        return;
      }
      const bytes = new Uint8Array(await blob.arrayBuffer());
      if (!this.live() || !this.enabled || generation !== this.generation) return;
      let binary = "";
      for (let offset = 0; offset < bytes.length; offset += 8192) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
      }
      const captured = { ...frame, image: `data:image/jpeg;base64,${btoa(binary)}` };
      const message = JSON.stringify({ type: "frame", ...captured });
      if (message.length > this.maxFrameBytes) {
        this.pause("画面输入已暂停：画面超出连接容量，语音通话可继续。", true);
        return;
      }
      this.onFrame(captured);
      if (!this.perceptionClosed) {
        this.pendingPerception = { message, sequence: frame.sequence };
        this.flushPerception();
      }
    } catch {
      if (this.live() && generation === this.generation) this.pause("画面输入已暂停：画面采集失败，语音通话可继续。", true);
    } finally {
      this.encoding = false;
      this.pump();
    }
  }

  flushPerception() {
    if (!this.live() || !this.ready || !this.enabled || this.perceptionClosed || !this.pendingPerception ||
        this.inFlight.size >= this.maxInFlight || this.socket.readyState !== WebSocket.OPEN ||
        this.socket.bufferedAmount + this.pendingPerception.message.length > this.maxFrameBytes * this.maxInFlight) return;
    this.socket.send(this.pendingPerception.message);
    this.inFlight.add(this.pendingPerception.sequence);
    this.pendingPerception = null;
  }

  control(type, enabled) {
    if (this.live() && this.ready && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type, enabled }));
    }
  }

  setEnabled(enabled) {
    this.enabled = enabled;
    this.generation++;
    this.latest = this.pendingPerception = null;
    this.control("camera", enabled);
    this.emit("perceptionstate", { state: !enabled ? "idle" : this.closed || this.perceptionClosed ? "paused" : this.ready ? "live" : "connecting" });
  }

  setRecall(enabled) {
    this.recall = enabled;
    this.control("recall", enabled);
  }

  sendState(state) {
    this.stateSnapshot = JSON.stringify({ type: "state", ...state });
    this.flushState();
  }

  flushState() {
    if (this.live() && this.ready && this.stateSnapshot !== null && this.stateSnapshot !== this.sentState &&
        this.socket.bufferedAmount <= this.maxFrameBytes * this.maxInFlight) {
      this.socket.send(this.stateSnapshot);
      this.sentState = this.stateSnapshot;
    }
  }

  pause(reason, stopCapture = false) {
    this.perceptionClosed = true;
    this.ready = false;
    this.pendingPerception = null;
    this.inFlight.clear();
    this.socket?.close();
    if (stopCapture) this.stop();
    this.emit("perceptionstate", { state: "paused", reason });
    this.emit("error", reason);
  }

  stop() {
    if (this.closed) return;
    this.closed = true;
    this.perceptionClosed = true;
    this.generation++;
    this.latest = this.pendingPerception = null;
    this.inFlight.clear();
    if (this.useVideoCallback) this.video.cancelVideoFrameCallback(this.callback);
    else cancelAnimationFrame(this.callback);
    this.socket?.close();
    this.emit("perceptionstate", { state: "idle" });
  }
}
