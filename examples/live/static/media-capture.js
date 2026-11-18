const MAX_UPLOAD_BYTES = 8 * 1024 * 1024;

export class MediaCapture {
  constructor({ stream, mimeType, current, emit, onPaused }) {
    Object.assign(this, { stream, current, emit, onPaused });
    this.queue = [];
    this.queuedBytes = 0;
    this.sending = this.stopped = this.endSent = false;
    this.finished = new Promise((resolve) => { this.resolve = resolve; });
    const url = new URL("/api/media", location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const socket = this.socket = new WebSocket(url);
    socket.onopen = () => {
      if (!this.current() || this.failed || this.cancelled) return;
      try {
        const recorder = new MediaRecorder(this.stream, { mimeType, videoKeyFrameIntervalDuration: 1000 });
        this.recorder = recorder;
        recorder.ondataavailable = ({ data }) => {
          if (!data.size || this.failed) return;
          if (this.queuedBytes + socket.bufferedAmount + data.size > MAX_UPLOAD_BYTES) {
            this.pause("记忆录制已暂停：音视频上传积压，通话可继续。未确认的片段不保证保存。");
            return;
          }
          this.queue.push(data);
          this.queuedBytes += data.size;
          void this.send();
        };
        recorder.onerror = () => this.pause("记忆录制已暂停：音视频采集失败，通话可继续。未确认的片段不保证保存。");
        recorder.onstop = () => {
          this.stopped = true;
          void this.send();
        };
        socket.send(JSON.stringify({ type: "start", started_at: new Date().toISOString() }));
        // Flush encoded bytes from one container; event boundaries belong to SenseMem.
        recorder.start(250);
        this.emit("recording", { state: "recording" });
      } catch {
        this.pause("记忆录制已暂停：无法开启音视频采集，通话可继续。");
      }
    };
    socket.onmessage = ({ data }) => {
      if (this.failed || this.cancelled) return;
      let event;
      try { event = JSON.parse(data); }
      catch { this.pause("记忆录制已暂停：无法读取流式确认，未确认的片段不保证保存；通话可继续。"); return; }
      if (event.type === "accepted" && this.endSent) {
        this.accepted = true;
        socket.close();
        this.resolve();
        this.emit("recording", { state: "saved" });
        this.emit("memorychange");
      } else if (event.type === "error") {
        this.pause("记忆录制已暂停：音视频流未能保存，未确认的片段不保证保存；通话可继续。");
      }
    };
    socket.onerror = socket.onclose = () => {
      if (this.accepted || this.cancelled) return;
      this.pause("记忆录制已暂停：音视频流已断开，未确认的片段不保证保存；通话可继续。");
    };
  }

  async send() {
    if (this.sending || this.failed) return;
    this.sending = true;
    try {
      while (this.queue.length && !this.failed) {
        const blob = this.queue[0];
        const data = await blob.arrayBuffer();
        if (this.failed) return;
        this.socket.send(data);
        this.queue.shift();
        this.queuedBytes -= blob.size;
      }
      if (this.stopped && !this.endSent && !this.failed) {
        this.endSent = true;
        this.socket.send(JSON.stringify({ type: "end" }));
      }
    } catch {
      this.pause("记忆录制已暂停：音视频上传失败，未确认的片段不保证保存；通话可继续。");
    } finally {
      this.sending = false;
    }
  }

  pause(message) {
    if (this.failed || this.accepted || this.cancelled) return;
    this.failed = true;
    this.onPaused();
    this.queue.length = this.queuedBytes = 0;
    if (this.recorder?.state !== "inactive") this.recorder?.stop();
    this.socket.close();
    this.resolve();
    this.emit("recording", { state: "paused", reason: message });
    this.emit("error", message);
  }

  finish() {
    if (!this.recorder && !this.failed) {
      this.cancelled = true;
      this.socket.close();
      this.resolve();
    } else if (this.recorder?.state !== "inactive") this.recorder?.stop();
    return this.finished;
  }
}
