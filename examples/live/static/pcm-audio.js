const OUTPUT_RATE = 24_000;

// Capture and playback run independently on the browser's audio render clock.
export class PcmAudio {
  constructor({ context, stream, send, onLevel = () => {} }) {
    this.context = context;
    this.inputStream = stream;
    this.send = send;
    this.onLevel = onLevel;
    this.stream = null;
    this.closed = false;
    this.sources = new Set();
    this.rendered = new Map();
    this.completedSeconds = this.nextTime = 0;
  }

  async start() {
    if (this.closed) return;
    if (!this.context.audioWorklet) throw new Error("浏览器不支持实时音频采集，请使用新版浏览器。");
    await this.context.audioWorklet.addModule(new URL("./pcm-worklet.js", import.meta.url));
    if (this.closed) return;
    this.destination = this.context.createMediaStreamDestination();
    this.stream = this.destination.stream;
    this.input = this.context.createMediaStreamSource(this.inputStream);
    this.processor = new AudioWorkletNode(this.context, "sensemem-pcm-input", {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1,
    });
    this.silence = this.context.createGain();
    this.silence.gain.value = 0;
    this.processor.port.onmessage = ({ data }) => {
      if (this.closed) return;
      const bytes = new Uint8Array(data);
      const view = new DataView(data);
      let energy = 0;
      for (let index = 0; index < view.byteLength; index += 2) {
        energy += (view.getInt16(index, true) / 32768) ** 2;
      }
      this.onLevel(Math.sqrt(energy / (view.byteLength / 2)));
      this.send({ type: "input_audio_buffer.append", audio: btoa(String.fromCharCode(...bytes)) });
    };
    // Keep the capture worklet rendering without feeding the microphone to speakers.
    this.input.connect(this.processor).connect(this.silence).connect(this.context.destination);
  }

  append(encoded, responseId = "") {
    if (this.closed) return;
    const decoded = atob(encoded);
    if (!decoded.length || decoded.length % 2) throw new Error("收到无法播放的语音数据。");
    const bytes = Uint8Array.from(decoded, (value) => value.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    const samples = new Float32Array(decoded.length / 2);
    for (let index = 0; index < samples.length; index++) {
      const value = view.getInt16(index * 2, true);
      samples[index] = value / (value < 0 ? 32768 : 32767);
    }
    const buffer = this.context.createBuffer(1, samples.length, OUTPUT_RATE);
    buffer.getChannelData(0).set(samples);
    const node = this.context.createBufferSource();
    node.buffer = buffer;
    node.connect(this.destination);
    const start = Math.max(this.context.currentTime + 0.012, this.nextTime);
    const segment = { node, start, duration: buffer.duration, samples, responseId, recorded: 0 };
    this.sources.add(segment);
    this.nextTime = start + buffer.duration;
    node.onended = () => {
      this.collectRendered(segment);
      if (this.sources.delete(segment)) this.completedSeconds += segment.duration;
      node.disconnect();
    };
    node.start(start);
  }

  playedFor(responseId) {
    let count = (this.rendered.get(responseId) || []).reduce((sum, chunk) => sum + chunk.length, 0);
    for (const segment of this.sources) if (segment.responseId === responseId) {
      count += Math.max(0, Math.min(segment.samples.length,
        Math.floor((this.context.currentTime - segment.start) * OUTPUT_RATE)) - segment.recorded);
    }
    return Math.floor(count / OUTPUT_RATE * 1000);
  }

  get playedMs() {
    let seconds = this.completedSeconds;
    for (const segment of this.sources) {
      seconds += Math.min(segment.duration, Math.max(0, this.context.currentTime - segment.start));
    }
    // This is the Web Audio render clock, not a physical speaker or heard-word acknowledgment.
    return Math.floor(seconds * 1000);
  }

  get pending() {
    return [...this.sources].some(({ start, duration }) => start + duration > this.context.currentTime);
  }

  get responseId() {
    return [...this.sources].find(({ start, duration }) => start + duration > this.context.currentTime)?.responseId || null;
  }

  collectRendered(segment) {
    const count = Math.min(segment.samples.length,
      Math.max(0, Math.floor((this.context.currentTime - segment.start) * OUTPUT_RATE)));
    if (!segment.responseId || count <= segment.recorded) return;
    const chunks = this.rendered.get(segment.responseId) || [];
    chunks.push(segment.samples.subarray(segment.recorded, count));
    this.rendered.set(segment.responseId, chunks);
    segment.recorded = count;
  }

  takePlayback(responseId) {
    for (const segment of this.sources) if (segment.responseId === responseId) this.collectRendered(segment);
    const chunks = this.rendered.get(responseId) || [];
    this.rendered.delete(responseId);
    const count = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    if (!count) return null;
    const bytes = new Uint8Array(count * 2);
    const view = new DataView(bytes.buffer);
    let offset = 0;
    for (const chunk of chunks) for (const value of chunk) {
      view.setInt16(offset, Math.round(value * (value < 0 ? 32768 : 32767)), true);
      offset += 2;
    }
    let binary = "";
    for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    return { sample_rate: OUTPUT_RATE, rendered_samples: count, pcm_s16le: btoa(binary) };
  }

  clear() {
    const played = this.playedMs;
    for (const segment of this.sources) {
      this.collectRendered(segment);
      const { node } = segment;
      node.onended = null;
      node.stop();
      node.disconnect();
    }
    this.sources.clear();
    this.completedSeconds = this.nextTime = 0;
    return played;
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    this.clear();
    if (this.processor) this.processor.port.onmessage = null;
    this.input?.disconnect();
    this.processor?.disconnect();
    this.silence?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
  }
}
