import assert from "node:assert/strict";
import fs from "node:fs/promises";
import vm from "node:vm";
import test from "node:test";

const audioSource = await fs.readFile(new URL("../../examples/live/static/pcm-audio.js", import.meta.url), "utf8");
const workletSource = await fs.readFile(new URL("../../examples/live/static/pcm-worklet.js", import.meta.url), "utf8");

function node() {
  return { connects: [], disconnected: false, connect(other) { this.connects.push(other); return other; }, disconnect() { this.disconnected = true; } };
}

function fixture(modulePromise = Promise.resolve()) {
  const events = [];
  const levels = [];
  const nodes = [];
  let inputStopped = false;
  let outputStopped = false;
  const context = {
    currentTime: 0, destination: node(), audioWorklet: { addModule: () => modulePromise },
    createMediaStreamDestination() { return { ...node(), stream: { getTracks: () => [{ stop: () => { outputStopped = true; } }] } }; },
    createMediaStreamSource() { return node(); },
    createGain() { return { ...node(), gain: { value: 1 } }; },
    createBuffer(_channels, length, rate) { return { sampleRate: rate, duration: length / rate, data: new Float32Array(length), getChannelData() { return this.data; } }; },
    createBufferSource() {
      const source = { ...node(), starts: [], stopped: false, start(at) { this.starts.push(at); }, stop() { this.stopped = true; } };
      nodes.push(source);
      return source;
    },
  };
  const scope = vm.createContext({ URL, Uint8Array, Float32Array, DataView, atob, btoa, AudioWorkletNode: class {
    constructor() { Object.assign(this, node(), { port: {} }); }
  } });
  vm.runInContext(audioSource.replace("export class", "class").replace("import.meta.url", '"https://demo.test/pcm-audio.js"') + "\nglobalThis.PcmAudio = PcmAudio;", scope);
  const audio = new scope.PcmAudio({ context, stream: { getTracks: () => [{ stop: () => { inputStopped = true; } }] }, send: event => events.push(event), onLevel: level => levels.push(level) });
  return { audio, context, events, levels, nodes, inputStopped: () => inputStopped, outputStopped: () => outputStopped };
}

function pcmBytes(samples) {
  const bytes = Buffer.alloc(samples.length * 2);
  samples.forEach((value, index) => bytes.writeInt16LE(Math.round(Math.max(-1, Math.min(1, value)) * (value < 0 ? 32768 : 32767)), index * 2));
  return bytes;
}

function pcm(samples) {
  return pcmBytes(samples).toString("base64");
}

function capture(audio, samples) {
  const bytes = pcmBytes(samples);
  audio.processor.port.onmessage({ data: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) });
}

function worklet(rate) {
  const packets = [];
  let Type;
  vm.runInNewContext(workletSource, {
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: buffer => packets.push(buffer.slice(0)) }; } },
    sampleRate: rate, registerProcessor(_name, value) { Type = value; },
  });
  return { input: new Type(), packets };
}

test("capture sends each 20ms PCM16 24kHz packet immediately while reporting its RMS", async () => {
  const f = fixture();
  await f.audio.start();
  assert.equal(f.audio.silence.gain.value, 0);
  const samples = [];
  for (let packet = 0; packet < 50; packet++) {
    const part = Array(480).fill(packet % 2 ? -0.5 : 0.25);
    samples.push(...part);
    capture(f.audio, part);
    assert.equal(f.levels.length, packet + 1);
    assert.equal(f.levels[packet], packet % 2 ? 0.5 : 0.25);
    assert.equal(f.events.length, packet + 1);
    assert.equal(f.events[packet].audio, pcm(part));
  }
  assert.equal(f.events[0].type, "input_audio_buffer.append");
  assert.equal(Buffer.from(f.events[0].audio, "base64").length, 960);
  f.audio.close();
  assert.equal(f.inputStopped(), false);
  assert.equal(f.outputStopped(), true);
});

test("closing prevents late microphone packets without waiting for an input window", async () => {
  const f = fixture();
  await f.audio.start();
  const onmessage = f.audio.processor.port.onmessage;
  capture(f.audio, Array(480).fill(0.5));
  f.audio.close();
  f.audio.close();
  onmessage({ data: new ArrayBuffer(480 * 2) });
  assert.equal(f.events.length, 1);
  assert.equal(f.levels.length, 1);
  assert.equal(f.audio.processor.port.onmessage, null);
});

test("PCM16 output retains samples and uses a contiguous 24kHz render timeline", async () => {
  const f = fixture();
  await f.audio.start();
  f.audio.append(pcm([-2, 0, 1.5]));
  f.audio.append(pcm([0.25, -0.25]));
  assert.deepEqual(Array.from(f.nodes[0].buffer.data), [-1, 0, 1]);
  assert.deepEqual(Array.from(f.nodes[1].buffer.data), [Math.fround(8192 / 32767), -0.25]);
  assert.equal(f.nodes[0].buffer.sampleRate, 24000);
  assert.equal(f.nodes[0].starts[0], 0.012);
  assert.equal(f.nodes[1].starts[0], 0.012 + 3 / 24000);
});

test("interruption clears every scheduled chunk and reports only rendered time", async () => {
  const f = fixture();
  await f.audio.start();
  f.audio.append(pcm(Array(480).fill(0.25)));
  f.audio.append(pcm(Array(480).fill(0.25)));
  assert.equal(f.audio.pending, true);
  f.context.currentTime = 0.0221;
  assert.equal(f.audio.clear(), 10);
  assert.ok(f.nodes.every(source => source.stopped && source.disconnected));
  assert.equal(f.audio.playedMs, 0);
  assert.equal(f.audio.sources.size, 0);
  assert.equal(f.audio.pending, false);
  f.audio.append(pcm(Array(480).fill(1)));
  assert.equal(f.nodes[2].starts[0], f.context.currentTime + 0.012);
});

test("completed chunks leave no nodes while rendered duration remains available for truncation", async () => {
  const f = fixture();
  await f.audio.start();
  f.audio.append(pcm(Array(480).fill(1)));
  f.context.currentTime = 0.04;
  assert.equal(f.audio.pending, false);
  f.nodes[0].onended();
  assert.equal(f.audio.sources.size, 0);
  assert.equal(f.audio.playedMs, 20);
  assert.equal(f.nodes[0].disconnected, true);
});

test("archive retains only the rendered prefix across interruption, excluding queued audio", async () => {
  const f = fixture();
  await f.audio.start();
  f.audio.append(pcm(Array(480).fill(0.5)), "r1");
  f.audio.append(pcm(Array(480).fill(-0.5)), "r1");
  f.context.currentTime = 0.0221;
  f.audio.clear();
  const archive = f.audio.takePlayback("r1");
  assert.equal(archive.sample_rate, 24000);
  assert.equal(archive.rendered_samples, 242);
  const bytes = Buffer.from(archive.pcm_s16le, "base64");
  assert.equal(bytes.length, 484);
  for (let index = 0; index < bytes.length; index += 2) assert.equal(bytes.readInt16LE(index), 16384);
  assert.equal(f.audio.takePlayback("r1"), null);
});

test("completed render archives are response-scoped and cannot capture an unplayed response", async () => {
  const f = fixture();
  await f.audio.start();
  f.audio.append(pcm([-1, 0, 1]), "r1");
  f.context.currentTime = 0.02;
  f.nodes[0].onended();
  f.audio.append(pcm([0.25, -0.25]), "r2");
  assert.equal(f.audio.responseId, "r2");
  assert.equal(f.audio.takePlayback("r2"), null);
  const archive = f.audio.takePlayback("r1");
  assert.equal(archive.rendered_samples, 3);
  const bytes = Buffer.from(archive.pcm_s16le, "base64");
  assert.deepEqual([bytes.readInt16LE(0), bytes.readInt16LE(2), bytes.readInt16LE(4)], [-32768, 0, 32767]);
  f.context.currentTime = 0.05;
  f.nodes[1].onended();
  assert.equal(f.audio.responseId, null);
  assert.equal(f.audio.takePlayback("r2").rendered_samples, 2);
  assert.equal(f.audio.takePlayback("r1"), null);
});

test("stopping during worklet loading creates no late capture or playback graph", async () => {
  let ready;
  const f = fixture(new Promise(resolve => { ready = resolve; }));
  const starting = f.audio.start();
  f.audio.close();
  ready();
  await starting;
  assert.equal(f.audio.processor, undefined);
  assert.equal(f.audio.stream, null);
  f.audio.append(pcm([1]));
  assert.equal(f.nodes.length, 0);
});

test("invalid PCM fails before it changes scheduled playback", async () => {
  const f = fixture();
  await f.audio.start();
  assert.throws(() => f.audio.append(Buffer.from([1]).toString("base64")));
  assert.throws(() => f.audio.append(""));
  assert.equal(f.nodes.length, 0);
});

test("worklet keeps 20ms PCM16 mono packets at 24kHz across render quantum boundaries", () => {
  const f = worklet(24000);
  for (let index = 0; index < 8; index++) f.input.process([[new Float32Array(128).fill(-1)]]);
  assert.equal(f.packets.length, 2);
  for (const packet of f.packets) {
    assert.equal(packet.byteLength, 960);
    const view = new DataView(packet);
    for (let index = 0; index < 480; index++) assert.equal(view.getInt16(index * 2, true), -32768);
  }
  assert.equal(f.input.position, 64);
});

test("worklet downmixes channels and resamples the actual 48kHz render clock continuously", () => {
  const f = worklet(48000);
  for (let index = 0; index < 8; index++) f.input.process([[
    new Float32Array(128).fill(1), new Float32Array(128).fill(-1),
  ]]);
  assert.equal(f.packets.length, 1);
  assert.ok(new Uint8Array(f.packets[0]).every(value => value === 0));
  assert.equal(f.input.position, 32);
  assert.equal(f.input.inputIndex, 1024);
});

test("44.1kHz input retains sample timing across uneven render boundaries", () => {
  const f = worklet(44100);
  for (let start = 0; start < 4410; start += 127) {
    f.input.process([[Float32Array.from({ length: Math.min(127, 4410 - start) }, (_value, index) => (start + index) / 4410)]]);
  }
  assert.equal(f.packets.length, 5);
  let output = 0;
  for (const packet of f.packets) {
    const view = new DataView(packet);
    for (let index = 0; index < 480; index++, output++) {
      const expected = output / 2400;
      assert.ok(Math.abs(view.getInt16(index * 2, true) / 32767 - expected) < 1 / 32767);
    }
  }
  assert.equal(output, 2400);
});
