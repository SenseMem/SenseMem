import assert from "node:assert/strict";
import test from "node:test";
import { VisualCapture } from "../../examples/live/static/visual-capture.js";

function harness(t, { fallback = false } = {}) {
  const names = ["document", "location", "performance", "WebSocket", "requestAnimationFrame", "cancelAnimationFrame", "setInterval", "setTimeout"];
  const globals = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  const callbacks = new Map(), raf = new Map(), encodings = [], sockets = [], events = [], direct = [];
  let now = 1000, callbackId = 0, active = true, sequence = 0;
  const canvases = [];
  globalThis.document = { createElement() {
    const canvas = { width: 0, height: 0, pixel: null,
      getContext: () => ({ drawImage(source) { canvas.pixel = source.frameId ?? source.pixel; } }),
      toBlob(callback, type, quality) { encodings.push({ callback, pixel: canvas.pixel, type, quality }); },
    };
    canvases.push(canvas);
    return canvas;
  } };
  globalThis.location = { href: "https://localhost:63258/" };
  Object.defineProperty(globalThis, "performance", { configurable: true, value: { now: () => now } });
  globalThis.requestAnimationFrame = callback => { raf.set(++callbackId, callback); return callbackId; };
  globalThis.cancelAnimationFrame = id => raf.delete(id);
  globalThis.setInterval = globalThis.setTimeout = () => { throw new Error("Frame capture must not use a timer"); };
  globalThis.WebSocket = class {
    static OPEN = 1;
    constructor(url) { this.url = String(url); this.readyState = 0; this.bufferedAmount = 0; this.sent = []; sockets.push(this); }
    open() { this.readyState = 1; this.onopen(); }
    message(event) { this.onmessage({ data: JSON.stringify(event) }); }
    send(value) { this.sent.push(JSON.parse(value)); }
    close() { this.readyState = 3; this.closed = true; this.onclose?.(); }
  };
  const video = { videoWidth: 640, videoHeight: 480, readyState: 2, currentTime: 0, frameId: 0 };
  if (!fallback) {
    video.requestVideoFrameCallback = callback => { callbacks.set(++callbackId, callback); return callbackId; };
    video.cancelVideoFrameCallback = id => callbacks.delete(id);
  }
  const clock = { start: 1000, startedAt: "2026-11-16T01:00:00.000Z",
    next(at = now) { return { sequence: ++sequence, timestamp_ms: at - this.start,
      captured_at: new Date(Date.parse(this.startedAt) + at - this.start).toISOString() }; },
  };
  const capture = new VisualCapture({ video, clock, current: () => active, onFrame: frame => direct.push(frame),
    emit: (type, detail) => events.push({ type, detail }) });
  capture.start(25);
  const socket = sockets[0];
  const ready = () => {
    socket.open();
    socket.message({ type: "ready", max_fps: 6, max_frame_bytes: 262144, max_in_flight: 2 });
  };
  const frame = (time, { mediaTime = (time - 1000) / 1000, pixel = time, ...metadata } = {}) => {
    now = time;
    video.currentTime = mediaTime;
    video.frameId = pixel;
    const queue = fallback ? raf : callbacks;
    const [id, callback] = queue.entries().next().value;
    queue.delete(id);
    if (fallback) callback(now);
    else callback(now, { mediaTime, presentationTime: time, ...metadata });
  };
  const finish = async (blob) => {
    const encoding = encodings.shift();
    assert.ok(encoding, "A frame should be waiting for JPEG encoding");
    await encoding.callback(blob === undefined ? new Blob([String(encoding.pixel)], { type: "image/jpeg" }) : blob);
    return encoding;
  };
  t.after(() => {
    capture.stop();
    for (const [name, descriptor] of globals) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
  return { capture, video, clock, socket, callbacks, raf, encodings, canvases, events, direct, ready, frame, finish,
    frames: () => socket.sent.filter(event => event.type === "frame"),
    advance(time) { now = time; }, expire() { active = false; },
  };
}

test("real video-frame callbacks share sequence and capture timestamps with audio independently of perception readiness", async t => {
  const h = harness(t);
  assert.equal(h.socket.url, "wss://localhost:63258/api/perception");
  assert.equal(h.callbacks.size, 1);
  h.socket.open();
  assert.deepEqual(h.socket.sent, [{ type: "start", started_at: h.clock.startedAt, audio_started_ms: 25 }]);
  h.socket.message({ type: "ready", max_fps: 6, max_frame_bytes: 262144, max_in_flight: 2 });
  assert.equal(h.callbacks.size, 1);
  assert.equal(h.clock.next(1020).sequence, 1);
  h.frame(1100, { captureTime: 1075, pixel: "captured-pixels" });
  assert.equal(h.frames().length, 0);
  const encoding = await h.finish();
  assert.equal(encoding.type, "image/jpeg");
  assert.equal(encoding.quality, 0.65);
  assert.deepEqual(h.frames(), [{ type: "frame", sequence: 2, timestamp_ms: 75,
    captured_at: "2026-11-16T01:00:00.075Z", image: `data:image/jpeg;base64,${btoa("captured-pixels")}` }]);
  h.frame(1200, { mediaTime: 0.1 });
  assert.equal(h.encodings.length, 0, "The same presented frame must not be encoded twice");
});

test("one encoder and one latest frame stay bounded while receipt acknowledgements are delayed", async t => {
  const h = harness(t);
  h.ready();
  h.frame(1100, { pixel: "first" });
  h.frame(1300, { pixel: "superseded" });
  h.frame(1500, { pixel: "latest" });
  assert.equal(h.encodings.length, 1);
  await h.finish();
  assert.equal(h.encodings.length, 1);
  await h.finish();
  assert.deepEqual(h.frames().map(frame => atob(frame.image.split(",")[1])), ["first", "latest"]);
  assert.deepEqual(h.frames().map(frame => frame.sequence), [1, 3]);
  h.frame(1700, { pixel: "older-pending" });
  h.frame(1900, { pixel: "newest-pending" });
  assert.equal(h.encodings.length, 1);
  await h.finish();
  await h.finish();
  assert.equal(h.direct.at(-1).sequence, 5, "Direct conversation frames do not wait for perception ACKs");
  assert.equal(h.capture.pendingPerception.sequence, 5);
  assert.equal(h.capture.inFlight.size, 2);
  h.socket.message({ type: "ack", sequence: 1 });
  assert.equal(h.encodings.length, 0, "Receipt ACK forwards the newest already encoded frame");
  assert.equal(atob(h.frames().at(-1).image.split(",")[1]), "newest-pending");
  assert.equal(h.frames().at(-1).sequence, 5);
  h.socket.message({ type: "ack", sequence: 1 });
  assert.equal(h.capture.inFlight.size, 2, "Duplicate ACKs cannot free another frame's slot");
});

test("the negotiated frame-rate cap is driven by new frames rather than a capture timer", async t => {
  const h = harness(t);
  h.ready();
  h.frame(1100);
  await h.finish();
  h.socket.message({ type: "ack", sequence: 1 });
  for (const time of [1116, 1132, 1148, 1164, 1180, 1196, 1212, 1228, 1244, 1260]) h.frame(time);
  assert.equal(h.encodings.length, 0);
  h.frame(1277, { pixel: "fresh" });
  await h.finish();
  assert.equal(h.frames().length, 2);
  assert.equal(atob(h.frames()[1].image.split(",")[1]), "fresh");
});

test("perception socket pressure retains one newest frame without delaying direct conversation images", async t => {
  const h = harness(t);
  h.ready();
  h.socket.bufferedAmount = 524289;
  h.frame(1100, { pixel: "old" });
  h.frame(1300, { pixel: "newer" });
  await h.finish();
  await h.finish();
  assert.equal(h.direct.length, 2);
  assert.equal(h.capture.pendingPerception.sequence, 2);
  assert.equal(h.frames().length, 0);
  h.socket.bufferedAmount = 0;
  h.frame(1500, { pixel: "freshest" });
  await h.finish();
  assert.equal(h.frames().length, 1);
  assert.equal(atob(h.frames()[0].image.split(",")[1]), "freshest");
});

test("camera-off discards late encodings and re-enabling waits for a fresh frame", async t => {
  const h = harness(t);
  h.ready();
  h.frame(1100, { pixel: "private-old-frame" });
  h.capture.setEnabled(false);
  h.capture.setEnabled(true);
  await h.finish();
  assert.deepEqual(h.frames(), []);
  assert.equal(h.encodings.length, 0);
  h.frame(1400, { pixel: "fresh-after-resume" });
  await h.finish();
  assert.equal(atob(h.frames()[0].image.split(",")[1]), "fresh-after-resume");
  assert.deepEqual(h.socket.sent.filter(event => event.type === "camera").map(event => event.enabled), [true, false, true]);
});

test("stopping cancels callbacks and ignores bytes whose conversion finishes after teardown", async t => {
  const h = harness(t);
  h.ready();
  h.frame(1100);
  let resolveBytes;
  const pending = h.finish({ type: "image/jpeg", size: 4, arrayBuffer: () => new Promise(resolve => { resolveBytes = resolve; }) });
  h.capture.stop();
  resolveBytes(new Uint8Array([1, 2, 3, 4]).buffer);
  await pending;
  assert.equal(h.socket.closed, true);
  assert.equal(h.callbacks.size, 0);
  assert.equal(h.frames().length, 0);
  h.socket.message({ type: "ack", sequence: 1 });
  assert.equal(h.encodings.length, 0);
});

test("the older-browser fallback observes currentTime changes and cancels its animation callback", async t => {
  const h = harness(t, { fallback: true });
  h.ready();
  assert.equal(h.callbacks.size, 0);
  h.frame(1100, { mediaTime: 0.1 });
  await h.finish();
  h.socket.message({ type: "ack", sequence: 1 });
  h.frame(1400, { mediaTime: 0.1 });
  assert.equal(h.encodings.length, 0);
  h.frame(1500, { mediaTime: 0.2 });
  await h.finish();
  assert.equal(h.frames().length, 2);
  h.capture.stop();
  assert.equal(h.raf.size, 0);
});

test("frame limits include JSON and base64 overhead and a visual failure does not stop the call owner", async t => {
  const h = harness(t);
  h.ready();
  h.frame(1100);
  await h.finish(new Blob([new Uint8Array(196608)], { type: "image/jpeg" }));
  assert.equal(h.frames().length, 0);
  assert.equal(h.socket.closed, true);
  assert.equal(h.events.at(-1).type, "error");
  assert.match(h.events.at(-1).detail, /语音通话可继续/);
});

test("control settings and state coalesce before ready without fabricated input or inference events", t => {
  const h = harness(t);
  h.capture.setRecall(true);
  h.capture.setEnabled(false);
  h.capture.sendState({ user_speaking: false, response_id: null });
  h.capture.sendState({ user_speaking: true, response_id: null });
  h.ready();
  assert.deepEqual(h.socket.sent.filter(event => event.type !== "start"), [
    { type: "camera", enabled: false }, { type: "recall", enabled: true },
    { type: "state", user_speaking: true, response_id: null },
  ]);
  h.capture.sendState({ user_speaking: true, response_id: null });
  assert.equal(h.socket.sent.filter(event => event.type === "state").length, 1);
});

test("invalid server capabilities pause only perception while direct frame capture continues", async t => {
  const h = harness(t);
  h.socket.open();
  h.socket.message({ type: "ready", max_fps: 6, max_frame_bytes: 1048576, max_in_flight: 100 });
  assert.equal(h.socket.closed, true);
  assert.equal(h.callbacks.size, 1);
  assert.match(h.events.at(-1).detail, /配置无效/);
  h.frame(1100, { pixel: "direct-still-live" });
  await h.finish();
  assert.equal(atob(h.direct[0].image.split(",")[1]), "direct-still-live");
  assert.equal(h.capture.closed, false);
});

test("the first JPEG reaches direct conversation before independent perception is ready", async t => {
  const h = harness(t);
  h.frame(1100, { pixel: "first-live-view" });
  await h.finish();
  assert.equal(h.direct.length, 1);
  assert.equal(h.frames().length, 0);
  h.ready();
  assert.equal(h.frames().length, 1);
  assert.equal(h.frames()[0].image, h.direct[0].image, "Both paths reuse the identical JPEG");
});

test("a perception disconnection cannot stop direct frames and teardown still cancels every callback", async t => {
  const h = harness(t);
  h.ready();
  h.socket.close();
  assert.equal(h.capture.perceptionClosed, true);
  assert.equal(h.capture.closed, false);
  h.frame(1100, { pixel: "after-independent-disconnect" });
  await h.finish();
  assert.equal(h.direct.length, 1);
  assert.equal(h.frames().length, 0);
  h.capture.stop();
  assert.equal(h.callbacks.size, 0);
});

test("a synchronous browser encoder failure pauses only perception and cancels the callback", t => {
  const h = harness(t);
  h.ready();
  h.canvases[1].toBlob = () => { throw new Error("Encoder unavailable"); };
  h.frame(1100);
  assert.equal(h.socket.closed, true);
  assert.equal(h.callbacks.size, 0);
  assert.match(h.events.at(-1).detail, /无法开启画面编码.*语音通话可继续/);
});

test("an encoder rejection from a stopped call cannot display a late failure in a newer call", async t => {
  const h = harness(t);
  h.ready();
  h.frame(1100);
  let rejectBytes;
  const pending = h.finish({ type: "image/jpeg", size: 4, arrayBuffer: () => new Promise((_, reject) => { rejectBytes = reject; }) });
  h.capture.stop();
  const afterStop = h.events.length;
  rejectBytes(new Error("Old encoder failed after teardown"));
  await pending;
  assert.equal(h.events.length, afterStop);
  assert.equal(h.frames().length, 0);
});
