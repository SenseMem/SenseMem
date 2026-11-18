import assert from "node:assert/strict";
import test from "node:test";
import { Presence } from "../../examples/live/static/presence.js";

function presenceHarness(t, reduced = false) {
  const names = ["document", "matchMedia", "ResizeObserver", "IntersectionObserver", "devicePixelRatio", "requestAnimationFrame", "cancelAnimationFrame", "setTimeout", "clearTimeout"];
  const original = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  const frames = new Map(), timers = new Map(), pageListeners = new Map(), motionListeners = new Map();
  let next = 0, level = 0, draws = 0, dots = [];
  let translateX = 0, translateY = 0;
  const transforms = [];
  const context = {
    clearRect() { draws++; dots = []; }, setTransform() {},
    save() { transforms.push([translateX, translateY]); },
    restore() { [translateX, translateY] = transforms.pop(); },
    translate(x, y) { translateX += x; translateY += y; }, rotate() {},
    createLinearGradient() { return { stops: [], addColorStop(stop, color) { this.stops.push([stop, color]); } }; }, beginPath() {}, moveTo() {}, lineTo() {}, closePath() {}, fill() {}, stroke() {},
    arc(x, y, radius) { dots.push([x + translateX, y + translateY, radius]); },
  };
  globalThis.document = {
    hidden: false,
    addEventListener: (name, callback) => pageListeners.set(name, callback),
    removeEventListener: name => pageListeners.delete(name),
  };
  const motion = {
    matches: reduced,
    addEventListener: (name, callback) => motionListeners.set(name, callback),
    removeEventListener: name => motionListeners.delete(name),
  };
  globalThis.matchMedia = () => motion;
  globalThis.devicePixelRatio = 1;
  globalThis.requestAnimationFrame = callback => { frames.set(++next, callback); return next; };
  globalThis.cancelAnimationFrame = id => frames.delete(id);
  globalThis.setTimeout = callback => { timers.set(++next, callback); return next; };
  globalThis.clearTimeout = id => timers.delete(id);
  let resize, intersection;
  globalThis.ResizeObserver = class {
    constructor(callback) { this.callback = callback; resize = this; }
    observe() {}
    disconnect() { this.disconnected = true; }
  };
  globalThis.IntersectionObserver = class {
    constructor(callback) { this.callback = callback; intersection = this; }
    observe() {}
    disconnect() { this.disconnected = true; }
  };
  const presence = new Presence({
    getContext: () => context,
    getBoundingClientRect: () => ({ width: 950, height: 560 }),
  }, () => level);
  t.after(() => {
    presence.destroy();
    for (const [name, descriptor] of original) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
  resize.callback();
  const flush = (queue, now) => {
    const [id, callback] = queue.entries().next().value;
    queue.delete(id);
    callback(now);
  };
  return {
    presence, frames, timers, pageListeners, motionListeners, motion, resize, intersection,
    get draws() { return draws; },
    get dots() { return dots; },
    setLevel(value) { level = value; },
    frame(now) { flush(frames, now); },
    timer() { flush(timers); },
  };
}

test("reduced motion freezes ambient clocks while sound and call state still update", t => {
  const h = presenceHarness(t, true);
  assert.equal(h.frames.size, 0);
  assert.equal(h.timers.size, 1);
  h.timer();
  const settledDraws = h.draws;
  h.timer();
  assert.equal(h.draws, settledDraws);
  h.setLevel(1);
  h.presence.setState("listening");
  h.timer();
  assert.ok(h.draws > settledDraws);
  assert.ok(h.presence.level > 0);
  assert.equal(h.presence.time, 0);
  assert.equal(h.presence.flow, 0);
  assert.equal(h.frames.size, 0);
  h.presence.destroy();
  assert.equal(h.timers.size, 0);
});

test("one gesture fully gathers, holds a filled ball, and restores the complete seahorse", t => {
  const h = presenceHarness(t);
  h.frame(0);
  const particles = h.presence.particles;
  const addresses = particles.map(({ address, across }) => [address, across]);
  const initialGeometry = [...h.presence.geometry];
  const initialHeight = Math.max(...h.dots.map(dot => dot[1])) - Math.min(...h.dots.map(dot => dot[1]));
  assert.equal(particles.length, 1700);
  h.presence.setState("thinking");
  for (let frame = 1; frame <= 510; frame++) {
    h.frame(frame * 1000 / 60);
    assert.equal(h.presence.particles, particles);
    if (frame === 48) {
      const { head, body, tail } = h.presence.pose;
      assert.ok(head < body && body < tail, "The head should lead the gathering wave");
    }
    if (frame === 210 || frame === 237) {
      assert.deepEqual(h.presence.pose, { head: 0, body: 0, tail: 0 });
      const xs = h.dots.map(dot => dot[0]), ys = h.dots.map(dot => dot[1]);
      const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
      const width = right - left, height = bottom - top;
      assert.ok(Math.max(width, height) / Math.min(width, height) < 1.25, `Closed shape aspect is ${width / height}`);
      assert.ok(height < initialHeight * 0.75, "The ball must visibly compact the tall seahorse");
      const centerX = (left + right) / 2, centerY = (top + bottom) / 2;
      const innerRadius = Math.min(width, height) * 0.15;
      const central = h.dots.filter(([x, y]) => Math.hypot(x - centerX, y - centerY) < innerRadius);
      assert.ok(central.length >= particles.length * 0.05, "The center must contain material rather than a ring-shaped hole");
      assert.ok(central.length < particles.length * 0.15, "The ball must distribute material rather than pile particles at its center");
      for (const [dx, dy] of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) {
        assert.ok(central.some(([x, y]) => (x - centerX) * dx > 0 && (y - centerY) * dy > 0));
      }
    }
    if (frame === 300) {
      const { head, body, tail } = h.presence.pose;
      assert.ok(head > body && body > tail, "The head should lead the unfurling wave");
    }
  }
  assert.equal(h.presence.cycleTime, null);
  assert.deepEqual(h.presence.pose, { head: 1, body: 1, tail: 1 });
  assert.deepEqual([...h.presence.geometry], initialGeometry);
  assert.deepEqual(particles.map(({ address, across }) => [address, across]), addresses);
});

test("sound follows elapsed time consistently at 60 and 120 Hz", t => {
  const h = presenceHarness(t);
  const follow = rate => {
    h.presence.previous = 0;
    h.presence.level = 0;
    h.setLevel(0);
    h.frame(1000);
    h.setLevel(1);
    for (let frame = 1; frame <= rate / 2; frame++) h.frame(1000 + frame * 1000 / rate);
    return h.presence.level;
  };
  const standard = follow(60), fast = follow(120);
  assert.ok(standard > 0.99);
  assert.ok(Math.abs(standard - fast) < 1e-9);
});

test("rapid duplex and recall chatter merges into one irreversible gesture without a queued restart", t => {
  const h = presenceHarness(t);
  h.frame(0);
  h.presence.setState("thinking");
  let prior = { ...h.presence.pose };
  for (let frame = 1; frame <= 510; frame++) {
    if (frame < 450) {
      const elapsed = h.presence.cycleTime, pose = { ...h.presence.pose };
      h.presence.setState(frame % 2 ? "speaking" : "thinking");
      h.presence.setRecall(frame % 3 === 0);
      assert.equal(h.presence.cycleTime, elapsed, "New transport updates must not reset the gesture clock");
      assert.deepEqual(h.presence.pose, pose, "State delivery must not move particles outside the render clock");
    }
    h.frame(frame * 1000 / 60);
    for (const part of Object.keys(prior)) {
      if (frame <= 246) assert.ok(h.presence.pose[part] <= prior[part] + 1e-12, "Gathering cannot reverse on a fast reply");
      else assert.ok(h.presence.pose[part] >= prior[part] - 1e-12, "Unfurling cannot reverse on another draft");
    }
    prior = { ...h.presence.pose };
  }
  assert.equal(h.presence.cycleTime, null);
  assert.deepEqual(h.presence.pose, { head: 1, body: 1, tail: 1 });
  for (let frame = 511; frame <= 570; frame++) h.frame(frame * 1000 / 60);
  assert.equal(h.presence.cycleTime, null, "Chatter received during the gesture must not enqueue another cycle");
  assert.deepEqual(h.presence.pose, { head: 1, body: 1, tail: 1 });
});

test("repeated recall updates do not restart the cycle and a later new recall can start another", t => {
  const h = presenceHarness(t);
  h.presence.setState("speaking");
  h.frame(0);
  h.presence.setRecall(true);
  for (let frame = 1; frame <= 30; frame++) h.frame(frame * 1000 / 60);
  const previous = h.presence.previous, pose = { ...h.presence.pose }, cycleTime = h.presence.cycleTime;
  h.presence.setRecall(true);
  h.presence.setState("speaking");
  assert.equal(h.presence.previous, previous);
  assert.deepEqual(h.presence.pose, pose);
  assert.equal(h.presence.cycleTime, cycleTime);
  for (let frame = 31; frame <= 510; frame++) h.frame(frame * 1000 / 60);
  assert.equal(h.presence.cycleTime, null);
  h.presence.setRecall(true);
  assert.equal(h.presence.cycleTime, null, "An unchanged ready result is not a new recall onset");
  h.presence.setRecall(false);
  h.presence.setRecall(true);
  assert.equal(h.presence.cycleTime, 0);
});

test("reduced motion keeps the complete seahorse static across every call and recall state", t => {
  const h = presenceHarness(t, true);
  const initial = h.dots;
  for (const phase of ["connecting", "listening", "thinking", "speaking", "stopping", "idle"]) {
    h.presence.setState(phase);
    for (const active of [true, false]) {
      h.presence.setRecall(active);
      assert.deepEqual(h.presence.pose, { head: 1, body: 1, tail: 1 });
      assert.deepEqual(h.dots, initial);
      assert.equal(h.presence.cycleTime, null);
    }
  }
  assert.equal(h.presence.time, 0);
  assert.equal(h.presence.flow, 0);
  assert.equal(h.frames.size, 0);
});

test("memory warmth changes pigment without moving the reduced-motion seahorse or replacing particles", t => {
  const h = presenceHarness(t, true);
  const particles = h.presence.particles;
  const addresses = particles.map(({ address, across }) => [address, across]);
  const initial = h.dots;
  const pigment = h.presence.context.fillStyle.stops;
  assert.equal(particles.length, 1700);
  assert.equal(initial.length, particles.length);
  assert.equal(h.presence.pose.head, 1);
  h.presence.pulseMemory();
  h.timer();
  assert.ok(h.presence.memoryGlow > 0);
  assert.deepEqual(h.dots, initial);
  const warmPigment = h.presence.context.fillStyle.stops;
  assert.notDeepEqual(warmPigment, pigment);
  const before = pigment[0][1].match(/\d+/g).map(Number), after = warmPigment[0][1].match(/\d+/g).map(Number);
  assert.ok(after[0] > before[0] && after[0] - after[2] > before[0] - before[2]);
  h.presence.setRecall(true);
  assert.deepEqual(h.dots, initial);
  h.presence.setRecall(false);
  assert.deepEqual(h.dots, initial);
  h.presence.width = 390;
  h.presence.height = 440;
  h.presence.draw();
  assert.equal(h.dots.length, particles.length);
  assert.equal(h.presence.particles, particles);
  assert.deepEqual(particles.map(({ address, across }) => [address, across]), addresses);
});

test("particle positions stay continuous across a flow cycle boundary", t => {
  const h = presenceHarness(t, true);
  h.presence.flow = 0.999;
  h.presence.draw();
  const before = h.dots;
  h.presence.flow = 1.001;
  h.presence.draw();
  assert.equal(h.dots.length, before.length);
  for (const [index, [x, y]] of h.dots.entries()) {
    assert.ok(Math.hypot(x - before[index][0], y - before[index][1]) < 0.05);
  }
});

test("gathering preserves the material width instead of flattening the body", t => {
  const h = presenceHarness(t, true);
  const material = h.presence.particles;
  h.presence.particles = [-0.7, 0.7].map(across => ({ address: 0.5, across, depth: 0, phase: 0, radius: 1 }));
  for (const openness of [1, 0.75, 0.5, 0.25, 0]) {
    h.presence.pose = { head: openness, body: openness, tail: openness };
    h.presence.draw();
    assert.ok(Math.hypot(h.dots[0][0] - h.dots[1][0], h.dots[0][1] - h.dots[1][1]) > 30);
  }
  h.presence.particles = material;
});

test("held postures carry material locally without replacing particles", t => {
  const h = presenceHarness(t);
  h.frame(1000);
  const before = h.dots, particles = h.presence.particles;
  for (let now = 1016; now <= 5000; now += 16) h.frame(now);
  const travel = h.dots.map(([x, y], index) => Math.hypot(x - before[index][0], y - before[index][1]));
  assert.equal(h.presence.particles, particles);
  assert.equal(h.dots.length, particles.length);
  assert.ok(travel.filter(distance => distance > 1).length > particles.length / 2);
  assert.ok(travel.every(distance => distance < 24));
});

test("the full gesture keeps every particle finite, inside the viewport, and below five pixels per 60Hz frame", t => {
  const h = presenceHarness(t);
  const particles = h.presence.particles;
  h.frame(0);
  h.presence.setState("thinking");
  let before = h.dots, largest = { distance: 0, frame: 0, particle: 0 };
  for (let frame = 1; frame <= 510; frame++) {
    h.frame(frame * 1000 / 60);
    assert.equal(h.presence.particles, particles);
    assert.equal(h.dots.length, 1700);
    for (const [index, dot] of h.dots.entries()) {
      assert.ok(dot.every(Number.isFinite));
      assert.ok(dot[0] - dot[2] > 0 && dot[0] + dot[2] < 950);
      assert.ok(dot[1] - dot[2] > 0 && dot[1] + dot[2] < 560);
      const distance = Math.hypot(dot[0] - before[index][0], dot[1] - before[index][1]);
      if (distance > largest.distance) largest = { distance, frame, particle: index };
    }
    before = h.dots;
  }
  assert.ok(largest.distance <= 5, `Particle ${largest.particle} moved ${largest.distance.toFixed(4)}px at frame ${largest.frame} (${(largest.frame * 1000 / 60).toFixed(1)}ms)`);
});

test("the entire particle trajectory matches at shared timestamps on 60 and 120Hz displays", t => {
  const h = presenceHarness(t);
  const changes = new Map([[400, "speaking"], [900, "thinking"], [1300, "listening"], [2600, "thinking"], [3200, "speaking"], [4200, "thinking"]]);
  const follow = rate => {
    Object.assign(h.presence, { previous: 0, time: 0, flow: 0, flowRate: 0.17, phase: "idle", level: 0, energy: 0, recalling: false, cycleTime: null });
    h.presence.pose = { head: 1, body: 1, tail: 1 };
    h.frame(0);
    h.presence.setState("thinking");
    const snapshots = [];
    for (let frame = 1; frame <= rate * 8.5; frame++) {
      const now = frame * 1000 / rate;
      h.frame(now);
      if (frame % (rate / 60) === 0) {
        const points = new Float64Array(h.dots.length * 2);
        for (const [index, [x, y]] of h.dots.entries()) { points[index * 2] = x; points[index * 2 + 1] = y; }
        snapshots.push({ points, flow: h.presence.flow, pose: { ...h.presence.pose } });
      }
      if (changes.has(now)) h.presence.setState(changes.get(now));
    }
    return snapshots;
  };
  const standard = follow(60), fast = follow(120);
  assert.equal(standard.length, 510);
  assert.equal(fast.length, standard.length);
  for (const [frame, snapshot] of standard.entries()) {
    assert.ok(Math.abs(snapshot.flow - fast[frame].flow) < 1e-10);
    for (const part of Object.keys(snapshot.pose)) assert.ok(Math.abs(snapshot.pose[part] - fast[frame].pose[part]) < 1e-10);
    for (let index = 0; index < snapshot.points.length; index += 2) {
      assert.ok(Math.hypot(snapshot.points[index] - fast[frame].points[index], snapshot.points[index + 1] - fast[frame].points[index + 1]) < 1e-7,
        `Particle ${index / 2} differs between refresh rates at shared frame ${frame}`);
    }
  }
});

test("memory events warm existing particles without restarting the gesture clock", t => {
  const h = presenceHarness(t);
  h.presence.setState("thinking");
  h.frame(1000);
  h.frame(1080);
  const pose = { ...h.presence.pose }, cycleTime = h.presence.cycleTime;
  const previous = h.presence.previous, particles = h.presence.particles;
  h.presence.pulseMemory();
  h.presence.pulseMemory();
  assert.equal(h.presence.memoryImpulse, 1);
  assert.equal(h.presence.memoryGlow, 0);
  assert.equal(h.presence.previous, previous);
  assert.deepEqual(h.presence.pose, pose);
  assert.equal(h.presence.cycleTime, cycleTime);
  h.frame(1096);
  assert.ok(h.presence.memoryGlow > 0 && h.presence.memoryGlow < 0.5);
  for (let now = 1112; now < 8100; now += 16) h.frame(now);
  assert.ok(h.presence.memoryGlow < 0.005);
  assert.equal(h.presence.particles, particles);
});

test("hidden pages and offscreen canvases stop scheduling and resume without a time jump", t => {
  const h = presenceHarness(t);
  h.frame(1000);
  h.frame(1050);
  assert.ok(h.presence.time > 0);
  document.hidden = true;
  h.pageListeners.get("visibilitychange")();
  assert.equal(h.frames.size, 0);
  document.hidden = false;
  h.pageListeners.get("visibilitychange")();
  const pausedTime = h.presence.time;
  h.frame(100000);
  assert.ok(h.presence.time - pausedTime <= 1 / 60 + 1e-12);
  h.intersection.callback([{ isIntersecting: false }]);
  assert.equal(h.frames.size, 0);
  h.intersection.callback([{ isIntersecting: true }]);
  assert.equal(h.frames.size, 1);
});

test("destroy removes both observers, event listeners and the pending animation", t => {
  const h = presenceHarness(t);
  h.presence.destroy();
  assert.equal(h.frames.size, 0);
  assert.equal(h.timers.size, 0);
  assert.equal(h.resize.disconnected, true);
  assert.equal(h.intersection.disconnected, true);
  assert.equal(h.pageListeners.size, 0);
  assert.equal(h.motionListeners.size, 0);
  const draws = h.draws;
  h.presence.tick(2000);
  assert.equal(h.draws, draws);
  assert.equal(h.frames.size, 0);
});
