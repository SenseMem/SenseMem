const TAU = Math.PI * 2;
const FLOW_PACE = { idle: 0.17, connecting: 0.24, listening: 0.2, thinking: 0.28, speaking: 0.23, stopping: 0.14 };
const CYCLE = { gather: 3.2, hold: 0.9, unfurl: 3.8 };
const LAG = { head: 0, body: 0.14, tail: 0.28 };
const PIGMENTS = [
  [0, [83, 86, 78], [151, 101, 69]],
  [0.46, [128, 97, 77], [181, 130, 85]],
  [1, [177, 124, 72], [208, 171, 115]],
];
const smoothstep = (from, to, value) => {
  const t = Math.max(0, Math.min(1, (value - from) / (to - from)));
  return t * t * (3 - 2 * t);
};
const smootherstep = (value) => {
  const t = Math.max(0, Math.min(1, value));
  return t ** 3 * (10 + t * (-15 + t * 6));
};

// Paired contours follow the crown, right-facing snout, S-body and curled tail of
// the SenseMem mark. Shared cubic stations keep every fibre inside that anatomy.
const CONTOURS = [
  [
    [[0.44, -0.56], [0.27, -0.56], [0.10, -0.72], [0.08, -0.94]],
    [[0.08, -0.94], [-0.02, -0.77], [-0.41, -0.91], [-0.34, -0.40]],
    [[-0.34, -0.40], [-0.31, -0.17], [-0.07, -0.18], [-0.03, 0.12]],
    [[-0.03, 0.12], [0.07, 0.37], [-0.01, 0.52], [-0.14, 0.61]],
    [[-0.14, 0.61], [-0.29, 0.72], [-0.37, 0.62], [-0.34, 0.48]],
    [[-0.34, 0.48], [-0.31, 0.33], [-0.17, 0.36], [-0.15, 0.46]],
    [[-0.15, 0.46], [-0.13, 0.53], [-0.20, 0.54], [-0.22, 0.51]],
  ],
  [
    [[0.44, -0.46], [0.26, -0.46], [0.07, -0.54], [0.02, -0.55]],
    [[0.02, -0.55], [-0.05, -0.53], [-0.08, -0.42], [-0.01, -0.34]],
    [[-0.01, -0.34], [0.08, -0.22], [0.31, -0.19], [0.36, 0.04]],
    [[0.36, 0.04], [0.44, 0.37], [0.28, 0.71], [0.04, 0.83]],
    [[0.04, 0.83], [-0.24, 1.02], [-0.58, 0.82], [-0.51, 0.47]],
    [[-0.51, 0.47], [-0.47, 0.21], [-0.07, 0.20], [-0.04, 0.46]],
    [[-0.04, 0.46], [-0.01, 0.63], [-0.25, 0.66], [-0.22, 0.51]],
  ],
];
// Match both handles at each station so a posture blend retains a continuous
// tangent, including when head, body and tail are moving at different speeds.
const CURVES = CONTOURS.map(contour => {
  const curves = structuredClone(contour);
  for (let station = 1; station < curves.length; station++) {
    const previous = curves[station - 1], next = curves[station], joint = next[0];
    const incoming = joint.map((value, axis) => value - previous[2][axis]);
    const outgoing = joint.map((value, axis) => next[1][axis] - value);
    const before = Math.hypot(...incoming), after = Math.hypot(...outgoing);
    const tangent = incoming.map((value, axis) => value / before + outgoing[axis] / after);
    const length = Math.hypot(...tangent);
    const handle = Math.min(before, after) * 1.15;
    previous[2] = joint.map((value, axis) => value - tangent[axis] / length * handle);
    next[1] = joint.map((value, axis) => value + tangent[axis] / length * handle);
  }
  return curves;
});
const TAIL = CURVES.map(contour => contour.at(-1));
const TERMINAL = TAIL[0][2].map((value, axis) => (value + TAIL[1][2][axis]) / 2);
for (const curve of TAIL) curve[2] = TERMINAL;
const SAMPLES = CONTOURS[0].flatMap((_, segment) =>
  Array.from({ length: segment === CONTOURS[0].length - 1 ? 81 : 80 }, (_, index) => {
    const t = index / 80, rest = 1 - t;
    const [left, right] = CURVES.map(contour => {
      const [start, first, second, end] = contour[segment];
      return [0, 1].map(axis => rest ** 3 * start[axis] + 3 * rest ** 2 * t * first[axis] + 3 * rest * t ** 2 * second[axis] + t ** 3 * end[axis]);
    });
    const nose = CURVES[0][0][0].map((value, axis) => (value + CURVES[1][0][0][axis] - CURVES[0][0][1][axis] - CURVES[1][0][1][axis]) / 2);
    const length = Math.hypot(...nose);
    return { x: (left[0] + right[0]) / 2, y: (left[1] + right[1]) / 2, dx: (right[0] - left[0]) / 2, dy: (right[1] - left[1]) / 2, nx: nose[0] / length, ny: nose[1] / length, distance: 0, area: 0 };
  }));
for (let index = 1; index < SAMPLES.length; index++) {
  const before = SAMPLES[index - 1], sample = SAMPLES[index];
  const length = Math.hypot(sample.x - before.x, sample.y - before.y);
  sample.distance = before.distance + length;
  sample.area = before.area + length * (Math.hypot(sample.dx, sample.dy) + Math.hypot(before.dx, before.dy));
}
const LENGTH = SAMPLES.at(-1).distance, AREA = SAMPLES.at(-1).area;
// Curl in the same direction as the open spine, facing into the broad belly.
// Aligning the ball to the snout instead would swing the body around its center;
// opposite winding adds almost two unnecessary revolutions at the tail.
const CURL_START = Math.atan2(SAMPLES[1].y - SAMPLES[0].y, SAMPLES[1].x - SAMPLES[0].x) - Math.PI * 1.5;
let arcStation = 1, previousAngle;
const RIBBON = Array.from({ length: 321 }, (_, index) => {
  const distance = index / 320 * LENGTH;
  while (arcStation < SAMPLES.length - 1 && SAMPLES[arcStation].distance < distance) arcStation++;
  const before = SAMPLES[arcStation - 1], after = SAMPLES[arcStation];
  const mix = (distance - before.distance) / (after.distance - before.distance);
  const address = (before.area + (after.area - before.area) * mix) / AREA;
  const sample = Object.fromEntries(["x", "y", "dx", "dy"].map(key => [key, before[key] + (after[key] - before[key]) * mix]));
  const angle = CURL_START + address * TAU;
  return { ...sample, address, head: 1 - smoothstep(0.18, 0.42, index / 320), tail: smoothstep(0.58, 0.82, index / 320), curlX: 0.28 * Math.cos(angle), curlY: 0.28 * Math.sin(angle), curlWidth: 0.32 };
});
for (const [index, sample] of RIBBON.entries()) {
  const before = RIBBON[Math.max(0, index - 1)], after = RIBBON[Math.min(320, index + 1)];
  const next = index < 320 ? after : before;
  let angle = Math.atan2(next.y - sample.y, next.x - sample.x) + (index === 320 ? Math.PI : 0);
  if (previousAngle !== undefined) {
    while (angle - previousAngle > Math.PI) angle -= TAU;
    while (angle - previousAngle < -Math.PI) angle += TAU;
  }
  previousAngle = angle;
  sample.angle = angle;
  sample.length = Math.hypot(next.x - sample.x, next.y - sample.y);
  sample.curlLength = Math.hypot(next.curlX - sample.curlX, next.curlY - sample.curlY);
  sample.curlAngle = CURL_START + (sample.address + next.address) / 2 * TAU + Math.PI / 2;
  sample.width = Math.hypot(sample.dx, sample.dy);
  const normal = Math.atan2(sample.dy, sample.dx) - angle + Math.PI / 2;
  sample.skew = Math.atan2(Math.sin(normal), Math.cos(normal));
  sample.weight = (after.address - before.address) / 2;
}
const CENTER = RIBBON.reduce((center, sample) => ({ x: center.x + sample.x * sample.weight, y: center.y + sample.y * sample.weight, curlX: center.curlX + sample.curlX * sample.weight, curlY: center.curlY + sample.curlY * sample.weight }), { x: 0, y: 0, curlX: 0, curlY: 0 });
// Arc length removes the cubic stations from the material. The area coordinate
// gives an even optical density through the broad belly and narrow curled tail.
let station = 1;
const FIELD = Array.from({ length: 2049 }, (_, index) => {
  const address = index / 2048, area = address * AREA;
  while (station < SAMPLES.length - 1 && SAMPLES[station].area < area) station++;
  const before = SAMPLES[station - 1], after = SAMPLES[station];
  const mix = (area - before.area) / (after.area - before.area);
  const sample = Object.fromEntries(["x", "y", "dx", "dy", "nx", "ny"].map(key => [key, before[key] + (after[key] - before[key]) * mix]));
  const progress = (before.distance + (after.distance - before.distance) * mix) / LENGTH;
  return { ...sample, progress };
});
// A deterministic blue-noise rejection pass avoids both rows and random clumps.
// These permanent material addresses are visual carriers, never an event count.
const PARTICLES = [];
const CELLS = new Map();
let randomState = 0x73656e73;
const random = () => {
  randomState += 0x6d2b79f5;
  let value = Math.imul(randomState ^ randomState >>> 15, 1 | randomState);
  value ^= value + Math.imul(value ^ value >>> 7, 61 | value);
  return ((value ^ value >>> 14) >>> 0) / 4294967296;
};
for (let candidate = 0; PARTICLES.length < 1700; candidate++) {
  const address = random();
  const across = random() * 1.94 - 0.97;
  const sample = FIELD[Math.round(address * 2048)];
  const x = sample.x + sample.dx * across, y = sample.y + sample.dy * across;
  const cellX = Math.floor(x / 0.011), cellY = Math.floor(y / 0.011);
  let clear = true;
  for (let dx = -1; dx <= 1 && clear; dx++) for (let dy = -1; dy <= 1 && clear; dy++) {
    for (const neighbor of CELLS.get(`${cellX + dx},${cellY + dy}`) || []) {
      if (Math.hypot(x - neighbor.x, y - neighbor.y) < 0.011) { clear = false; break; }
    }
  }
  if (!clear) continue;
  const particle = Object.freeze({ address, across, depth: random() * 2 - 1, phase: random() * TAU, radius: 0.64 + random() * 0.46 });
  PARTICLES.push(particle);
  const key = `${cellX},${cellY}`;
  if (!CELLS.has(key)) CELLS.set(key, []);
  CELLS.get(key).push({ x, y });
}
Object.freeze(PARTICLES);
CELLS.clear();

export class Presence {
  constructor(canvas, sampleLevel = () => 0) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.sampleLevel = sampleLevel;
    this.phase = "idle";
    this.recalling = false;
    this.pose = { head: 1, body: 1, tail: 1 };
    this.cycleTime = null;
    this.level = 0;
    this.energy = 0;
    this.particles = PARTICLES;
    this.geometry = new Float64Array(RIBBON.length * 7);
    this.memoryImpulse = 0;
    this.memoryGlow = 0;
    this.time = 0;
    this.flow = 0;
    this.flowRate = FLOW_PACE.idle;
    this.previous = null;
    this.dirty = true;
    this.visible = true;
    this.disposed = false;
    this.motion = matchMedia("(prefers-reduced-motion: reduce)");
    this.tick = (now) => {
      if (this.disposed) return;
      const elapsed = this.previous === null ? 1 / 60 : Math.max(0, Math.min((now - this.previous) / 1000, 0.05));
      this.previous = now;
      const level = Math.max(0, Math.min(1, this.sampleLevel()));
      const changed = Math.abs(level - this.level) > 0.012;
      this.level += (level - this.level) * (1 - Math.exp(-13.4 * elapsed));
      this.energy += ((this.phase === "idle" ? 0 : this.level) - this.energy) * (1 - Math.exp(-3.2 * elapsed));
      const previousGlow = this.memoryGlow;
      const impulseDecay = Math.exp(-1.2 * elapsed), glowDecay = Math.exp(-5.5 * elapsed);
      this.memoryGlow = this.memoryGlow * glowDecay + this.memoryImpulse * 5.5 / 4.3 * (impulseDecay - glowDecay);
      this.memoryImpulse *= impulseDecay;
      const memoryChanged = this.memoryGlow > 0.005 || Math.abs(previousGlow - this.memoryGlow) > 0.001;
      if (!this.motion.matches || changed || memoryChanged || this.dirty) {
        if (!this.motion.matches) {
          this.time += elapsed;
          const rate = FLOW_PACE[this.phase] + this.energy * 0.2;
          const decay = Math.exp(-2.2 * elapsed);
          this.flow += rate * elapsed + (this.flowRate - rate) * (1 - decay) / 2.2;
          this.flowRate = rate + (this.flowRate - rate) * decay;
          if (this.cycleTime !== null) {
            this.cycleTime += elapsed;
            const opening = this.cycleTime - CYCLE.gather - CYCLE.hold;
            for (const [part, lag] of Object.entries(LAG)) {
              // One committed clock carries a small wave along the anatomy.
              // Both ends of each leg have zero velocity and acceleration.
              this.pose[part] = opening >= 0
                ? smootherstep((opening - lag) / (CYCLE.unfurl - LAG.tail))
                : 1 - smootherstep((this.cycleTime - lag) / (CYCLE.gather - LAG.tail));
            }
            if (opening >= CYCLE.unfurl) this.cycleTime = null;
          }
        }
        this.draw();
        this.dirty = false;
      }
      this.schedule();
    };
    this.restart = () => {
      cancelAnimationFrame(this.frame);
      clearTimeout(this.timer);
      this.previous = null;
      this.invalidate();
      this.schedule();
    };
    this.resize = new ResizeObserver(() => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(devicePixelRatio || 1, 2);
      this.width = bounds.width;
      this.height = bounds.height;
      this.materials = null;
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.draw();
      this.restart();
    });
    this.intersection = new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
      this.restart();
    });
    this.resize.observe(canvas);
    this.intersection.observe(canvas);
    this.motion.addEventListener("change", this.restart);
    document.addEventListener("visibilitychange", this.restart);
  }

  schedule() {
    if (this.disposed || !this.visible || document.hidden) return;
    if (this.motion.matches) this.timer = setTimeout(() => this.tick(performance.now()), 100);
    else this.frame = requestAnimationFrame(this.tick);
  }

  setState(phase) {
    if (this.phase === phase) return;
    this.phase = phase;
    if (phase === "connecting" || phase === "thinking") this.gather();
    this.invalidate();
  }

  setRecall(active) {
    if (this.recalling === active) return;
    this.recalling = active;
    if (active) this.gather();
    this.invalidate();
  }

  gather() {
    // ASR hypotheses and duplex call states can change many times per second.
    // Coalesce them into the current gesture; never reverse or queue a new leg.
    if (!this.motion.matches && this.cycleTime === null) this.cycleTime = 0;
  }

  invalidate() {
    this.dirty = true;
    if (!this.motion.matches || this.disposed) return;
    this.cycleTime = null;
    for (const part of Object.keys(LAG)) this.pose[part] = 1;
    if (this.visible && !document.hidden) {
      this.draw();
      this.dirty = false;
    }
  }

  pulseMemory() {
    this.memoryImpulse = Math.min(1, this.memoryImpulse + 0.75);
    this.dirty = true;
  }

  draw() {
    const context = this.context;
    const width = this.width || 0, height = this.height || 0;
    context.clearRect(0, 0, width, height);
    if (!width || !height) return;
    const flow = this.flow;
    const energy = this.motion.matches ? 0 : this.energy;
    // One fixed frame keeps the presence at its full scale through interruptions;
    // fitting each posture separately would make the presence breathe in scale.
    const scale = Math.min(width * 0.7, height * (width < 480 ? 0.34 : 0.43));
    const geometry = this.geometry;
    let centerX = 0, centerY = 0, sumX = 0, sumY = 0;
    for (const [index, sample] of RIBBON.entries()) {
      const openness = sample.head * this.pose.head + (1 - sample.head - sample.tail) * this.pose.body + sample.tail * this.pose.tail;
      // Interpolate tangent and arc length, then integrate the spine. Transporting
      // its cross-section keeps a real volume while the tail unwinds; a Cartesian
      // blend would flatten opposing normals to a string halfway through a curl.
      const angle = sample.curlAngle + (sample.angle - sample.curlAngle) * openness;
      const section = angle - Math.PI / 2 + sample.skew * openness;
      const width = sample.curlWidth + (sample.width - sample.curlWidth) * openness;
      const offset = index * 7;
      geometry[offset] = centerX;
      geometry[offset + 1] = centerY;
      geometry[offset + 2] = Math.cos(section) * width;
      geometry[offset + 3] = Math.sin(section) * width;
      geometry[offset + 4] = openness;
      geometry[offset + 5] = -Math.cos(angle);
      geometry[offset + 6] = -Math.sin(angle);
      sumX += centerX * sample.weight;
      sumY += centerY * sample.weight;
      const length = sample.curlLength + (sample.length - sample.curlLength) * openness;
      centerX += Math.cos(angle) * length;
      centerY += Math.sin(angle) * length;
    }
    const anchorX = CENTER.curlX + (CENTER.x - CENTER.curlX) * this.pose.body - sumX;
    const anchorY = CENTER.curlY + (CENTER.y - CENTER.curlY) * this.pose.body - sumY;
    context.save();
    context.translate(width * 0.51, height * 0.5);
    if (!this.materials) {
      this.materials = Array.from({ length: 33 }, (_, index) => {
        const material = context.createLinearGradient(-scale * 0.3, -scale * 0.65, scale * 0.25, scale * 0.8);
        for (const [stop, base, warm] of PIGMENTS) {
          const channels = base.map((value, channel) => Math.round(value + (warm[channel] - value) * index / 32));
          material.addColorStop(stop, `rgb(${channels.join(",")})`);
        }
        return material;
      });
    }
    context.fillStyle = this.materials[Math.round(Math.min(1, this.memoryGlow * 1.4) * 32)];
    const dotScale = Math.max(0.8, Math.min(1.2, scale / 220));
    for (const particle of this.particles) {
      const { address, across, depth, phase, radius } = particle;
      // Closed local streamlines move material along and across the body. Their
      // sine gates vanish at both ends: no modulo, respawn, or tail-to-head jump.
      const current = flow * 1.15 - address * 11;
      // Each permanent grain drifts at its own phase within the shared current.
      // Small edge-gated excursions keep the silhouette intact without a moving skin.
      const drift = flow * (0.5 + depth * 0.08) + phase;
      const along = address + Math.sin(Math.PI * address) * (0.008 * Math.sin(current + across * 2.1) + 0.003 * Math.sin(drift));
      const transverse = across + (1 - across * across) * (0.08 * Math.cos(current) + 0.03 * Math.sin(flow * 0.55 - address * 16) + 0.035 * Math.sin(drift * 0.83 + phase));
      const lookup = along * 2048, index = Math.min(2047, Math.floor(lookup)), mix = lookup - index;
      const before = FIELD[index], after = FIELD[index + 1];
      const progress = before.progress + (after.progress - before.progress) * mix;
      const location = progress * 320, segment = Math.min(319, Math.floor(location)), blend = location - segment;
      const offset = segment * 7, next = offset + 7;
      const openness = geometry[offset + 4] + (geometry[next + 4] - geometry[offset + 4]) * blend;
      const voice = Math.exp(-(((progress - 0.4) / 0.24) ** 2));
      const breath = 1 + energy * voice * 0.03 + (this.motion.matches ? 0 : this.memoryGlow * voice * 0.012);
      // Distribute the gathered material by circular area, not linear radius:
      // otherwise the polar fold piles particles onto a dense central spoke.
      const gathered = (0.6 * Math.sqrt((transverse + 1) / 2) - 0.28) / 0.32;
      const thread = (transverse + (gathered - transverse) * (1 - openness)) * breath;
      const cap = Math.sqrt(1 - transverse * transverse) * Math.exp(-progress * 100) * 0.035;
      const bodyX = geometry[offset] + (geometry[next] - geometry[offset]) * blend + (geometry[offset + 2] + (geometry[next + 2] - geometry[offset + 2]) * blend) * thread;
      const bodyY = geometry[offset + 1] + (geometry[next + 1] - geometry[offset + 1]) * blend + (geometry[offset + 3] + (geometry[next + 3] - geometry[offset + 3]) * blend) * thread;
      const volume = Math.sqrt(1 - transverse * transverse);
      const z = volume * (depth * 0.88 + Math.sin(flow * 0.24 - address * 6 + phase) * 0.12);
      const x = (bodyX + anchorX + cap * geometry[offset + 5] * openness + z * 0.009) * scale;
      const y = (bodyY + anchorY + cap * geometry[offset + 6] * openness - z * 0.014) * scale;
      const focus = 0.5 + z * 0.5;
      const grain = (radius - 0.64) / 0.46;
      context.globalAlpha = 0.34 + volume * 0.12 + focus * 0.26 + grain * 0.12 + this.memoryGlow * 0.03;
      context.beginPath();
      context.arc(x, y, (0.48 + 0.78 * grain ** 2.1) * dotScale * (0.78 + focus * 0.4), 0, TAU);
      context.fill();
    }
    context.restore();
    context.globalAlpha = 1;
  }

  destroy() {
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    clearTimeout(this.timer);
    this.resize.disconnect();
    this.intersection.disconnect();
    this.motion.removeEventListener("change", this.restart);
    document.removeEventListener("visibilitychange", this.restart);
  }
}
