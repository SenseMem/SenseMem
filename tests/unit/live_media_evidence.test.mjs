import assert from "node:assert/strict";
import test from "node:test";
import { createMediaEvidence } from "../../examples/live/static/media-evidence.js";

class Node extends EventTarget {
  constructor(tagName) {
    super();
    this.tagName = tagName;
    this.children = [];
    this.attributes = new Map();
    this.selectedIndex = 0;
    this.duration = NaN;
    this.actions = [];
  }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this.attributes.set(name, value); }
  pause() { this.actions.push("pause"); }
  set src(value) { this.source = value; this.actions.push(["src", value]); }
  get src() { return this.source; }
  load() { this.actions.push("load"); this.duration = NaN; }
}

function render(t, assets) {
  const original = Object.getOwnPropertyDescriptor(globalThis, "document");
  globalThis.document = {
    createDocumentFragment: () => new Node("fragment"),
    createElement: (tagName) => new Node(tagName),
  };
  t.after(() => {
    if (original) Object.defineProperty(globalThis, "document", original);
    else delete globalThis.document;
  });
  return createMediaEvidence(assets);
}

function asset(url, media_type = "video/webm") { return { url, media_type }; }

test("segments share one native player and keep their original order without autoplay", (t) => {
  const result = render(t, [asset("/third"), asset("/first"), asset("/second")]);
  assert.equal(result.children.length, 1);
  const [media, heading] = result.children[0].children;
  const selector = heading.children.find((node) => node.tagName === "select");
  assert.equal(media.tagName, "video");
  assert.equal(media.controls, true);
  assert.equal(media.preload, "metadata");
  assert.equal(Boolean(media.autoplay), false);
  assert.equal(selector.attributes.get("aria-label"), "选择原片片段");
  assert.deepEqual(selector.children.map((node) => node.textContent), ["片段 1 / 3", "片段 2 / 3", "片段 3 / 3"]);
  assert.equal(media.src, "/third");
  selector.selectedIndex = 1;
  selector.dispatchEvent(new Event("change"));
  assert.deepEqual(media.actions.slice(-3), ["pause", ["src", "/first"], "load"]);
  selector.selectedIndex = 2;
  selector.dispatchEvent(new Event("change"));
  assert.equal(media.src, "/second");
});

test("duration comes from current metadata, clears on switching, and never invents unknown durations", (t) => {
  const result = render(t, [asset("/one"), asset("/two")]);
  const [media, heading] = result.children[0].children;
  const duration = heading.children.find((node) => node.className === "media-duration");
  const selector = heading.children.find((node) => node.tagName === "select");
  assert.equal(duration.hidden, true);
  media.duration = 3.055;
  media.dispatchEvent(new Event("loadedmetadata"));
  assert.equal(duration.textContent, "3.1 秒");
  assert.equal(duration.hidden, false);
  selector.selectedIndex = 1;
  selector.dispatchEvent(new Event("change"));
  assert.equal(duration.textContent, "");
  assert.equal(duration.hidden, true);
  for (const seconds of [NaN, Infinity]) {
    media.duration = seconds;
    media.dispatchEvent(new Event("loadedmetadata"));
    assert.equal(duration.textContent, "");
    assert.equal(duration.hidden, true);
  }
  media.duration = 2.9;
  media.dispatchEvent(new Event("loadedmetadata"));
  assert.equal(duration.textContent, "2.9 秒");
  assert.equal(duration.hidden, false);
});

test("images remain visible and mixed audio/video sources all remain reachable", (t) => {
  const result = render(t, [asset("/video-one"), asset("/image-one", "image/png"), asset("/audio", "audio/ogg"), asset("/video-two"), asset("/image-two", "image/jpeg")]);
  assert.deepEqual(result.children.map((figure) => figure.children[0].tagName), ["video", "img", "audio", "img"]);
  const images = result.children.filter((figure) => figure.children[0].tagName === "img");
  assert.deepEqual(images.map((figure) => figure.children[0].src), ["/image-one", "/image-two"]);
  assert.ok(images.every((figure) => figure.children[0].loading === "lazy"));
  const [audio, audioHeading] = result.children[2].children;
  assert.equal(audio.src, "/audio");
  assert.equal(audioHeading.children.some((node) => node.tagName === "select"), false);
  const [video, videoHeading] = result.children[0].children;
  const selector = videoHeading.children.find((node) => node.tagName === "select");
  selector.selectedIndex = 1;
  selector.dispatchEvent(new Event("change"));
  assert.equal(video.src, "/video-two");
});

test("a single original has no selector and empty evidence stays empty", (t) => {
  const result = render(t, [asset("/original")]);
  assert.equal(result.children[0].children[1].children.some((node) => node.tagName === "select"), false);
  assert.equal(createMediaEvidence([]).children.length, 0);
});
