import assert from "node:assert/strict";
import test from "node:test";
import { groupRecords, hasNewlyFormedMemory, reconcileRecall, recordDescription, recordState, updateRecall } from "../../examples/live/static/memory-state.js";

test("dialogue presentation uses attributed text instead of legacy boilerplate", () => {
  const record = { content: "旧版记录(内容真实性及实际听到未核实)", assets: [], dialogue: {
    user: { item_id: "user-1", text: "请解释‘未核实’是什么意思。" },
    assistant: { item_ids: ["reply-1"], text: "表示还没有确认。" }, completion: "interrupted",
  } };
  assert.equal(recordDescription(record), "你：\n请解释‘未核实’是什么意思。\n\nSenseMem：\n表示还没有确认。");
  assert.equal(recordDescription({ ...record, dialogue: { ...record.dialogue, user: null, assistant: { item_ids: ["reply-1"], text: "" } } }), "SenseMem：\n语音回复");
  assert.equal(recordDescription({ ...record, dialogue: null }), record.content);
});

test("memory formation feedback follows new settled records, not polling or metadata changes", () => {
  const formed = { id: "past", pending: false, error: null };
  assert.equal(hasNewlyFormedMemory([formed], [{ ...formed, visible: false }]), false);
  assert.equal(hasNewlyFormedMemory([formed], [formed, { id: "new", pending: true }]), false);
  assert.equal(hasNewlyFormedMemory([formed], [formed, { id: "new", error: "unavailable" }]), false);
  assert.equal(hasNewlyFormedMemory([formed], [formed, { id: "new", pending: false }]), true);
  assert.equal(hasNewlyFormedMemory([{ id: "new", pending: true }], [{ id: "new", pending: false }]), true);
  assert.equal(hasNewlyFormedMemory([{ id: "new", error: "unavailable" }], [{ id: "new", error: null }]), true);
  assert.equal(hasNewlyFormedMemory([formed], []), false);
});

test("cautious trait presentation follows native visibility, not source count", () => {
  const trait = { kind: "trait", basis: "model_inference", evidence_ids: ["same-source-1", "same-source-2"] };
  assert.equal(recordState({ ...trait, visible: false }).label, "证据不足");
  assert.equal(recordState({ ...trait, visible: true }).label, "模型推断");
  assert.equal(recordState({ ...trait, forgotten_at: "2026-11-16T00:00:00Z" }).label, "已遗忘");
});

test("forgotten and superseded records remain discoverable separately from current understanding", () => {
  const records = [
    { id: "clip", kind: "observation" }, { id: "preference", kind: "state" },
    { id: "old", kind: "trait", retired_at: "2026-11-16T00:00:00Z" },
    { id: "forgotten", kind: "observation", forgotten_at: "2026-11-16T00:00:00Z" },
  ];
  const groups = groupRecords(records);
  assert.deepEqual(groups.experiences.map(({ id }) => id), ["clip"]);
  assert.deepEqual(groups.profile.map(({ id }) => id), ["preference"]);
  assert.deepEqual(groups.history.map(({ id }) => id), ["old", "forgotten"]);
});

test("a changed draft clears preceding evidence and stale completion cannot become final", () => {
  const ready = { revision: 1, query: "上次的钥匙", status: "ready", hits: [{ id: "old" }], final: false };
  const searching = updateRecall(ready, { revision: 2, query: "上次的钥匙放在", status: "searching" });
  assert.deepEqual(searching.hits, []);
  assert.equal(searching.final, false);
  assert.equal(updateRecall(searching, { revision: 1, query: "上次的钥匙", status: "final", hits: [{ id: "old" }] }), searching);
  const final = updateRecall(searching, { revision: 2, query: "上次的钥匙放在哪里", status: "final", hits: [{ id: "key" }] });
  assert.equal(final.final, true);
});

test("new speech clears final evidence while an HTTP final keeps the stream revision", () => {
  const ready = { revision: 8, query: "钥匙在哪", status: "ready", hits: [{ id: "key" }], final: false };
  const final = updateRecall(ready, { query: "钥匙在哪", status: "final", hits: [{ id: "key" }], media_review: { source_ids: ["key"], abstained: false } });
  assert.equal(final.revision, 8);
  assert.equal(final.final, true);
  assert.deepEqual(final.media_review.source_ids, ["key"]);
  assert.equal(updateRecall(final, { revision: 8, query: "钥匙在哪", status: "ready", hits: [] }), final);
  const next = updateRecall(final, { status: "listening" });
  assert.equal(next.revision, -1);
  assert.deepEqual(next.hits, []);
  assert.equal(next.final, false);
});

test("native event summaries remain explicitly unverified model interpretations", () => {
  assert.equal(recordState({ kind: "observation", event_summary: "model synopsis", event_summary_verified: false }).label, "模型理解，待核实");
  assert.equal(recordState({ event_summary: "model synopsis", forgotten_at: "2026-11-16T00:00:00Z" }).label, "已遗忘");
});

test("native refresh withdraws recalled items and undo restores the same query without stale evidence", () => {
  const hit = { id: "remembered", content: "Earlier interpretation", visible: true, assets: [] };
  const recalled = { revision: 2, query: "habit", status: "final", final: true, hits: [hit], media_review: { source_ids: ["remembered"], abstained: false } };
  const forgotten = reconcileRecall(recalled, [{ ...hit, forgotten_at: "2026-11-16T00:00:00Z" }]);
  assert.deepEqual(forgotten.hits, []);
  assert.equal(forgotten.media_review, null);
  const restored = reconcileRecall(forgotten, [{ ...hit, content: "Current native interpretation", forgotten_at: null }]);
  assert.deepEqual(restored.hits.map(({ id, content }) => ({ id, content })), [{ id: "remembered", content: "Current native interpretation" }]);
  assert.equal(restored.media_review, null);
  assert.deepEqual(reconcileRecall(restored, [{ ...hit, retired_at: "2026-11-16T00:00:00Z" }]).hits, []);
  assert.deepEqual(reconcileRecall(restored, [{ ...hit, visible: false }]).hits, []);
});

test("late cached results obey current eligibility and failed unknown refreshes cannot retain old hits", () => {
  const hit = { id: "outside-list", content: "Earlier cached result", visible: true };
  const incoming = updateRecall({ revision: -1, hits: [] }, { revision: 3, query: "past", status: "ready", hits: [hit] });
  assert.deepEqual(reconcileRecall(incoming, [{ ...hit, visible: false }]).hits, []);
  assert.deepEqual(reconcileRecall(incoming, [], { excludeUnknown: true }).hits, []);
  assert.equal(reconcileRecall(incoming, []).hits.length, 1);
  assert.equal(reconcileRecall(incoming, [{ ...hit, content: "Fresh native result" }], { excludeUnknown: true }).hits[0].content, "Fresh native result");
});
