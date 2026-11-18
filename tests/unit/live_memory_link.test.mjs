import assert from "node:assert/strict";
import test from "node:test";
import { MemoryLink } from "../../examples/live/static/memory-link.js";

function fixture() {
  const sent = [], events = [], interruptions = [];
  const state = { userSpeaking: false, playing: false, generating: false, responseId: null,
    lastUserAt: Date.now() - 2000, lastOutputAt: Date.now() - 2000 };
  const link = new MemoryLink({ sourceId: "live:test", current: () => true, getState: () => state,
    instructions: "base policy", send: value => { sent.push(value); return true; },
    emit: (name, detail) => events.push({ name, detail }), interrupt: () => {
      interruptions.push(state.responseId); state.playing = false; state.generating = true;
    } });
  const feedback = (extra = {}) => ({ type: "memory_feedback", status: "ready", source_id: "live:test",
    memory_revision: 0, feedback_id: "feedback", expires_at: new Date(Date.now() + 30_000).toISOString(),
    facts: [{ memory_id: "record", text: "钥匙在绿色盒子里。" }], ...extra });
  const candidate = (extra = {}) => ({ type: "proactive", source_id: "live:test", memory_revision: 0,
    candidate_id: "candidate", event_id: "event", utterance: "钥匙在绿色盒子里。",
    context: "reviewed original", expires_at: new Date(Date.now() + 30_000).toISOString(), ...extra });
  const ack = () => link.ack({ type: "session.updated", session: { instructions: link.pending.instructions } });
  return { link, sent, events, state, feedback, candidate, ack, interruptions };
}

test("native updates preserve policy, coalesce pending evidence and require exact server acknowledgement", () => {
  const f = fixture();
  f.link.handle(f.feedback());
  assert.equal(f.sent[0].type, "session.update");
  assert.ok(f.sent[0].session.instructions.startsWith("base policy"));
  assert.ok(f.sent[0].session.instructions.includes("绿色盒子"));
  assert.equal(f.link.ack({ session: { instructions: "different" } }), null);
  assert.equal(f.events.length, 0);
  f.link.handle(f.feedback({ feedback_id: "new", facts: [{ memory_id: "new-record", text: "现在是蓝色盒子。" }] }));
  assert.equal(f.sent.length, 1);
  assert.equal(f.ack().feedback_id, "feedback");
  assert.equal(f.sent.length, 2);
  assert.equal(f.ack().feedback_id, "new");
  assert.equal(f.events.at(-1).detail.status, "received");
  assert.ok(!("adopted" in f.events.at(-1).detail));
});

test("revision changes withdraw previous evidence immediately and stale acknowledgements confer no receipt", () => {
  const f = fixture();
  f.link.handle(f.feedback());
  const old = f.link.pending.instructions;
  f.link.handle({ type: "invalidated", memory_revision: 1 });
  assert.equal(f.link.responseStarted(), null);
  assert.equal(f.link.ack({ session: { instructions: old } }), null);
  assert.ok(!f.sent.at(-1).session.instructions.includes("绿色盒子"));
  assert.equal(f.ack().operation, "withdraw");
  f.link.handle(f.feedback());
  assert.equal(f.sent.length, 2);
});

test("source and expiry checks reject unrelated or expired evidence", () => {
  const f = fixture();
  f.link.handle(f.feedback({ source_id: "other" }));
  f.link.handle(f.feedback({ expires_at: "2000-01-01T00:00:00Z" }));
  assert.equal(f.sent.length, 0);
  f.link.handle(f.feedback()); f.ack();
  f.link.feedback.expires_at = "2000-01-01T00:00:00Z";
  f.link.tick();
  assert.equal(f.link.responseStarted(), null);
  assert.ok(!f.sent.at(-1).session.instructions.includes("绿色盒子"));
});

test("proactive responses yield to the user and preserve evidence without fabricating speech", () => {
  const f = fixture();
  f.link.handle(f.feedback()); f.ack();
  f.state.userSpeaking = true;
  f.link.handle(f.candidate());
  assert.equal(f.sent.length, 1);
  f.state.userSpeaking = false;
  f.link.tick();
  assert.equal(f.sent.at(-1).type, "response.create");
  assert.ok(f.sent.at(-1).response.instructions.includes("绿色盒子"));
  assert.ok(!f.sent.some(value => value.type === "conversation.item.create"));
  assert.equal(f.link.responseStarted({ sensemem_candidate_id: "candidate" }).candidate_id, "candidate");
  assert.equal(f.link.responseStarted().action, "quiet");
  f.link.handle(f.candidate());
  assert.equal(f.sent.length, 2);
});

test("a visual correction cancels its owner once and waits for native cancellation before replacement", () => {
  const f = fixture();
  Object.assign(f.state, { generating: true, playing: true, responseId: "response" });
  f.link.handle(f.candidate({ type: "visual_response", action: "interrupt", target_response_id: "response" }));
  assert.deepEqual(f.interruptions, ["response"]);
  assert.equal(f.sent.length, 0);
  f.link.tick();
  assert.deepEqual(f.interruptions, ["response"]);
  f.state.generating = false;
  f.link.tick();
  assert.equal(f.sent[0].type, "response.create");
  assert.equal(f.sent.length, 1);
});

test("a mismatched correction cannot cancel another response and stop prevents late work", () => {
  const f = fixture();
  Object.assign(f.state, { generating: true, responseId: "current" });
  f.link.handle(f.candidate({ type: "visual_response", action: "interrupt", target_response_id: "old" }));
  assert.deepEqual(f.interruptions, []);
  f.link.stop();
  f.link.handle(f.feedback());
  f.link.tick();
  assert.equal(f.sent.length, 0);
});

test("acknowledgements identify the exact final ASR and distinguish visual observation from sensory recall", () => {
  const f = fixture();
  const expires_at = new Date(Date.now() + 20_000).toISOString();
  f.link.handle({ type: "asr_recall", source_id: "live:test", memory_revision: 0,
    turn_id: "turn", revision: 3, query: "钥匙", final: true, input_final: true,
    context: "检索证据", expires_at });
  const receipt = f.ack();
  assert.deepEqual(receipt.asr_recall, { turn_id: "turn", revision: 3, query: "钥匙", final: true,
    input_final: true, status: "ready", expires_at });
  assert.equal(receipt.perception, null);
  f.link.handle({ type: "sensory_recall", source_id: "live:test", memory_revision: 0,
    context: "感官预检索", expires_at });
  assert.equal(f.ack().perception, null);
  const occurred_end = new Date(Date.now() - 200).toISOString();
  f.link.handle({ type: "visual_context", source_id: "live:test", memory_revision: 0,
    perception_id: "observed", context: "蓝色方块", occurred_end, expires_at });
  const visual = f.ack();
  assert.deepEqual(visual.perception, { source_id: "live:test", perception_id: "observed", occurred_end, expires_at });
  assert.equal(visual.asr_recall.turn_id, "turn");
});

test("proactive speech waits while the current user answer is waiting for SenseMem", () => {
  const f = fixture();
  f.state.awaitingAnswer = true;
  f.link.handle(f.candidate());
  assert.equal(f.sent.length, 0);
  f.state.awaitingAnswer = false;
  f.link.tick();
  assert.equal(f.sent.at(-1).type, "response.create");
});
