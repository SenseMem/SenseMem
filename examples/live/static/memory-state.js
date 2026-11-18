const profileKinds = new Set(["entity", "state", "relation", "affect", "trait", "response_policy"]);

export function hasNewlyFormedMemory(previous, records) {
  const known = new Map(previous.map(record => [record.id, record]));
  return records.some(record => {
    if (record.pending || record.error) return false;
    const before = known.get(record.id);
    return !before || Boolean(before.pending || before.error);
  });
}

export function recordState(record) {
  if (record.forgotten_at) return { label: "已遗忘", tone: "inactive" };
  if (record.retired_at) return { label: "历史版本", tone: "inactive" };
  if (record.error) return { label: "分析未完成", tone: "failed" };
  if (record.pending) return { label: "理解中", tone: "processing" };
  if (record.visible === false) return {
    label: record.kind === "trait" && record.basis === "model_inference" ? "证据不足" : "暂不召回",
    tone: "cautious",
  };
  if (record.event_summary && !record.event_summary_verified) return { label: "模型理解，待核实", tone: "cautious" };
  return record.basis === "model_inference"
    ? { label: "模型推断", tone: "inference" }
    : { label: "已保存", tone: "ready" };
}

export function groupRecords(records) {
  const groups = { experiences: [], profile: [], history: [] };
  for (const record of records) {
    if (record.forgotten_at || record.retired_at) groups.history.push(record);
    else if (profileKinds.has(record.kind)) groups.profile.push(record);
    else groups.experiences.push(record);
  }
  return groups;
}

export function updateRecall(current, incoming) {
  if (incoming.status === "listening" && !incoming.query) return {
    revision: -1, query: "", status: "listening", hits: [], final: false, media_review: null,
  };
  const revision = incoming.revision ?? current.revision;
  if (revision < current.revision) return current;
  const query = incoming.query ?? current.query;
  const changed = revision !== current.revision || query !== current.query;
  const final = incoming.final === true || incoming.status === "final";
  if (current.final && !changed && !final) return current;
  return {
    revision,
    query,
    status: incoming.status,
    hits: incoming.hits || (changed ? [] : current.hits),
    candidates: incoming.hits || (changed ? [] : current.candidates || current.hits),
    final,
    media_review: incoming.media_review || (changed ? null : current.media_review),
  };
}

export function reconcileRecall(current, records, { excludeUnknown = false } = {}) {
  const latest = new Map(records.map((record) => [record.id, record]));
  const candidates = current.candidates || current.hits;
  const eligible = (record) => record.visible !== false && !record.retired_at && !record.forgotten_at && !record.pending && !record.error;
  const hits = [];
  for (const candidate of candidates) {
    const record = latest.get(candidate.id);
    if (!record && excludeUnknown) continue;
    const fresh = record ? { ...candidate, ...record } : candidate;
    if (eligible(fresh)) hits.push(fresh);
  }
  const review = current.media_review;
  const invalidReview = hits.length !== candidates.length || review?.source_ids.some((id) => latest.has(id) && !eligible(latest.get(id)));
  return { ...current, candidates, hits, media_review: invalidReview ? null : review };
}
export function recordDescription(record) {
  if (record.dialogue) {
    const { user, assistant } = record.dialogue;
    return (user ? `你：\n${user.text}\n\n` : "") + `SenseMem：\n${assistant.text.trim() ? assistant.text : "语音回复"}`;
  }
  if (record.event_summary) return record.event_summary;
  if (record.content) return record.content;
  if (record.pending) return "原始内容已保存，正在理解。";
  if (record.assets.some((asset) => asset.media_type.startsWith("image/"))) return "留存的原始画面";
  if (record.assets.some((asset) => asset.media_type.startsWith("video/"))) return "留存的原始音视频";
  if (record.assets.some((asset) => asset.media_type.startsWith("audio/"))) return "留存的原始音频";
  return "保存的记忆";
}
