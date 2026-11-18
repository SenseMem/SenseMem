import { Presence } from "./presence.js";
import { CallSession } from "./session.js";
import { groupRecords, hasNewlyFormedMemory, reconcileRecall, recordDescription, recordState, updateRecall } from "./memory-state.js";
import { createMediaEvidence } from "./media-evidence.js";

const $ = (id) => document.getElementById(id);
const workspace = $("workspace");
const video = $("local-video");
const callButton = $("toggle-call");
const micButton = $("toggle-mic");
const cameraButton = $("toggle-camera");
const memoryButton = $("toggle-memory");
const tabs = [$("timeline-tab"), $("profile-tab"), $("retrieval-tab"), $("history-tab")];
const timeFormat = new Intl.DateTimeFormat("zh-CN", {
  month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
});
const kindLabels = {
  observation: "共同经历", event: "事件", entity: "事物", state: "近况与偏好",
  relation: "关系", affect: "此刻的感受", trait: "长期特征", response_policy: "相处方式",
};
const intentLabels = { reinforce: "强化", consolidate: "整合", correct: "纠正", forget: "遗忘", identify: "识别" };
let call;
let saved = [];
let memoriesLoaded = false;
let latestRecallCards = [];
let operations = [];
let selectedTab = 0;
let refreshTask = null;
let captureNoticeShown = false;
let refreshQueued = false;
let sensoryRecall = { revision: -1, query: "", status: "idle", hits: [], final: false };
let feedbackReceipt = null;
const detailsCache = new Map();
const detailRequests = new Map();
const presence = new Presence($("presence"), () => call?.getLevel() || 0);
const recordList = document.createElement("div");
recordList.className = "record-list";
const operationList = document.createElement("div");
operationList.className = "operation-list";
$("memory-content").replaceChildren(recordList, operationList);

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showNotification(message, action) {
  $("notice-text").textContent = message;
  const button = $("notice-action");
  button.hidden = !action;
  button.textContent = action?.label || "";
  button.onclick = action ? () => { $("notice").hidden = true; void action.run(); } : null;
  $("notice").hidden = false;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error("操作未完成，请检查本地模型服务后重试。");
  return response.json();
}

function setMemoryPanelVisible(visible) {
  $("memory-rail").hidden = !visible;
  workspace.classList.toggle("memory-hidden", !visible);
  memoryButton.setAttribute("aria-expanded", String(visible));
  if (!visible) $("memory-content").querySelectorAll("video, audio").forEach((media) => media.pause());
}

function renderCameraState() {
  const enabled = call?.active && video.srcObject?.getVideoTracks()[0]?.enabled;
  $("camera-empty").hidden = Boolean(enabled);
  $("camera-empty").querySelector("span").textContent = call?.phase === "connecting"
    ? "正在开启…" : call?.active ? "摄像头已关闭" : "摄像头未开启";
}

function renderCallState(phase) {
  const messages = {
    idle: ["未连接", "开始通话"], connecting: ["连接中", "取消连接"],
    listening: ["聆听中", "结束通话"], thinking: ["回忆中", "结束通话"],
    speaking: ["回应中", "结束通话"], stopping: ["保存中", "正在结束"],
  };
  const [connection, button] = messages[phase];
  $("connection-state").textContent = connection;
  $("call-label").textContent = button;
  const connected = ["listening", "thinking", "speaking"].includes(phase);
  $("connection-dot").classList.toggle("connected", connected);
  callButton.classList.toggle("active", connected);
  callButton.setAttribute("aria-busy", String(phase === "connecting" || phase === "stopping"));
  callButton.disabled = phase === "stopping";
  micButton.disabled = cameraButton.disabled = !connected;
  presence.setState(phase);
  if (!connected) presence.setRecall(false);
  if (phase === "connecting") {
    feedbackReceipt = null;
    sensoryRecall = { revision: -1, query: "", status: "idle", hits: [], final: false };
    $("live-observation").hidden = true;
    renderRecall();
  }
  if (phase === "idle") {
    $("live-observation").hidden = true;
    micButton.setAttribute("aria-pressed", "true");
    cameraButton.setAttribute("aria-pressed", "true");
    micButton.querySelector("span:last-child").textContent = "麦克风";
  }
  renderCameraState();
}

function renderRecordingState(detail) {
  const indicator = $("recording-indicator");
  indicator.hidden = detail.state === "idle";
  indicator.dataset.state = detail.state;
  $("recording-label").textContent = {
    recording: "正在记录", paused: "记忆录制已暂停", saved: "本次原片已保存",
  }[detail.state] || "";
  indicator.title = detail.reason || "";
  $("capture-paused").hidden = detail.state !== "paused";
  $("capture-paused").title = detail.reason || "";
}

function selectTab(index, focus = false) {
  selectedTab = index;
  tabs.forEach((tab, item) => {
    tab.setAttribute("aria-selected", String(index === item));
    tab.tabIndex = index === item ? 0 : -1;
  });
  $("memory-content").setAttribute("aria-labelledby", tabs[index].id);
  recordList.querySelectorAll("video, audio").forEach((media) => media.pause());
  recordList.replaceChildren();
  renderMemories();
  if (index === 3) void refreshOperations();
  if (focus) tabs[index].focus();
}

function renderRecall() {
  const displayed = sensoryRecall;
  presence.setRecall(["listening", "thinking", "speaking"].includes(call?.phase)
    && !displayed.final && ["searching", "listening", "ready"].includes(displayed.status));
  const strip = $("recall-strip");
  strip.hidden = displayed.status === "idle" || !displayed.query;
  strip.dataset.status = displayed.status;
  strip.dataset.final = String(displayed.final);
  const delivered = feedbackReceipt?.status === "received" &&
    displayed.hits.some(hit => feedbackReceipt.memory_ids.includes(hit.id));
  $("recall-label").textContent = delivered ? "记忆已送入通话" : "正在回忆";
  strip.title = delivered ? "模型已接收这些历史证据；接收回执不代表回答已正确采用。" : "异步查找相关经历";
  $("recall-query").textContent = displayed.query;
  $("recall-count").textContent = displayed.hits.length || displayed.status === "ready" || displayed.final ? String(displayed.hits.length) : "";
  const announcement = delivered ? "历史证据已送入当前通话" : displayed.status === "ready" ? `找到 ${displayed.hits.length} 条相关记忆` : "";
  if ($("recall-announcement").textContent !== announcement) $("recall-announcement").textContent = announcement;
  if (selectedTab === 2) renderMemories();
}

function renderEmptyMemories() {
  const displayed = sensoryRecall;
  const empty = createElement("div", "memory-empty");
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("class", "icon");
  icon.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", selectedTab === 2 ? "#icon-search" : selectedTab === 3 ? "#icon-history" : "#icon-memory");
  icon.append(use);
  empty.append(icon);
  let text = ["还没有共同经历", "理解会随相处慢慢形成", "等待一次回忆", "还没有变化记录"][selectedTab];
  if (selectedTab === 2 && displayed.query) text = displayed.status === "error" ? "召回未完成，请继续通话后重试"
    : ["searching", "listening"].includes(displayed.status) ? "正在寻找相关经历" : "未找到相关记忆";
  empty.append(createElement("p", "", text));
  return empty;
}

function createTimestampElement(record) {
  const time = createElement("time", "record-time");
  const supplied = record.occurred_at || record.created_at;
  if (supplied) {
    time.dateTime = supplied;
    time.textContent = timeFormat.format(new Date(supplied));
  } else time.textContent = "保存的记忆";
  return time;
}

function createOriginalContent(record) {
  const fragment = document.createDocumentFragment();
  if (record.dialogue) {
    const info = createElement("details", "raw-record");
    const states = { completed: "播放完成", interrupted: "已打断", stopped: "已停止", text_only: "仅字幕", generated: "已生成" };
    const notes = [record.dialogue.user ? "输入：语音转写" : "", "回复：生成字幕", `状态：${states[record.dialogue.completion]}`];
    info.append(createElement("summary", "", "记录信息"), createElement("p", "", notes.filter(Boolean).join(" · ")));
    fragment.append(info);
    return fragment;
  }
  if (!record.event_summary) return fragment;
  const scope = record.event_summary_shared
    ? `这段理解覆盖整段事件，由 ${record.event_summary_part_count} 段原片共享，尚未逐段核实。当前是第 ${record.event_summary_part_index + 1} 段。`
    : "这是模型对本段事件的理解，尚未核实；请以原片为准。";
  fragment.append(createElement("p", "detail-note", scope));
  if (record.content) {
    const raw = createElement("details", "raw-record");
    raw.append(createElement("summary", "", "原始记录与采集信息"), createElement("p", "", record.content));
    fragment.append(raw);
  }
  return fragment;
}

async function submitMemoryChange(url, body, success) {
  const result = await requestJson(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });
  detailsCache.clear();
  if (refreshTask) await refreshTask;
  await refreshMemories();
  if (selectedTab === 3) await refreshOperations();
  for (const card of recordList.querySelectorAll(".memory-item[open]")) {
    card.dataset.loaded = "";
    void loadDetail(card);
  }
  if (success) showNotification(success, result.operation?.can_rollback ? { label: "撤销", run: () => rollbackOperation(result.operation.id) } : undefined);
  return result;
}

async function rollbackOperation(operationId) {
  try {
    const result = await submitMemoryChange(`/api/operations/${encodeURIComponent(operationId)}/rollback`);
    showNotification(result.restored ? "已撤销这次更改。" : "这次更改已无法撤销，请查看后续历史。");
  } catch (error) { showNotification(error.message); }
}

function createOperationEntry(operation) {
  const entry = createElement("div", "operation-entry");
  const heading = createElement("div", "operation-heading");
  heading.append(createElement("span", "", intentLabels[operation.intent] || "记忆更新"));
  if (operation.rolled_back_at) heading.append(createElement("span", "operation-reverted", "已撤销"));
  heading.append(createTimestampElement({ created_at: operation.applied_at }));
  entry.append(heading);
  if (operation.rationale) entry.append(createElement("p", "operation-rationale", operation.rationale));
  if (operation.can_rollback) {
    const undo = createElement("button", "quiet-button", "撤销这次更改");
    undo.addEventListener("click", async () => { undo.disabled = true; await rollbackOperation(operation.id); undo.disabled = false; });
    entry.append(undo);
  }
  return entry;
}

function createRecordActions(record) {
  const actions = createElement("div", "record-actions");
  const url = `/api/memories/${encodeURIComponent(record.id)}`;
  if (record.can_correct) {
    const correct = createElement("button", "quiet-button", "纠正理解");
    const form = createElement("form", "correction-form");
    form.hidden = true;
    const label = createElement("label", "", "你希望它记住什么？");
    const input = createElement("textarea", "");
    input.name = "content";
    input.required = true;
    input.maxLength = 2000;
    input.rows = 3;
    input.placeholder = "补充正确的信息…";
    label.append(input);
    const row = createElement("div", "form-actions");
    const submit = createElement("button", "small-primary", "保存纠正");
    submit.type = "submit";
    const cancel = createElement("button", "quiet-button", "取消");
    cancel.type = "button";
    cancel.addEventListener("click", () => { form.hidden = true; correct.focus(); });
    row.append(submit, cancel);
    form.append(label, createElement("p", "form-note", "原始经历与之前的理解都会保留。"), row);
    correct.addEventListener("click", () => { form.hidden = !form.hidden; if (!form.hidden) input.focus(); });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!input.value.trim()) return;
      submit.disabled = cancel.disabled = true;
      submit.textContent = "保存中…";
      try {
        await submitMemoryChange(`${url}/corrections`, { content: input.value.trim() }, "纠正已保存，新的理解正在形成。");
        form.hidden = true;
      } catch (error) { showNotification(error.message); }
      finally { submit.disabled = cancel.disabled = false; submit.textContent = "保存纠正"; }
    });
    actions.append(correct, form);
  }
  if (record.can_confirm) {
    const confirm = createElement("button", "quiet-button", "确认这条理解");
    confirm.addEventListener("click", async () => {
      confirm.disabled = true;
      try { await submitMemoryChange(`${url}/confirmations`, {}, "确认已作为新的来源保存。"); }
      catch (error) { showNotification(error.message); }
      finally { confirm.disabled = false; }
    });
    actions.append(confirm);
  }
  if (!record.forgotten_at && !record.retired_at) {
    const forget = createElement("button", "quiet-button forget-button", "暂时遗忘");
    forget.addEventListener("click", async () => {
      forget.disabled = true;
      try { await submitMemoryChange(`${url}/forget`, {}, "已移出召回，原始记录与历史保留。"); }
      catch (error) { showNotification(error.message); }
      finally { forget.disabled = false; }
    });
    actions.append(forget);
  }
  return actions;
}

function createMemoryDetail(detail) {
  const fragment = document.createDocumentFragment();
  const record = detail.item;
  if (record.error) fragment.append(createElement("p", "detail-note", "原片已保留，记忆分析尚未完成。请检查模型服务。"));
  if (record.basis === "model_inference" && record.visible === false) fragment.append(createElement("p", "detail-note", record.kind === "trait"
    ? "长期特征需要至少两组独立经历支持。" : "这条理解暂未用于召回。"));
  fragment.append(createOriginalContent(record), createMediaEvidence(record.assets));
  if (detail.evidence.length) {
    const sources = createElement("section", "evidence-section");
    sources.append(createElement("h3", "detail-title", "来源"));
    for (const source of detail.evidence) {
      const evidence = createElement("details", "source-evidence");
      const summary = createElement("summary", "");
      summary.append(createTimestampElement(source), document.createTextNode(` · ${source.dialogue ? "对话" : kindLabels[source.kind] || "共同经历"}`));
      evidence.append(summary);
      let loaded = false;
      evidence.addEventListener("toggle", () => {
        if (!evidence.open) evidence.querySelectorAll("video, audio").forEach((media) => media.pause());
        else if (!loaded) {
          loaded = true;
          evidence.append(createElement("p", "", recordDescription(source)), createOriginalContent(source), createMediaEvidence(source.assets));
        }
      });
      sources.append(evidence);
    }
    fragment.append(sources);
  }
  const versions = detail.versions.filter((version) => version.id !== record.id);
  if (versions.length) {
    const history = createElement("section", "version-history");
    history.append(createElement("h3", "detail-title", "理解的变化"));
    for (const version of versions) {
      const step = createElement("div", "version-step");
      const status = recordState(version);
      step.append(createTimestampElement(version), createElement("span", `record-status ${status.tone}`, status.label), createElement("p", "", recordDescription(version)));
      history.append(step);
    }
    fragment.append(history);
  }
  if (detail.operations.length) {
    const history = createElement("section", "detail-operations");
    history.append(createElement("h3", "detail-title", "更改历史"));
    detail.operations.forEach((operation) => history.append(createOperationEntry(operation)));
    fragment.append(history);
  }
  fragment.append(createRecordActions(record));
  return fragment;
}

async function loadDetail(card) {
  const id = card.dataset.memoryId;
  if (card.dataset.loaded === "true") return;
  const body = card.querySelector(".record-detail");
  body.setAttribute("aria-busy", "true");
  body.replaceChildren(createElement("p", "detail-loading", "正在取回原始证据…"));
  try {
    if (!detailsCache.has(id)) {
      if (!detailRequests.has(id)) detailRequests.set(id, requestJson(`/api/memories/${encodeURIComponent(id)}`).finally(() => detailRequests.delete(id)));
      detailsCache.set(id, await detailRequests.get(id));
    }
    body.replaceChildren(createMemoryDetail(detailsCache.get(id)));
    card.dataset.loaded = "true";
  } catch (error) {
    const retry = createElement("button", "quiet-button", "重试");
    retry.addEventListener("click", () => { void loadDetail(card); });
    body.replaceChildren(createElement("p", "detail-note", "原始证据暂未取回。"), retry);
  } finally { body.removeAttribute("aria-busy"); }
}

function updateCard(card, record) {
  card.querySelector(".memory-description").textContent = recordDescription(record);
  const status = recordState(record);
  const badge = card.querySelector(".record-status");
  badge.className = `record-status ${status.tone}`;
  badge.textContent = status.label;
  card.classList.toggle("inactive-memory", Boolean(record.forgotten_at || record.retired_at));
  card.querySelector(".record-kind").textContent = kindLabels[record.kind] || "共同经历";
  const scope = card.querySelector(".event-summary-scope");
  scope.hidden = !record.event_summary_shared;
  scope.textContent = record.event_summary_shared ? "整段事件的共享理解" : "";
}

function createMemoryCard(record) {
  const card = createElement("details", "memory-item");
  card.dataset.memoryId = record.id;
  const summary = createElement("summary", "");
  const meta = createElement("div", "record-meta");
  meta.append(createTimestampElement(record), createElement("span", "record-status"));
  const foot = createElement("div", "record-foot");
  foot.append(createElement("span", "record-kind"), createElement("span", "event-summary-scope"));
  if (record.assets.length) foot.append(createElement("span", "record-original", "原片"));
  else if (record.evidence_ids?.length) foot.append(createElement("span", "record-original", `${record.evidence_ids.length} 个来源`));
  summary.append(meta, createElement("p", "memory-description"), foot);
  card.append(summary, createElement("div", "record-detail"));
  updateCard(card, record);
  card.addEventListener("toggle", () => {
    if (card.open) void loadDetail(card);
    else card.querySelectorAll("video, audio").forEach((media) => media.pause());
  });
  return card;
}

function renderMemories() {
  const displayed = sensoryRecall;
  const groups = groupRecords(saved);
  const records = [groups.experiences, groups.profile, displayed.hits, groups.history][selectedTab];
  const context = $("view-context");
  const copy = selectedTab === 2 && displayed.query ? `召回：${displayed.query}` : "";
  context.hidden = !copy;
  context.replaceChildren(createElement("span", "", copy));
  context.dataset.final = "false";
  operationList.hidden = selectedTab !== 3;
  if (!records.length) {
    recordList.replaceChildren(selectedTab === 3 && operations.length ? createElement("div") : renderEmptyMemories());
    return;
  }
  const retained = new Map([...recordList.querySelectorAll(":scope > .memory-item")].map((card) => [card.dataset.memoryId, card]));
  recordList.querySelector(".memory-empty")?.remove();
  const ids = new Set(records.map((record) => record.id));
  for (const [id, card] of retained) if (!ids.has(id)) card.remove();
  records.forEach((record, index) => {
    const card = retained.get(record.id) || createMemoryCard(record);
    updateCard(card, record);
    // Keep a playing original attached when background memory formation adds records.
    if (recordList.children[index] !== card) recordList.insertBefore(card, recordList.children[index] || null);
  });
}

async function refreshOperations() {
  try {
    operations = (await requestJson("/api/operations")).items;
    operationList.replaceChildren();
    if (operations.length) {
      operationList.append(createElement("h3", "detail-title", "记忆更改"));
      operations.forEach((operation) => operationList.append(createOperationEntry(operation)));
    }
    if (selectedTab === 3) renderMemories();
  } catch (error) { showNotification("更改历史暂不可用，请稍后重试。"); }
}

function refreshMemories() {
  if (refreshTask) return refreshTask;
  refreshTask = (async () => {
    try {
      const result = await requestJson("/api/memories?include_inactive=true");
      if (memoriesLoaded && hasNewlyFormedMemory(saved, result.items)) presence.pulseMemory();
      memoriesLoaded = true;
      saved = result.items;
      const listed = new Set(saved.map((record) => record.id));
      const recalled = sensoryRecall.candidates || sensoryRecall.hits;
      const unknownIds = [...new Set(recalled.map((record) => record.id))].filter((id) => !listed.has(id));
      const unknown = await Promise.allSettled(unknownIds.map((id) => requestJson(`/api/memories/${encodeURIComponent(id)}`)));
      latestRecallCards = [...saved];
      for (const item of unknown) {
        if (item.status === "fulfilled") latestRecallCards.push(item.value.item);
      }
      // Unknown hits need a current SDK read after a mutation; failed reads withdraw them.
      sensoryRecall = reconcileRecall(sensoryRecall, latestRecallCards, { excludeUnknown: true });
      $("memory-progress").textContent = result.processing ? "正在形成" : "";
      if (result.capture_failed && !captureNoticeShown) {
        captureNoticeShown = true;
        showNotification("原始音视频已保留，记忆分析尚未完成。请检查模型服务；通话可继续。");
      }
      saved.sort((a, b) => (b.occurred_at || b.created_at || "").localeCompare(a.occurred_at || a.created_at || ""));
      $("memory-count").textContent = String(saved.filter((record) => !record.forgotten_at && !record.retired_at).length);
      renderRecall();
      if (selectedTab !== 2) renderMemories();
    } catch (error) {
      $("memory-count").textContent = "—";
      showNotification("记忆暂不可用，请检查本地 Demo 服务。");
    } finally {
      refreshTask = null;
      if (refreshQueued) { refreshQueued = false; queueMicrotask(() => { void refreshMemories(); }); }
    }
  })();
  return refreshTask;
}

function scheduleMemoryRefresh() {
  if (refreshTask) refreshQueued = true;
  else void refreshMemories();
}

$("show-help").addEventListener("click", () => $("help-dialog").showModal());
$("close-help").addEventListener("click", () => $("help-dialog").close());
$("dismiss-notice").addEventListener("click", () => { $("notice").hidden = true; });
$("close-memory").addEventListener("click", () => { setMemoryPanelVisible(false); memoryButton.focus(); });
memoryButton.addEventListener("click", () => setMemoryPanelVisible($("memory-rail").hidden));
$("recall-strip").addEventListener("click", () => { selectTab(2); setMemoryPanelVisible(true); tabs[2].focus(); });
video.addEventListener("playing", renderCameraState);
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(index));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
      : (selectedTab + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    selectTab(next, true);
  });
});
setMemoryPanelVisible(!matchMedia("(max-width: 900px)").matches);

try {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error("无法连接本地 Demo 服务，请检查终端中的启动状态。");
  const config = await response.json();
  call = new CallSession({ video, audio: $("remote-audio"), config });
  call.addEventListener("state", ({ detail }) => renderCallState(detail));
  call.addEventListener("recording", ({ detail }) => renderRecordingState(detail));
  call.addEventListener("perceptionreset", () => { $("live-observation").hidden = true; });
  call.addEventListener("perception", ({ detail }) => {
    $("live-observation").hidden = false;
    $("live-observation").dataset.perceptionId = detail.perception_id;
    $("observation-memory-state").textContent = "持续观察中";
    $("observation-summary").textContent = detail.summary;
    const end = new Date(detail.occurred_end);
    $("observation-time").dateTime = detail.occurred_end;
    $("observation-time").textContent = timeFormat.format(end);
    $("observation-time").title = `${timeFormat.format(new Date(detail.occurred_at))} — ${timeFormat.format(end)}`;
  });
  call.addEventListener("visualmemory", ({ detail }) => {
    if ($("live-observation").dataset.perceptionId === detail.perception_id) {
      $("observation-memory-state").textContent = "本段画面已保存";
    }
  });
  call.addEventListener("sensoryrecall", ({ detail }) => {
    if (detail.status === "idle") feedbackReceipt = null;
    sensoryRecall = detail.status === "idle"
      ? { revision: -1, query: "", status: "idle", hits: [], final: false }
      : reconcileRecall(updateRecall(sensoryRecall, { ...detail, final: false }), latestRecallCards);
    renderRecall();
  });
  call.addEventListener("memoryfeedback", ({ detail }) => {
    feedbackReceipt = detail;
    renderRecall();
  });
  call.addEventListener("error", ({ detail }) => showNotification(detail));
  call.addEventListener("caption", ({ detail }) => {
    $("caption-role").hidden = false;
    $("caption-role").textContent = detail.role === "user" ? "你" : "SenseMem";
    $("caption-text").textContent = detail.text + (detail.interrupted ? "（已打断）" : "");
  });
  call.addEventListener("memorychange", scheduleMemoryRefresh);
  callButton.addEventListener("click", async () => {
    $("notice").hidden = true;
    if (call.active) await call.stop();
    else await call.start();
  });
  micButton.addEventListener("click", () => {
    const enabled = call.toggleMic();
    micButton.setAttribute("aria-pressed", String(enabled));
    micButton.querySelector("span:last-child").textContent = enabled ? "麦克风" : "已静音";
  });
  cameraButton.addEventListener("click", () => {
    const enabled = call.toggleCamera();
    cameraButton.setAttribute("aria-pressed", String(enabled));
    renderCallState(call.phase);
  });
  await refreshMemories();
  const updates = new EventSource("/api/events");
  updates.addEventListener("memory_changed", scheduleMemoryRefresh);
  updates.addEventListener("open", scheduleMemoryRefresh);
  window.addEventListener("pagehide", () => {
    updates.close();
    presence.destroy();
    void call.stop({ unloading: true });
  }, { once: true });
} catch (error) {
  callButton.disabled = true;
  showNotification(error.message);
}
