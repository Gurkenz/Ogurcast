const form = document.getElementById("jobForm");
const startButton = document.getElementById("startButton");
const formError = document.getElementById("formError");
const progress = document.getElementById("progress");
const result = document.getElementById("result");
const health = document.getElementById("health");
const statusBadge = document.getElementById("statusBadge");
const fileInput = document.getElementById("file");
const fileName = document.getElementById("fileName");
const dropZone = document.getElementById("dropZone");
const workbenchView = document.getElementById("workbenchView");
const modeTabs = document.getElementById("modeTabs");
const runQwenButton = document.getElementById("runQwenButton");
const applySafeButton = document.getElementById("applySafeButton");
const rollbackLatestButton = document.getElementById("rollbackLatestButton");
const segmentRail = document.getElementById("segmentRail");
const transcriptEditor = document.getElementById("transcriptEditor");
const reviewQueue = document.getElementById("reviewQueue");
const editorTitle = document.getElementById("editorTitle");
const editorStats = document.getElementById("editorStats");
const queueTitle = document.getElementById("queueTitle");
const popoverLayer = document.getElementById("popoverLayer");

const state = {
  mode: "text",
  activeFilter: "all",
  transcript: [],
  words: [],
  corrections: [],
  entities: [],
  speakers: [],
  selectedCorrectionId: null,
  selectedEntityId: null,
  selectedSegmentId: null,
  currentTime: 0,
  appliedBatches: [],
  currentJobId: null,
};

const modes = [
  { id: "text", label: "Text" },
  { id: "asr", label: "ASR Review" },
  { id: "entity", label: "Entity Review" },
  { id: "speaker", label: "Speaker Review", disabled: true },
  { id: "style", label: "Style Review", disabled: true },
  { id: "listen", label: "Listen Review", disabled: true },
  { id: "final", label: "Final Preview" },
];

let pollTimer = null;

function el(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) {
    node.className = className;
  }
  if (text !== undefined && text !== null) {
    node.textContent = String(text);
  }
  return node;
}

function button(text, className, onClick, disabled = false) {
  const node = el("button", className, text);
  node.type = "button";
  node.disabled = disabled;
  if (onClick) {
    node.addEventListener("click", onClick);
  }
  return node;
}

function setError(message) {
  formError.textContent = message || "";
}

function boolText(value) {
  return value ? "да" : "нет";
}

function formatTime(value) {
  const total = Math.max(0, Math.floor(Number(value) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function truncate(text, limit = 90) {
  const value = String(text || "").trim();
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, limit - 1)}…`;
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Ошибка сервера: ${response.status}`);
  }
  return data;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  return readJson(response);
}

async function sendJson(url, method, payload = {}) {
  return requestJson(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function renderFacts(data) {
  health.replaceChildren();
  const rows = [
    ["CUDA", boolText(data.cuda_available)],
    ["GPU", data.gpu || "NO CUDA"],
    ["HF token", boolText(data.hf_token_present)],
    ["FFmpeg", boolText(data.ffmpeg_ok)],
    ["WhisperX", boolText(data.whisperx_ok)],
    ["HF_HOME", data.hf_home || ""],
    ["TMP", data.tmp || ""],
  ];
  for (const [name, value] of rows) {
    health.append(el("dt", "", name), el("dd", "", value));
  }
}

async function loadHealth() {
  try {
    const data = await requestJson("/api/health");
    renderFacts(data);
    statusBadge.textContent = data.cuda_available ? "CUDA доступна" : "CUDA недоступна";
    statusBadge.className = data.cuda_available ? "badge ok" : "badge warn";
  } catch (error) {
    statusBadge.textContent = "Ошибка health";
    statusBadge.className = "badge bad";
    health.replaceChildren(el("dt", "", "Ошибка"), el("dd", "", error.message));
  }
}

function selectedSpeakerHint() {
  const checked = document.querySelector("input[name='speaker_hint']:checked");
  return checked ? checked.value : "auto";
}

function collectFormData() {
  if (!fileInput.files.length) {
    throw new Error("Выберите файл аудио или видео.");
  }

  const data = new FormData();
  data.append("file", fileInput.files[0]);
  data.append("output_dir", document.getElementById("output_dir").value);
  data.append("model", document.getElementById("model").value);
  data.append("language", document.getElementById("language").value);
  data.append("device", document.getElementById("device").value);
  data.append("compute_type", document.getElementById("compute_type").value);
  data.append("batch_size", document.getElementById("batch_size").value);
  data.append("diarize", document.getElementById("diarize").checked ? "true" : "false");

  let minSpeakers = document.getElementById("min_speakers").value;
  let maxSpeakers = document.getElementById("max_speakers").value;
  const speakerHint = selectedSpeakerHint();
  if (speakerHint === "one") {
    minSpeakers = "1";
    maxSpeakers = "1";
  } else if (speakerHint === "two") {
    minSpeakers = "2";
    maxSpeakers = "2";
  } else if (speakerHint === "many") {
    minSpeakers = "2";
    maxSpeakers = "4";
  }
  data.append("min_speakers", minSpeakers);
  data.append("max_speakers", maxSpeakers);

  const token = document.getElementById("hf_token").value.trim();
  if (token) {
    data.append("hf_token", token);
  }
  return data;
}

function renderProgress(lines) {
  progress.textContent = lines.length ? lines.join("\n") : "Ожидание...";
  progress.scrollTop = progress.scrollHeight;
}

function appendProgressLine(line) {
  const current = progress.textContent && progress.textContent !== "Ожидание..." ? progress.textContent : "";
  progress.textContent = current ? `${current}\n${line}` : line;
  progress.scrollTop = progress.scrollHeight;
}

function renderFileList(data, jobId) {
  result.replaceChildren();
  const output = el("p");
  output.append(el("strong", "", "Папка вывода: "), el("code", "", data.output_dir));
  const list = el("ul", "files");
  for (const item of data.files || []) {
    const row = el("li");
    row.append(el("span", "", item.name), el("code", "", item.path));
    list.append(row);
  }
  const openButton = button("Открыть текст", "primary-inline", () => loadReview(jobId));
  result.append(output, list, openButton);
}

async function loadFiles(jobId) {
  const data = await requestJson(`/api/jobs/${jobId}/files`);
  renderFileList(data, jobId);
}

async function pollJob(jobId) {
  try {
    const job = await requestJson(`/api/jobs/${jobId}`);
    renderProgress(job.progress || []);

    if (job.status === "done") {
      clearInterval(pollTimer);
      pollTimer = null;
      startButton.disabled = false;
      state.currentJobId = jobId;
      await loadFiles(jobId);
    }

    if (job.status === "error") {
      clearInterval(pollTimer);
      pollTimer = null;
      startButton.disabled = false;
      result.textContent = job.error || "Ошибка обработки.";
    }
  } catch (error) {
    clearInterval(pollTimer);
    pollTimer = null;
    startButton.disabled = false;
    setError(error.message);
  }
}

function updateFileName() {
  fileName.textContent = fileInput.files.length ? fileInput.files[0].name : "или нажмите для выбора аудио/видео";
}

fileInput.addEventListener("change", updateFileName);

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  const droppedFile = event.dataTransfer.files[0];
  if (!droppedFile) {
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(droppedFile);
  fileInput.files = transfer.files;
  updateFileName();
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  result.textContent = "Задача запускается...";
  progress.textContent = "Задача запускается...";
  startButton.disabled = true;

  try {
    const data = await requestJson("/api/jobs", {
      method: "POST",
      body: collectFormData(),
    });
    state.currentJobId = data.job_id;
    pollTimer = setInterval(() => pollJob(data.job_id), 2000);
    await pollJob(data.job_id);
  } catch (error) {
    startButton.disabled = false;
    result.textContent = "Пока нет результата.";
    setError(error.message);
  }
});

function loadReviewState(data) {
  state.lastReviewBundle = data;
  state.lastApprovedSegments = data.approvedTranscript?.segments || [];
  state.transcript = data.transcript?.segments || [];
  state.words = data.words || [];
  state.corrections = data.corrections?.corrections || [];
  state.entities = data.entities?.entities || [];
  state.speakers = data.speakers?.speakers || [];
  state.appliedBatches = (data.editBatches?.batches || []).filter((batch) => batch.status === "applied");
  if (!state.selectedSegmentId && state.transcript.length) {
    state.selectedSegmentId = state.transcript[0].id;
  }
}

async function loadReview(jobId) {
  try {
    const data = await requestJson(`/api/jobs/${jobId}/review`);
    state.currentJobId = jobId;
    loadReviewState(data);
    workbenchView.hidden = false;
    renderWorkbench();
    workbenchView.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setError(error.message);
  }
}

async function reloadReview() {
  if (!state.currentJobId) {
    return;
  }
  const data = await requestJson(`/api/jobs/${state.currentJobId}/review`);
  loadReviewState(data);
  renderWorkbench();
}

function approvedSegment(segmentId) {
  const approved = state.reviewApprovedMap;
  if (approved && approved.has(segmentId)) {
    return approved.get(segmentId);
  }
  return null;
}

function rebuildApprovedMap() {
  const bundle = new Map();
  const approvedSegments = state.lastApprovedSegments || [];
  for (const segment of approvedSegments) {
    bundle.set(segment.id, segment);
  }
  state.reviewApprovedMap = bundle;
}

function safeCorrectionCount() {
  return state.corrections.filter((correction) => (
    ["ASR_ERROR", "TYPO", "PUNCTUATION"].includes(correction.category) &&
    Number(correction.confidence || 0) >= 0.95 &&
    correction.severity === "low" &&
    correction.requiresAudioReview === false &&
    correction.canBatchApply === true &&
    correction.status === "pending"
  )).length;
}

function correctionsForSegment(segmentId) {
  return state.corrections.filter((correction) => correction.segmentId === segmentId && correction.status !== "rejected");
}

function entitiesForSegment(segmentId) {
  return state.entities.filter((entity) => (entity.segmentIds || []).includes(segmentId) && entity.status !== "rejected");
}

function renderWorkbench() {
  rebuildApprovedMap();
  renderModeTabs();
  renderSegmentRail();
  renderEditor();
  renderQueue();

  const safeCount = safeCorrectionCount();
  applySafeButton.textContent = `Применить безопасные ASR (${safeCount})`;
  applySafeButton.disabled = safeCount === 0;
  rollbackLatestButton.disabled = state.appliedBatches.length === 0;
  runQwenButton.disabled = !state.currentJobId;
}

function renderModeTabs() {
  modeTabs.replaceChildren();
  for (const mode of modes) {
    const tab = button(mode.label, "mode-tab", () => {
      state.mode = mode.id;
      closePopover();
      renderWorkbench();
    }, Boolean(mode.disabled));
    if (mode.id === state.mode) {
      tab.classList.add("active");
    }
    if (mode.disabled) {
      tab.title = "Будет добавлено позже";
    }
    modeTabs.append(tab);
  }
}

function renderSegmentRail() {
  segmentRail.replaceChildren();
  for (const segment of state.transcript) {
    const item = button("", "segment-nav", () => {
      state.selectedSegmentId = segment.id;
      scrollToSegment(segment.id);
      renderSegmentRail();
    });
    if (state.selectedSegmentId === segment.id) {
      item.classList.add("active");
    }
    const meta = el("span", "segment-nav-meta", `${formatTime(segment.start)} · ${segment.speaker}`);
    const text = el("span", "segment-nav-text", truncate(segment.text, 54));
    const count = correctionsForSegment(segment.id).length + entitiesForSegment(segment.id).length;
    if (count > 0) {
      item.append(meta, text, el("span", "segment-nav-count", String(count)));
    } else {
      item.append(meta, text);
    }
    segmentRail.append(item);
  }
}

function scrollToSegment(segmentId) {
  const target = transcriptEditor.querySelector(`[data-segment-id="${segmentId}"]`);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function segmentHeader(segment) {
  const header = el("div", "segment-meta");
  header.append(el("span", "", formatTime(segment.start)), el("span", "", segment.speaker));
  return header;
}

function approvedText(segment) {
  const approved = approvedSegment(segment.id);
  return approved ? approved.text : segment.text;
}

function renderPlainSegment(segment, finalMode = false) {
  const card = el("article", finalMode ? "segment-card final" : "segment-card");
  card.dataset.segmentId = segment.id;
  card.append(segmentHeader(segment), el("p", "segment-text", approvedText(segment)));
  return card;
}

function correctionChip(correction) {
  const chip = button("", `correction-unit ${correction.status || "pending"}`, () => {
    state.selectedCorrectionId = correction.id;
    showCorrectionPopover(correction);
  });
  if (correction.category === "NEEDS_LISTENING") {
    chip.classList.add("needs-listening");
    chip.append(el("span", "warn-chip", correction.originalText || "прослушать"));
    return chip;
  }
  chip.append(
    el("span", "before-chip", correction.originalText),
    el("span", "arrow", "→"),
    el("span", "after-chip", correction.suggestedText),
  );
  return chip;
}

function renderAsrSegment(segment) {
  const card = el("article", "segment-card");
  card.dataset.segmentId = segment.id;
  card.append(segmentHeader(segment));
  const body = el("p", "segment-text inline-review");
  const source = segment.text || "";
  const corrections = correctionsForSegment(segment.id).filter((correction) => (
    ["ASR_ERROR", "TYPO", "PUNCTUATION", "FILLER_GARBAGE", "NEEDS_LISTENING"].includes(correction.category)
  ));
  let cursor = 0;
  const trailing = [];

  for (const correction of corrections) {
    const original = correction.originalText || "";
    const index = original ? source.indexOf(original, cursor) : -1;
    if (index >= cursor) {
      body.append(document.createTextNode(source.slice(cursor, index)));
      body.append(correctionChip(correction));
      cursor = index + original.length;
    } else {
      trailing.push(correction);
    }
  }

  body.append(document.createTextNode(source.slice(cursor)));
  for (const correction of trailing) {
    body.append(document.createTextNode(" "));
    body.append(correctionChip(correction));
  }
  card.append(body);
  return card;
}

function entityBadge(entity) {
  const badge = button("", `entity-badge ${entity.verificationStatus || "new"}`, () => {
    state.selectedEntityId = entity.id;
    showEntityPopover(entity);
  });
  badge.append(el("span", "", entity.surface), el("span", "entity-status", entity.verificationStatus || "new"));
  return badge;
}

function renderEntitySegment(segment) {
  const card = el("article", "segment-card");
  card.dataset.segmentId = segment.id;
  card.append(segmentHeader(segment));
  const body = el("p", "segment-text inline-review");
  const source = approvedText(segment) || "";
  const entities = entitiesForSegment(segment.id);
  let cursor = 0;
  const trailing = [];

  for (const entity of entities) {
    const surface = entity.surface || "";
    const index = surface ? source.indexOf(surface, cursor) : -1;
    if (index >= cursor) {
      body.append(document.createTextNode(source.slice(cursor, index)));
      body.append(entityBadge(entity));
      cursor = index + surface.length;
    } else {
      trailing.push(entity);
    }
  }

  body.append(document.createTextNode(source.slice(cursor)));
  for (const entity of trailing) {
    body.append(document.createTextNode(" "));
    body.append(entityBadge(entity));
  }
  card.append(body);
  return card;
}

function renderEditor() {
  transcriptEditor.replaceChildren();
  const safeCount = safeCorrectionCount();
  const pendingCorrections = state.corrections.filter((correction) => correction.status === "pending").length;
  const pendingEntities = state.entities.filter((entity) => entity.status === "pending").length;

  if (state.mode === "asr") {
    editorTitle.textContent = "ASR Review";
    editorStats.textContent = `Safe fixes: ${safeCount} · Needs review: ${pendingCorrections - safeCount}`;
    for (const segment of state.transcript) {
      transcriptEditor.append(renderAsrSegment(segment));
    }
    return;
  }

  if (state.mode === "entity") {
    editorTitle.textContent = "Entity Review";
    editorStats.textContent = `Сущности: ${state.entities.length} · pending: ${pendingEntities}`;
    for (const segment of state.transcript) {
      transcriptEditor.append(renderEntitySegment(segment));
    }
    return;
  }

  if (state.mode === "final") {
    editorTitle.textContent = "Final Preview";
    editorStats.textContent = "Чистый текст после принятых правок";
    for (const segment of state.transcript) {
      transcriptEditor.append(renderPlainSegment(segment, true));
    }
    return;
  }

  editorTitle.textContent = "Текст";
  editorStats.textContent = `Сегменты: ${state.transcript.length} · спикеры: ${state.speakers.length}`;
  for (const segment of state.transcript) {
    transcriptEditor.append(renderPlainSegment(segment));
  }
}

function queueItem(primary, secondary, onClick, tone = "") {
  const item = button("", `queue-item ${tone}`, onClick);
  item.append(el("span", "queue-primary", primary), el("span", "queue-secondary", secondary));
  return item;
}

function renderTextQueue() {
  queueTitle.textContent = "Очередь проверки";
  const pending = state.corrections.filter((correction) => correction.status === "pending").length;
  reviewQueue.append(
    queueItem("ASR", `${pending} ожидает`, () => {
      state.mode = "asr";
      renderWorkbench();
    }),
    queueItem("Сущности", `${state.entities.filter((entity) => entity.status === "pending").length} ожидает`, () => {
      state.mode = "entity";
      renderWorkbench();
    }),
    queueItem("Спикеры", `${state.speakers.length} профилей`, null, "muted-item"),
  );
}

function renderAsrQueue() {
  queueTitle.textContent = "ASR queue";
  const corrections = state.corrections.filter((correction) => correction.status !== "rejected");
  for (const correction of corrections) {
    const safe = safeCorrectionCount() && correction.status === "pending" && correction.canBatchApply ? "safe" : correction.status;
    const primary = `${correction.category} · ${safe}`;
    const secondary = `${formatTime(correction.startTime)} ${truncate(correction.originalText, 26)} → ${truncate(correction.suggestedText, 26)}`;
    reviewQueue.append(queueItem(primary, secondary, () => {
      state.selectedCorrectionId = correction.id;
      state.selectedSegmentId = correction.segmentId;
      scrollToSegment(correction.segmentId);
      showCorrectionPopover(correction);
    }, correction.category === "NEEDS_LISTENING" ? "warn-item" : ""));
  }
}

function renderEntityQueue() {
  queueTitle.textContent = "Сущности";
  for (const entity of state.entities) {
    reviewQueue.append(queueItem(
      `${entity.surface} · ${entity.type}`,
      `${entity.verificationStatus} · ${entity.status}`,
      () => {
        state.selectedEntityId = entity.id;
        state.selectedSegmentId = entity.segmentIds?.[0] || state.selectedSegmentId;
        if (state.selectedSegmentId) {
          scrollToSegment(state.selectedSegmentId);
        }
        showEntityPopover(entity);
      },
      entity.verificationStatus === "contradicted" ? "bad-item" : "",
    ));
  }
}

function renderFinalQueue() {
  queueTitle.textContent = "История";
  const batches = state.lastReviewBundle?.editBatches?.batches || [];
  if (!batches.length) {
    reviewQueue.append(el("p", "muted", "История изменений пуста."));
    return;
  }
  for (const batch of batches.slice().reverse()) {
    reviewQueue.append(queueItem(
      `${batch.type} · ${batch.status}`,
      `${batch.appliedCount || 0} правок · ${batch.id}`,
      null,
      batch.status === "rolled_back" ? "muted-item" : "",
    ));
  }
}

function renderQueue() {
  reviewQueue.replaceChildren();
  if (state.mode === "asr") {
    renderAsrQueue();
  } else if (state.mode === "entity") {
    renderEntityQueue();
  } else if (state.mode === "final") {
    renderFinalQueue();
  } else {
    renderTextQueue();
  }
}

function closePopover() {
  popoverLayer.hidden = true;
  popoverLayer.replaceChildren();
}

function popoverCard(title) {
  const card = el("div", "popover-card");
  const head = el("div", "popover-head");
  head.append(el("h2", "", title), button("×", "icon-button", closePopover));
  card.append(head);
  popoverLayer.replaceChildren(card);
  popoverLayer.hidden = false;
  return card;
}

async function patchCorrection(correction, payload) {
  await sendJson(`/api/jobs/${state.currentJobId}/corrections/${correction.id}`, "PATCH", payload);
  await reloadReview();
  closePopover();
}

function showCorrectionPopover(correction) {
  const card = popoverCard("CorrectionUnit");
  card.append(
    el("p", "muted", `${correction.category} · confidence ${Math.round(Number(correction.confidence || 0) * 100)}% · ${correction.severity}`),
    el("p", "", correction.reason || ""),
    el("p", "diff-line", `${correction.originalText || ""} → ${correction.suggestedText || ""}`),
  );
  const actions = el("div", "popover-actions");
  actions.append(
    button("Применить", "", () => patchCorrection(correction, { status: "accepted" })),
    button("Пропуск", "secondary", () => patchCorrection(correction, { status: "rejected" })),
    button("Прослушать", "secondary", null, true),
    button("Свой вариант", "secondary", async () => {
      const value = window.prompt("Свой вариант", correction.suggestedText || correction.originalText || "");
      if (value !== null && value.trim()) {
        await patchCorrection(correction, { status: "modified", suggestedText: value.trim() });
      }
    }),
  );
  card.append(actions);
}

async function patchEntity(entity, payload) {
  await sendJson(`/api/jobs/${state.currentJobId}/entities/${entity.id}`, "PATCH", payload);
  await reloadReview();
  closePopover();
}

function showEntityPopover(entity) {
  const card = popoverCard("Сущность");
  card.append(
    el("p", "muted", `${entity.type} · ${entity.verificationStatus} · ${entity.status}`),
    el("p", "", entity.surface || ""),
    el("p", "diff-line", `Каноническая форма: ${entity.canonical || entity.surface || ""}`),
  );
  const evidenceList = el("ul", "evidence-list");
  for (const evidence of entity.evidence || []) {
    evidenceList.append(el("li", "", evidence.snippet || evidence.source || ""));
  }
  card.append(evidenceList);

  const actions = el("div", "popover-actions");
  actions.append(
    button("Применить", "", () => patchEntity(entity, { status: "accepted", verificationStatus: "manual_confirmed" })),
    button("Искать еще", "secondary", null, true),
    button("Игнор", "secondary", () => patchEntity(entity, { status: "rejected", verificationStatus: "manual_rejected" })),
    button("Canonical", "secondary", async () => {
      const value = window.prompt("Каноническая форма", entity.canonical || entity.surface || "");
      if (value !== null && value.trim()) {
        await patchEntity(entity, { status: "modified", canonical: value.trim() });
      }
    }),
  );
  card.append(actions);
}

applySafeButton.addEventListener("click", async () => {
  try {
    await sendJson(`/api/jobs/${state.currentJobId}/corrections/apply-safe-asr`, "POST");
    await reloadReview();
  } catch (error) {
    setError(error.message);
  }
});

rollbackLatestButton.addEventListener("click", async () => {
  try {
    await sendJson(`/api/jobs/${state.currentJobId}/corrections/rollback-batch`, "POST");
    await reloadReview();
  } catch (error) {
    setError(error.message);
  }
});

runQwenButton.addEventListener("click", async () => {
  if (!state.currentJobId) {
    setError("Сначала откройте завершенный результат.");
    return;
  }
  setError("");
  runQwenButton.disabled = true;
  runQwenButton.textContent = "Qwen3: обработка...";
  try {
    const data = await sendJson(`/api/jobs/${state.currentJobId}/llm/postprocess`, "POST");
    await reloadReview();
    state.mode = "asr";
    renderWorkbench();
    appendProgressLine(`Qwen3: добавлено ${data.addedCount || 0}, пропущено ${data.skippedCount || 0}.`);
  } catch (error) {
    setError(error.message);
  } finally {
    runQwenButton.textContent = "Qwen3: найти правки";
    runQwenButton.disabled = !state.currentJobId;
  }
});

popoverLayer.addEventListener("click", (event) => {
  if (event.target === popoverLayer) {
    closePopover();
  }
});

loadHealth();
