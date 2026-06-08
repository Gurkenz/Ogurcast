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
const audioPlayer = document.getElementById("audioPlayer");
const playbackSpeed = document.getElementById("playbackSpeed");
const llmStatus = document.getElementById("llmStatus");
const llmModel = document.getElementById("llmModel");
const llmTemperature = document.getElementById("llmTemperature");
const llmTimeout = document.getElementById("llmTimeout");
const llmStageDescription = document.getElementById("llmStageDescription");
const llmSystemInstruction = document.getElementById("llmSystemInstruction");
const llmUserTemplate = document.getElementById("llmUserTemplate");
const llmSchemaNotes = document.getElementById("llmSchemaNotes");
const saveLlmProfileButton = document.getElementById("saveLlmProfileButton");
const llmEventLog = document.getElementById("llmEventLog");

const state = {
  mode: "text",
  activeFilter: "all",
  transcript: [],
  words: [],
  corrections: [],
  audioFlags: [],
  entities: [],
  speakers: [],
  speakerTurns: [],
  segmentById: new Map(),
  approvedById: new Map(),
  availableModels: [],
  llmReady: false,
  selectedCorrectionId: null,
  selectedEntityId: null,
  selectedSegmentId: null,
  currentTime: 0,
  audioStopAt: null,
  appliedBatches: [],
  currentJobId: null,
  llmProfile: null,
  llmRunPollTimer: null,
};

const modes = [
  { id: "text", label: "Text" },
  { id: "asr", label: "ASR Review" },
  { id: "entity", label: "Entity Review" },
  { id: "speaker", label: "Speaker Review" },
  { id: "style", label: "Style Review", disabled: true },
  { id: "listen", label: "Listen Review" },
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

function formatSeconds(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "0.0";
  }
  return number.toFixed(1);
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

function renderLlmStatus(data) {
  llmStatus.replaceChildren();
  const availableModels = data.availableModels || [];
  state.availableModels = availableModels;
  state.llmReady = Boolean(data.reachable && data.modelLoaded && availableModels.length);
  const rows = [
    ["Base URL", data.baseUrl || ""],
    ["Reachable", boolText(data.reachable)],
    ["Configured", data.configuredModel || ""],
    ["Loaded", boolText(data.modelLoaded)],
    ["Available", (data.availableModels || []).join(", ") || "нет"],
  ];
  if (data.error) {
    rows.push(["Ошибка", data.error]);
  }
  for (const [name, value] of rows) {
    llmStatus.append(el("dt", "", name), el("dd", "", value));
  }
  renderLlmModelOptions(data.configuredModel || "");
}

function renderLlmModelOptions(preferredModel = "") {
  const previous = llmModel.value || preferredModel;
  llmModel.replaceChildren();
  const models = state.availableModels.length ? state.availableModels : [preferredModel].filter(Boolean);
  for (const model of models) {
    const option = el("option", "", model);
    option.value = model;
    llmModel.append(option);
  }
  llmModel.disabled = !state.availableModels.length;
  if (models.includes(previous)) {
    llmModel.value = previous;
  } else if (models.includes(preferredModel)) {
    llmModel.value = preferredModel;
  }
  runQwenButton.disabled = !state.currentJobId || !state.llmReady;
}

function fillLlmProfile(profile) {
  if (!profile) {
    return;
  }
  state.llmProfile = profile;
  renderLlmModelOptions(profile.defaultModel || "");
  llmTemperature.value = profile.temperature ?? "";
  llmTimeout.value = profile.timeoutSec ?? "";
  llmStageDescription.value = profile.stageDescription || "";
  llmSystemInstruction.value = profile.systemInstruction || "";
  llmUserTemplate.value = profile.userPayloadTemplate || "";
  llmSchemaNotes.value = profile.schemaNotes || "";
}

async function loadLlmDeveloperData() {
  try {
    const [status, profiles] = await Promise.all([
      requestJson("/api/llm/status"),
      requestJson("/api/llm/profiles"),
    ]);
    renderLlmStatus(status);
    const profile = profiles.effective?.asr_correction || profiles.defaults?.asr_correction;
    fillLlmProfile(profile);
  } catch (error) {
    llmStatus.replaceChildren(el("dt", "", "Ошибка"), el("dd", "", error.message));
  }
}

function collectLlmProfilePayload() {
  return {
    profileId: state.llmProfile?.profileId || "local-asr-correction",
    version: Number(state.llmProfile?.version || 1),
    label: state.llmProfile?.label || "ASR correction",
    stageDescription: llmStageDescription.value.trim(),
    systemInstruction: llmSystemInstruction.value.trim(),
    userPayloadTemplate: llmUserTemplate.value.trim(),
    schemaNotes: llmSchemaNotes.value.trim(),
    defaultModel: llmModel.value,
    temperature: Number(llmTemperature.value),
    timeoutSec: Number(llmTimeout.value),
    maxInputChars: Number(state.llmProfile?.maxInputChars || 18000),
  };
}

function renderLlmEvents(run) {
  const events = run?.events || [];
  if (!events.length) {
    llmEventLog.textContent = "Нет LLM events.";
    return;
  }
  llmEventLog.textContent = events.map((event) => {
    const parts = [event.time, event.type];
    if (event.chunkIndex) {
      parts.push(`chunk=${event.chunkIndex}/${event.chunkTotal || "?"}`);
    }
    if (event.model) {
      parts.push(`model=${event.model}`);
    }
    if (event.addedCount !== undefined) {
      parts.push(`added=${event.addedCount}`);
    }
    if (event.skippedCount !== undefined) {
      parts.push(`skipped=${event.skippedCount}`);
    }
    if (event.error) {
      parts.push(`error=${event.error}`);
    }
    return parts.join(" · ");
  }).join("\n");
  llmEventLog.scrollTop = llmEventLog.scrollHeight;
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
  state.audioFlags = data.audioFlags?.flags || [];
  state.entities = data.entities?.entities || [];
  state.speakers = data.speakers?.speakers || [];
  state.speakerTurns = data.speakerTurns?.turns || [];
  state.segmentById = new Map(state.transcript.map((segment) => [segment.id, segment]));
  state.approvedById = new Map(state.lastApprovedSegments.map((segment) => [segment.id, segment]));
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
    audioPlayer.src = `/api/jobs/${jobId}/audio`;
    audioPlayer.load();
    workbenchView.hidden = false;
    renderWorkbench();
    await loadLlmDeveloperData();
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
  runQwenButton.disabled = !state.currentJobId || !state.llmReady;
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

function playRange(start, end, pad = 3) {
  if (!audioPlayer.src) {
    setError("Аудио недоступно для текущей задачи.");
    return;
  }
  const safeStart = Math.max(0, Number(start) - pad);
  const safeEnd = Math.max(safeStart, Number(end) + pad);
  state.audioStopAt = Number.isFinite(safeEnd) ? safeEnd : null;
  audioPlayer.currentTime = Number.isFinite(safeStart) ? safeStart : 0;
  audioPlayer.play().catch((error) => setError(error.message));
}

function segmentHeader(segment) {
  const header = el("div", "segment-meta");
  header.append(
    button("▶", "play-button", () => playRange(segment.start, segment.end)),
    el("span", "", formatTime(segment.start)),
    el("span", "", segment.speaker),
  );
  return header;
}

function approvedText(segment) {
  const approved = approvedSegment(segment.id);
  return approved ? approved.text : segment.text;
}

function appendWordTokens(container, segment) {
  const words = Array.isArray(segment.words) ? segment.words : [];
  if (!words.length) {
    container.append(document.createTextNode(approvedText(segment)));
    return;
  }
  for (const word of words) {
    const token = el("span", "word-token", `${word.word || ""} `);
    token.dataset.wordStart = word.start;
    token.dataset.wordEnd = word.end;
    token.dataset.segmentId = segment.id;
    container.append(token);
  }
}

function renderPlainSegment(segment, finalMode = false) {
  const card = el("article", finalMode ? "segment-card final" : "segment-card");
  card.dataset.segmentId = segment.id;
  card.dataset.segmentStart = segment.start;
  card.dataset.segmentEnd = segment.end;
  const body = el("p", "segment-text");
  appendWordTokens(body, segment);
  card.append(segmentHeader(segment), body);
  return card;
}

function renderSpeakerTurn(turn, finalMode = false) {
  const card = el("article", finalMode ? "segment-card final speaker-turn" : "segment-card speaker-turn");
  card.dataset.turnId = turn.id;
  card.dataset.segmentStart = turn.start;
  card.dataset.segmentEnd = turn.end;
  const pseudoSegment = {
    id: turn.segmentIds?.[0],
    start: turn.start,
    end: turn.end,
    speaker: turn.displayName || turn.speaker,
  };
  card.append(segmentHeader(pseudoSegment));
  for (const segmentId of turn.segmentIds || []) {
    const segment = state.segmentById.get(segmentId);
    if (!segment) {
      continue;
    }
    const body = el("p", "segment-text");
    appendWordTokens(body, segment);
    card.append(body);
  }
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
  card.dataset.segmentStart = segment.start;
  card.dataset.segmentEnd = segment.end;
  card.append(segmentHeader(segment));
  const body = el("p", "segment-text inline-review");
  const corrections = correctionsForSegment(segment.id).filter((correction) => (
    ["ASR_ERROR", "TYPO", "PUNCTUATION", "FILLER_GARBAGE"].includes(correction.category)
  ));
  appendWordTokens(body, segment);
  card.append(body);
  if (corrections.length) {
    const chips = el("div", "correction-strip");
    for (const correction of corrections) {
      chips.append(correctionChip(correction));
    }
    card.append(chips);
  }
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
  card.dataset.segmentStart = segment.start;
  card.dataset.segmentEnd = segment.end;
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

function speakerOptions() {
  const labels = state.speakers.map((speaker) => speaker.label).filter(Boolean);
  return labels.length ? labels : ["SPEAKER_00", "SPEAKER_01"];
}

function renderSpeakerTurnEditor(turn) {
  const card = renderSpeakerTurn(turn);
  const actions = el("div", "popover-actions");
  actions.append(
    button("Переименовать", "secondary", async () => {
      const value = window.prompt("Имя спикера", turn.displayName || turn.speaker || "");
      if (value !== null) {
        await sendJson(`/api/jobs/${state.currentJobId}/speakers/${encodeURIComponent(turn.speaker)}`, "PATCH", {
          displayName: value.trim() || null,
          verificationStatus: value.trim() ? "manual_confirmed" : "new",
        });
        await reloadReview();
      }
    }),
  );
  for (const segmentId of turn.segmentIds || []) {
    const segment = state.segmentById.get(segmentId);
    if (!segment) {
      continue;
    }
    const row = el("div", "speaker-segment-row");
    row.append(el("span", "", `${formatTime(segment.start)} ${truncate(segment.text, 52)}`));
    const select = el("select");
    for (const speaker of speakerOptions()) {
      const option = el("option", "", speaker);
      option.value = speaker;
      select.append(option);
    }
    select.value = segment.speaker;
    select.addEventListener("change", async () => {
      await sendJson(`/api/jobs/${state.currentJobId}/segments/${segment.id}/speaker`, "PATCH", { speaker: select.value });
      await reloadReview();
    });
    row.append(select);
    card.append(row);
  }
  card.append(actions);
  return card;
}

function renderListenSegment(segment) {
  const card = el("article", "segment-card");
  card.dataset.segmentId = segment.id;
  card.dataset.segmentStart = segment.start;
  card.dataset.segmentEnd = segment.end;
  card.append(segmentHeader(segment));
  const body = el("p", "segment-text inline-review");
  const words = Array.isArray(segment.words) ? segment.words : [];
  if (!words.length) {
    body.textContent = segment.text || "";
  } else {
    for (const word of words) {
      const token = el("span", "word-token", `${word.word || ""} `);
      token.dataset.wordStart = word.start;
      token.dataset.wordEnd = word.end;
      body.append(token);
    }
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

  if (state.mode === "speaker") {
    editorTitle.textContent = "Speaker Review";
    editorStats.textContent = `Turns: ${state.speakerTurns.length} · speakers: ${state.speakers.length}`;
    for (const turn of state.speakerTurns) {
      transcriptEditor.append(renderSpeakerTurnEditor(turn));
    }
    updatePlaybackHighlight();
    return;
  }

  if (state.mode === "listen") {
    editorTitle.textContent = "Listen Review";
    editorStats.textContent = `Аудио: ${formatSeconds(state.currentTime)} sec`;
    for (const segment of state.transcript) {
      transcriptEditor.append(renderListenSegment(segment));
    }
    updatePlaybackHighlight();
    return;
  }

  if (state.mode === "final") {
    editorTitle.textContent = "Final Preview";
    editorStats.textContent = "Чистый текст после принятых правок";
    for (const turn of state.speakerTurns) {
      transcriptEditor.append(renderSpeakerTurn(turn, true));
    }
    updatePlaybackHighlight();
    return;
  }

  editorTitle.textContent = "Текст";
  editorStats.textContent = `Реплики: ${state.speakerTurns.length} · сегменты: ${state.transcript.length} · спикеры: ${state.speakers.length}`;
  for (const turn of state.speakerTurns) {
    transcriptEditor.append(renderSpeakerTurn(turn));
  }
  updatePlaybackHighlight();
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
    queueItem("Прослушивание", `${state.transcript.length} сегментов`, () => {
      state.mode = "listen";
      renderWorkbench();
    }),
    queueItem("Спикеры", `${state.speakers.length} профилей`, () => {
      state.mode = "speaker";
      renderWorkbench();
    }),
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

function renderListenQueue() {
  queueTitle.textContent = "Listen queue";
  for (const flag of state.audioFlags.filter((item) => item.status !== "ignored")) {
    reviewQueue.append(queueItem(
      `${flag.category} · ${flag.status}`,
      `${formatTime(flag.startTime)} ${truncate(flag.text, 40)}`,
      () => {
        state.selectedSegmentId = flag.segmentId;
        scrollToSegment(flag.segmentId);
        playRange(flag.startTime, flag.endTime);
      },
      "warn-item",
    ));
  }
}

function renderQueue() {
  reviewQueue.replaceChildren();
  if (state.mode === "asr") {
    renderAsrQueue();
  } else if (state.mode === "entity") {
    renderEntityQueue();
  } else if (state.mode === "listen") {
    renderListenQueue();
  } else if (state.mode === "speaker") {
    renderTextQueue();
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
    button(
      "Прослушать",
      "secondary",
      () => playRange(correction.startTime, correction.endTime),
      correction.startTime === undefined || correction.endTime === undefined,
    ),
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

function updatePlaybackHighlight() {
  state.currentTime = Number(audioPlayer.currentTime || 0);
  const time = state.currentTime;

  for (const card of transcriptEditor.querySelectorAll(".segment-card[data-segment-start]")) {
    const start = Number(card.dataset.segmentStart);
    const end = Number(card.dataset.segmentEnd);
    card.classList.toggle("active-audio", Number.isFinite(start) && Number.isFinite(end) && time >= start && time <= end);
  }

  for (const token of transcriptEditor.querySelectorAll(".word-token[data-word-start]")) {
    const start = Number(token.dataset.wordStart);
    const end = Number(token.dataset.wordEnd);
    token.classList.toggle("active-word", Number.isFinite(start) && Number.isFinite(end) && time >= start && time <= end);
  }

  if (state.mode === "listen") {
    editorStats.textContent = `Аудио: ${formatSeconds(time)} sec`;
  }
}

audioPlayer.addEventListener("timeupdate", () => {
  updatePlaybackHighlight();
  if (state.audioStopAt !== null && audioPlayer.currentTime >= state.audioStopAt) {
    audioPlayer.pause();
    state.audioStopAt = null;
  }
});

audioPlayer.addEventListener("seeked", updatePlaybackHighlight);

playbackSpeed.addEventListener("change", () => {
  audioPlayer.playbackRate = Number(playbackSpeed.value) || 1;
});

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

saveLlmProfileButton.addEventListener("click", async () => {
  setError("");
  saveLlmProfileButton.disabled = true;
  try {
    const data = await sendJson("/api/llm/profiles/asr_correction", "PUT", collectLlmProfilePayload());
    fillLlmProfile(data.profile);
    await loadLlmDeveloperData();
    appendProgressLine("LLM profile сохранен.");
  } catch (error) {
    setError(error.message);
  } finally {
    saveLlmProfileButton.disabled = false;
  }
});

async function pollLlmRun(runId) {
  const run = await requestJson(`/api/jobs/${state.currentJobId}/llm/runs/${runId}`);
  renderLlmEvents(run);
  if (run.status === "done") {
    clearInterval(state.llmRunPollTimer);
    state.llmRunPollTimer = null;
    await reloadReview();
    state.mode = "asr";
    renderWorkbench();
    const summary = run.summary || {};
    appendProgressLine(`LLM: добавлено ${summary.addedCount || 0}, пропущено ${summary.skippedCount || 0}.`);
    runQwenButton.textContent = "LLM: найти правки";
    runQwenButton.disabled = !state.currentJobId;
  } else if (run.status === "failed") {
    clearInterval(state.llmRunPollTimer);
    state.llmRunPollTimer = null;
    setError(run.error || "Ошибка LLM run.");
    runQwenButton.textContent = "LLM: найти правки";
    runQwenButton.disabled = !state.currentJobId;
  }
}

runQwenButton.addEventListener("click", async () => {
  if (!state.currentJobId) {
    setError("Сначала откройте завершенный результат.");
    return;
  }
  setError("");
  runQwenButton.disabled = true;
  runQwenButton.textContent = "LLM: обработка...";
  try {
    const payload = {
      stage: "asr_correction",
      model: llmModel.value.trim(),
      temperature: Number(llmTemperature.value),
      timeoutSec: Number(llmTimeout.value),
      profileId: state.llmProfile?.profileId || "default-asr-correction",
    };
    const run = await sendJson(`/api/jobs/${state.currentJobId}/llm/runs`, "POST", payload);
    renderLlmEvents(run);
    state.llmRunPollTimer = setInterval(() => {
      pollLlmRun(run.run_id).catch((error) => {
        clearInterval(state.llmRunPollTimer);
        state.llmRunPollTimer = null;
        setError(error.message);
        runQwenButton.textContent = "LLM: найти правки";
        runQwenButton.disabled = !state.currentJobId;
      });
    }, 1000);
    await pollLlmRun(run.run_id);
  } catch (error) {
    setError(error.message);
    runQwenButton.textContent = "LLM: найти правки";
    runQwenButton.disabled = !state.currentJobId;
  }
});

popoverLayer.addEventListener("click", (event) => {
  if (event.target === popoverLayer) {
    closePopover();
  }
});

loadHealth();
loadLlmDeveloperData();
