const form = document.getElementById("jobForm");
const startButton = document.getElementById("startButton");
const formError = document.getElementById("formError");
const progress = document.getElementById("progress");
const result = document.getElementById("result");
const health = document.getElementById("health");
const statusBadge = document.getElementById("statusBadge");

let pollTimer = null;

function setError(message) {
  formError.textContent = message || "";
}

function boolText(value) {
  return value ? "да" : "нет";
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Ошибка сервера: ${response.status}`);
  }
  return data;
}

async function loadHealth() {
  try {
    const data = await readJson(await fetch("/api/health"));
    health.innerHTML = `
      <dt>CUDA</dt><dd>${boolText(data.cuda_available)}</dd>
      <dt>GPU</dt><dd>${data.gpu || "NO CUDA"}</dd>
      <dt>HF token</dt><dd>${boolText(data.hf_token_present)}</dd>
      <dt>FFmpeg</dt><dd>${boolText(data.ffmpeg_ok)}</dd>
      <dt>WhisperX</dt><dd>${boolText(data.whisperx_ok)}</dd>
      <dt>HF_HOME</dt><dd>${data.hf_home || ""}</dd>
      <dt>TMP</dt><dd>${data.tmp || ""}</dd>
    `;
    statusBadge.textContent = data.cuda_available ? "CUDA доступна" : "CUDA недоступна";
    statusBadge.className = data.cuda_available ? "badge ok" : "badge warn";
  } catch (error) {
    statusBadge.textContent = "Ошибка health";
    statusBadge.className = "badge bad";
    health.innerHTML = `<dt>Ошибка</dt><dd>${error.message}</dd>`;
  }
}

function collectFormData() {
  const data = new FormData();
  const fileInput = document.getElementById("file");
  data.append("file", fileInput.files[0]);
  data.append("output_dir", document.getElementById("output_dir").value);
  data.append("model", document.getElementById("model").value);
  data.append("language", document.getElementById("language").value);
  data.append("device", document.getElementById("device").value);
  data.append("compute_type", document.getElementById("compute_type").value);
  data.append("batch_size", document.getElementById("batch_size").value);
  data.append("diarize", document.getElementById("diarize").checked ? "true" : "false");
  data.append("min_speakers", document.getElementById("min_speakers").value);
  data.append("max_speakers", document.getElementById("max_speakers").value);

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

async function loadFiles(jobId) {
  const data = await readJson(await fetch(`/api/jobs/${jobId}/files`));
  const links = data.files
    .map((item) => `<li><span>${item.name}</span><code>${item.path}</code></li>`)
    .join("");
  result.innerHTML = `
    <p><strong>Папка вывода:</strong> <code>${data.output_dir}</code></p>
    <ul class="files">${links}</ul>
  `;
}

async function pollJob(jobId) {
  try {
    const job = await readJson(await fetch(`/api/jobs/${jobId}`));
    renderProgress(job.progress || []);

    if (job.status === "done") {
      clearInterval(pollTimer);
      pollTimer = null;
      startButton.disabled = false;
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  result.textContent = "Задача запускается...";
  progress.textContent = "Задача запускается...";
  startButton.disabled = true;

  try {
    const data = await readJson(await fetch("/api/jobs", {
      method: "POST",
      body: collectFormData(),
    }));
    pollTimer = setInterval(() => pollJob(data.job_id), 2000);
    await pollJob(data.job_id);
  } catch (error) {
    startButton.disabled = false;
    result.textContent = "Пока нет результата.";
    setError(error.message);
  }
});

loadHealth();
