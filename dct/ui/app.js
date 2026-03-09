async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

const STORAGE_KEY = "dct_control_surface_state_v2";

const PROVIDER_DEFAULT_URLS = {
  openai_compatible: "http://localhost:11434/v1",
  openai: "https://api.openai.com/v1",
  azure_openai: "https://YOUR_RESOURCE.openai.azure.com",
  xai: "https://api.x.ai/v1",
  deepseek: "https://api.deepseek.com/v1",
  groq: "https://api.groq.com/openai/v1",
  mistral: "https://api.mistral.ai/v1",
  together: "https://api.together.xyz/v1",
  fireworks: "https://api.fireworks.ai/inference/v1",
  openrouter: "https://openrouter.ai/api/v1",
  ollama: "http://localhost:11434/v1",
  lmstudio: "http://localhost:1234/v1",
  vllm: "http://localhost:8000/v1",
  llamacpp: "http://localhost:8080/v1",
  anthropic: "https://api.anthropic.com",
  gemini: "https://generativelanguage.googleapis.com",
};

const els = {
  healthBadge: document.getElementById("health-badge"),
  progressBadge: document.getElementById("progress-badge"),
  refreshAll: document.getElementById("refresh-all"),
  runForm: document.getElementById("run-form"),
  startRunBtn: document.getElementById("start-run-btn"),
  stopCurrentJobBtn: document.getElementById("stop-current-job-btn"),
  providerHint: document.getElementById("provider-url-hint"),
  modelFetchHint: document.getElementById("model-fetch-hint"),
  modelNameOptions: document.getElementById("model-name-options"),
  reasonerModelNameOptions: document.getElementById("reasoner-model-name-options"),

  jobsList: document.getElementById("jobs-list"),
  jobTemplate: document.getElementById("job-item-template"),

  runsList: document.getElementById("runs-list"),
  runTemplate: document.getElementById("run-item-template"),
  reloadRuns: document.getElementById("reload-runs"),

  runDetail: document.getElementById("run-detail"),
  runTitle: document.getElementById("run-title"),
  summaryCards: document.getElementById("summary-cards"),
  upliftTable: document.getElementById("uplift-table"),
  methodsTable: document.getElementById("methods-table"),
  plotsGrid: document.getElementById("plots-grid"),
  artifactsList: document.getElementById("artifacts-list"),

  explainFocus: document.getElementById("explain-focus"),
  explainRunBtn: document.getElementById("explain-run-btn"),
  runExplainOutput: document.getElementById("run-explain-output"),

  tabRunsBtn: document.getElementById("tab-runs-btn"),
  tabStreamBtn: document.getElementById("tab-stream-btn"),
  tabCollisionBtn: document.getElementById("tab-collision-btn"),
  tabRuns: document.getElementById("tab-runs"),
  tabStream: document.getElementById("tab-stream"),
  tabCollision: document.getElementById("tab-collision"),

  streamOutputHead: document.getElementById("stream-output-head"),
  streamOutput: document.getElementById("stream-output"),

  collisionForm: document.getElementById("collision-form"),
  collisionAddDiscovery: document.getElementById("collision-add-discovery"),
  discoveriesList: document.getElementById("discoveries-list"),
  discoveryTemplate: document.getElementById("discovery-item-template"),
  knownTheories: document.getElementById("known-theories"),
  memoryExpressions: document.getElementById("memory-expressions"),
  maxCollisions: document.getElementById("max-collisions"),
  collisionOutput: document.getElementById("collision-output"),

  quickConfig: document.getElementById("quick-config"),
  fullConfig: document.getElementById("full-config"),
  openworldConfig: document.getElementById("openworld-config"),
  readmeContent: document.getElementById("readme-content"),
};

let currentRunName = null;
let currentJobId = null;
let jobsPollHandle = null;
let activeTab = "runs";
let latestJobs = [];
let runFormRestoring = false;
let modelsFetchTimer = null;

function hasElement(el) {
  return el !== null && el !== undefined;
}

function setText(el, text) {
  if (hasElement(el)) {
    el.textContent = text;
  }
}

function toNumber(value) {
  if (typeof value !== "number") return null;
  if (!Number.isFinite(value)) return null;
  return value;
}

function formatNumber(value) {
  if (value == null) return "-";
  return Number(value).toFixed(4);
}

function formatJSON(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (err) {
    return String(value);
  }
}

function parseLines(text) {
  if (typeof text !== "string") return [];
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function isRemoteUrl(urlText) {
  try {
    const parsed = new URL(urlText);
    const host = (parsed.hostname || "").trim().toLowerCase();
    return host !== "localhost" && host !== "127.0.0.1" && host !== "::1";
  } catch (err) {
    return false;
  }
}

function readStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed ? parsed : {};
  } catch (err) {
    return {};
  }
}

function writeStorage(partial) {
  try {
    const merged = { ...readStorage(), ...partial };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
  } catch (err) {
    return;
  }
}

function serializeRunForm() {
  if (!hasElement(els.runForm)) return {};
  const out = {};
  const controls = els.runForm.querySelectorAll("input, select, textarea");
  controls.forEach((ctrl) => {
    const name = ctrl.name;
    if (!name) return;
    if (ctrl.type === "checkbox") {
      out[name] = ctrl.checked;
      return;
    }
    out[name] = ctrl.value;
  });
  return out;
}

function applyRunFormState(state) {
  if (!hasElement(els.runForm) || !state || typeof state !== "object") return;
  runFormRestoring = true;
  const controls = els.runForm.querySelectorAll("input, select, textarea");
  controls.forEach((ctrl) => {
    const name = ctrl.name;
    if (!name || !(name in state)) return;
    if (ctrl.type === "checkbox") {
      ctrl.checked = Boolean(state[name]);
    } else {
      ctrl.value = String(state[name] ?? "");
    }
  });
  runFormRestoring = false;
}

function persistUIState() {
  writeStorage({
    activeTab,
    currentJobId,
    currentRunName,
    runForm: serializeRunForm(),
    collisionDraft: serializeCollisionDraft(),
  });
}

function restoreUIState() {
  const saved = readStorage();
  if (saved.activeTab) activeTab = saved.activeTab;
  if (saved.currentJobId) currentJobId = saved.currentJobId;
  if (saved.currentRunName) currentRunName = saved.currentRunName;
  if (saved.runForm) applyRunFormState(saved.runForm);
  if (saved.collisionDraft) applyCollisionDraft(saved.collisionDraft);
}

function updateProviderHint(provider) {
  if (!hasElement(els.providerHint)) return;
  const suggested = PROVIDER_DEFAULT_URLS[provider] || "(no default)";
  let targetField = "openai_base_url";
  if (provider === "anthropic") targetField = "anthropic_base_url";
  if (provider === "gemini") targetField = "google_base_url";
  setText(
    els.providerHint,
    `Auto Base URL target: ${targetField} -> ${suggested}`
  );
}

function providerBaseUrlFieldName(provider) {
  if (provider === "anthropic") return "anthropic_base_url";
  if (provider === "gemini") return "google_base_url";
  return "openai_base_url";
}

function inputByName(name) {
  if (!hasElement(els.runForm)) return null;
  return els.runForm.querySelector(`[name="${name}"]`);
}

function autoFillBaseUrl(provider, force = false) {
  const defaultUrl = PROVIDER_DEFAULT_URLS[provider];
  if (!defaultUrl) return;
  const fieldName = providerBaseUrlFieldName(provider);
  const target = inputByName(fieldName);
  if (!target) return;

  const current = (target.value || "").trim();
  if (force || !current) {
    target.value = defaultUrl;
  }

  if (force && provider !== "anthropic" && provider !== "gemini") {
    const otherAnth = inputByName("anthropic_base_url");
    const otherGem = inputByName("google_base_url");
    if (otherAnth && !otherAnth.value.trim()) otherAnth.value = PROVIDER_DEFAULT_URLS.anthropic;
    if (otherGem && !otherGem.value.trim()) otherGem.value = PROVIDER_DEFAULT_URLS.gemini;
  }
  updateProviderHint(provider);
}

function switchTab(name) {
  activeTab = name;

  if (hasElement(els.tabRunsBtn)) els.tabRunsBtn.classList.toggle("active", name === "runs");
  if (hasElement(els.tabStreamBtn)) els.tabStreamBtn.classList.toggle("active", name === "stream");
  if (hasElement(els.tabCollisionBtn)) els.tabCollisionBtn.classList.toggle("active", name === "collision");
  if (hasElement(els.tabRuns)) els.tabRuns.classList.toggle("active", name === "runs");
  if (hasElement(els.tabStream)) els.tabStream.classList.toggle("active", name === "stream");
  if (hasElement(els.tabCollision)) els.tabCollision.classList.toggle("active", name === "collision");
  persistUIState();
}

function normalizePayload(formData) {
  const payload = {};
  for (const [key, value] of formData.entries()) {
    if (value === "") continue;
    payload[key] = value;
  }

  payload.allow_remote_inference = formData.get("allow_remote_inference") === "on";
  payload.skip_model_check = formData.get("skip_model_check") === "on";
  payload.use_reasoner = formData.get("use_reasoner") === "on";

  const tempRaw = formData.get("model_temperature");
  if (typeof tempRaw === "string" && tempRaw.trim() !== "") {
    const parsed = parseFloat(tempRaw);
    if (!Number.isNaN(parsed)) payload.model_temperature = parsed;
  }

  if (!payload.config_path) delete payload.config_path;
  if (!payload.output_dir) delete payload.output_dir;
  if (!payload.model_name) delete payload.model_name;
  if (!payload.reasoner_model_name) delete payload.reasoner_model_name;

  return payload;
}

function buildRuntimePayloadFromRunForm() {
  if (!hasElement(els.runForm)) return {};
  const formData = new FormData(els.runForm);
  const normalized = normalizePayload(formData);
  const keys = [
    "model_provider",
    "model_access_mode",
    "model_name",
    "model_temperature",
    "use_reasoner",
    "reasoner_model_name",
    "allow_remote_inference",
    "openai_base_url",
    "openai_api_key",
    "anthropic_base_url",
    "anthropic_api_key",
    "google_base_url",
    "google_api_key",
  ];
  const runtime = {};
  for (const key of keys) {
    if (key in normalized) runtime[key] = normalized[key];
  }
  return runtime;
}

function selectedProviderAndUrl() {
  const runtime = buildRuntimePayloadFromRunForm();
  const provider = runtime.model_provider || "openai_compatible";
  let baseUrl = runtime.openai_base_url || "";
  if (provider === "anthropic") baseUrl = runtime.anthropic_base_url || "";
  if (provider === "gemini") baseUrl = runtime.google_base_url || "";
  return { provider, baseUrl, runtime };
}

function setModelFetchHint(message, isError = false) {
  if (!hasElement(els.modelFetchHint)) return;
  els.modelFetchHint.textContent = message;
  els.modelFetchHint.style.color = isError ? "#ff8f8f" : "";
}

function fillDatalist(el, values) {
  if (!hasElement(el)) return;
  el.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    el.appendChild(option);
  }
}

function pickFirstNonEmpty(values) {
  for (const item of values || []) {
    const value = String(item || "").trim();
    if (value) return value;
  }
  return "";
}

function applyModelSuggestions(models, reasonerModels) {
  fillDatalist(els.modelNameOptions, models);
  fillDatalist(els.reasonerModelNameOptions, reasonerModels);

  const modelInput = inputByName("model_name");
  const reasonerInput = inputByName("reasoner_model_name");
  if (hasElement(modelInput) && !(modelInput.value || "").trim()) {
    const suggestion = pickFirstNonEmpty(models);
    if (suggestion) modelInput.value = suggestion;
  }
  if (hasElement(reasonerInput) && !(reasonerInput.value || "").trim()) {
    const suggestion = pickFirstNonEmpty(reasonerModels) || pickFirstNonEmpty(models);
    if (suggestion) reasonerInput.value = suggestion;
  }
}

function buildProviderModelsPayload() {
  const runtime = buildRuntimePayloadFromRunForm();
  const payload = {
    model_provider: runtime.model_provider || "openai_compatible",
    model_access_mode: runtime.model_access_mode || "local",
    allow_remote_inference: Boolean(runtime.allow_remote_inference),
    openai_base_url: runtime.openai_base_url || "",
    openai_api_key: runtime.openai_api_key || "",
    anthropic_base_url: runtime.anthropic_base_url || "",
    anthropic_api_key: runtime.anthropic_api_key || "",
    google_base_url: runtime.google_base_url || "",
    google_api_key: runtime.google_api_key || "",
  };
  return payload;
}

function scheduleProviderModelFetch(delayMs = 350) {
  if (modelsFetchTimer) clearTimeout(modelsFetchTimer);
  modelsFetchTimer = setTimeout(() => {
    loadProviderModels(false);
  }, delayMs);
}

async function loadProviderModels(force = false) {
  const payload = buildProviderModelsPayload();
  const provider = payload.model_provider || "openai_compatible";
  if (!force && runFormRestoring) return;

  if (provider === "anthropic" && !(payload.anthropic_api_key || "").trim()) {
    fillDatalist(els.modelNameOptions, []);
    fillDatalist(els.reasonerModelNameOptions, []);
    setModelFetchHint("Model list: fill ANTHROPIC_API_KEY to fetch models.");
    return;
  }
  if (provider === "gemini" && !(payload.google_api_key || "").trim()) {
    fillDatalist(els.modelNameOptions, []);
    fillDatalist(els.reasonerModelNameOptions, []);
    setModelFetchHint("Model list: fill GOOGLE_API_KEY to fetch models.");
    return;
  }

  setModelFetchHint(`Loading models for ${provider}...`);
  try {
    const result = await api("/api/provider-models", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const models = Array.isArray(result.models) ? result.models : [];
    const reasonerModels = Array.isArray(result.reasoner_models) ? result.reasoner_models : [];
    applyModelSuggestions(models, reasonerModels);
    setModelFetchHint(
      `Model list loaded: ${models.length} models, ${reasonerModels.length} reasoner candidates (${provider}).`
    );
    persistUIState();
  } catch (err) {
    fillDatalist(els.modelNameOptions, []);
    fillDatalist(els.reasonerModelNameOptions, []);
    setModelFetchHint(`Model list fetch failed (${provider}): ${err.message}`, true);
  }
}

function setHealth(statusText, ok = false) {
  if (!hasElement(els.healthBadge)) return;
  els.healthBadge.textContent = statusText;
  els.healthBadge.style.color = ok ? "#6efac9" : "#ffb97a";
}

async function loadHealth() {
  try {
    const data = await api("/api/health");
    setHealth(`${data.status} @ ${new Date(data.time).toLocaleTimeString()}`, true);
  } catch (err) {
    setHealth(`offline: ${err.message}`, false);
  }
}

function updateProgressBadge(jobs) {
  if (!hasElement(els.progressBadge)) return;
  const completed = jobs.filter((job) => job.status === "completed").length;
  const running = jobs.filter((job) => job.status === "running").length;
  const stopping = jobs.filter((job) => job.status === "stopping").length;
  const cancelled = jobs.filter((job) => job.status === "cancelled").length;
  const failed = jobs.filter((job) => job.status === "failed").length;
  els.progressBadge.textContent = `Progress: ${completed} done | ${running} running | ${stopping} stopping | ${cancelled} cancelled | ${failed} failed`;
}

function canStopJob(job) {
  if (!job || typeof job !== "object") return false;
  return job.status === "queued" || job.status === "running" || job.status === "stopping";
}

function selectedJobFromLatest() {
  if (!Array.isArray(latestJobs) || !latestJobs.length) return null;
  return latestJobs.find((j) => j.job_id === currentJobId) || latestJobs[0] || null;
}

function updateRunActionButtons(jobs = latestJobs) {
  if (!hasElement(els.stopCurrentJobBtn)) return;
  const selected = Array.isArray(jobs)
    ? jobs.find((j) => j.job_id === currentJobId) || jobs[0] || null
    : null;
  const stoppable = canStopJob(selected);
  els.stopCurrentJobBtn.disabled = !stoppable;
  els.stopCurrentJobBtn.textContent = selected?.status === "stopping" ? "Stopping..." : "Stop Current Job";
}

async function requestStopJob(jobId) {
  if (!jobId) return;
  try {
    const updated = await api(`/api/jobs/${encodeURIComponent(jobId)}/stop`, {
      method: "POST",
    });
    currentJobId = updated.job_id || jobId;
    persistUIState();
    await loadJobs();
  } catch (err) {
    alert(`Failed to stop job: ${err.message}`);
  }
}

async function handleStopCurrentJob() {
  const selected = selectedJobFromLatest();
  if (!selected) {
    alert("No jobs to stop.");
    return;
  }
  if (!canStopJob(selected)) {
    alert(`Selected job cannot be stopped from status: ${selected.status}`);
    return;
  }
  await requestStopJob(selected.job_id);
}

function renderStreamOutput(job) {
  if (!hasElement(els.streamOutputHead) || !hasElement(els.streamOutput)) return;

  if (!job) {
    setText(els.streamOutputHead, "Select a job to view live model outputs.");
    setText(els.streamOutput, "");
    return;
  }

  const req = job.request || {};
  const headBits = [
    `job=${job.job_id}`,
    `status=${job.status}`,
    `provider=${req.model_provider || "(env/default)"}`,
    `model=${req.model_name || "(env/default)"}`,
    `temp=${req.model_temperature ?? "(env/default)"}`,
    `reasoner=${req.use_reasoner ? "on" : "off"}`,
  ];
  if (job.run_name) headBits.push(`run=${job.run_name}`);
  setText(els.streamOutputHead, headBits.join(" | "));

  const logs = Array.isArray(job.logs) ? job.logs : [];
  if (!logs.length) {
    setText(els.streamOutput, "No logs yet...");
    return;
  }

  const modelLogs = logs.filter((l) => l.level === "model" && l.event && l.event.type === "model_output");
  const targetLogs = modelLogs.length ? modelLogs : logs;

  const blocks = targetLogs.map((log) => {
    const ts = log.time ? new Date(log.time).toLocaleTimeString() : "--:--:--";
    const ev = log.event || {};
    if (ev.type === "model_output") {
      const provider = ev.provider || "provider";
      const model = ev.model || "model";
      const phase = ev.phase || "primary";
      const agent = ev.agent_hint || "agent";
      const text = typeof ev.text === "string" ? ev.text : log.message || "";
      return `[${ts}] ${provider}/${model} [${phase}] ${agent}\n${text}`;
    }
    return `[${ts}] ${(log.level || "info").toUpperCase()} ${log.message || ""}`;
  });

  if (job.error) {
    blocks.push(`[ERROR] ${job.error}`);
  }

  setText(els.streamOutput, blocks.join("\n\n------------------------------\n\n"));
  els.streamOutput.scrollTop = els.streamOutput.scrollHeight;
}

function renderJobs(jobs) {
  if (!hasElement(els.jobsList)) return;
  els.jobsList.innerHTML = "";
  latestJobs = jobs;
  updateProgressBadge(jobs);

  if (!jobs.length) {
    els.jobsList.innerHTML = '<p class="sub">No run jobs yet.</p>';
    renderStreamOutput(null);
    updateRunActionButtons([]);
    return;
  }

  for (const job of jobs) {
    let node = null;
    if (hasElement(els.jobTemplate) && els.jobTemplate.content && els.jobTemplate.content.firstElementChild) {
      node = els.jobTemplate.content.firstElementChild.cloneNode(true);
    } else {
      node = document.createElement("article");
      node.className = "item";
      node.innerHTML =
        '<div class="item-top"><strong class="name"></strong><div class="inline-actions"><span class="status"></span><button class="btn btn-ghost job-stop-btn" type="button">Stop</button></div></div><p class="meta"></p>';
    }

    setText(node.querySelector(".name"), job.job_id);

    const statusEl = node.querySelector(".status") || node.querySelector(".status-chip");
    if (hasElement(statusEl)) {
      statusEl.textContent = job.status;
      statusEl.classList.add(job.status);
    }

    const stopBtn = node.querySelector(".job-stop-btn");
    if (hasElement(stopBtn)) {
      const stoppable = canStopJob(job);
      stopBtn.disabled = !stoppable;
      stopBtn.textContent = job.status === "stopping" ? "Stopping..." : "Stop";
      stopBtn.addEventListener("click", async (event) => {
        event.stopPropagation();
        await requestStopJob(job.job_id);
      });
    }

    const meta = [];
    if (job.run_name) meta.push(`run=${job.run_name}`);
    if (job.error) meta.push(`error=${job.error}`);
    const req = job.request || {};
    if (req.use_reasoner) meta.push(`reasoner=${req.reasoner_model_name || "on"}`);

    const lastMessage = Array.isArray(job.logs) && job.logs.length ? job.logs[job.logs.length - 1].message : null;
    if (lastMessage) meta.push(`last=${lastMessage}`);
    meta.push(`updated=${new Date(job.updated_at).toLocaleString()}`);

    setText(node.querySelector(".meta"), meta.join(" | "));

    if (job.job_id === currentJobId) node.classList.add("selected");

    node.addEventListener("click", () => {
      currentJobId = job.job_id;
      persistUIState();
      renderJobs(jobs);
      renderStreamOutput(job);
      switchTab("stream");
      if (job.run_name) {
        currentRunName = job.run_name;
        persistUIState();
        loadRunDetail(job.run_name);
      }
    });

    els.jobsList.appendChild(node);
  }
  updateRunActionButtons(jobs);
}

async function loadJobs() {
  try {
    const jobs = await api("/api/jobs");
    renderJobs(jobs);

    if (!currentJobId && jobs.length) currentJobId = jobs[0].job_id;

    const selected = jobs.find((j) => j.job_id === currentJobId) || jobs[0] || null;
    renderStreamOutput(selected);
    updateRunActionButtons(jobs);

    if (selected?.run_name && selected.status === "completed" && selected.run_name !== currentRunName) {
      currentRunName = selected.run_name;
      persistUIState();
      await loadRunDetail(selected.run_name);
      await loadRuns();
    }
  } catch (err) {
    if (hasElement(els.jobsList)) {
      els.jobsList.innerHTML = `<p class="sub">Failed to load jobs: ${err.message}</p>`;
    }
    updateRunActionButtons([]);
  }
}

function renderRuns(runs) {
  if (!hasElement(els.runsList)) return;
  els.runsList.innerHTML = "";

  if (!runs.length) {
    els.runsList.innerHTML = '<p class="sub">No runs found in output root yet.</p>';
    return;
  }

  for (const run of runs) {
    let node = null;
    if (hasElement(els.runTemplate) && els.runTemplate.content && els.runTemplate.content.firstElementChild) {
      node = els.runTemplate.content.firstElementChild.cloneNode(true);
    } else {
      node = document.createElement("button");
      node.type = "button";
      node.className = "item run-item";
      node.innerHTML = '<div class="item-top"><strong class="name"></strong><span class="meta-time"></span></div><p class="meta"></p>';
    }

    setText(node.querySelector(".name"), run.run_name);
    setText(node.querySelector(".meta-time"), new Date(run.updated_at).toLocaleString());
    setText(node.querySelector(".meta"), run.run_dir);

    node.addEventListener("click", () => {
      currentRunName = run.run_name;
      persistUIState();
      switchTab("runs");
      loadRunDetail(run.run_name);
    });

    els.runsList.appendChild(node);
  }
}

async function loadRuns() {
  try {
    const runs = await api("/api/runs");
    renderRuns(runs);

    if (!currentRunName && runs.length) {
      currentRunName = runs[0].run_name;
      persistUIState();
      loadRunDetail(currentRunName);
    }
  } catch (err) {
    if (hasElement(els.runsList)) {
      els.runsList.innerHTML = `<p class="sub">Failed to load runs: ${err.message}</p>`;
    }
  }
}

function metricCard(label, value) {
  const card = document.createElement("article");
  card.className = "metric-card";
  card.innerHTML = `<div class="k">${label}</div><div class="v">${value}</div>`;
  return card;
}

function renderUpliftTable(uplift) {
  if (!hasElement(els.upliftTable)) return;
  const baselines = Object.keys(uplift || {});
  if (!baselines.length) {
    els.upliftTable.innerHTML = '<p class="sub">No uplift data.</p>';
    return;
  }

  const metricSet = new Set();
  for (const b of baselines) Object.keys(uplift[b] || {}).forEach((m) => metricSet.add(m));
  const metrics = Array.from(metricSet);

  const rows = baselines
    .map((b) => {
      const cells = metrics.map((m) => `<td>${formatNumber(toNumber(uplift[b]?.[m]))}</td>`).join("");
      return `<tr><td>${b}</td>${cells}</tr>`;
    })
    .join("");

  els.upliftTable.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr><th>Baseline</th>${metrics.map((m) => `<th>${m}</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderMethodsTable(methodSummaries) {
  if (!hasElement(els.methodsTable)) return;
  if (!methodSummaries?.length) {
    els.methodsTable.innerHTML = '<p class="sub">No method summary data.</p>';
    return;
  }

  const rows = methodSummaries
    .map(
      (m) => `
        <tr>
          <td>${m.method}</td>
          <td>${m.trial_index}</td>
          <td>${formatNumber(toNumber(m.validity_rate))}</td>
          <td>${formatNumber(toNumber(m.heldout_predictive_accuracy))}</td>
          <td>${formatNumber(toNumber(m.ood_predictive_accuracy))}</td>
          <td>${formatNumber(toNumber(m.stress_predictive_accuracy))}</td>
          <td>${formatNumber(toNumber(m.transfer_generalization_score))}</td>
          <td>${formatNumber(toNumber(m.open_world_readiness_score))}</td>
          <td>${formatNumber(toNumber(m.rule_recovery_exact_match_rate))}</td>
          <td>${formatNumber(toNumber(m.compression_score))}</td>
          <td>${formatNumber(toNumber(m.novelty_score))}</td>
          <td>${formatNumber(toNumber(m.time_to_valid_discovery))}</td>
          <td>${formatNumber(toNumber(m.cumulative_improvement))}</td>
        </tr>
      `
    )
    .join("");

  els.methodsTable.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Trial</th>
            <th>Validity</th>
            <th>Heldout Acc</th>
            <th>OOD Acc</th>
            <th>Stress Acc</th>
            <th>Transfer</th>
            <th>OpenWorld</th>
            <th>Exact Match</th>
            <th>Compression</th>
            <th>Novelty</th>
            <th>Time-to-Valid</th>
            <th>Cumulative</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderArtifacts(artifactsResponse) {
  if (!hasElement(els.plotsGrid) || !hasElement(els.artifactsList)) return;

  const artifacts = artifactsResponse.artifacts || [];
  const plotFiles = artifacts.filter((a) => a.relative_path.includes("/plots/") && a.name.endsWith(".png"));

  els.plotsGrid.innerHTML = "";
  if (!plotFiles.length) {
    els.plotsGrid.innerHTML = '<p class="sub">No plot images found.</p>';
  } else {
    for (const plot of plotFiles) {
      const card = document.createElement("article");
      card.className = "plot-card";
      card.innerHTML = `<img src="${plot.url}" alt="${plot.name}" /><a href="${plot.url}" target="_blank">${plot.name}</a>`;
      els.plotsGrid.appendChild(card);
    }
  }

  els.artifactsList.innerHTML = "";
  if (!artifacts.length) {
    els.artifactsList.innerHTML = '<p class="sub">No artifacts found.</p>';
    return;
  }

  for (const file of artifacts) {
    const link = document.createElement("a");
    link.href = file.url;
    link.target = "_blank";
    link.textContent = `${file.relative_path} (${file.size_bytes} bytes)`;
    els.artifactsList.appendChild(link);
  }
}

async function loadRunDetail(runName) {
  if (!hasElement(els.runDetail)) return;
  try {
    const [summary, artifacts] = await Promise.all([
      api(`/api/runs/${encodeURIComponent(runName)}`),
      api(`/api/runs/${encodeURIComponent(runName)}/artifacts`),
    ]);

    els.runDetail.classList.remove("hidden");
    setText(els.runTitle, `Run Detail: ${summary.run_name}`);
    if (hasElement(els.runExplainOutput)) {
      setText(els.runExplainOutput, "Click `Explain This Run` to generate an online LLM interpretation.");
    }

    const methodSummaries = summary.method_summaries || [];
    const methods = new Set(methodSummaries.map((m) => m.method));
    const trials = new Set(methodSummaries.map((m) => m.trial_index));

    if (hasElement(els.summaryCards)) {
      els.summaryCards.innerHTML = "";
      els.summaryCards.appendChild(metricCard("Methods", methods.size));
      els.summaryCards.appendChild(metricCard("Trials", trials.size));
      els.summaryCards.appendChild(metricCard("Run Name", summary.run_name || "-"));
      els.summaryCards.appendChild(metricCard("Output Dir", artifacts.run_dir || "-"));
    }

    renderUpliftTable(summary.uplift || {});
    renderMethodsTable(methodSummaries);
    renderArtifacts(artifacts);
  } catch (err) {
    els.runDetail.classList.remove("hidden");
    setText(els.runTitle, `Run Detail Error: ${runName}`);
    if (hasElement(els.summaryCards)) els.summaryCards.innerHTML = `<p class="sub">${err.message}</p>`;
    if (hasElement(els.upliftTable)) els.upliftTable.innerHTML = "";
    if (hasElement(els.methodsTable)) els.methodsTable.innerHTML = "";
    if (hasElement(els.plotsGrid)) els.plotsGrid.innerHTML = "";
    if (hasElement(els.artifactsList)) els.artifactsList.innerHTML = "";
  }
}

async function handleExplainRun() {
  if (!currentRunName) {
    alert("Select a run first.");
    return;
  }
  if (!hasElement(els.runExplainOutput)) return;

  const { provider, baseUrl, runtime } = selectedProviderAndUrl();
  const accessMode = runtime.model_access_mode || "local";
  const allowRemote = Boolean(runtime.allow_remote_inference);
  if (accessMode !== "online" || !allowRemote || !isRemoteUrl(baseUrl)) {
    setText(
      els.runExplainOutput,
      "Online explanation is blocked. Set provider to a remote URL, model_access_mode=online, and allow_remote_inference=true."
    );
    return;
  }

  const payload = {
    ...runtime,
    focus: hasElement(els.explainFocus) ? (els.explainFocus.value || "").trim() : "",
  };

  setText(els.runExplainOutput, "Generating explanation...");
  try {
    const response = await api(`/api/runs/${encodeURIComponent(currentRunName)}/explain`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setText(els.runExplainOutput, formatJSON(response.explanation || response));
  } catch (err) {
    setText(els.runExplainOutput, `Failed to explain run (${provider} @ ${baseUrl}): ${err.message}`);
  }
}

function createDiscoveryNode() {
  if (hasElement(els.discoveryTemplate) && els.discoveryTemplate.content && els.discoveryTemplate.content.firstElementChild) {
    return els.discoveryTemplate.content.firstElementChild.cloneNode(true);
  }
  const node = document.createElement("article");
  node.className = "item discovery-item";
  node.innerHTML = `
    <div class="item-top">
      <strong class="name">Discovery</strong>
      <button class="btn btn-ghost remove-discovery" type="button">Remove</button>
    </div>
    <label><span>Expression</span><input type="text" class="discovery-expression" /></label>
  `;
  return node;
}

function hydrateDiscoveryNode(node, index, data = {}) {
  const idInput = node.querySelector(".discovery-id");
  const titleInput = node.querySelector(".discovery-title");
  const exprInput = node.querySelector(".discovery-expression");
  const rationaleInput = node.querySelector(".discovery-rationale");
  const confInput = node.querySelector(".discovery-confidence");
  const dx = node.querySelector(".direction-x");
  const dy = node.querySelector(".direction-y");
  const dz = node.querySelector(".direction-z");
  const name = node.querySelector(".name");
  const remove = node.querySelector(".remove-discovery");

  if (hasElement(name)) name.textContent = `Discovery ${index + 1}`;
  if (hasElement(idInput)) idInput.value = data.discovery_id || `d${index + 1}`;
  if (hasElement(titleInput)) titleInput.value = data.title || "";
  if (hasElement(exprInput)) exprInput.value = data.expression || "";
  if (hasElement(rationaleInput)) rationaleInput.value = data.rationale || "";
  if (hasElement(confInput)) confInput.value = data.confidence ?? 0.5;
  if (hasElement(dx)) dx.value = data.direction?.x ?? 1;
  if (hasElement(dy)) dy.value = data.direction?.y ?? 0;
  if (hasElement(dz)) dz.value = data.direction?.z ?? 0;

  if (hasElement(remove)) {
    remove.addEventListener("click", () => {
      node.remove();
      reindexDiscoveries();
      persistUIState();
    });
  }

  const inputs = node.querySelectorAll("input, textarea");
  inputs.forEach((input) => {
    input.addEventListener("input", () => persistUIState());
  });
}

function reindexDiscoveries() {
  if (!hasElement(els.discoveriesList)) return;
  const rows = Array.from(els.discoveriesList.querySelectorAll(".discovery-item"));
  rows.forEach((node, idx) => {
    const name = node.querySelector(".name");
    if (hasElement(name)) name.textContent = `Discovery ${idx + 1}`;
    const idInput = node.querySelector(".discovery-id");
    if (hasElement(idInput) && !(idInput.value || "").trim()) {
      idInput.value = `d${idx + 1}`;
    }
  });
}

function addDiscoveryRow(data = {}) {
  if (!hasElement(els.discoveriesList)) return;
  const node = createDiscoveryNode();
  const idx = els.discoveriesList.querySelectorAll(".discovery-item").length;
  hydrateDiscoveryNode(node, idx, data);
  els.discoveriesList.appendChild(node);
}

function collectDiscoveries() {
  if (!hasElement(els.discoveriesList)) return [];
  const rows = Array.from(els.discoveriesList.querySelectorAll(".discovery-item"));
  return rows
    .map((node, idx) => {
      const read = (selector) => {
        const el = node.querySelector(selector);
        return hasElement(el) ? el.value : "";
      };
      const expression = (read(".discovery-expression") || "").trim();
      return {
        discovery_id: (read(".discovery-id") || `d${idx + 1}`).trim(),
        title: (read(".discovery-title") || "").trim(),
        expression,
        rationale: (read(".discovery-rationale") || "").trim(),
        confidence: Number(read(".discovery-confidence") || 0.5),
        direction: {
          x: Number(read(".direction-x") || 0),
          y: Number(read(".direction-y") || 0),
          z: Number(read(".direction-z") || 0),
        },
      };
    })
    .filter((item) => item.expression.length > 0);
}

function serializeCollisionDraft() {
  return {
    discoveries: collectDiscoveries(),
    known_theories: hasElement(els.knownTheories) ? els.knownTheories.value : "",
    memory_expressions: hasElement(els.memoryExpressions) ? els.memoryExpressions.value : "",
    max_collisions: hasElement(els.maxCollisions) ? els.maxCollisions.value : "4",
    last_output: hasElement(els.collisionOutput) ? els.collisionOutput.textContent : "",
  };
}

function applyCollisionDraft(draft) {
  if (!draft || typeof draft !== "object") return;

  if (hasElement(els.discoveriesList)) {
    els.discoveriesList.innerHTML = "";
    const discoveries = Array.isArray(draft.discoveries) ? draft.discoveries : [];
    if (discoveries.length) {
      discoveries.forEach((item) => addDiscoveryRow(item));
    } else {
      addDiscoveryRow({ discovery_id: "d1", direction: { x: 1, y: 0, z: 0 } });
      addDiscoveryRow({ discovery_id: "d2", direction: { x: -1, y: 0.4, z: 0.2 } });
    }
  }
  if (hasElement(els.knownTheories)) els.knownTheories.value = draft.known_theories || "";
  if (hasElement(els.memoryExpressions)) els.memoryExpressions.value = draft.memory_expressions || "";
  if (hasElement(els.maxCollisions)) els.maxCollisions.value = draft.max_collisions || "4";
  if (hasElement(els.collisionOutput) && draft.last_output) els.collisionOutput.textContent = draft.last_output;
}

async function handleCollisionSubmit(event) {
  event.preventDefault();
  if (!hasElement(els.collisionOutput)) return;

  const discoveries = collectDiscoveries();
  if (discoveries.length < 2) {
    setText(els.collisionOutput, "Please provide at least 2 discoveries with valid expressions.");
    return;
  }

  const payload = {
    ...buildRuntimePayloadFromRunForm(),
    discoveries,
    known_theories: parseLines(hasElement(els.knownTheories) ? els.knownTheories.value : ""),
    memory_expressions: parseLines(hasElement(els.memoryExpressions) ? els.memoryExpressions.value : ""),
    max_collisions: Number(hasElement(els.maxCollisions) ? els.maxCollisions.value : 4) || 4,
  };

  setText(els.collisionOutput, "Running collision synthesis...");
  try {
    const result = await api("/api/discovery/collide", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setText(els.collisionOutput, formatJSON(result));
  } catch (err) {
    setText(els.collisionOutput, `Collision failed: ${err.message}`);
  }
  persistUIState();
}

async function loadKnowledge() {
  if (
    !hasElement(els.readmeContent) ||
    !hasElement(els.quickConfig) ||
    !hasElement(els.fullConfig) ||
    !hasElement(els.openworldConfig)
  ) {
    return;
  }
  try {
    const [readme, configs] = await Promise.all([api("/api/readme"), api("/api/configs")]);
    setText(els.readmeContent, readme.content || "");
    setText(els.quickConfig, configs.quickstart || "");
    setText(els.fullConfig, configs.full_experiment || "");
    setText(els.openworldConfig, configs.openworld_pathfinder || "");
  } catch (err) {
    setText(els.readmeContent, `Failed to load docs: ${err.message}`);
  }
}

async function handleRunSubmit(event) {
  event.preventDefault();
  if (!hasElement(els.runForm)) return;

  const formData = new FormData(els.runForm);
  const payload = normalizePayload(formData);

  try {
    const job = await api("/api/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    currentJobId = job.job_id;
    persistUIState();
    switchTab("stream");
    await loadJobs();
    if (!jobsPollHandle) jobsPollHandle = setInterval(loadJobs, 3000);
  } catch (err) {
    alert(`Failed to start run: ${err.message}`);
  }
}

function handleProviderChanged() {
  const providerSelect = inputByName("model_provider");
  if (!providerSelect) return;
  autoFillBaseUrl(providerSelect.value, true);
  loadProviderModels(true);
  if (!runFormRestoring) persistUIState();
}

function bindRunFormPersistence() {
  if (!hasElement(els.runForm)) return;
  const modelFetchTriggerFields = new Set([
    "openai_base_url",
    "openai_api_key",
    "anthropic_base_url",
    "anthropic_api_key",
    "google_base_url",
    "google_api_key",
    "model_access_mode",
  ]);
  const controls = els.runForm.querySelectorAll("input, select, textarea");
  controls.forEach((ctrl) => {
    const evt = ctrl.type === "checkbox" || ctrl.tagName === "SELECT" ? "change" : "input";
    ctrl.addEventListener(evt, () => {
      if (ctrl.name === "model_provider") return;
      if (modelFetchTriggerFields.has(ctrl.name)) {
        scheduleProviderModelFetch(450);
      }
      if (!runFormRestoring) persistUIState();
    });
  });
}

async function refreshAll() {
  await Promise.all([loadHealth(), loadRuns(), loadJobs(), loadKnowledge()]);
}

function bindEvents() {
  if (hasElement(els.runForm)) els.runForm.addEventListener("submit", handleRunSubmit);
  if (hasElement(els.stopCurrentJobBtn)) els.stopCurrentJobBtn.addEventListener("click", handleStopCurrentJob);
  if (hasElement(els.refreshAll)) els.refreshAll.addEventListener("click", refreshAll);
  if (hasElement(els.reloadRuns)) els.reloadRuns.addEventListener("click", loadRuns);
  if (hasElement(els.explainRunBtn)) els.explainRunBtn.addEventListener("click", handleExplainRun);

  if (hasElement(els.tabRunsBtn)) els.tabRunsBtn.addEventListener("click", () => switchTab("runs"));
  if (hasElement(els.tabStreamBtn)) els.tabStreamBtn.addEventListener("click", () => switchTab("stream"));
  if (hasElement(els.tabCollisionBtn)) els.tabCollisionBtn.addEventListener("click", () => switchTab("collision"));

  if (hasElement(els.collisionAddDiscovery)) {
    els.collisionAddDiscovery.addEventListener("click", () => {
      addDiscoveryRow({
        discovery_id: `d${(els.discoveriesList?.querySelectorAll(".discovery-item").length || 0) + 1}`,
        direction: { x: 1, y: 0, z: 0 },
      });
      persistUIState();
    });
  }
  if (hasElement(els.collisionForm)) {
    els.collisionForm.addEventListener("submit", handleCollisionSubmit);
    els.collisionForm.addEventListener("input", persistUIState);
  }

  const providerSelect = inputByName("model_provider");
  if (providerSelect) providerSelect.addEventListener("change", handleProviderChanged);

  bindRunFormPersistence();
}

function ensureInitialDiscoveries() {
  if (!hasElement(els.discoveriesList)) return;
  const existing = els.discoveriesList.querySelectorAll(".discovery-item").length;
  if (existing >= 2) return;
  addDiscoveryRow({ discovery_id: "d1", title: "Trajectory A Candidate", direction: { x: 1, y: 0, z: 0 } });
  addDiscoveryRow({ discovery_id: "d2", title: "Trajectory B Candidate", direction: { x: -1, y: 0.4, z: 0.2 } });
}

async function init() {
  restoreUIState();
  bindEvents();

  const providerSelect = inputByName("model_provider");
  if (providerSelect) {
    if (!providerSelect.value) providerSelect.value = "openai_compatible";
    autoFillBaseUrl(providerSelect.value, false);
    updateProviderHint(providerSelect.value);
  }
  await loadProviderModels(true);

  ensureInitialDiscoveries();
  switchTab(activeTab);
  await refreshAll();

  jobsPollHandle = setInterval(loadJobs, 3000);
}

init();
