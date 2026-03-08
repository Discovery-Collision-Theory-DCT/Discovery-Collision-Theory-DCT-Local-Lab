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

const els = {
  healthBadge: document.getElementById("health-badge"),
  refreshAll: document.getElementById("refresh-all"),
  runForm: document.getElementById("run-form"),

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

  tabRunsBtn: document.getElementById("tab-runs-btn"),
  tabStreamBtn: document.getElementById("tab-stream-btn"),
  tabRuns: document.getElementById("tab-runs"),
  tabStream: document.getElementById("tab-stream"),

  streamOutputHead: document.getElementById("stream-output-head"),
  streamOutput: document.getElementById("stream-output"),

  quickConfig: document.getElementById("quick-config"),
  fullConfig: document.getElementById("full-config"),
  readmeContent: document.getElementById("readme-content"),
};

let currentRunName = null;
let currentJobId = null;
let jobsPollHandle = null;
let activeTab = "runs";

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

function switchTab(name) {
  activeTab = name;

  if (hasElement(els.tabRunsBtn)) els.tabRunsBtn.classList.toggle("active", name === "runs");
  if (hasElement(els.tabStreamBtn)) els.tabStreamBtn.classList.toggle("active", name === "stream");
  if (hasElement(els.tabRuns)) els.tabRuns.classList.toggle("active", name === "runs");
  if (hasElement(els.tabStream)) els.tabStream.classList.toggle("active", name === "stream");
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

  if (!jobs.length) {
    els.jobsList.innerHTML = '<p class="sub">No run jobs yet.</p>';
    renderStreamOutput(null);
    return;
  }

  for (const job of jobs) {
    let node = null;
    if (hasElement(els.jobTemplate) && els.jobTemplate.content && els.jobTemplate.content.firstElementChild) {
      node = els.jobTemplate.content.firstElementChild.cloneNode(true);
    } else {
      node = document.createElement("article");
      node.className = "item";
      node.innerHTML = '<div class="item-top"><strong class="name"></strong><span class="status"></span></div><p class="meta"></p>';
    }

    setText(node.querySelector(".name"), job.job_id);

    const statusEl = node.querySelector(".status") || node.querySelector(".status-chip");
    if (hasElement(statusEl)) {
      statusEl.textContent = job.status;
      statusEl.classList.add(job.status);
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
      renderJobs(jobs);
      renderStreamOutput(job);
      switchTab("stream");
      if (job.run_name) {
        currentRunName = job.run_name;
        loadRunDetail(job.run_name);
      }
    });

    els.jobsList.appendChild(node);
  }
}

async function loadJobs() {
  try {
    const jobs = await api("/api/jobs");
    renderJobs(jobs);

    if (!currentJobId && jobs.length) currentJobId = jobs[0].job_id;

    const selected = jobs.find((j) => j.job_id === currentJobId) || jobs[0] || null;
    renderStreamOutput(selected);

    if (selected?.run_name && selected.status === "completed" && selected.run_name !== currentRunName) {
      currentRunName = selected.run_name;
      await loadRunDetail(selected.run_name);
      await loadRuns();
    }
  } catch (err) {
    if (hasElement(els.jobsList)) {
      els.jobsList.innerHTML = `<p class="sub">Failed to load jobs: ${err.message}</p>`;
    }
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

async function loadKnowledge() {
  if (!hasElement(els.readmeContent) || !hasElement(els.quickConfig) || !hasElement(els.fullConfig)) return;
  try {
    const [readme, configs] = await Promise.all([api("/api/readme"), api("/api/configs")]);
    setText(els.readmeContent, readme.content || "");
    setText(els.quickConfig, configs.quickstart || "");
    setText(els.fullConfig, configs.full_experiment || "");
  } catch (err) {
    setText(els.readmeContent, `Failed to load docs: ${err.message}`);
  }
}

async function handleRunSubmit(event) {
  event.preventDefault();
  const formData = new FormData(els.runForm);
  const payload = normalizePayload(formData);

  try {
    const job = await api("/api/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    currentJobId = job.job_id;
    switchTab("stream");
    await loadJobs();
    if (!jobsPollHandle) jobsPollHandle = setInterval(loadJobs, 3000);
  } catch (err) {
    alert(`Failed to start run: ${err.message}`);
  }
}

async function refreshAll() {
  await Promise.all([loadHealth(), loadRuns(), loadJobs(), loadKnowledge()]);
}

function bindEvents() {
  if (hasElement(els.runForm)) els.runForm.addEventListener("submit", handleRunSubmit);
  if (hasElement(els.refreshAll)) els.refreshAll.addEventListener("click", refreshAll);
  if (hasElement(els.reloadRuns)) els.reloadRuns.addEventListener("click", loadRuns);

  if (hasElement(els.tabRunsBtn)) els.tabRunsBtn.addEventListener("click", () => switchTab("runs"));
  if (hasElement(els.tabStreamBtn)) els.tabStreamBtn.addEventListener("click", () => switchTab("stream"));
}

async function init() {
  bindEvents();
  switchTab(activeTab);
  await refreshAll();
  jobsPollHandle = setInterval(loadJobs, 3000);
}

init();
