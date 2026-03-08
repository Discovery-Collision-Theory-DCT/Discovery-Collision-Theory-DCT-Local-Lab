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
  jobOutputHead: document.getElementById("job-output-head"),
  jobOutput: document.getElementById("job-output"),
  runsList: document.getElementById("runs-list"),
  reloadRuns: document.getElementById("reload-runs"),
  runDetail: document.getElementById("run-detail"),
  runTitle: document.getElementById("run-title"),
  summaryCards: document.getElementById("summary-cards"),
  upliftTable: document.getElementById("uplift-table"),
  methodsTable: document.getElementById("methods-table"),
  plotsGrid: document.getElementById("plots-grid"),
  artifactsList: document.getElementById("artifacts-list"),
  quickConfig: document.getElementById("quick-config"),
  fullConfig: document.getElementById("full-config"),
  readmeContent: document.getElementById("readme-content"),
  jobTemplate: document.getElementById("job-item-template"),
  runTemplate: document.getElementById("run-item-template"),
};

let currentRunName = null;
let currentJobId = null;
let jobsPollHandle = null;

function toNumber(value) {
  if (typeof value !== "number") return null;
  if (!Number.isFinite(value)) return null;
  return value;
}

function formatNumber(value) {
  if (value == null) return "-";
  return Number(value).toFixed(4);
}

function normalizePayload(formData) {
  const payload = {};
  for (const [key, value] of formData.entries()) {
    if (value === "") continue;
    payload[key] = value;
  }

  payload.allow_remote_inference = formData.get("allow_remote_inference") === "on";
  payload.skip_model_check = formData.get("skip_model_check") === "on";

  const tempRaw = formData.get("model_temperature");
  if (typeof tempRaw === "string" && tempRaw.trim() !== "") {
    const parsed = parseFloat(tempRaw);
    if (!Number.isNaN(parsed)) payload.model_temperature = parsed;
  }

  if (!payload.config_path) delete payload.config_path;
  if (!payload.output_dir) delete payload.output_dir;
  if (!payload.model_name) delete payload.model_name;

  return payload;
}

function setHealth(statusText, ok = false) {
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

function renderJobOutput(job) {
  if (!job) {
    els.jobOutputHead.textContent = "Select a job to view live logs.";
    els.jobOutput.textContent = "";
    return;
  }

  const req = job.request || {};
  const headBits = [
    `job=${job.job_id}`,
    `status=${job.status}`,
    `provider=${req.model_provider || "(env/default)"}`,
    `model=${req.model_name || "(env/default)"}`,
    `temp=${req.model_temperature ?? "(env/default)"}`,
  ];
  if (job.run_name) headBits.push(`run=${job.run_name}`);
  els.jobOutputHead.textContent = headBits.join(" | ");

  const logs = Array.isArray(job.logs) ? job.logs : [];
  if (!logs.length) {
    els.jobOutput.textContent = "No logs yet...";
    return;
  }

  const lines = logs.map((log) => {
    const ts = log.time ? new Date(log.time).toLocaleTimeString() : "--:--:--";
    const lvl = (log.level || "info").toUpperCase();
    return `[${ts}] ${lvl} ${log.message || ""}`;
  });

  if (job.error) {
    lines.push(`\n[ERROR] ${job.error}`);
  }

  els.jobOutput.textContent = lines.join("\n");
  els.jobOutput.scrollTop = els.jobOutput.scrollHeight;
}

function renderJobs(jobs) {
  els.jobsList.innerHTML = "";
  if (!jobs.length) {
    els.jobsList.innerHTML = '<p class="sub">No run jobs yet.</p>';
    renderJobOutput(null);
    return;
  }

  for (const job of jobs) {
    const node = els.jobTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".name").textContent = job.job_id;

    const statusEl = node.querySelector(".status-chip");
    statusEl.textContent = job.status;
    statusEl.classList.add(job.status);

    const meta = [];
    if (job.run_name) meta.push(`run=${job.run_name}`);
    if (job.error) meta.push(`error=${job.error}`);
    const lastMessage = Array.isArray(job.logs) && job.logs.length ? job.logs[job.logs.length - 1].message : null;
    if (lastMessage) meta.push(`last=${lastMessage}`);
    meta.push(`updated=${new Date(job.updated_at).toLocaleString()}`);
    node.querySelector(".meta").textContent = meta.join(" | ");

    if (job.job_id === currentJobId) {
      node.classList.add("selected");
    }

    node.addEventListener("click", () => {
      currentJobId = job.job_id;
      renderJobs(jobs);
      renderJobOutput(job);
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

    if (!currentJobId && jobs.length) {
      currentJobId = jobs[0].job_id;
    }

    const selected = jobs.find((j) => j.job_id === currentJobId) || jobs[0] || null;
    renderJobOutput(selected);

    if (selected?.run_name && selected.status === "completed" && selected.run_name !== currentRunName) {
      currentRunName = selected.run_name;
      await loadRunDetail(selected.run_name);
      await loadRuns();
    }
  } catch (err) {
    els.jobsList.innerHTML = `<p class="sub">Failed to load jobs: ${err.message}</p>`;
  }
}

function renderRuns(runs) {
  els.runsList.innerHTML = "";
  if (!runs.length) {
    els.runsList.innerHTML = '<p class="sub">No runs found in output root yet.</p>';
    return;
  }

  for (const run of runs) {
    const node = els.runTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".name").textContent = run.run_name;
    node.querySelector(".meta-time").textContent = new Date(run.updated_at).toLocaleString();
    node.querySelector(".meta").textContent = run.run_dir;

    node.addEventListener("click", () => {
      currentRunName = run.run_name;
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
    els.runsList.innerHTML = `<p class="sub">Failed to load runs: ${err.message}</p>`;
  }
}

function metricCard(label, value) {
  const card = document.createElement("article");
  card.className = "metric-card";
  card.innerHTML = `<div class="k">${label}</div><div class="v">${value}</div>`;
  return card;
}

function renderUpliftTable(uplift) {
  const baselines = Object.keys(uplift || {});
  if (!baselines.length) {
    els.upliftTable.innerHTML = '<p class="sub">No uplift data.</p>';
    return;
  }

  const metricSet = new Set();
  for (const b of baselines) {
    Object.keys(uplift[b] || {}).forEach((m) => metricSet.add(m));
  }
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
  if (!methodSummaries?.length) {
    els.methodsTable.innerHTML = '<p class="sub">No method summary data.</p>';
    return;
  }

  const rows = methodSummaries
    .map((m) => {
      return `
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
      `;
    })
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
  try {
    const [summary, artifacts] = await Promise.all([
      api(`/api/runs/${encodeURIComponent(runName)}`),
      api(`/api/runs/${encodeURIComponent(runName)}/artifacts`),
    ]);

    els.runDetail.classList.remove("hidden");
    els.runTitle.textContent = `Run Detail: ${summary.run_name}`;

    const methodSummaries = summary.method_summaries || [];
    const methods = new Set(methodSummaries.map((m) => m.method));
    const trials = new Set(methodSummaries.map((m) => m.trial_index));

    els.summaryCards.innerHTML = "";
    els.summaryCards.appendChild(metricCard("Methods", methods.size));
    els.summaryCards.appendChild(metricCard("Trials", trials.size));
    els.summaryCards.appendChild(metricCard("Run Name", summary.run_name || "-"));
    els.summaryCards.appendChild(metricCard("Output Dir", artifacts.run_dir || "-"));

    renderUpliftTable(summary.uplift || {});
    renderMethodsTable(methodSummaries);
    renderArtifacts(artifacts);
  } catch (err) {
    els.runDetail.classList.remove("hidden");
    els.runTitle.textContent = `Run Detail Error: ${runName}`;
    els.summaryCards.innerHTML = `<p class="sub">${err.message}</p>`;
    els.upliftTable.innerHTML = "";
    els.methodsTable.innerHTML = "";
    els.plotsGrid.innerHTML = "";
    els.artifactsList.innerHTML = "";
  }
}

async function loadKnowledge() {
  try {
    const [readme, configs] = await Promise.all([api("/api/readme"), api("/api/configs")]);
    els.readmeContent.textContent = readme.content || "";
    els.quickConfig.textContent = configs.quickstart || "";
    els.fullConfig.textContent = configs.full_experiment || "";
  } catch (err) {
    els.readmeContent.textContent = `Failed to load docs: ${err.message}`;
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
    await loadJobs();
    if (!jobsPollHandle) {
      jobsPollHandle = setInterval(loadJobs, 3000);
    }
  } catch (err) {
    alert(`Failed to start run: ${err.message}`);
  }
}

async function refreshAll() {
  await Promise.all([loadHealth(), loadRuns(), loadJobs(), loadKnowledge()]);
}

function bindEvents() {
  els.runForm.addEventListener("submit", handleRunSubmit);
  els.refreshAll.addEventListener("click", refreshAll);
  els.reloadRuns.addEventListener("click", loadRuns);
}

async function init() {
  bindEvents();
  await refreshAll();
  jobsPollHandle = setInterval(loadJobs, 4000);
}

init();
