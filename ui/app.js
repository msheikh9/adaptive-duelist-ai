/* Adaptive Duelist AI — Dashboard JS */

const API = "";  // same origin

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

function showResult(elId, data, isError = false) {
  const el = document.getElementById(elId);
  el.classList.remove("hidden");
  el.style.color = isError ? "#fca5a5" : "#94a3b8";
  el.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = loading;
  btn.innerHTML = loading
    ? '<span class="spinner"></span>Running…'
    : btn.dataset.label || btn.textContent;
  if (!btn.dataset.label) btn.dataset.label = btn.textContent;
}

function winnerBadge(winner) {
  const cls = winner === "AI" ? "win" : winner === "PLAYER" ? "lose" : "draw";
  return `<span class="badge ${cls}">${winner}</span>`;
}

// ---------------------------------------------------------------------------
// Health + Stats
// ---------------------------------------------------------------------------

async function loadHealth() {
  const badge = document.getElementById("status-badge");
  try {
    const data = await apiFetch("/api/health");
    badge.textContent = `v${data.version} — ok`;
    badge.className = "ok";
  } catch (e) {
    badge.textContent = "offline";
    badge.className = "error";
  }
}

async function loadStats() {
  try {
    const data = await apiFetch("/api/stats");
    document.getElementById("stat-total-matches").textContent = data.total_matches;
    document.getElementById("stat-human").textContent = data.human_matches;
    document.getElementById("stat-selfplay").textContent = data.self_play_matches;
    document.getElementById("stat-model").textContent = data.active_model_version || "none";
  } catch (e) {
    console.error("stats error", e);
  }
}

// ---------------------------------------------------------------------------
// Recent Matches
// ---------------------------------------------------------------------------

async function loadRecent() {
  try {
    const data = await apiFetch("/api/matches/recent?limit=15");
    const tbody = document.getElementById("recent-tbody");
    if (!data.matches.length) {
      tbody.innerHTML = '<tr><td colspan="6" style="color:#4b5563">No matches yet.</td></tr>';
      return;
    }
    tbody.innerHTML = data.matches.map(m => `
      <tr>
        <td style="font-family:monospace;font-size:0.75rem">${m.match_id}</td>
        <td>${winnerBadge(m.winner)}</td>
        <td>${m.player_hp_final ?? m.player_hp ?? "—"}</td>
        <td>${m.ai_hp_final ?? m.ai_hp ?? "—"}</td>
        <td>${m.total_ticks ?? m.ticks ?? "—"}</td>
        <td style="font-size:0.75rem;color:#475569">${(m.started_at || m.created_at || "—").slice(0,19)}</td>
      </tr>`).join("");
  } catch (e) {
    document.getElementById("recent-tbody").innerHTML =
      `<tr><td colspan="6" style="color:#fca5a5">${e.message}</td></tr>`;
  }
}

// ---------------------------------------------------------------------------
// Self-Play
// ---------------------------------------------------------------------------

async function runSelfPlay() {
  const body = {
    n_matches: parseInt(document.getElementById("sp-matches").value),
    seed: parseInt(document.getElementById("sp-seed").value),
    max_ticks: parseInt(document.getElementById("sp-max-ticks").value),
  };
  setLoading("sp-run-btn", true);
  try {
    const data = await apiFetch("/api/matches/self-play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    showResult("sp-result", data);
    loadStats();
    loadRecent();
  } catch (e) {
    showResult("sp-result", e.message, true);
  } finally {
    setLoading("sp-run-btn", false);
  }
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------

async function runEvaluate() {
  const body = {
    n_matches: parseInt(document.getElementById("ev-matches").value),
    tier: document.getElementById("ev-tier").value,
    seed: parseInt(document.getElementById("ev-seed").value),
  };
  setLoading("ev-run-btn", true);
  try {
    const data = await apiFetch("/api/matches/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    showResult("ev-result", data);
  } catch (e) {
    showResult("ev-result", e.message, true);
  } finally {
    setLoading("ev-run-btn", false);
  }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

async function loadTrainingStatus() {
  try {
    const data = await apiFetch("/api/training/status");
    document.getElementById("ts-delta").textContent = data.delta;
    document.getElementById("ts-threshold").textContent = data.retrain_threshold;
    document.getElementById("ts-needed").textContent = data.retrain_needed ? "Yes" : "No";
  } catch (e) {
    console.error("training status error", e);
  }
}

async function runTraining(autoPromote) {
  const btnId = autoPromote ? "train-promote-btn" : "train-run-btn";
  setLoading(btnId, true);
  try {
    const data = await apiFetch("/api/training/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auto_promote: autoPromote }),
    });
    showResult("train-result", data);
    loadStats();
    loadTrainingStatus();
    loadModels();
  } catch (e) {
    showResult("train-result", e.message, true);
  } finally {
    setLoading(btnId, false);
  }
}

async function runCurriculum() {
  setLoading("curriculum-btn", true);
  try {
    const data = await apiFetch("/api/training/curriculum", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_matches: 20, auto_promote: false }),
    });
    showResult("train-result", data);
    loadStats();
    loadTrainingStatus();
  } catch (e) {
    showResult("train-result", e.message, true);
  } finally {
    setLoading("curriculum-btn", false);
  }
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

async function loadModels() {
  try {
    const data = await apiFetch("/api/models/status");
    const el = document.getElementById("model-list");
    if (!data.all_versions.length) {
      el.textContent = "No models registered.";
      return;
    }
    el.innerHTML = `<table>
      <thead><tr><th>Version</th><th>Accuracy</th><th>Active</th><th>Created</th></tr></thead>
      <tbody>
        ${data.all_versions.map(m => `
          <tr>
            <td style="font-family:monospace">${m.version}</td>
            <td>${m.eval_accuracy != null ? (m.eval_accuracy * 100).toFixed(1) + "%" : "—"}</td>
            <td>${m.is_active ? '<span class="badge pass">active</span>' : ""}</td>
            <td style="font-size:0.75rem;color:#475569">${m.created_at ? m.created_at.slice(0,19) : "—"}</td>
          </tr>`).join("")}
      </tbody>
    </table>`;
  } catch (e) {
    document.getElementById("model-list").textContent = `Error: ${e.message}`;
  }
}

// ---------------------------------------------------------------------------
// Baseline & Regression
// ---------------------------------------------------------------------------

async function createBaseline() {
  const body = {
    n_matches: parseInt(document.getElementById("bl-matches").value),
    tag: document.getElementById("bl-tag").value,
    tier: document.getElementById("bl-tier").value,
    seed: 0,
  };
  try {
    const data = await apiFetch("/api/models/baseline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    showResult("bl-result", data);
  } catch (e) {
    showResult("bl-result", e.message, true);
  }
}

async function checkRegression() {
  const body = {
    n_matches: parseInt(document.getElementById("bl-matches").value),
    tier: document.getElementById("bl-tier").value,
    baseline_tag: document.getElementById("bl-tag").value,
    seed: 0,
  };
  try {
    const data = await apiFetch("/api/models/check-regression", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    showResult("bl-result", data);
  } catch (e) {
    showResult("bl-result", e.message, true);
  }
}

// ---------------------------------------------------------------------------
// Match Report
// ---------------------------------------------------------------------------

async function fetchReport() {
  const matchId = document.getElementById("report-match-id").value.trim();
  if (!matchId) return;
  try {
    const data = await apiFetch(`/api/matches/${encodeURIComponent(matchId)}/report`);
    showResult("report-result", data);
  } catch (e) {
    showResult("report-result", e.message, true);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

(async () => {
  await Promise.all([
    loadHealth(),
    loadStats(),
    loadRecent(),
    loadTrainingStatus(),
    loadModels(),
  ]);
})();
