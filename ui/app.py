"""
PathSnap Evaluator User Interface (Laptop 2).
Communicates with Laptop 1's FastAPI Backend (/trips, /predict/{trip_id}),
displays the 10 evaluation trip options, sends the selected trip ID to the API,
and renders the reconstructed route map and evaluation results.
"""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

DEFAULT_API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")

ui_app = FastAPI(title="PathSnap Evaluator Interface (Laptop 2)")

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PathSnap — Distributed Map-Matching & Route Reconstruction UI</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --success: #10b981;
      --text: #f8fafc;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #0b1120;
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      border: 1px solid #3b82f6;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    input, select, button {
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 14px;
    }
    input { width: 240px; }
    select { min-width: 210px; cursor: pointer; }
    button {
      background: var(--accent);
      border-color: var(--accent);
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    button:hover { background: var(--accent-hover); }
    button.secondary {
      background: #334155;
      border-color: #475569;
    }
    main {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
      padding: 16px 24px;
      flex: 1;
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }
    .card h3 {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 12px;
    }
    .trip-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
    }
    .trip-btn {
      background: #0f172a;
      border: 1px solid var(--border);
      padding: 9px 10px;
      border-radius: 6px;
      text-align: left;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .trip-btn:hover, .trip-btn.active {
      border-color: var(--accent);
      background: rgba(59, 130, 246, 0.18);
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .metric-box {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .metric-label {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .metric-val {
      font-size: 18px;
      font-weight: 700;
      color: #38bdf8;
    }
    .metric-val.highlight { color: var(--success); }
    .map-container {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 640px;
    }
    .map-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
    }
    iframe {
      width: 100%;
      height: 680px;
      flex: 1;
      border: none;
      background: #fff;
    }
    .route-seq {
      font-family: monospace;
      font-size: 12px;
      color: #cbd5e1;
      max-height: 110px;
      overflow-y: auto;
      background: #0f172a;
      padding: 8px;
      border-radius: 6px;
      border: 1px solid var(--border);
      word-break: break-all;
    }
    #statusBanner {
      font-size: 13px;
      color: #fbbf24;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h2>PathSnap</h2>
      <span class="badge">Laptop 2 — Evaluator Interface</span>
      <span id="statusBanner">Connecting to API...</span>
    </div>
    <div class="controls">
      <label style="font-size:13px;color:var(--muted);">Laptop 1 API URL:</label>
      <input type="text" id="apiUrlInput" value="__DEFAULT_API_URL__" />
      <button class="secondary" onclick="loadTrips()">Connect & Refresh Trips</button>
      <select id="tripSelect"></select>
      <button onclick="runSelectedTrip()">Run Route Reconstruction</button>
    </div>
  </header>

  <main>
    <div class="sidebar">
      <div class="card">
        <h3>10 Reserved Evaluation Trips</h3>
        <div id="tripButtons" class="trip-grid"></div>
      </div>

      <div class="card">
        <h3>Server-Side Evaluation Metrics</h3>
        <div class="metric-grid">
          <div class="metric-box">
            <div class="metric-label">HMM+Viterbi Top-1 Acc</div>
            <div class="metric-val highlight" id="m_hmm_top1">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Classifier-Only Top-1</div>
            <div class="metric-val" id="m_clf_top1">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Route Match Accuracy</div>
            <div class="metric-val highlight" id="m_route_match">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Route Continuity</div>
            <div class="metric-val highlight" id="m_continuity">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Candidate Macro F1</div>
            <div class="metric-val" id="m_macro_f1">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Sequence Route F1</div>
            <div class="metric-val" id="m_seq_f1">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Log Loss</div>
            <div class="metric-val" id="m_logloss">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Pipeline Latency</div>
            <div class="metric-val" id="m_latency">—</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h3>Reconstructed Route Sequence (.arcs)</h3>
        <div style="font-size:12px;color:var(--muted);margin-bottom:6px;" id="routeMeta">Select a trip to reconstruct its route.</div>
        <div class="route-seq" id="routeSeqBox">No route loaded yet.</div>
      </div>
    </div>

    <div class="map-container">
      <div class="map-header">
        <span id="mapTitle"><b>Interactive Folium Map</b> — Original GPS (.track), Road Network (.arcs), and Predicted Route</span>
        <span id="modelBadge" style="color:var(--muted);font-size:12px;">Model: Tuned LightGBM + HMM/Viterbi</span>
      </div>
      <iframe id="mapFrame"></iframe>
    </div>
  </main>

  <script>
    function getApiBase() {
      return document.getElementById('apiUrlInput').value.replace(/\/+$/, '');
    }

    async function loadTrips() {
      const base = getApiBase();
      const status = document.getElementById('statusBanner');
      status.textContent = 'Fetching trips from ' + base + '/trips ...';
      status.style.color = '#fbbf24';
      try {
        const res = await fetch(base + '/trips');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const sel = document.getElementById('tripSelect');
        const grid = document.getElementById('tripButtons');
        sel.innerHTML = '';
        grid.innerHTML = '';

        data.trips.forEach((t, idx) => {
          const opt = document.createElement('option');
          opt.value = t.trip_id;
          opt.textContent = `Trip ${t.trip_id} (${t.n_gps_observations} GPS pts)`;
          sel.appendChild(opt);

          const btn = document.createElement('button');
          btn.className = 'trip-btn';
          btn.id = 'btn_' + t.trip_id;
          btn.innerHTML = `<b>${t.trip_id}</b><br><span style="color:#94a3b8;font-size:11px;">${t.n_gps_observations} GPS pts</span>`;
          btn.onclick = () => {
            sel.value = t.trip_id;
            runSelectedTrip();
          };
          grid.appendChild(btn);
        });

        status.textContent = `Connected to API (${data.count} evaluation trips loaded)`;
        status.style.color = '#10b981';
        if (data.trips.length > 0) {
          runSelectedTrip();
        }
      } catch (err) {
        status.textContent = 'API Connection Error: ' + err.message;
        status.style.color = '#ef4444';
      }
    }

    async function runSelectedTrip() {
      const base = getApiBase();
      const tripId = document.getElementById('tripSelect').value;
      if (!tripId) return;

      document.querySelectorAll('.trip-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('btn_' + tripId);
      if (activeBtn) activeBtn.classList.add('active');

      const status = document.getElementById('statusBanner');
      status.textContent = `Running prediction pipeline on Trip ${tripId}...`;
      status.style.color = '#fbbf24';

      try {
        const res = await fetch(base + '/predict/' + encodeURIComponent(tripId));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const m = data.evaluation_metrics || {};

        const pct = v => (v !== undefined && v !== null) ? (v * 100).toFixed(2) + '%' : 'N/A';
        document.getElementById('m_hmm_top1').textContent = pct(m.hmm_viterbi_top1_accuracy);
        document.getElementById('m_clf_top1').textContent = pct(m.classifier_top1_accuracy);
        document.getElementById('m_route_match').textContent = pct(m.hmm_viterbi_route_match_accuracy);
        document.getElementById('m_continuity').textContent = pct(m.hmm_route_continuity);
        document.getElementById('m_macro_f1').textContent = m.candidate_macro_f1 !== undefined ? m.candidate_macro_f1.toFixed(4) : 'N/A';
        document.getElementById('m_seq_f1').textContent = m.sequence_f1 !== undefined ? m.sequence_f1.toFixed(4) : 'N/A';
        document.getElementById('m_logloss').textContent = m.candidate_log_loss !== undefined ? m.candidate_log_loss.toFixed(4) : 'N/A';
        document.getElementById('m_latency').textContent = (data.timing ? data.timing.total_pipeline_s : 0) + 's';

        document.getElementById('routeMeta').textContent =
          `Trip ${data.trip_id}: ${data.n_gps_observations} GPS observations -> ${data.n_reconstructed_route_segments} connected road segments`;
        document.getElementById('routeSeqBox').textContent =
          data.reconstructed_route_arcs.join(' -> ');

        document.getElementById('mapTitle').innerHTML =
          `<b>Trip ${data.trip_id}</b> — ${data.n_gps_observations} GPS Observations | Road Network: ${data.n_road_nodes} nodes, ${data.n_road_segments} arcs`;
        document.getElementById('modelBadge').textContent = data.model_used;

        const iframe = document.getElementById('mapFrame');
        iframe.srcdoc = data.folium_map_html;

        status.textContent = `Completed Trip ${tripId} in ${data.timing.total_pipeline_s}s`;
        status.style.color = '#10b981';
      } catch (err) {
        status.textContent = 'Inference Error: ' + err.message;
        status.style.color = '#ef4444';
      }
    }

    window.addEventListener('DOMContentLoaded', loadTrips);
  </script>
</body>
</html>
"""


@ui_app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML.replace("__DEFAULT_API_URL__", DEFAULT_API_URL))


if __name__ == "__main__":
    import uvicorn
    ui_port = int(os.environ.get("UI_PORT", 8501))
    uvicorn.run("ui.app:ui_app", host="0.0.0.0", port=ui_port, reload=False)
