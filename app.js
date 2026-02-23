(() => {
  "use strict";

  const dom = {
    startBtn: document.getElementById("start-btn"),
    pauseBtn: document.getElementById("pause-btn"),
    resetBtn: document.getElementById("reset-btn"),
    checkBtn: document.getElementById("check-btn"),
    sweepBtn: document.getElementById("sweep-btn"),
    seedInput: document.getElementById("seed-input"),
    intervalInput: document.getElementById("interval-input"),
    payloadInput: document.getElementById("payload-input"),
    siteInput: document.getElementById("site-count-input"),
    speedInput: document.getElementById("speed-input"),
    speedLabel: document.getElementById("speed-label"),
    simState: document.getElementById("sim-state"),
    paramSummary: document.getElementById("param-summary"),
    simTime: document.getElementById("sim-time"),
    currentSiteCount: document.getElementById("current-site-count"),
    currentPayloadCapacity: document.getElementById("current-payload-capacity"),
    insight: document.getElementById("insight-text"),
    mapCanvasAstar: document.getElementById("map-canvas-astar"),
    mapCanvasMappo: document.getElementById("map-canvas-mappo"),
    completedChart: document.getElementById("chart-completed"),
    waitChart: document.getElementById("chart-wait"),
    scatterChart: document.getElementById("chart-scatter"),
    ganttChart: document.getElementById("chart-gantt"),
    astarCard: document.getElementById("card-astar"),
    mappoCard: document.getElementById("card-mappo"),
    astarBatchView: document.getElementById("astar-batch-view"),
    mappoBatchView: document.getElementById("mappo-batch-view"),
    astarDecisionLog: document.getElementById("astar-decision-log"),
    mappoDecisionLog: document.getElementById("mappo-decision-log"),
    astarEventLog: document.getElementById("astar-event-log"),
    mappoEventLog: document.getElementById("mappo-event-log"),
  };

  const ctxMapAstar = dom.mapCanvasAstar.getContext("2d");
  const ctxMapMappo = dom.mapCanvasMappo.getContext("2d");
  const ctxCompleted = dom.completedChart.getContext("2d");
  const ctxWait = dom.waitChart.getContext("2d");
  const ctxScatter = dom.scatterChart.getContext("2d");
  const ctxGantt = dom.ganttChart.getContext("2d");

  const COLORS = {
    bg0: "#091727",
    bg1: "#0e2b44",
    grid: "rgba(160, 208, 246, 0.08)",
    obstacle: "#bc6b45",
    obstacleStroke: "rgba(255, 181, 71, 0.48)",
    site: "#8ad5ff",
    task: "#ffd447",
    drone0: "#36d1ff",
    drone1: "#ff5e5e",
    route: "rgba(140, 241, 255, 0.55)",
    route2: "rgba(255, 124, 124, 0.55)",
    batchRoute0: "rgba(54, 209, 255, 0.9)",
    batchRoute1: "rgba(255, 94, 94, 0.9)",
    phaseIdle: "rgba(140, 160, 178, 0.86)",
    phaseDelivering: "rgba(77, 205, 255, 0.9)",
    phaseReturning: "rgba(255, 161, 77, 0.92)",
    label: "#dff1ff",
  };

  const stateStore = {
    latest: null,
    fetching: false,
    evalRunning: false,
    sweepRunning: false,
    manualInsight: null,
    manualInsightUntil: 0,
  };

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  function fmtSec(v) {
    if (!Number.isFinite(v)) return "-";
    return `${v.toFixed(1)}s`;
  }

  function fmtDistance(v) {
    if (!Number.isFinite(v)) return "-";
    return `${v.toFixed(0)} px`;
  }

  function fmtPercent(v) {
    if (!Number.isFinite(v)) return "-";
    return `${(v * 100).toFixed(1)}%`;
  }

  function fmtNum(v, digits = 2) {
    if (!Number.isFinite(v)) return "-";
    return Number(v).toFixed(digits);
  }

  function readNumberInput(el, fallback) {
    if (!el) return fallback;
    const raw = String(el.value ?? "").trim();
    if (raw === "") return fallback;
    const val = Number(raw);
    return Number.isFinite(val) ? val : fallback;
  }

  function phaseLabel(phase) {
    if (phase === "delivering") return "配送中";
    if (phase === "returning") return "返航中";
    if (phase === "idle") return "待命";
    return phase || "-";
  }

  function setPanelCollapsed(toggleBtn, collapsed) {
    const panel = toggleBtn.closest(".panel");
    if (!panel) return;
    panel.classList.toggle("collapsed", collapsed);
    toggleBtn.textContent = collapsed ? "展开" : "收起";
    toggleBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    try {
      localStorage.setItem(`panel:${toggleBtn.id}`, collapsed ? "1" : "0");
    } catch (_err) {}
  }

  function initPanelToggles() {
    const buttons = document.querySelectorAll(".panel-toggle");
    buttons.forEach((btn) => {
      let collapsed = false;
      try {
        collapsed = localStorage.getItem(`panel:${btn.id}`) === "1";
      } catch (_err) {}
      setPanelCollapsed(btn, collapsed);
      btn.addEventListener("click", () => {
        const panel = btn.closest(".panel");
        const isCollapsed = panel ? panel.classList.contains("collapsed") : false;
        setPanelCollapsed(btn, !isCollapsed);
      });
    });
  }

  async function apiRequest(path, method = "GET", body) {
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const res = await fetch(path, options);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${text}`);
    }
    return res.json();
  }

  function drawBackgroundGrid(ctx, map) {
    const width = map.width;
    const height = map.height;
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, COLORS.bg1);
    gradient.addColorStop(1, COLORS.bg0);

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let x = 0; x <= width; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    }
    for (let y = 0; y <= height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(width, y + 0.5);
      ctx.stroke();
    }
  }

  function drawObstacles(ctx, obstacles) {
    ctx.fillStyle = COLORS.obstacle;
    ctx.strokeStyle = COLORS.obstacleStroke;
    ctx.lineWidth = 2;
    for (const o of obstacles) {
      ctx.fillRect(o.x, o.y, o.w, o.h);
      ctx.strokeRect(o.x, o.y, o.w, o.h);
    }
  }

  function drawSites(ctx, sites) {
    ctx.font = "12px Space Grotesk";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const site of sites) {
      ctx.beginPath();
      ctx.fillStyle = COLORS.site;
      ctx.arc(site.x, site.y, 9, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.strokeStyle = "rgba(138, 213, 255, 0.3)";
      ctx.lineWidth = 6;
      ctx.arc(site.x, site.y, 12, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = COLORS.label;
      ctx.fillText(`${site.id} ${site.name}`, site.x + 14, site.y - 2);
    }
  }

  function drawArrow(ctx, from, to, color, width = 1.5, dashed = false) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const angle = Math.atan2(dy, dx);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    if (dashed) ctx.setLineDash([8, 7]);

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(to.x, to.y);
    ctx.lineTo(to.x - 8 * Math.cos(angle - 0.35), to.y - 8 * Math.sin(angle - 0.35));
    ctx.lineTo(to.x - 8 * Math.cos(angle + 0.35), to.y - 8 * Math.sin(angle + 0.35));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawTasks(ctx, sim, sites) {
    for (const task of sim.open_tasks || []) {
      if (task.status !== "pending") continue;
      const src = sites[task.source_index];
      const dst = sites[task.target_index];
      if (!src || !dst) continue;

      const alpha = clamp(0.25 + task.age_s / 18, 0.25, 0.92);
      drawArrow(
        ctx,
        src,
        dst,
        `rgba(255, 212, 71, ${alpha.toFixed(2)})`,
        task.priority ? 2.6 : 1.7,
        true
      );

      ctx.beginPath();
      ctx.fillStyle = task.priority ? "#ff7f50" : COLORS.task;
      ctx.arc(src.x, src.y, task.priority ? 4.2 : 3.1, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawDronePath(ctx, drone, color) {
    const path = drone.path_remaining || [];
    if (path.length === 0) return;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(drone.x, drone.y);
    for (const p of path) {
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawBatchRouteOverlay(ctx, sim) {
    const decisions = sim.decision_log || [];
    if (decisions.length === 0) return;

    const latestByDrone = new Map();
    for (const row of decisions) {
      if (row && Number.isFinite(Number(row.drone_id))) {
        latestByDrone.set(Number(row.drone_id), row);
      }
    }

    for (const [droneId, row] of latestByDrone.entries()) {
      const points = (row.route_points || []).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
      if (points.length < 2) continue;
      const color = droneId === 0 ? COLORS.batchRoute0 : COLORS.batchRoute1;

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 3.4;
      ctx.setLineDash([10, 7]);
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i += 1) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = color;
      for (const p of points) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3.3, 0, Math.PI * 2);
        ctx.fill();
      }

      const label = `${row.batch_id || "B?"}(${Number(row.batch_size || 0)})`;
      const anchor = points[Math.min(1, points.length - 1)];
      ctx.fillStyle = "rgba(7, 20, 31, 0.92)";
      ctx.fillRect(anchor.x - 16, anchor.y - 22, 38, 14);
      ctx.fillStyle = "#dff4ff";
      ctx.font = "10px IBM Plex Mono";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(label, anchor.x - 14, anchor.y - 21);
      ctx.restore();
    }
  }

  function drawDrone(ctx, drone, color, label) {
    ctx.save();
    ctx.translate(drone.x, drone.y);
    ctx.rotate(drone.heading_rad || 0);

    ctx.fillStyle = color;
    ctx.strokeStyle = "rgba(12, 25, 37, 0.85)";
    ctx.lineWidth = 1.4;

    ctx.beginPath();
    ctx.moveTo(10, 0);
    ctx.lineTo(-8, 6.5);
    ctx.lineTo(-4, 0);
    ctx.lineTo(-8, -6.5);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.restore();

    ctx.fillStyle = "rgba(7, 20, 31, 0.9)";
    ctx.fillRect(drone.x - 14, drone.y - 20, 28, 12);
    ctx.fillStyle = "#f4fbff";
    ctx.font = "10px IBM Plex Mono";
    ctx.textAlign = "center";
    ctx.fillText(label, drone.x, drone.y - 10.5);
  }

  function renderMap(ctx, map, sim, mapLabel, payloadCapacity, siteCount) {
    if (!map || !sim) return;

    ctx.clearRect(0, 0, map.width, map.height);
    drawBackgroundGrid(ctx, map);
    drawObstacles(ctx, map.obstacles || []);
    drawTasks(ctx, sim, map.sites || []);
    drawBatchRouteOverlay(ctx, sim);
    drawSites(ctx, map.sites || []);

    for (const drone of sim.drones || []) {
      const routeColor = drone.id === 0 ? COLORS.route : COLORS.route2;
      drawDronePath(ctx, drone, routeColor);
    }

    if (sim.drones && sim.drones[0]) {
      drawDrone(
        ctx,
        sim.drones[0],
        COLORS.drone0,
        `D1(${Number(sim.drones[0].cargo_count || 0)})`
      );
    }
    if (sim.drones && sim.drones[1]) {
      drawDrone(
        ctx,
        sim.drones[1],
        COLORS.drone1,
        `D2(${Number(sim.drones[1].cargo_count || 0)})`
      );
    }

    const m = sim.metrics;
    ctx.fillStyle = "rgba(4, 14, 23, 0.8)";
    ctx.fillRect(12, 12, 420, 74);
    ctx.strokeStyle = "rgba(152, 212, 255, 0.25)";
    ctx.strokeRect(12, 12, 420, 74);

    ctx.fillStyle = "#e6f5ff";
    ctx.font = "13px IBM Plex Mono";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(`算法: ${mapLabel}  载荷上限: ${payloadCapacity}  任务点: ${siteCount}`, 22, 22);
    ctx.fillText(`待处理任务: ${m.pending}    累计完成: ${m.completed}`, 22, 42);
    ctx.fillText(`平均等待: ${m.avg_wait_s.toFixed(1)}s    完成率: ${(m.completion_rate * 100).toFixed(1)}%`, 22, 61);
  }

  function renderMetricCard(card, metrics) {
    const fields = card.querySelectorAll("[data-metric]");
    for (const field of fields) {
      const key = field.dataset.metric;
      switch (key) {
        case "avg-completion":
          field.textContent = fmtSec(metrics.avg_completion_s);
          break;
        case "distance":
          field.textContent = fmtDistance(metrics.total_distance_px);
          break;
        case "completion-rate":
          field.textContent = fmtPercent(metrics.completion_rate);
          break;
        case "avg-wait":
          field.textContent = fmtSec(metrics.avg_wait_s);
          break;
        case "completed":
          field.textContent = `${metrics.completed}`;
          break;
        default:
          field.textContent = "-";
      }
    }
  }

  function drawAxis(ctx, x, y, w, h) {
    ctx.strokeStyle = "rgba(194, 222, 248, 0.32)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.stroke();
  }

  function drawSeries(ctx, points, color, radius = 2.1) {
    if (points.length === 0) return;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();

    for (const p of points) {
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function renderLineChart(ctx, width, height, dataA, dataB, valueKey, yLabel) {
    ctx.clearRect(0, 0, width, height);

    const g = ctx.createLinearGradient(0, 0, width, height);
    g.addColorStop(0, "rgba(15, 34, 53, 0.86)");
    g.addColorStop(1, "rgba(9, 18, 31, 0.94)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, width, height);

    const margin = { left: 48, right: 16, top: 16, bottom: 30 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    drawAxis(ctx, margin.left, margin.top, plotW, plotH);

    const merged = [...dataA, ...dataB];
    if (merged.length === 0) {
      ctx.fillStyle = "rgba(198, 222, 245, 0.8)";
      ctx.font = "11px IBM Plex Mono";
      ctx.fillText("等待数据...", margin.left + 16, margin.top + 24);
      return;
    }

    const maxT = Math.max(1, merged[merged.length - 1].t);
    const maxYRaw = Math.max(...merged.map((p) => Number(p[valueKey] || 0)), 1);
    const maxY = maxYRaw * 1.12;

    const mapPoint = (p) => ({
      x: margin.left + (p.t / maxT) * plotW,
      y: margin.top + plotH - ((Number(p[valueKey] || 0) / maxY) * plotH),
    });

    drawSeries(ctx, dataA.map(mapPoint), COLORS.drone0);
    drawSeries(ctx, dataB.map(mapPoint), COLORS.drone1);

    ctx.fillStyle = "rgba(220, 238, 252, 0.85)";
    ctx.font = "11px IBM Plex Mono";
    ctx.textAlign = "left";
    ctx.fillText("0", margin.left - 10, margin.top + plotH + 14);
    ctx.fillText(`${maxT.toFixed(0)}s`, margin.left + plotW - 26, margin.top + plotH + 14);

    ctx.textAlign = "right";
    ctx.fillText("0", margin.left - 7, margin.top + plotH + 3);
    ctx.fillText(maxYRaw.toFixed(1), margin.left - 7, margin.top + 4);

    ctx.textAlign = "left";
    ctx.fillText(yLabel, 10, 12);

    ctx.fillStyle = COLORS.drone0;
    ctx.fillRect(width - 126, 10, 10, 10);
    ctx.fillStyle = "rgba(225, 240, 252, 0.92)";
    ctx.fillText("A* + 贪心", width - 112, 19);

    ctx.fillStyle = COLORS.drone1;
    ctx.fillRect(width - 126, 26, 10, 10);
    ctx.fillStyle = "rgba(225, 240, 252, 0.92)";
    ctx.fillText("A* 基线", width - 112, 35);
  }

  function renderScatterChart(ctx, width, height, dataA, dataB) {
    ctx.clearRect(0, 0, width, height);

    const g = ctx.createLinearGradient(0, 0, width, height);
    g.addColorStop(0, "rgba(16, 35, 55, 0.9)");
    g.addColorStop(1, "rgba(8, 18, 30, 0.95)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, width, height);

    const margin = { left: 48, right: 16, top: 16, bottom: 30 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    drawAxis(ctx, margin.left, margin.top, plotW, plotH);

    const pA = (dataA || []).map((row) => ({
      x: Number(row.batch_route_est_distance || 0),
      y: Number(row.batch_elapsed_s || 0),
      size: Number(row.batch_size || 1),
    }));
    const pB = (dataB || []).map((row) => ({
      x: Number(row.batch_route_est_distance || 0),
      y: Number(row.batch_elapsed_s || 0),
      size: Number(row.batch_size || 1),
    }));
    const merged = [...pA, ...pB].filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (merged.length === 0) {
      ctx.fillStyle = "rgba(198, 222, 245, 0.8)";
      ctx.font = "11px IBM Plex Mono";
      ctx.fillText("等待批次结果数据...", margin.left + 14, margin.top + 24);
      return;
    }

    const maxXRaw = Math.max(...merged.map((p) => p.x), 1);
    const maxYRaw = Math.max(...merged.map((p) => p.y), 1);
    const maxX = maxXRaw * 1.1;
    const maxY = maxYRaw * 1.15;

    const mapX = (v) => margin.left + (v / maxX) * plotW;
    const mapY = (v) => margin.top + plotH - (v / maxY) * plotH;

    const drawDots = (arr, color) => {
      for (const p of arr) {
        const radius = clamp(2 + p.size * 0.65, 2.3, 6.2);
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(mapX(p.x), mapY(p.y), radius, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    drawDots(pA, "rgba(54, 209, 255, 0.82)");
    drawDots(pB, "rgba(255, 94, 94, 0.82)");

    ctx.fillStyle = "rgba(220, 238, 252, 0.85)";
    ctx.font = "11px IBM Plex Mono";
    ctx.textAlign = "left";
    ctx.fillText("0", margin.left - 10, margin.top + plotH + 14);
    ctx.fillText(`${maxXRaw.toFixed(0)}px`, margin.left + plotW - 38, margin.top + plotH + 14);
    ctx.textAlign = "right";
    ctx.fillText("0", margin.left - 7, margin.top + plotH + 3);
    ctx.fillText(`${maxYRaw.toFixed(1)}s`, margin.left - 7, margin.top + 4);
    ctx.textAlign = "left";
    ctx.fillText("x: route_est, y: batch_elapsed", 10, 12);

    ctx.fillStyle = COLORS.drone0;
    ctx.fillRect(width - 126, 10, 10, 10);
    ctx.fillStyle = "rgba(225, 240, 252, 0.92)";
    ctx.fillText("A* + 贪心", width - 112, 19);
    ctx.fillStyle = COLORS.drone1;
    ctx.fillRect(width - 126, 26, 10, 10);
    ctx.fillStyle = "rgba(225, 240, 252, 0.92)";
    ctx.fillText("A* 基线", width - 112, 35);
  }

  function phaseColor(phase) {
    if (phase === "delivering") return COLORS.phaseDelivering;
    if (phase === "returning") return COLORS.phaseReturning;
    return COLORS.phaseIdle;
  }

  function buildPhaseSegments(trace, droneId, tMin, tMax, currentT) {
    if (!Array.isArray(trace) || trace.length === 0) return [];
    const arr = trace
      .filter((row) => Number.isFinite(Number(row.t)))
      .map((row) => ({ t: Number(row.t), phases: Array.isArray(row.phases) ? row.phases : [] }))
      .sort((a, b) => a.t - b.t);
    if (arr.length === 0) return [];

    const segments = [];
    for (let i = 0; i < arr.length - 1; i += 1) {
      const start = arr[i].t;
      const end = arr[i + 1].t;
      const phase = arr[i].phases[droneId] || "idle";
      if (end <= tMin || start >= tMax || end <= start) continue;
      segments.push({
        phase,
        start: Math.max(start, tMin),
        end: Math.min(end, tMax),
      });
    }

    const last = arr[arr.length - 1];
    const lastPhase = last.phases[droneId] || "idle";
    const lastStart = Math.max(last.t, tMin);
    if (currentT > lastStart) {
      segments.push({
        phase: lastPhase,
        start: lastStart,
        end: Math.min(currentT, tMax),
      });
    }

    const merged = [];
    for (const seg of segments) {
      if (seg.end - seg.start <= 0.001) continue;
      const prev = merged[merged.length - 1];
      if (prev && prev.phase === seg.phase && Math.abs(prev.end - seg.start) < 0.06) {
        prev.end = seg.end;
      } else {
        merged.push(seg);
      }
    }
    return merged;
  }

  function renderGanttChart(ctx, width, height, state) {
    ctx.clearRect(0, 0, width, height);

    const g = ctx.createLinearGradient(0, 0, width, height);
    g.addColorStop(0, "rgba(16, 35, 55, 0.9)");
    g.addColorStop(1, "rgba(8, 18, 30, 0.95)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, width, height);

    const margin = { left: 82, right: 16, top: 16, bottom: 28 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    drawAxis(ctx, margin.left, margin.top, plotW, plotH);

    if (!state?.algorithms) {
      ctx.fillStyle = "rgba(198, 222, 245, 0.8)";
      ctx.font = "11px IBM Plex Mono";
      ctx.fillText("等待状态数据...", margin.left + 14, margin.top + 24);
      return;
    }

    const tNow = Number(state.time_s || 0);
    const tMax = Math.max(1, tNow);
    const tMin = Math.max(0, tMax - 60);
    const span = Math.max(1, tMax - tMin);

    const astarTrace = state.algorithms.astar?.phase_trace || [];
    const baselineTrace = state.algorithms.mappo?.phase_trace || [];
    const rows = [
      { label: "A* D1", segments: buildPhaseSegments(astarTrace, 0, tMin, tMax, tNow) },
      { label: "A* D2", segments: buildPhaseSegments(astarTrace, 1, tMin, tMax, tNow) },
      { label: "基线 D1", segments: buildPhaseSegments(baselineTrace, 0, tMin, tMax, tNow) },
      { label: "基线 D2", segments: buildPhaseSegments(baselineTrace, 1, tMin, tMax, tNow) },
    ];

    const hasAny = rows.some((r) => r.segments.length > 0);
    if (!hasAny) {
      ctx.fillStyle = "rgba(198, 222, 245, 0.8)";
      ctx.font = "11px IBM Plex Mono";
      ctx.fillText("等待事件时间轴数据...", margin.left + 14, margin.top + 24);
      return;
    }

    const rowGap = 6;
    const rowH = (plotH - rowGap * (rows.length - 1)) / rows.length;
    const mapX = (t) => margin.left + ((t - tMin) / span) * plotW;

    rows.forEach((row, idx) => {
      const y = margin.top + idx * (rowH + rowGap);
      ctx.fillStyle = "rgba(109, 144, 176, 0.16)";
      ctx.fillRect(margin.left, y, plotW, rowH);

      for (const seg of row.segments) {
        const x0 = mapX(seg.start);
        const x1 = mapX(seg.end);
        const w = Math.max(1, x1 - x0);
        ctx.fillStyle = phaseColor(seg.phase);
        ctx.fillRect(x0, y, w, rowH);
      }

      ctx.fillStyle = "rgba(220, 238, 252, 0.9)";
      ctx.font = "11px IBM Plex Mono";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(row.label, margin.left - 8, y + rowH * 0.5);
    });

    ctx.fillStyle = "rgba(220, 238, 252, 0.85)";
    ctx.font = "11px IBM Plex Mono";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`${tMin.toFixed(0)}s`, margin.left, margin.top + plotH + 16);
    ctx.fillText(`${tMax.toFixed(0)}s`, margin.left + plotW - 24, margin.top + plotH + 16);
    ctx.fillText("phase timeline", 10, 12);

    const legend = [
      { label: "idle", color: COLORS.phaseIdle },
      { label: "delivering", color: COLORS.phaseDelivering },
      { label: "returning", color: COLORS.phaseReturning },
    ];
    legend.forEach((item, i) => {
      const x = width - 220 + i * 72;
      ctx.fillStyle = item.color;
      ctx.fillRect(x, 10, 10, 10);
      ctx.fillStyle = "rgba(225, 240, 252, 0.92)";
      ctx.fillText(item.label, x + 13, 19);
    });
  }

  function buildInsight(astarSim, baselineSim) {
    const astar = astarSim.metrics || {};
    const baseline = baselineSim.metrics || {};

    if ((astar.generated || 0) < 4) {
      return "样本任务较少，建议先运行 20-30 秒后再比较。";
    }

    const lines = [];
    if (baseline.avg_wait_s + 0.12 < astar.avg_wait_s) {
      lines.push("结果差异: A* 基线 当前平均等待时长更低。");
    } else if (astar.avg_wait_s + 0.12 < baseline.avg_wait_s) {
      lines.push("结果差异: A* + 贪心 当前平均等待时长更低。");
    } else {
      lines.push("结果差异: 两者等待时长接近。");
    }

    if (astar.total_distance_px + 80 < baseline.total_distance_px) {
      lines.push("结果差异: A* + 贪心 当前总路程更短。");
    } else if (baseline.total_distance_px + 80 < astar.total_distance_px) {
      lines.push("结果差异: A* 基线 当前总路程更短。");
    }

    if (Math.abs((astar.avg_batch_size || 0) - (baseline.avg_batch_size || 0)) > 0.2) {
      const batchWinner = (astar.avg_batch_size || 0) > (baseline.avg_batch_size || 0) ? "A* + 贪心" : "A* 基线";
      lines.push(`行为差异: ${batchWinner} 的平均批次载荷更高（更倾向连续多点投递）。`);
    }
    if (Math.abs((astar.avg_batch_route_est_distance || 0) - (baseline.avg_batch_route_est_distance || 0)) > 45) {
      const routeWinner =
        (astar.avg_batch_route_est_distance || 0) < (baseline.avg_batch_route_est_distance || 0) ? "A* + 贪心" : "A* 基线";
      lines.push(`行为差异: ${routeWinner} 的单批次预估路径更紧凑。`);
    }

    if (astar.avg_wait_s + 0.25 < baseline.avg_wait_s && astar.avg_batch_size > baseline.avg_batch_size + 0.15) {
      lines.push("行为->结果映射: A* 批量装载更高，与更低任务等待时长同时出现。");
    } else if (baseline.avg_wait_s + 0.25 < astar.avg_wait_s && baseline.avg_batch_size > astar.avg_batch_size + 0.15) {
      lines.push("行为->结果映射: A* 基线 批量装载更高，与更低任务等待时长同时出现。");
    }

    return lines.join("\n");
  }

  function formatEventType(eventName) {
    if (eventName === "load_batch") return "装载批次";
    if (eventName === "deliver_task") return "完成配送";
    if (eventName === "batch_completed_returning") return "批次结束返航";
    if (eventName === "arrive_center") return "回到物流中心";
    if (eventName === "reposition_to_center") return "重定位到中心";
    return eventName || "未知事件";
  }

  function formatEventMetricBrief(metrics) {
    if (!metrics) return "pending=-, wait=-, done=-";
    const pending = Number(metrics.pending || 0);
    const avgWait = fmtNum(metrics.avg_wait_s || 0, 1);
    const rate = fmtPercent(metrics.completion_rate || 0);
    return `pending=${pending}, wait=${avgWait}s, done=${rate}`;
  }

  const SCORE_KEYS = {
    score: "s",
    distance_term: "dist",
    age_term: "age*",
    priority_term: "prio*",
    mode: "mode",
    age: "age",
    center_to_dst: "d0",
    priority: "prio",
    load_factor: "load",
  };

  function formatScoreItem(item) {
    if (!item || !item.task_id) return "-";
    const parts = [];
    for (const [k, v] of Object.entries(item)) {
      if (k === "task_id") continue;
      const key = SCORE_KEYS[k] || k;
      if (typeof v === "number") {
        parts.push(`${key}:${fmtNum(v, 2)}`);
      } else if (v === null || v === undefined) {
        parts.push(`${key}:NA`);
      } else {
        parts.push(`${key}:${v}`);
      }
    }
    return `${item.task_id}${parts.length ? `(${parts.join(", ")})` : ""}`;
  }

  function renderBehaviorCard(sim, payloadCapacity, elements) {
    const metrics = sim.metrics || {};
    const drones = sim.drones || [];
    const events = sim.event_log || [];
    const decisions = sim.decision_log || [];
    const deliverEvents = events.filter((e) => e.event === "deliver_task");
    const chainDeliveries = deliverEvents.filter((e) => Number(e.remaining_cargo || 0) > 0).length;
    const chainRate = deliverEvents.length > 0 ? chainDeliveries / deliverEvents.length : 0;

    const batchLines = [
      `平均批次装载: ${fmtNum(metrics.avg_batch_size || 0, 2)} / ${payloadCapacity}`,
      `平均批次预估路径: ${fmtNum(metrics.avg_batch_route_est_distance || 0, 1)} px`,
      `批次数: ${Number(metrics.batch_count || 0)} | 连续投递占比: ${fmtPercent(chainRate)}`,
      "",
    ];
    for (const drone of drones) {
      const queue = (drone.cargo_task_ids || []).join(" -> ") || "空";
      batchLines.push(
        `D${Number(drone.id) + 1} | ${phaseLabel(drone.phase)} | 载荷 ${Number(drone.cargo_count || 0)}/${payloadCapacity} | 队列 ${queue}`
      );
    }
    elements.batch.textContent = batchLines.join("\n");

    const decisionLines = decisions
      .slice(-6)
      .reverse()
      .map((row) => {
        const tasks = (row.selected_tasks || []).join(" -> ") || "空";
        const top = (row.ranking_top || []).map((item) => formatScoreItem(item)).join(" ; ");
        const scoreBlock = top ? `\n  候选评分: ${top}` : "";
        return `[t=${fmtNum(row.t, 1)}s] ${row.batch_id || "B?"} D${Number(row.drone_id) + 1} 选中 ${tasks} | 预估路程 ${fmtNum(row.batch_route_est_distance, 1)} px${scoreBlock}`;
      });
    elements.decision.textContent = decisionLines.length ? decisionLines.join("\n") : "暂无决策日志";

    const eventLines = events
      .slice(-10)
      .reverse()
      .map((row) => {
        let detail = "";
        if (row.event === "load_batch") {
          detail = `${row.batch_id || "B?"} size=${Number(row.batch_size || 0)}, tasks=${(row.tasks || []).join("->")}`;
        } else if (row.event === "deliver_task") {
          detail = `${row.batch_id || "B?"} task=${row.task_id || "-"}, remain=${Number(row.remaining_cargo || 0)}`;
        } else if (row.event === "batch_completed_returning") {
          detail = `${row.batch_id || "B?"} last=${row.last_task_id || "-"}, elapsed=${fmtNum(row.batch_elapsed_s || 0, 1)}s`;
        }
        const metricBrief = formatEventMetricBrief(row.metrics);
        return `[t=${fmtNum(row.t, 1)}s] D${Number(row.drone_id) + 1} ${formatEventType(row.event)} ${detail} | ${metricBrief}`;
      });
    elements.event.textContent = eventLines.length ? eventLines.join("\n") : "暂无行为事件";
  }

  function renderBehaviorPanels(astar, mappo, payloadCapacity) {
    renderBehaviorCard(astar, payloadCapacity, {
      batch: dom.astarBatchView,
      decision: dom.astarDecisionLog,
      event: dom.astarEventLog,
    });
    renderBehaviorCard(mappo, payloadCapacity, {
      batch: dom.mappoBatchView,
      decision: dom.mappoDecisionLog,
      event: dom.mappoEventLog,
    });
  }

  function setManualInsight(text, holdMs = 12000) {
    stateStore.manualInsight = text;
    stateStore.manualInsightUntil = Date.now() + holdMs;
  }

  function formatSweepConclusion(report) {
    const lines = [];
    const rows = report.capacity_reports || [];
    for (const row of rows) {
      const winner =
        row.winner_by_completion === "astar"
          ? "A*更优"
          : row.winner_by_completion === "baseline"
            ? "A*基线更优"
            : "无显著差异";
      lines.push(
        `载荷${row.payload_capacity}: ${winner} | 完成率 A*+贪心=${(row.astar_completion_mean * 100).toFixed(1)}% A*基线=${(row.baseline_completion_mean * 100).toFixed(1)}%`
      );
    }
    lines.push(report.overall_conclusion || "");
    return lines.join("\n");
  }

  function renderDashboard() {
    const state = stateStore.latest;
    if (!state) return;

    const astar = state.algorithms.astar;
    const mappo = state.algorithms.mappo;

    const payloadCapacity = Number(state.controls?.payload_capacity || 1);
    const siteCount = Number(state.controls?.site_count || 3);
    renderMap(ctxMapAstar, state.map, astar, "A* + 贪心", payloadCapacity, siteCount);
    renderMap(ctxMapMappo, state.map, mappo, "A* 基线", payloadCapacity, siteCount);

    renderMetricCard(dom.astarCard, astar.metrics);
    renderMetricCard(dom.mappoCard, mappo.metrics);

    renderLineChart(
      ctxCompleted,
      dom.completedChart.width,
      dom.completedChart.height,
      astar.history || [],
      mappo.history || [],
      "completed",
      "completed"
    );

    renderLineChart(
      ctxWait,
      dom.waitChart.width,
      dom.waitChart.height,
      astar.history || [],
      mappo.history || [],
      "avg_wait",
      "avg wait (s)"
    );

    renderScatterChart(
      ctxScatter,
      dom.scatterChart.width,
      dom.scatterChart.height,
      astar.batch_results || [],
      mappo.batch_results || []
    );

    renderGanttChart(ctxGantt, dom.ganttChart.width, dom.ganttChart.height, state);

    if (stateStore.manualInsight && Date.now() < stateStore.manualInsightUntil) {
      dom.insight.textContent = stateStore.manualInsight;
    } else {
      stateStore.manualInsight = null;
      dom.insight.textContent = buildInsight(astar, mappo);
    }
    renderBehaviorPanels(astar, mappo, payloadCapacity);
    dom.simState.textContent = state.mode === "running" ? "运行中" : "已暂停";
    dom.simState.classList.toggle("running", state.mode === "running");
    if (dom.paramSummary) {
      dom.paramSummary.textContent = `点位=${siteCount} 载荷=${payloadCapacity}`;
    }
    if (dom.currentSiteCount) dom.currentSiteCount.textContent = String(siteCount);
    if (dom.currentPayloadCapacity) dom.currentPayloadCapacity.textContent = String(payloadCapacity);
    dom.simTime.textContent = `t = ${(state.time_s || 0).toFixed(1)}s`;
  }

  async function refreshState() {
    if (stateStore.fetching) return;
    stateStore.fetching = true;
    try {
      const data = await apiRequest("/api/state");
      stateStore.latest = data;
      dom.speedLabel.textContent = `${Number(data.controls.speed_multiplier || 1.5).toFixed(1)}x`;
      if (dom.payloadInput && document.activeElement !== dom.payloadInput) {
        dom.payloadInput.value = `${Number(data.controls.payload_capacity || 2)}`;
      }
      if (dom.siteInput && document.activeElement !== dom.siteInput) {
        dom.siteInput.value = `${Number(data.controls.site_count || 3)}`;
      }
    } catch (err) {
      dom.insight.textContent = `状态更新失败: ${String(err.message || err)}`;
    } finally {
      stateStore.fetching = false;
    }
  }

  let configTimer = null;
  function scheduleConfigUpdate() {
    if (configTimer) clearTimeout(configTimer);
    configTimer = setTimeout(async () => {
      try {
        await apiRequest("/api/control/config", "POST", {
          speed: readNumberInput(dom.speedInput, 1.5),
          interval_s: readNumberInput(dom.intervalInput, 4),
          payload_capacity: readNumberInput(dom.payloadInput, 2),
          site_count: readNumberInput(dom.siteInput, 3),
        });
      } catch (err) {
        dom.insight.textContent = `参数更新失败: ${String(err.message || err)}`;
      }
    }, 120);
  }

  function attachHandlers() {
    dom.startBtn.addEventListener("click", async () => {
      await apiRequest("/api/control/start", "POST");
      await refreshState();
    });

    dom.pauseBtn.addEventListener("click", async () => {
      await apiRequest("/api/control/pause", "POST");
      await refreshState();
    });

    dom.resetBtn.addEventListener("click", async () => {
      await apiRequest("/api/control/reset", "POST", {
        seed: readNumberInput(dom.seedInput, 42),
        interval_s: readNumberInput(dom.intervalInput, 4),
        speed: readNumberInput(dom.speedInput, 1.5),
        payload_capacity: readNumberInput(dom.payloadInput, 2),
        site_count: readNumberInput(dom.siteInput, 3),
      });
      await refreshState();
    });

    dom.speedInput.addEventListener("input", () => {
      dom.speedLabel.textContent = `${Number(dom.speedInput.value || 1.5).toFixed(1)}x`;
      scheduleConfigUpdate();
    });

    dom.intervalInput.addEventListener("change", scheduleConfigUpdate);
    dom.payloadInput.addEventListener("change", scheduleConfigUpdate);
    dom.siteInput.addEventListener("change", scheduleConfigUpdate);
    dom.payloadInput.addEventListener("input", scheduleConfigUpdate);
    dom.siteInput.addEventListener("input", scheduleConfigUpdate);

    dom.checkBtn.addEventListener("click", async () => {
      if (stateStore.evalRunning || stateStore.sweepRunning) return;
      stateStore.evalRunning = true;
      dom.checkBtn.disabled = true;
      const original = dom.insight.textContent;
      setManualInsight("正在执行多随机种子结果检查（约 5-15 秒）...", 60000);

      try {
        const data = await apiRequest("/api/evaluate", "POST", {
          episodes: 24,
          horizon_s: 120,
          interval_s: readNumberInput(dom.intervalInput, 4),
          base_seed: 100,
          payload_capacity: readNumberInput(dom.payloadInput, 2),
          site_count: readNumberInput(dom.siteInput, 3),
        });
        const report = data.report;
        const checks = report.checks;
        const checksOK = Object.values(checks).every(Boolean);
        setManualInsight(`${checksOK ? "校验通过" : "校验存在异常"} | ${report.conclusion}`, 30000);
      } catch (err) {
        setManualInsight(`结果检查失败: ${String(err.message || err)}\n${original}`, 15000);
      } finally {
        stateStore.evalRunning = false;
        dom.checkBtn.disabled = false;
      }
    });

    dom.sweepBtn.addEventListener("click", async () => {
      if (stateStore.sweepRunning || stateStore.evalRunning) return;
      stateStore.sweepRunning = true;
      dom.sweepBtn.disabled = true;
      setManualInsight("正在执行载荷上限扫描分析（1..5）...", 90000);

      try {
        const data = await apiRequest("/api/evaluate", "POST", {
          episodes: 24,
          horizon_s: 120,
          interval_s: readNumberInput(dom.intervalInput, 4),
          base_seed: 100,
          site_count: readNumberInput(dom.siteInput, 3),
          payload_values: [1, 2, 3, 4, 5],
        });
        const report = data.report;
        setManualInsight(formatSweepConclusion(report), 40000);
      } catch (err) {
        setManualInsight(`载荷分析失败: ${String(err.message || err)}`, 15000);
      } finally {
        stateStore.sweepRunning = false;
        dom.sweepBtn.disabled = false;
      }
    });

    window.addEventListener("keydown", async (event) => {
      if (event.key.toLowerCase() === "f") {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      }
      if (event.code === "Space") {
        event.preventDefault();
        const mode = stateStore.latest?.mode;
        if (mode === "running") {
          await apiRequest("/api/control/pause", "POST");
        } else {
          await apiRequest("/api/control/start", "POST");
        }
        await refreshState();
      }
    });

    window.render_game_to_text = () => JSON.stringify(stateStore.latest || { mode: "loading" });

    window.advanceTime = async (ms) => {
      const dt = Number(ms);
      if (!Number.isFinite(dt) || dt <= 0) return;
      await apiRequest("/api/control/advance", "POST", { ms: dt });
      await refreshState();
    };
  }

  function frame() {
    renderDashboard();
    requestAnimationFrame(frame);
  }

  attachHandlers();
  initPanelToggles();
  refreshState();
  setInterval(refreshState, 180);
  requestAnimationFrame(frame);
})();
