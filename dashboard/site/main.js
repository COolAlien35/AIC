/* AIC results dashboard · charts + bindings.
 *
 * Data is exposed by data.js as window.AIC_DATA. Everything is sync.
 */
(function () {
  "use strict";

  const D = window.AIC_DATA;
  if (!D) return;

  // Hugging Face Spaces (Hub git) rejects large binary blobs unless stored via Xet/LFS.
  // To keep the dashboard Space lightweight, we do NOT push PNG plots there.
  // When running on *.hf.space we instead load plot images from GitHub raw.
  const IS_HF_SPACE = /\.hf\.space$/i.test(window.location.hostname || "");
  const PLOTS_BASE = IS_HF_SPACE
    ? "https://raw.githubusercontent.com/COolAlien35/AIC/main/dashboard/site/plots/"
    : "plots/";

  /* ──────────── Chart.js global tuning ──────────── */
  const C = window.Chart;
  C.defaults.color = "#aab2a7";
  C.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  C.defaults.font.size = 12;
  C.defaults.borderColor = "rgba(255, 255, 255, 0.06)";
  C.defaults.animation.duration = 420;
  C.defaults.animation.easing = "easeOutCubic";
  C.defaults.responsive = true;
  C.defaults.maintainAspectRatio = false;

  const palette = {
    text:     "#e7ece5",
    text2:    "#aab2a7",
    text3:    "#6f7a6c",
    line:     "rgba(255, 255, 255, 0.06)",
    lineHard: "rgba(255, 255, 255, 0.12)",
    emerald:  "#10b981",
    emerald2: "#34d399",
    emerald3: "rgba(52, 211, 153, 0.18)",
    amber:    "#fbbf24",
    rose:     "#fb7185",
    violet:   "#a78bfa",
    teal:     "#2dd4bf",
    grey:     "#64748b",
  };

  const POLICY_LABEL = {
    baseline_frozen:   "Baseline (frozen)",
    baseline_adaptive: "Baseline (adaptive)",
    trained_grpo:      "Trained · GRPO",
  };
  const POLICY_COLOR = {
    baseline_frozen:   palette.grey,
    baseline_adaptive: palette.teal,
    trained_grpo:      palette.emerald2,
  };

  /* ──────────── small formatters ──────────── */
  const fmt1 = (x, sign = false) => {
    const n = Number(x);
    if (!isFinite(n)) return "—";
    const v = n.toFixed(1);
    return sign && n > 0 ? "+" + v : v;
  };
  const fmt2 = (x, sign = false) => {
    const n = Number(x);
    if (!isFinite(n)) return "—";
    const v = n.toFixed(2);
    return sign && n > 0 ? "+" + v : v;
  };
  const fmtPct = (x) => {
    const n = Number(x);
    if (!isFinite(n)) return "—";
    const sign = n > 0 ? "+" : "";
    return sign + n.toFixed(1) + " %";
  };
  const fmtP = (p) => {
    if (p == null || !isFinite(p)) return "—";
    if (p === 0) return "≪ 0.001";
    if (p < 1e-6) return p.toExponential(2).replace("e", " × 10^");
    if (p < 0.001) return p.toExponential(2);
    return p.toFixed(3);
  };
  const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

  /* ──────────── tooltip / chart base options ──────────── */
  const baseOpts = (overrides = {}) => ({
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: "rgba(11, 15, 12, 0.94)",
        borderColor: palette.lineHard,
        borderWidth: 1,
        titleColor: palette.text,
        bodyColor: palette.text2,
        titleFont: { weight: 600, size: 12.5 },
        bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
        padding: 10,
        cornerRadius: 8,
        displayColors: false,
      },
    },
    scales: {
      x: {
        grid: { color: palette.line, drawTicks: false, drawBorder: false },
        ticks: { color: palette.text3, font: { size: 11 } },
        border: { display: false },
      },
      y: {
        grid: { color: palette.line, drawTicks: false, drawBorder: false },
        ticks: { color: palette.text3, font: { size: 11, family: "'JetBrains Mono', monospace" } },
        border: { display: false },
      },
    },
    ...overrides,
  });

  /* ──────────── HERO + meta ──────────── */
  function bindHero() {
    const stats = D.stats || {};
    const grpo  = D.grpo_summary || {};
    const ins   = D.insights || {};
    const totalEpisodes = (D.benchmark_summary || []).reduce((a, r) => a + (r.num_episodes || 0), 0);
    const scenarios = new Set((D.benchmark_by_scenario || []).map(r => r.scenario)).size;
    const trainingRuns = new Set((D.episodes || []).map(r => r.training_run_id)).size;

    setText("meta-episodes",  totalEpisodes || "—");
    setText("meta-runs",      trainingRuns  || "—");
    setText("meta-scenarios", scenarios     || "—");

    setText("kpi-uplift",      fmt2(stats.improvement, true));
    setText("kpi-uplift-pct",  fmtPct(stats.improvement_pct));
    setText("kpi-cohens-d",    fmt2(stats.cohens_d));
    setText("kpi-cohens-label",(stats.effect_size_label || "").toLowerCase());
    setText("kpi-pvalue",      fmtP(stats.p_value));
    setText("kpi-grpo-delta",  fmt2(grpo.reward_delta, true));
    setText("kpi-grpo-steps",  grpo.total_steps || "—");
    setText("kpi-grpo-time",   grpo.training_time_minutes ? (grpo.training_time_minutes/60).toFixed(2) + " h" : "—");

    setText("training-headline",  `−${Math.abs(grpo.initial_reward).toFixed(2)} → −${Math.abs(grpo.final_reward).toFixed(2)}  ·  Δ ${fmt2(grpo.reward_delta, true)}`);
    setText("training-final-loss",`final ${grpo.final_loss}`);
    setText("training-final-kl",  `final ${(D.grpo_progress?.[D.grpo_progress.length-1]?.kl ?? 0).toFixed(4)}`);
  }

  /* ──────────── HEADLINE bar + table ──────────── */
  function chartHeadline() {
    const rows = (D.benchmark_summary || []).slice().sort((a, b) => b.avg_reward - a.avg_reward);
    const labels  = rows.map(r => POLICY_LABEL[r.policy] || r.policy);
    const values  = rows.map(r => r.avg_reward);
    const stds    = rows.map(r => r.std_reward);
    const ns      = rows.map(r => r.num_episodes);
    const ci95    = rows.map((r, i) => 1.96 * (stds[i] / Math.sqrt(ns[i] || 1)));
    const colors  = rows.map(r => POLICY_COLOR[r.policy] || palette.grey);
    const best = rows[0];
    setText("headline-best-policy", `Top: ${POLICY_LABEL[best.policy] || best.policy}  ·  μ = ${fmt1(best.avg_reward)}`);

    const ctx = document.getElementById("chart-headline").getContext("2d");
    new C(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: colors.map(c => c + "cc"),
          borderColor: colors,
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
          maxBarThickness: 78,
        }],
      },
      options: baseOpts({
        layout: { padding: { top: 18, right: 8 } },
        scales: {
          x: { grid: { display: false }, ticks: { color: palette.text2, font: { size: 12, weight: "500" } }, border: { display: false } },
          y: { grid: { color: palette.line, drawTicks: false }, ticks: { color: palette.text3, font: { size: 11, family: "'JetBrains Mono', monospace" } }, border: { display: false }, suggestedMax: 0 },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...baseOpts().plugins.tooltip,
            callbacks: {
              title: (items) => items[0].label,
              label: (item) => {
                const i = item.dataIndex;
                return [
                  `μ      ${values[i].toFixed(2)}`,
                  `±1.96·SE  ${ci95[i].toFixed(2)}`,
                  `n      ${ns[i]}`,
                ];
              },
            },
          },
        },
      }),
      // Custom error bar plugin
      plugins: [{
        id: "errorbars",
        afterDatasetsDraw(chart) {
          const { ctx, scales: { x, y } } = chart;
          ctx.save();
          ctx.strokeStyle = palette.lineHard;
          ctx.lineWidth = 1.5;
          values.forEach((v, i) => {
            const cx = x.getPixelForValue(i);
            const top = y.getPixelForValue(v + ci95[i]);
            const bot = y.getPixelForValue(v - ci95[i]);
            ctx.beginPath(); ctx.moveTo(cx, top); ctx.lineTo(cx, bot); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx - 6, top); ctx.lineTo(cx + 6, top); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx - 6, bot); ctx.lineTo(cx + 6, bot); ctx.stroke();
          });
          ctx.restore();
        },
      }],
    });
  }

  function tableSummary() {
    const tbody = document.getElementById("tbody-summary");
    const rows = (D.benchmark_summary || []).slice().sort((a, b) => b.avg_reward - a.avg_reward);
    const trainedReward = (rows.find(r => r.policy === "trained_grpo") || {}).avg_reward;
    tbody.innerHTML = rows.map(r => {
      const isTrained = r.policy === "trained_grpo";
      const tag = isTrained
        ? `<span class="row-tag good">trained · best</span>`
        : `<span class="row-tag neutral">baseline</span>`;
      return `<tr>
        <td>${POLICY_LABEL[r.policy] || r.policy}</td>
        <td class="num">${r.avg_reward.toFixed(2)}</td>
        <td class="num">${r.std_reward.toFixed(2)}</td>
        <td class="num">${r.num_episodes}</td>
        <td>${tag}</td>
      </tr>`;
    }).join("");
  }

  /* ──────────── TRAINING charts ──────────── */
  function chartGRPO(canvasId, key, color, fillAlpha) {
    const data = D.grpo_progress || [];
    const labels = data.map(r => r.step);
    const values = data.map(r => r[key]);
    const ctx = document.getElementById(canvasId).getContext("2d");

    // gradient fill
    const c = ctx.canvas;
    const grad = ctx.createLinearGradient(0, 0, 0, c.height || 220);
    grad.addColorStop(0, color + (fillAlpha || "33"));
    grad.addColorStop(1, color + "00");

    new C(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          data: values,
          borderColor: color,
          backgroundColor: grad,
          borderWidth: 1.8,
          fill: true,
          tension: 0.28,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: color,
          pointHoverBorderColor: "rgba(255,255,255,0.9)",
          pointHoverBorderWidth: 2,
        }],
      },
      options: baseOpts({
        interaction: { mode: "index", intersect: false },
        scales: {
          x: {
            grid: { display: false }, ticks: { color: palette.text3, maxTicksLimit: 8 }, border: { display: false },
            title: { display: true, text: "Step", color: palette.text3, font: { size: 10.5, family: "'JetBrains Mono', monospace" } },
          },
          y: {
            grid: { color: palette.line, drawTicks: false }, ticks: { color: palette.text3, font: { size: 10.5, family: "'JetBrains Mono', monospace" } }, border: { display: false },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...baseOpts().plugins.tooltip,
            callbacks: {
              title: (items) => `step ${items[0].label}`,
              label: (item) => `${key.padEnd(8)} ${Number(item.parsed.y).toFixed(4)}`,
            },
          },
        },
      }),
    });
  }

  /* ──────────── PER-SCENARIO ──────────── */
  function aggregateByScenario() {
    const map = {};
    (D.benchmark_by_scenario || []).forEach(r => {
      map[r.scenario] = map[r.scenario] || {};
      map[r.scenario][r.policy] = r.avg_reward;
    });
    const scenarios = Object.keys(map);
    return scenarios.map(s => ({
      scenario: s,
      baseline: map[s]["baseline_frozen"],
      adaptive: map[s]["baseline_adaptive"],
      trained:  map[s]["trained_grpo"],
      delta: map[s]["trained_grpo"] - map[s]["baseline_frozen"],
    })).sort((a, b) => b.delta - a.delta);
  }

  function chartScenarioDelta() {
    const rows = aggregateByScenario();
    const labels = rows.map(r => r.scenario);
    const values = rows.map(r => r.delta);
    const colors = values.map(v => v >= 0 ? palette.emerald2 : palette.rose);
    setText("scenarios-best", `top: ${rows[0].scenario}  ·  Δ ${fmt2(rows[0].delta, true)}`);

    const ctx = document.getElementById("chart-scenario-delta").getContext("2d");
    new C(ctx, {
      type: "bar",
      data: { labels, datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + "cc"),
        borderColor: colors,
        borderWidth: 1.4,
        borderRadius: 6,
        maxBarThickness: 36,
      }] },
      options: baseOpts({
        indexAxis: "y",
        scales: {
          y: { grid: { display: false }, ticks: { color: palette.text2, font: { size: 12 } }, border: { display: false } },
          x: { grid: { color: palette.line, drawTicks: false }, ticks: { color: palette.text3, font: { size: 11, family: "'JetBrains Mono', monospace" }, callback: (v) => "+" + v }, border: { display: false } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...baseOpts().plugins.tooltip,
            callbacks: {
              title: (items) => items[0].label,
              label: (item) => {
                const r = rows[item.dataIndex];
                return [
                  `Δ        ${fmt2(r.delta, true)}`,
                  `baseline ${r.baseline.toFixed(2)}`,
                  `trained  ${r.trained.toFixed(2)}`,
                ];
              },
            },
          },
        },
      }),
    });
  }

  function chartScenarioGrouped() {
    const rows = aggregateByScenario();
    const labels = rows.map(r => r.scenario);
    const datasets = [
      { label: "Baseline (frozen)",   data: rows.map(r => r.baseline), color: palette.grey },
      { label: "Baseline (adaptive)", data: rows.map(r => r.adaptive), color: palette.teal },
      { label: "Trained · GRPO",     data: rows.map(r => r.trained),  color: palette.emerald2 },
    ];

    const ctx = document.getElementById("chart-scenario-grouped").getContext("2d");
    new C(ctx, {
      type: "bar",
      data: { labels, datasets: datasets.map(d => ({
        label: d.label,
        data: d.data,
        backgroundColor: d.color + "cc",
        borderColor: d.color,
        borderWidth: 1.2,
        borderRadius: 4,
        maxBarThickness: 22,
      })) },
      options: baseOpts({
        scales: {
          x: { grid: { display: false }, ticks: { color: palette.text2, font: { size: 11 }, autoSkip: false, maxRotation: 0, minRotation: 0 }, border: { display: false } },
          y: { grid: { color: palette.line, drawTicks: false }, ticks: { color: palette.text3, font: { size: 11, family: "'JetBrains Mono', monospace" } }, border: { display: false }, suggestedMax: 0 },
        },
        plugins: {
          legend: {
            display: true,
            position: "top",
            align: "start",
            labels: {
              color: palette.text2, font: { size: 11.5 }, boxWidth: 10, boxHeight: 10, usePointStyle: true,
              pointStyle: "rectRounded",
            },
          },
          tooltip: { ...baseOpts().plugins.tooltip, callbacks: {
            label: (item) => `${item.dataset.label.padEnd(20)} ${item.parsed.y.toFixed(2)}`,
          } },
        },
      }),
    });
  }

  function tableScenarios() {
    const tbody = document.getElementById("tbody-scenarios");
    const rows = aggregateByScenario();
    tbody.innerHTML = rows.map(r => {
      const dir = r.delta >= 0
        ? `<span class="row-tag good">↑ trained better</span>`
        : `<span class="row-tag neutral">↓ baseline better</span>`;
      return `<tr>
        <td>${r.scenario}</td>
        <td class="num">${r.baseline.toFixed(2)}</td>
        <td class="num">${r.trained.toFixed(2)}</td>
        <td class="num accent">${fmt2(r.delta, true)}</td>
        <td>${dir}</td>
      </tr>`;
    }).join("");
  }

  /* ──────────── STATISTICS ──────────── */
  function bindStats() {
    const s = D.stats || {};
    const n = (D.benchmark_summary || []).reduce((a, r) => a + (r.num_episodes || 0), 0);
    setText("stat-t",                fmt2(s.t_statistic));
    setText("stat-p",                fmtP(s.p_value));
    setText("stat-d",                fmt2(s.cohens_d));
    setText("stat-improvement",      fmt2(s.improvement, true));
    setText("stat-improvement-pct",  fmtPct(s.improvement_pct));
    setText("stat-n",                n || "—");

    setText("quote-n",   n || "—");
    setText("quote-imp", fmt2(s.improvement, true));
    setText("quote-pct", fmtPct(s.improvement_pct));
    setText("quote-d",   fmt2(s.cohens_d));
    setText("quote-p",   fmtP(s.p_value));
  }

  /* ──────────── TASKS ──────────── */
  function renderTasks() {
    const grid = document.getElementById("task-grid");
    const tg = D.task_grader || { tasks: {} };
    const taskTable = D.task_grader_table || [];

    // Pick the best baseline score for each task as the "current score" — same value
    // shows up in inference_summary.json.  Show that against the threshold.
    const ORDER = ["db_pool_recovery", "canary_blackout", "adversarial_misroute"];
    const FRIENDLY = {
      db_pool_recovery:    "DB pool recovery",
      canary_blackout:     "Canary blackout",
      adversarial_misroute:"Adversarial misroute",
    };
    grid.innerHTML = ORDER.map(id => {
      const t = tg.tasks?.[id] || {};
      const tableRow = taskTable.find(r => r.task_id === id) || {};
      const score = t.mean_score ?? tableRow.mean_score_0_1 ?? 0;
      const threshold = tableRow.success_threshold ?? 0.5;
      const difficulty = (t.difficulty || tableRow.difficulty || "—").toLowerCase();
      const scenarioName = tableRow.scenario_name || "";
      const pct = Math.max(0, Math.min(1, score)) * 100;
      const thresholdPct = Math.max(0, Math.min(1, threshold)) * 100;
      return `
        <article class="task-card">
          <header class="task-card-head">
            <span class="task-id mono">${id}</span>
            <span class="task-difficulty ${difficulty}">${difficulty}</span>
          </header>
          <div class="task-name">${FRIENDLY[id] || id}${scenarioName ? `<br><span style="color:var(--text-3); font-size:11.5px;">${scenarioName}</span>` : ""}</div>
          <div class="task-bar" aria-hidden="true">
            <div class="task-bar-fill" data-pct="${pct}"></div>
            <div class="task-bar-threshold" style="left:${thresholdPct}%"></div>
          </div>
          <div class="task-foot">
            <span>baseline score</span>
            <span><span class="mono">${score.toFixed(2)}</span> / 1.00</span>
          </div>
          <div class="task-foot" style="margin-top:4px;">
            <span>success threshold</span>
            <span class="mono">${threshold.toFixed(2)}</span>
          </div>
        </article>
      `;
    }).join("");

    // Animate fills in next frame
    requestAnimationFrame(() => {
      grid.querySelectorAll(".task-bar-fill").forEach(el => {
        const p = parseFloat(el.dataset.pct);
        el.style.width = p + "%";
      });
    });
  }

  /* ──────────── FIGURE GALLERY ──────────── */
  const FIGURE_ORDER = [
    { file: "fig01_headline_policy_bar_ci.png",      caption: "Mean reward by policy with 95 % CI (merged episodes, n = 120)." },
    { file: "fig05_dumbbell_baseline_to_trained.png",caption: "Per-scenario baseline → trained dumbbell — every scenario improves." },
    { file: "fig06_delta_by_scenario.png",           caption: "Per-scenario uplift (trained − frozen baseline)." },
    { file: "fig04_heatmap_scenario_policy.png",     caption: "Scenario × policy mean reward heatmap." },
    { file: "fig07_ecdf_baseline_vs_trained.png",    caption: "ECDF — stochastic ordering of trained over baseline." },
    { file: "fig08_kde_baseline_vs_trained.png",     caption: "KDE overlay of baseline vs trained reward distributions." },
    { file: "fig02_box_strip.png",                   caption: "Box + strip plot showing per-policy spread and outliers." },
    { file: "fig03_violin_trained_runs.png",         caption: "Trained policy spread across two independent training runs." },
    { file: "fig10_line_mean_by_scenario.png",       caption: "Mean reward by scenario — the benchmark differentiates conditions." },
    { file: "fig11_faceted_by_training_run.png",     caption: "Boxplots faceted by training run × policy (replication check)." },
    { file: "appendix_bootstrap_mean_diff.png",      caption: "Appendix · bootstrap distribution of mean(trained) − mean(baseline)." },
    { file: "fig12_paired_trained_runs.png",         caption: "Appendix · paired y = x plot for trained on matching (scenario, episode)." },
  ];
  function renderFigures() {
    const grid = document.getElementById("figure-grid");
    grid.innerHTML = FIGURE_ORDER.map(f => `
      <figure class="figure-card">
        <img class="figure-img" loading="lazy" src="${PLOTS_BASE}${f.file}" alt="${f.caption}" />
        <figcaption class="figure-cap"><span class="mono">${f.file}</span><br>${f.caption}</figcaption>
      </figure>
    `).join("");
  }

  /* ──────────── boot ──────────── */
  function init() {
    bindHero();
    chartHeadline();
    tableSummary();
    chartGRPO("chart-grpo-reward", "reward", palette.emerald2, "33");
    chartGRPO("chart-grpo-loss",   "loss",   palette.amber,   "1f");
    chartGRPO("chart-grpo-kl",     "kl",     palette.violet,  "1f");
    chartScenarioDelta();
    chartScenarioGrouped();
    tableScenarios();
    bindStats();
    renderTasks();
    renderFigures();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
