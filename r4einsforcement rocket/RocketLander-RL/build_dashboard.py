import os
import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_training_log(log_path):
    episodes = []
    if not os.path.exists(log_path):
        return episodes

    with open(log_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episodes.append({
                    "episode": int(row["episode_number"]),
                    "return": float(row["return"]),
                    "length": int(row["length"]),
                    "outcome": row["outcome"],
                    "vy": float(row["final_vy"]),
                    "vx": float(row["final_vx"]),
                    "tilt": float(row["final_tilt"]),
                    "x": float(row["final_x"]),
                    "fuel": float(row["fuel_left"]),
                    "is_success": 1 if row["outcome"] in ["Perfect landing", "Soft landing"] else 0
                })
            except (ValueError, KeyError):
                continue
    return episodes


def generate_comparison_png(replays_data, training_episodes, output_path="outputs/comparison.png"):
    """
    Export outputs/comparison.png (matplotlib grid: trajectory plots of all milestones + learning curve).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    milestones = replays_data.get("milestones", {})
    if not milestones:
        print("[Comparison PNG] No milestones data to plot.")
        return

    fig = plt.figure(figsize=(16, 10), facecolor="#0a0e17")
    plt.rcParams["text.color"] = "#e2e8f0"
    plt.rcParams["axes.labelcolor"] = "#94a3b8"
    plt.rcParams["xtick.color"] = "#94a3b8"
    plt.rcParams["ytick.color"] = "#94a3b8"

    # Color palette
    colors = ["#ef4444", "#f97316", "#eab308", "#06b6d4", "#3b82f6", "#10b981", "#8b5cf6"]

    # 1. Trajectory (Altitude vs Horizontal Position)
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#0f172a")
    ax1.set_title("Showcase Trajectory (Seed 123)", fontsize=13, fontweight="bold", pad=10)
    ax1.axhline(0, color="#64748b", linestyle="--", linewidth=1.5, label="Ground")
    ax1.plot([-5, 5], [0, 0], color="#22c55e", linewidth=4, label="Pad Target")

    for idx, (m_key, m_val) in enumerate(milestones.items()):
        steps = m_val.get("showcase", {}).get("steps", [])
        if not steps:
            continue
        xs = [s["x"] for s in steps]
        ys = [s["y"] for s in steps]
        c = colors[idx % len(colors)]
        label = f"{m_val.get('label', m_key)} ({m_val['showcase']['outcome']})"
        ax1.plot(xs, ys, color=c, linewidth=2, label=label)
        # Touchdown point
        ax1.scatter([xs[-1]], [ys[-1]], color=c, s=50, edgecolors="#ffffff", zorder=5)

    ax1.set_xlabel("Horizontal Position x (m)")
    ax1.set_ylabel("Altitude y (m)")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#475569")
    ax1.legend(loc="upper right", fontsize=8, facecolor="#1e293b", edgecolor="#334155")

    # 2. Velocity Profile (Vertical Velocity vs Time)
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#0f172a")
    ax2.set_title("Vertical Velocity vs Time", fontsize=13, fontweight="bold", pad=10)
    ax2.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax2.axhspan(-2, 0, color="#10b981", alpha=0.15, label="Soft Safe Band (|vy|<2)")

    for idx, (m_key, m_val) in enumerate(milestones.items()):
        steps = m_val.get("showcase", {}).get("steps", [])
        if not steps:
            continue
        ts = [s["time_sec"] for s in steps]
        vys = [s["vy"] for s in steps]
        c = colors[idx % len(colors)]
        ax2.plot(ts, vys, color=c, linewidth=2, label=m_val.get('label', m_key))
        ax2.scatter([ts[-1]], [vys[-1]], color=c, s=40, zorder=5)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vertical Velocity vy (m/s)")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#475569")
    ax2.legend(loc="lower right", fontsize=8, facecolor="#1e293b", edgecolor="#334155")

    # 3. Learning Curve: Rolling Success Rate
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#0f172a")
    ax3.set_title("Training Learning Curve (Rolling Success Rate)", fontsize=13, fontweight="bold", pad=10)

    if training_episodes:
        ep_nums = [e["episode"] for e in training_episodes]
        successes = [e["is_success"] for e in training_episodes]
        # Compute rolling window of 100
        window = 100
        rolling_sr = []
        for i in range(len(successes)):
            start = max(0, i - window + 1)
            chunk = successes[start:i + 1]
            rolling_sr.append(sum(chunk) / len(chunk) * 100.0)

        ax3.plot(ep_nums, rolling_sr, color="#38bdf8", linewidth=2.5, label=f"Rolling Success Rate ({window} eps)")
        ax3.axhline(90, color="#22c55e", linestyle="--", linewidth=1.5, label="90% Target")

        # Mark milestone markers
        for m_key in milestones.keys():
            if m_key.isdigit():
                m_ep = int(m_key)
                if m_ep <= ep_nums[-1]:
                    ax3.axvline(m_ep, color="#f59e0b", linestyle=":", alpha=0.7)
                    ax3.text(m_ep, 10, f" T{m_ep}", color="#f59e0b", fontsize=8, rotation=90)

    ax3.set_xlabel("Episode (Trial Number)")
    ax3.set_ylabel("Success Rate (%)")
    ax3.set_ylim(-5, 105)
    ax3.grid(True, linestyle=":", alpha=0.3, color="#475569")
    ax3.legend(loc="lower right", fontsize=8, facecolor="#1e293b", edgecolor="#334155")

    # 4. Statistical Evaluation: 100 Random Starts Success & Outcomes
    ax4 = fig.add_subplot(2, 2, 4, facecolor="#0f172a")
    ax4.set_title("Milestone 100-Seed Benchmark Success", fontsize=13, fontweight="bold", pad=10)

    labels = [m_val.get("label", m_key) for m_key, m_val in milestones.items()]
    sr_vals = [m_val.get("eval_100", {}).get("success_rate_pct", 0.0) for m_val in milestones.values()]
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

    bars = ax4.bar(labels, sr_vals, color=bar_colors, width=0.55, edgecolor="#ffffff", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax4.set_ylabel("Success Rate on 100 Seeds (%)")
    ax4.set_ylim(0, 115)
    ax4.grid(True, axis="y", linestyle=":", alpha=0.3, color="#475569")
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout(pad=2.5)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Comparison PNG Generated] -> {output_path}")


def generate_dashboard_html(replays_data, training_episodes, output_path="outputs/dashboard.html"):
    """
    Generate ONE self-contained dashboard.html with embedded JSON, pure vanilla JS and Canvas.
    Zero external CDN links, works completely offline by double-click!
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process training episodes for the learning curve (subsample if large to keep HTML compact)
    episodes_summary = []
    step_skip = max(1, len(training_episodes) // 500)
    for i in range(0, len(training_episodes), step_skip):
        e = training_episodes[i]
        # Compute rolling window of 100
        start = max(0, i - 100 + 1)
        sub = [x["is_success"] for x in training_episodes[start:i+1]]
        sr = sum(sub) / len(sub) * 100.0 if sub else 0.0
        ret_sub = [x["return"] for x in training_episodes[start:i+1]]
        mean_ret = sum(ret_sub) / len(ret_sub) if ret_sub else 0.0
        episodes_summary.append({
            "ep": e["episode"],
            "sr": round(sr, 1),
            "ret": round(mean_ret, 1)
        })

    # Add the final episode if skipped
    if training_episodes and (not episodes_summary or episodes_summary[-1]["ep"] != training_episodes[-1]["episode"]):
        last_e = training_episodes[-1]
        sub = [x["is_success"] for x in training_episodes[max(0, len(training_episodes)-100):]]
        sr = sum(sub) / len(sub) * 100.0 if sub else 0.0
        ret_sub = [x["return"] for x in training_episodes[max(0, len(training_episodes)-100):]]
        mean_ret = sum(ret_sub) / len(ret_sub) if ret_sub else 0.0
        episodes_summary.append({
            "ep": last_e["episode"],
            "sr": round(sr, 1),
            "ret": round(mean_ret, 1)
        })

    embedded_payload = {
        "hero_summary": replays_data.get("hero_summary", ""),
        "milestones": replays_data.get("milestones", {}),
        "learning_curve": episodes_summary
    }
    json_str = json.dumps(embedded_payload)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RocketLander-RL | Autonomous Deep RL Flight Replay Dashboard</title>
  <style>
    :root {{
      --bg: #070b12;
      --card-bg: rgba(15, 23, 42, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --green: #10b981;
      --green-glow: rgba(16, 185, 129, 0.3);
      --red: #ef4444;
      --orange: #f97316;
      --yellow: #f59e0b;
      --purple: #a855f7;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      background: radial-gradient(circle at 50% 0%, #111e38 0%, var(--bg) 70%);
      color: var(--text);
      padding: 24px 20px 60px 20px;
      min-height: 100vh;
      overflow-x: hidden;
    }}
    .container {{
      max-width: 1540px;
      margin: 0 auto;
    }}
    header {{
      text-align: center;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
    }}
    .badge {{
      display: inline-block;
      padding: 4px 12px;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid var(--accent);
      color: var(--accent);
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      margin-bottom: 10px;
    }}
    h1 {{
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 15px;
      max-width: 860px;
      margin: 0 auto;
      line-height: 1.5;
    }}

    /* Section 1: Hero Summary */
    .hero-card {{
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
      border: 1px solid var(--card-border);
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
      border-radius: 16px;
      padding: 22px 28px;
      margin-bottom: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      backdrop-filter: blur(12px);
    }}
    .hero-title-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .hero-story {{
      font-size: 17px;
      font-weight: 600;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .hero-story-arrow {{
      color: var(--text-muted);
      font-size: 14px;
    }}
    .hero-stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .stat-box {{
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 10px;
      padding: 12px 16px;
      text-align: center;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .stat-box:hover {{
      transform: translateY(-2px);
      border-color: var(--accent);
    }}
    .stat-label {{
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }}
    .stat-value {{
      font-size: 20px;
      font-weight: 800;
      color: var(--text);
    }}
    .stat-sub {{
      font-size: 11px;
      margin-top: 2px;
      font-weight: 500;
    }}

    /* Global Section Card */
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 22px;
      margin-bottom: 24px;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .section-title {{
      font-size: 18px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .section-title span.num {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 26px;
      height: 26px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border-radius: 8px;
      font-size: 13px;
      font-weight: 800;
    }}

    /* Controls Bar */
    .controls-bar {{
      display: flex;
      align-items: center;
      gap: 14px;
      background: rgba(0, 0, 0, 0.4);
      padding: 12px 18px;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .btn {{
      background: #1e293b;
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.12);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}
    .btn:hover {{
      background: #334155;
      border-color: var(--accent);
    }}
    .btn.active {{
      background: var(--accent);
      color: #070b12;
      border-color: var(--accent);
      box-shadow: 0 0 12px var(--accent-glow);
    }}
    .timeline-slider-wrapper {{
      flex: 1;
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 220px;
    }}
    .timeline-slider {{
      flex: 1;
      -webkit-appearance: none;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      cursor: pointer;
    }}
    .timeline-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
      cursor: pointer;
      transition: transform 0.1s;
    }}
    .timeline-slider::-webkit-slider-thumb:hover {{
      transform: scale(1.3);
    }}
    .time-badge {{
      font-family: monospace;
      font-size: 13px;
      color: var(--accent);
      font-weight: bold;
      min-width: 125px;
    }}

    /* Section 2: Canvases Grid */
    .canvases-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .canvas-card {{
      background: #020617;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
      box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    .canvas-card-header {{
      background: rgba(15, 23, 42, 0.95);
      padding: 8px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      font-weight: 700;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .canvas-card canvas {{
      width: 100%;
      height: 280px;
      display: block;
      background: #060d19;
    }}
    .canvas-footer {{
      background: rgba(15, 23, 42, 0.95);
      padding: 8px 12px;
      font-size: 11px;
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
      border-top: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* Outcome Badges */
    .outcome-pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }}
    .outcome-perfect {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .outcome-soft {{ background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid #06b6d4; }}
    .outcome-hard {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
    .outcome-crash {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
    .outcome-flipped {{ background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #a855f7; }}
    .outcome-flew {{ background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }}
    .outcome-timeout {{ background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid #64748b; }}

    /* Section 3: Decision Log Table */
    .table-controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .select-input {{
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--card-border);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
      cursor: pointer;
    }}
    .notice-box {{
      font-size: 12px;
      color: #94a3b8;
      background: rgba(255, 255, 255, 0.03);
      padding: 8px 14px;
      border-radius: 8px;
      border-left: 3px solid var(--yellow);
    }}
    .decision-table-wrap {{
      max-height: 380px;
      overflow-y: auto;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      background: rgba(0, 0, 0, 0.25);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #1e293b;
      color: var(--text-muted);
      font-weight: 700;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      z-index: 2;
    }}
    td {{
      padding: 7px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-family: monospace;
    }}
    tr.step-row {{
      cursor: pointer;
      transition: background 0.15s;
    }}
    tr.step-row:hover {{
      background: rgba(56, 189, 248, 0.08);
    }}
    tr.step-row.highlight {{
      background: rgba(56, 189, 248, 0.25) !important;
      font-weight: bold;
    }}
    .exp-cell {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #cbd5e1;
    }}

    /* Section 4: Telemetry Charts */
    .charts-tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .chart-canvas-container {{
      position: relative;
      background: #020617;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px;
      height: 320px;
    }}
    .chart-canvas-container canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .legend-toggles {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      padding: 4px 10px;
      background: rgba(0,0,0,0.3);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 6px;
      cursor: pointer;
      user-select: none;
    }}
    .legend-color {{
      width: 12px;
      height: 4px;
      border-radius: 2px;
    }}

    /* Section 6: Milestone Comparison Table */
    .milestone-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .milestone-table th {{
      background: #1e293b;
      padding: 10px 14px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .milestone-table td {{
      padding: 10px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .stacked-bar-cell {{
      min-width: 180px;
    }}
    .stacked-bar {{
      height: 16px;
      border-radius: 4px;
      overflow: hidden;
      display: flex;
      background: #0f172a;
      border: 1px solid rgba(255,255,255,0.1);
    }}
    .stacked-segment {{
      height: 100%;
      transition: width 0.3s;
    }}

    /* Section 7: How It Works */
    .how-it-works-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .info-subcard {{
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 12px;
      padding: 16px;
    }}
    .info-subcard h3 {{
      font-size: 15px;
      color: var(--accent);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .info-subcard p, .info-subcard li {{
      font-size: 13px;
      line-height: 1.6;
      color: #cbd5e1;
    }}
    .info-subcard ul {{
      padding-left: 20px;
      margin-top: 6px;
    }}
    .code-chip {{
      font-family: monospace;
      background: rgba(56, 189, 248, 0.1);
      color: #38bdf8;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 12px;
    }}

    footer {{
      text-align: center;
      margin-top: 30px;
      color: var(--text-muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <header>
    <div class="badge">Deep Reinforcement Learning (PPO) Mission Control</div>
    <h1>RocketLander-RL Visual Replay Dashboard</h1>
    <p class="subtitle">
      Autonomous rocket landing powered by Proximal Policy Optimization (PPO) with real 2D dynamics,
      thrust-vectoring gimbal control, potential-based reward shaping, and milestone replay telemetry.
    </p>
  </header>

  <!-- SECTION 1: HERO SUMMARY -->
  <div class="hero-card" id="heroSummaryCard">
    <div class="hero-title-row">
      <div class="hero-story" id="heroStoryText">Loading flight trajectory outcomes...</div>
      <div class="badge" style="margin: 0; background: rgba(16, 185, 129, 0.15); border-color: var(--green); color: var(--green);">
        Fixed Start Seed: 123
      </div>
    </div>
    <div class="hero-stats-grid" id="heroStatsGrid">
      <!-- Populated via JS -->
    </div>
  </div>

  <!-- SECTION 2: SIDE-BY-SIDE REPLAY -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">
        <span class="num">2</span>
        Side-by-Side Replay (Simultaneous Multi-Milestone Physics Simulation)
      </div>
      <div style="font-size: 12px; color: var(--text-muted);">
        All models tested on identical initial conditions (Seed 123)
      </div>
    </div>

    <!-- Controls Bar -->
    <div class="controls-bar">
      <button class="btn" id="playPauseBtn" onclick="togglePlay()">
        <svg id="playIcon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
        <span id="playText">Play</span>
      </button>
      <button class="btn" onclick="restartReplay()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
        Restart
      </button>

      <div style="display: flex; gap: 4px;">
        <button class="btn speed-btn" onclick="setSpeed(0.25)">0.25x</button>
        <button class="btn speed-btn" onclick="setSpeed(0.5)">0.5x</button>
        <button class="btn speed-btn active" id="speed1x" onclick="setSpeed(1.0)">1.0x</button>
        <button class="btn speed-btn" onclick="setSpeed(2.0)">2.0x</button>
      </div>

      <div class="timeline-slider-wrapper">
        <input type="range" class="timeline-slider" id="timelineSlider" min="0" max="100" value="0" oninput="onScrub(this.value)">
        <div class="time-badge" id="timeBadge">T+0.00s / 0.00s</div>
      </div>
    </div>

    <!-- Canvases Grid -->
    <div class="canvases-grid" id="canvasesGrid">
      <!-- Generated dynamically for each milestone -->
    </div>
  </div>

  <!-- SECTION 3: DECISION LOG -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">
        <span class="num">3</span>
        Decision Log: "What Did the AI Decide?"
      </div>
      <div class="table-controls">
        <label style="font-size: 13px; color: var(--text-muted);">Inspect Milestone:</label>
        <select class="select-input" id="milestoneSelect" onchange="onMilestoneSelectChange(this.value)">
          <!-- Options populated via JS -->
        </select>
        <label style="font-size: 13px; color: var(--text-muted); margin-left: 10px;">Subsample:</label>
        <select class="select-input" id="stepFilterSelect" onchange="renderDecisionTable()">
          <option value="1">All Steps (50 Hz)</option>
          <option value="2">Every 2nd Step</option>
          <option value="5" selected>Every 5th Step (10 Hz)</option>
          <option value="10">Every 10th Step (5 Hz)</option>
        </select>
      </div>
    </div>

    <div class="notice-box" style="margin-bottom: 12px;">
      <strong>Note:</strong> Explanations are rule-based interpretations of the network's outputs; the network itself does not produce reasons. Click any row to jump the simulation to that timestamp.
    </div>

    <div class="decision-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Time (s)</th>
            <th>Alt (m)</th>
            <th>Vy (m/s)</th>
            <th>Tilt (°)</th>
            <th>Throttle %</th>
            <th>Thrust (N)</th>
            <th>Gimbal (°)</th>
            <th>Reward</th>
            <th>Cumulative</th>
            <th>AI Action Interpretation</th>
          </tr>
        </thead>
        <tbody id="decisionTableBody">
          <!-- Populated via JS -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- SECTION 4: TELEMETRY COMPARISON CHARTS -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">
        <span class="num">4</span>
        Telemetry Comparison Charts (Multi-Milestone Overlay)
      </div>
      <div style="font-size: 12px; color: var(--text-muted);">
        Interactive comparison across all flight attempts
      </div>
    </div>

    <div class="charts-tabs" id="chartTabs">
      <button class="btn active" onclick="switchChart('altitude')">Altitude vs Time</button>
      <button class="btn" onclick="switchChart('vy')">Vertical Velocity (vy)</button>
      <button class="btn" onclick="switchChart('tilt')">Tilt Angle (°)</button>
      <button class="btn" onclick="switchChart('throttle')">Throttle %</button>
      <button class="btn" onclick="switchChart('gimbal')">Gimbal Angle (°)</button>
      <button class="btn" onclick="switchChart('fuel')">Fuel Left (kg)</button>
      <button class="btn" onclick="switchChart('reward')">Cumulative Reward</button>
    </div>

    <div class="chart-canvas-container">
      <canvas id="telemetryChartCanvas"></canvas>
    </div>

    <div class="legend-toggles" id="legendToggles">
      <!-- Generated via JS -->
    </div>
  </div>

  <!-- SECTION 5: LEARNING CURVE -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">
        <span class="num">5</span>
        Learning Curve: Rolling Success Rate & Episode Return
      </div>
      <div style="font-size: 12px; color: var(--text-muted);">
        Target: >= 90% Success Rate across 200 episodes
      </div>
    </div>

    <div class="chart-canvas-container" style="height: 280px;">
      <canvas id="learningCurveCanvas"></canvas>
    </div>
  </div>

  <!-- SECTION 6: MILESTONE COMPARISON TABLE -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">
        <span class="num">6</span>
        Milestone Comparison & 100-Seed Benchmark Table
      </div>
      <div style="font-size: 12px; color: var(--text-muted);">
        Rigorous statistical evaluation on 100 randomized starting states
      </div>
    </div>

    <div style="overflow-x: auto;">
      <table class="milestone-table">
        <thead>
          <tr>
            <th>Milestone</th>
            <th>Showcase Outcome</th>
            <th>100-Seed Success</th>
            <th>Avg |Vy| Landing</th>
            <th>Avg Tilt</th>
            <th>Avg |X| Error</th>
            <th>Avg Fuel Left</th>
            <th>Avg Total Reward</th>
            <th class="stacked-bar-cell">100-Trial Outcome Breakdown</th>
          </tr>
        </thead>
        <tbody id="milestoneTableBody">
          <!-- Populated via JS -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- SECTION 7: HOW IT WORKS -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">
        <span class="num">7</span>
        How It Works: RL Dynamics, Reward Shaping & Equations
      </div>
    </div>

    <div class="how-it-works-grid">
      <div class="info-subcard">
        <h3>1. Agent-Environment Loop</h3>
        <p>
          At every step (dt = 0.02s, 50 Hz physics), the agent observes an 8D normalized state vector
          <span class="code-chip">[x/50, y/100, vx/10, vy/10, sin(θ), cos(θ), ω, fuel/fuel0]</span>.
          The PPO actor-critic network outputs continuous actions in <span class="code-chip">[-1, 1]²</span>:
        </p>
        <ul>
          <li>Throttle: <span class="code-chip">u = (a₀ + 1) / 2 ∈ [0, 1]</span>, mapping to 0 to 20,000 N thrust.</li>
          <li>Gimbal angle: <span class="code-chip">φ = a₁ · 15°</span> vectoring the rocket engine.</li>
        </ul>
      </div>

      <div class="info-subcard">
        <h3>2. Reward & Potential-Based Shaping</h3>
        <p>
          The dense potential function shapes the policy to smoothly guide the rocket towards pad coordinates:
        </p>
        <p style="margin: 6px 0; font-family: monospace; font-size: 12px; color: #38bdf8;">
          Φ(s) = -(dist/10 + speed/2 + 2·|θ|)<br>
          r = γ·Φ(s') - Φ(s) - 0.02·u - 0.01
        </p>
        <ul>
          <li><strong>Touchdown Success:</strong> +100 bonus for landing with <span class="code-chip">|vy|&lt;2 m/s</span>, <span class="code-chip">|tilt|&lt;0.1 rad</span>.</li>
          <li><strong>Crash / Flip / Out of bounds:</strong> -100 severe terminal penalty.</li>
        </ul>
      </div>

      <div class="info-subcard">
        <h3>3. Real Physics Equations</h3>
        <p>
          The rigid body dynamics are governed by Newton-Euler 2D planar equations:
        </p>
        <p style="margin: 6px 0; font-family: monospace; font-size: 11px; color: #a5b4fc;">
          m = m_dry + fuel, &nbsp; I = m·H² / 12<br>
          a_x = -(T/m)·sin(θ + φ)<br>
          a_y = (T/m)·cos(θ + φ) - g<br>
          α = -(L·T·sin(φ)) / I<br>
          dm = T / (I_sp · g₀) · dt
        </p>
        <p>Integrated via semi-implicit Euler to ensure energy preservation.</p>
      </div>
    </div>
  </div>

  <footer>
    RocketLander-RL Autonomous Deep RL Dashboard • Built with PyTorch, Gymnasium, Stable-Baselines3, Vanilla JS & Canvas
  </footer>

</div>

<!-- EMBEDDED FLIGHT DATA -->
<script>
const FLIGHT_DATA = {json_str};

// Global playback state
let currentStep = 0;
let maxSteps = 0;
let isPlaying = false;
let playSpeed = 1.0;
let playInterval = null;
let activeMilestoneKey = null;
let currentChartType = "altitude";
let enabledMilestones = {{}};

const MILESTONE_COLORS = [
  "#ef4444", "#f97316", "#eab308", "#06b6d4", "#3b82f6", "#10b981", "#8b5cf6"
];

function initDashboard() {{
  const milestones = FLIGHT_DATA.milestones || {{}};
  const keys = Object.keys(milestones);
  if (keys.length === 0) return;

  // Determine maximum showcase steps
  maxSteps = 0;
  keys.forEach((k, idx) => {{
    enabledMilestones[k] = true;
    const steps = milestones[k].showcase?.steps || [];
    if (steps.length > maxSteps) maxSteps = steps.length;
  }});

  // Set default active milestone (prefer final, else last)
  activeMilestoneKey = milestones["final"] ? "final" : keys[keys.length - 1];

  // 1. Render Hero Summary
  renderHeroSummary();

  // 2. Setup Replay Canvases
  setupReplayCanvases();

  // 3. Setup Decision Log Select
  setupDecisionLogControls();
  renderDecisionTable();

  // 4. Setup Telemetry Chart & Legend
  setupChartLegend();
  drawTelemetryChart();

  // 5. Draw Learning Curve
  drawLearningCurve();

  // 6. Render Milestone Comparison Table
  renderMilestoneTable();

  // Update slider bounds
  const slider = document.getElementById("timelineSlider");
  slider.max = maxSteps > 0 ? maxSteps - 1 : 0;
  slider.value = 0;

  // Draw initial frame
  drawReplayFrame(0);
}}

function renderHeroSummary() {{
  const storyEl = document.getElementById("heroStoryText");
  storyEl.innerHTML = FLIGHT_DATA.hero_summary || "Multi-Milestone Training Evolution";

  const grid = document.getElementById("heroStatsGrid");
  grid.innerHTML = "";

  const milestones = FLIGHT_DATA.milestones || {{}};
  Object.keys(milestones).forEach((k, idx) => {{
    const m = milestones[k];
    const sc = m.showcase || {{}};
    const ev = m.eval_100 || {{}};
    const color = MILESTONE_COLORS[idx % MILESTONE_COLORS.length];

    const box = document.createElement("div");
    box.className = "stat-box";
    box.innerHTML = `
      <div class="stat-label">${{m.label}}</div>
      <div class="stat-value" style="color: ${{color}}">${{ev.success_rate_pct != null ? ev.success_rate_pct.toFixed(0) + "%" : "--"}}</div>
      <div class="stat-sub" style="color: ${{getOutcomeColor(sc.outcome)}}">${{sc.outcome || "N/A"}}</div>
    `;
    grid.appendChild(box);
  }});
}}

function getOutcomeColor(outcome) {{
  if (!outcome) return "#94a3b8";
  if (outcome.includes("Perfect")) return "#34d399";
  if (outcome.includes("Soft")) return "#22d3ee";
  if (outcome.includes("Hard")) return "#fbbf24";
  if (outcome.includes("Crash")) return "#f87171";
  if (outcome.includes("Flipped")) return "#c084fc";
  if (outcome.includes("Flew")) return "#fb923c";
  return "#94a3b8";
}}

function getOutcomePillClass(outcome) {{
  if (!outcome) return "outcome-timeout";
  if (outcome.includes("Perfect")) return "outcome-perfect";
  if (outcome.includes("Soft")) return "outcome-soft";
  if (outcome.includes("Hard")) return "outcome-hard";
  if (outcome.includes("Crash")) return "outcome-crash";
  if (outcome.includes("Flipped")) return "outcome-flipped";
  if (outcome.includes("Flew")) return "outcome-flew";
  return "outcome-timeout";
}}

function setupReplayCanvases() {{
  const grid = document.getElementById("canvasesGrid");
  grid.innerHTML = "";

  const milestones = FLIGHT_DATA.milestones || {{}};
  Object.keys(milestones).forEach((k, idx) => {{
    const m = milestones[k];
    const sc = m.showcase || {{}};
    const card = document.createElement("div");
    card.className = "canvas-card";
    card.id = "canvasCard_" + k;

    const color = MILESTONE_COLORS[idx % MILESTONE_COLORS.length];

    card.innerHTML = `
      <div class="canvas-card-header">
        <span style="color: ${{color}};">● ${{m.label}}</span>
        <span class="outcome-pill ${{getOutcomePillClass(sc.outcome)}}" id="badge_${{k}}">${{sc.outcome}}</span>
      </div>
      <canvas id="canvas_${{k}}" width="360" height="340"></canvas>
      <div class="canvas-footer" id="footer_${{k}}">
        <span>Alt: 0m | Vy: 0m/s</span>
        <span>Throt: 0%</span>
      </div>
    `;
    grid.appendChild(card);
  }});
}}

function drawReplayFrame(stepIdx) {{
  const milestones = FLIGHT_DATA.milestones || {{}};
  let currentTimeSec = stepIdx * 0.02;
  let maxTimeSec = maxSteps * 0.02;

  document.getElementById("timeBadge").innerText = `T+${{currentTimeSec.toFixed(2)}}s / ${{maxTimeSec.toFixed(2)}}s`;
  document.getElementById("timelineSlider").value = stepIdx;

  Object.keys(milestones).forEach((k, idx) => {{
    const m = milestones[k];
    const steps = m.showcase?.steps || [];
    const canvas = document.getElementById("canvas_" + k);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    // Pick active step for this milestone (clamp if ended)
    const curStep = steps.length > 0 ? (stepIdx < steps.length ? steps[stepIdx] : steps[steps.length - 1]) : null;
    const hasEnded = stepIdx >= steps.length;

    // Clear background: space twilight gradient
    const sky = ctx.createLinearGradient(0, 0, 0, H);
    sky.addColorStop(0, "#030712");
    sky.addColorStop(0.7, "#0f172a");
    sky.addColorStop(1, "#1e293b");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, H);

    // Stars background
    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
    for (let s = 0; s < 15; s++) {{
      let sx = ((s * 47 + 13) % W);
      let sy = ((s * 31 + 7) % (H * 0.7));
      ctx.fillRect(sx, sy, 1.5, 1.5);
    }}

    // World coordinate transformation
    // World bounds: x from -35 to +35 m, y from -5 to +140 m
    const worldW = 70.0;
    const worldH = 145.0;
    const scale = Math.min(W / worldW, (H - 40) / worldH);
    const originX = W / 2.0;
    const originY = H - 35.0;

    function toScreen(wx, wy) {{
      return {{
        x: originX + wx * scale,
        y: originY - wy * scale
      }};
    }}

    // Draw Ground
    const groundP1 = toScreen(-40, 0);
    const groundP2 = toScreen(40, 0);
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, groundP1.y);
    ctx.lineTo(W, groundP1.y);
    ctx.stroke();

    // Ground fill
    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, groundP1.y, W, H - groundP1.y);

    // Draw Landing Pad (x: -5 to +5 m)
    const padP1 = toScreen(-5.0, 0);
    const padP2 = toScreen(5.0, 0);
    const padW = padP2.x - padP1.x;

    // Pad concrete slab
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(padP1.x, padP1.y, padW, 8);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.strokeRect(padP1.x, padP1.y, padW, 8);

    // Target crosshair / pad marking
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(originX - 10, padP1.y + 4);
    ctx.lineTo(originX + 10, padP1.y + 4);
    ctx.moveTo(originX, padP1.y - 4);
    ctx.lineTo(originX, padP1.y + 12);
    ctx.stroke();

    // Trajectory fading trail
    if (steps.length > 0) {{
      const limit = Math.min(stepIdx, steps.length - 1);
      ctx.beginPath();
      for (let i = 0; i <= limit; i++) {{
        const pt = toScreen(steps[i].x, steps[i].y);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }}
      ctx.strokeStyle = MILESTONE_COLORS[idx % MILESTONE_COLORS.length];
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    // Draw Rocket if current step exists
    if (curStep) {{
      const pos = toScreen(curStep.x, curStep.y);
      const theta = curStep.theta_rad; // angle in rad, 0 = upright
      const rocketH = 10.0 * scale; // rocket height in pixels
      const rocketW = 2.2 * scale; // rocket width in pixels

      ctx.save();
      ctx.translate(pos.x, pos.y);
      // In canvas, y points down. Positive theta tilts rocket clockwise.
      ctx.rotate(-theta);

      // Draw Engine Exhaust Flame if throttle > 0 and not ended
      if (curStep.throttle_pct > 0 && !hasEnded && curStep.fuel_kg > 0) {{
        const throt = curStep.throttle_pct / 100.0;
        const gimbal = curStep.gimbal_deg * Math.PI / 180.0;
        const flameLen = (12.0 + throt * 26.0) * (scale / 4.0);
        const flameW = (2.0 + throt * 4.0) * (scale / 4.0);

        ctx.save();
        ctx.translate(0, rocketH / 2.0);
        ctx.rotate(-gimbal); // rotate flame by gimbal angle

        // Outer Flame (orange/yellow)
        const flameGrad = ctx.createLinearGradient(0, 0, 0, flameLen);
        flameGrad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
        flameGrad.addColorStop(0.2, "rgba(255, 200, 50, 0.9)");
        flameGrad.addColorStop(0.6, "rgba(255, 100, 20, 0.8)");
        flameGrad.addColorStop(1, "rgba(255, 50, 0, 0)");

        ctx.fillStyle = flameGrad;
        ctx.beginPath();
        ctx.moveTo(-flameW, 0);
        ctx.lineTo(flameW, 0);
        ctx.lineTo(0, flameLen + (Math.random() * 4 - 2));
        ctx.closePath();
        ctx.fill();

        // Inner Core Flame (plasma blue)
        ctx.fillStyle = "rgba(56, 189, 248, 0.85)";
        ctx.beginPath();
        ctx.moveTo(-flameW * 0.4, 0);
        ctx.lineTo(flameW * 0.4, 0);
        ctx.lineTo(0, flameLen * 0.45);
        ctx.closePath();
        ctx.fill();

        ctx.restore();
      }}

      // Rocket Fuselage (Falcon 9 aesthetic)
      const grad = ctx.createLinearGradient(-rocketW / 2, 0, rocketW / 2, 0);
      grad.addColorStop(0, "#cbd5e1");
      grad.addColorStop(0.5, "#ffffff");
      grad.addColorStop(1, "#94a3b8");

      ctx.fillStyle = grad;
      ctx.fillRect(-rocketW / 2, -rocketH / 2, rocketW, rocketH);
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1;
      ctx.strokeRect(-rocketW / 2, -rocketH / 2, rocketW, rocketH);

      // Interstage / nose cone
      ctx.fillStyle = "#0f172a";
      ctx.beginPath();
      ctx.moveTo(-rocketW / 2, -rocketH / 2);
      ctx.lineTo(rocketW / 2, -rocketH / 2);
      ctx.lineTo(0, -rocketH / 2 - rocketW * 0.8);
      ctx.closePath();
      ctx.fill();

      // Engine Nozzle at base
      ctx.fillStyle = "#1e293b";
      ctx.fillRect(-rocketW * 0.35, rocketH / 2 - 2, rocketW * 0.7, 4);

      // Deployable Landing Legs
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      // Left leg
      ctx.moveTo(-rocketW / 2, rocketH / 2 - 4);
      ctx.lineTo(-rocketW * 1.1, rocketH / 2 + 5);
      // Right leg
      ctx.moveTo(rocketW / 2, rocketH / 2 - 4);
      ctx.lineTo(rocketW * 1.1, rocketH / 2 + 5);
      ctx.stroke();

      ctx.restore();

      // End-of-flight Big Outcome Badge on canvas when finished
      if (hasEnded) {{
        const outcome = m.showcase?.outcome || "Ended";
        ctx.save();
        ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
        ctx.fillRect(W / 2 - 75, H / 2 - 22, 150, 44);
        ctx.strokeStyle = getOutcomeColor(outcome);
        ctx.lineWidth = 2;
        ctx.strokeRect(W / 2 - 75, H / 2 - 22, 150, 44);

        ctx.fillStyle = getOutcomeColor(outcome);
        ctx.font = "bold 13px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(outcome.toUpperCase(), W / 2, H / 2 + 5);
        ctx.restore();
      }}

      // Canvas footer HUD telemetry
      const footer = document.getElementById("footer_" + k);
      if (footer) {{
        footer.innerHTML = `
          <span>Alt: ${{curStep.y.toFixed(1)}}m | Vy: ${{curStep.vy.toFixed(1)}}m/s | Tilt: ${{curStep.theta_deg.toFixed(1)}}°</span>
          <span style="color: ${{curStep.throttle_pct > 50 ? '#f97316' : '#38bdf8'}}">Throt: ${{curStep.throttle_pct.toFixed(0)}}%</span>
        `;
      }}
    }}
  }});

  // Update Decision Log active highlight
  highlightDecisionTableRow(stepIdx);
}}

function togglePlay() {{
  isPlaying = !isPlaying;
  const btn = document.getElementById("playPauseBtn");
  const icon = document.getElementById("playIcon");
  const text = document.getElementById("playText");

  if (isPlaying) {{
    btn.classList.add("active");
    text.innerText = "Pause";
    icon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
    startPlaybackLoop();
  }} else {{
    btn.classList.remove("active");
    text.innerText = "Play";
    icon.innerHTML = '<path d="M8 5v14l11-7z"/>';
    clearInterval(playInterval);
  }}
}}

function startPlaybackLoop() {{
  clearInterval(playInterval);
  // Base dt is 0.02s (50 Hz). Interval ms = 20 / playSpeed
  const intervalMs = Math.max(10, Math.round(20 / playSpeed));
  playInterval = setInterval(() => {{
    if (currentStep >= maxSteps - 1) {{
      togglePlay(); // Pause at end
      return;
    }}
    currentStep++;
    drawReplayFrame(currentStep);
  }}, intervalMs);
}}

function restartReplay() {{
  currentStep = 0;
  drawReplayFrame(0);
  if (!isPlaying) togglePlay();
}}

function setSpeed(speed) {{
  playSpeed = speed;
  document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
  event.target.classList.add("active");
  if (isPlaying) startPlaybackLoop();
}}

function onScrub(val) {{
  currentStep = parseInt(val);
  drawReplayFrame(currentStep);
}}

// Section 3: Decision Log Logic
function setupDecisionLogControls() {{
  const select = document.getElementById("milestoneSelect");
  select.innerHTML = "";
  const milestones = FLIGHT_DATA.milestones || {{}};
  Object.keys(milestones).forEach(k => {{
    const opt = document.createElement("option");
    opt.value = k;
    opt.innerText = `${{milestones[k].label}} (${{milestones[k].showcase?.outcome || 'Replay'}})`;
    if (k === activeMilestoneKey) opt.selected = true;
    select.appendChild(opt);
  }});
}}

function onMilestoneSelectChange(key) {{
  activeMilestoneKey = key;
  renderDecisionTable();
}}

function renderDecisionTable() {{
  const tbody = document.getElementById("decisionTableBody");
  tbody.innerHTML = "";
  const m = FLIGHT_DATA.milestones?.[activeMilestoneKey];
  if (!m || !m.showcase?.steps) return;

  const steps = m.showcase.steps;
  const stepMod = parseInt(document.getElementById("stepFilterSelect").value) || 1;

  steps.forEach((s, idx) => {{
    if (idx % stepMod !== 0 && idx !== steps.length - 1) return;
    const tr = document.createElement("tr");
    tr.className = "step-row";
    tr.id = "stepRow_" + s.t;
    tr.onclick = () => {{
      currentStep = idx;
      drawReplayFrame(idx);
    }};

    tr.innerHTML = `
      <td>${{s.time_sec.toFixed(2)}}s</td>
      <td>${{s.y.toFixed(1)}}</td>
      <td style="color: ${{s.vy < -2 ? '#f87171' : '#34d399'}}">${{s.vy.toFixed(2)}}</td>
      <td style="color: ${{Math.abs(s.theta_deg) > 4 ? '#fbbf24' : '#cbd5e1'}}">${{s.theta_deg.toFixed(1)}}°</td>
      <td>${{s.throttle_pct.toFixed(0)}}%</td>
      <td>${{s.thrust_N.toFixed(0)}}</td>
      <td>${{s.gimbal_deg.toFixed(1)}}°</td>
      <td>${{s.step_reward.toFixed(2)}}</td>
      <td>${{s.cumulative_reward.toFixed(1)}}</td>
      <td class="exp-cell">${{s.explanation}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

function highlightDecisionTableRow(stepIdx) {{
  const m = FLIGHT_DATA.milestones?.[activeMilestoneKey];
  if (!m || !m.showcase?.steps) return;
  const steps = m.showcase.steps;
  const activeS = steps[stepIdx < steps.length ? stepIdx : steps.length - 1];
  if (!activeS) return;

  // Clear past highlights
  document.querySelectorAll("tr.step-row.highlight").forEach(r => r.classList.remove("highlight"));

  // Find exact or closest preceding row
  const row = document.getElementById("stepRow_" + activeS.t);
  if (row) {{
    row.classList.add("highlight");
    row.scrollIntoView({{ block: "nearest", behavior: "smooth" }});
  }}
}}

// Section 4: Telemetry Charts
function setupChartLegend() {{
  const wrap = document.getElementById("legendToggles");
  wrap.innerHTML = "";
  const milestones = FLIGHT_DATA.milestones || {{}};

  Object.keys(milestones).forEach((k, idx) => {{
    const m = milestones[k];
    const color = MILESTONE_COLORS[idx % MILESTONE_COLORS.length];
    const item = document.createElement("div");
    item.className = "legend-item";
    item.id = "legendItem_" + k;
    item.innerHTML = `
      <div class="legend-color" style="background: ${{color}}"></div>
      <span>${{m.label}}</span>
    `;
    item.onclick = () => {{
      enabledMilestones[k] = !enabledMilestones[k];
      item.style.opacity = enabledMilestones[k] ? "1" : "0.35";
      drawTelemetryChart();
    }};
    wrap.appendChild(item);
  }});
}}

function switchChart(type) {{
  currentChartType = type;
  document.querySelectorAll("#chartTabs .btn").forEach(b => b.classList.remove("active"));
  event.target.classList.add("active");
  drawTelemetryChart();
}}

function drawTelemetryChart() {{
  const canvas = document.getElementById("telemetryChartCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  const W = canvas.width;
  const H = canvas.height;

  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, W, H);

  const padLeft = 70;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;
  const plotW = W - padLeft - padRight;
  const plotH = H - padTop - padBottom;

  const milestones = FLIGHT_DATA.milestones || {{}};
  let maxT = 0.1;
  let minY = Infinity;
  let maxY = -Infinity;

  // Compute domains
  Object.keys(milestones).forEach(k => {{
    if (!enabledMilestones[k]) return;
    const steps = milestones[k].showcase?.steps || [];
    steps.forEach(s => {{
      if (s.time_sec > maxT) maxT = s.time_sec;
      let val = 0;
      if (currentChartType === "altitude") val = s.y;
      else if (currentChartType === "vy") val = s.vy;
      else if (currentChartType === "tilt") val = s.theta_deg;
      else if (currentChartType === "throttle") val = s.throttle_pct;
      else if (currentChartType === "gimbal") val = s.gimbal_deg;
      else if (currentChartType === "fuel") val = s.fuel_kg;
      else if (currentChartType === "reward") val = s.cumulative_reward;

      if (val < minY) minY = val;
      if (val > maxY) maxY = val;
    }});
  }});

  if (minY === Infinity) {{ minY = 0; maxY = 100; }}
  if (maxY === minY) {{ maxY += 10; minY -= 10; }}
  const ySpan = maxY - minY;

  // Grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.font = `${{11 * window.devicePixelRatio}}px sans-serif`;
  ctx.fillStyle = "#64748b";
  ctx.textAlign = "right";

  for (let g = 0; g <= 5; g++) {{
    const gy = padTop + plotH * (1 - g / 5.0);
    const gVal = minY + ySpan * (g / 5.0);
    ctx.beginPath();
    ctx.moveTo(padLeft, gy);
    ctx.lineTo(W - padRight, gy);
    ctx.stroke();
    ctx.fillText(gVal.toFixed(1), padLeft - 10, gy + 4);
  }}

  // Zero reference line if within bounds
  if (minY <= 0 && maxY >= 0) {{
    const zeroY = padTop + plotH * (1 - (0 - minY) / ySpan);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, zeroY);
    ctx.lineTo(W - padRight, zeroY);
    ctx.stroke();
  }}

  // Plot Curves
  Object.keys(milestones).forEach((k, idx) => {{
    if (!enabledMilestones[k]) return;
    const steps = milestones[k].showcase?.steps || [];
    if (steps.length === 0) return;

    const color = MILESTONE_COLORS[idx % MILESTONE_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * window.devicePixelRatio;
    ctx.beginPath();

    steps.forEach((s, sIdx) => {{
      let val = 0;
      if (currentChartType === "altitude") val = s.y;
      else if (currentChartType === "vy") val = s.vy;
      else if (currentChartType === "tilt") val = s.theta_deg;
      else if (currentChartType === "throttle") val = s.throttle_pct;
      else if (currentChartType === "gimbal") val = s.gimbal_deg;
      else if (currentChartType === "fuel") val = s.fuel_kg;
      else if (currentChartType === "reward") val = s.cumulative_reward;

      const px = padLeft + (s.time_sec / maxT) * plotW;
      const py = padTop + plotH * (1 - (val - minY) / ySpan);

      if (sIdx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }});
    ctx.stroke();

    // Mark Touchdown Point with dot
    const lastS = steps[steps.length - 1];
    let lastVal = 0;
    if (currentChartType === "altitude") lastVal = lastS.y;
    else if (currentChartType === "vy") lastVal = lastS.vy;
    else if (currentChartType === "tilt") lastVal = lastS.theta_deg;
    else if (currentChartType === "throttle") lastVal = lastS.throttle_pct;
    else if (currentChartType === "gimbal") lastVal = lastS.gimbal_deg;
    else if (currentChartType === "fuel") lastVal = lastS.fuel_kg;
    else if (currentChartType === "reward") lastVal = lastS.cumulative_reward;

    const endX = padLeft + (lastS.time_sec / maxT) * plotW;
    const endY = padTop + plotH * (1 - (lastVal - minY) / ySpan);

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(endX, endY, 5 * window.devicePixelRatio, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }});

  // X Axis labels
  ctx.textAlign = "center";
  ctx.fillStyle = "#64748b";
  for (let t = 0; t <= 5; t++) {{
    const tx = padLeft + plotW * (t / 5.0);
    const tVal = (maxT * (t / 5.0)).toFixed(1);
    ctx.fillText(`${{tVal}}s`, tx, H - padBottom + 20);
  }}
}}

// Section 5: Learning Curve Chart
function drawLearningCurve() {{
  const canvas = document.getElementById("learningCurveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  const W = canvas.width;
  const H = canvas.height;

  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, W, H);

  const data = FLIGHT_DATA.learning_curve || [];
  if (data.length === 0) return;

  const padLeft = 60;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;
  const plotW = W - padLeft - padRight;
  const plotH = H - padTop - padBottom;

  const maxEp = data[data.length - 1].ep;

  // Grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.font = `${{11 * window.devicePixelRatio}}px sans-serif`;
  ctx.fillStyle = "#64748b";
  ctx.textAlign = "right";

  for (let pct = 0; pct <= 100; pct += 25) {{
    const py = padTop + plotH * (1 - pct / 100.0);
    ctx.beginPath();
    ctx.moveTo(padLeft, py);
    ctx.lineTo(W - padRight, py);
    ctx.stroke();
    ctx.fillText(pct + "%", padLeft - 10, py + 4);
  }}

  // 90% Success Target Line
  const y90 = padTop + plotH * (1 - 0.9);
  ctx.strokeStyle = "rgba(16, 185, 129, 0.6)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(padLeft, y90);
  ctx.lineTo(W - padRight, y90);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#10b981";
  ctx.textAlign = "left";
  ctx.fillText("90% Target", W - padRight - 75, y90 - 6);

  // Plot Rolling Success Rate
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5 * window.devicePixelRatio;
  ctx.beginPath();
  data.forEach((pt, idx) => {{
    const px = padLeft + (pt.ep / maxEp) * plotW;
    const py = padTop + plotH * (1 - pt.sr / 100.0);
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }});
  ctx.stroke();

  // Milestone vertical markers
  const milestones = FLIGHT_DATA.milestones || {{}};
  Object.keys(milestones).forEach(k => {{
    if (!k.match(/^\\d+$/)) return;
    const ep = parseInt(k);
    if (ep <= maxEp) {{
      const mx = padLeft + (ep / maxEp) * plotW;
      ctx.strokeStyle = "rgba(245, 158, 11, 0.5)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(mx, padTop);
      ctx.lineTo(mx, padTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#f59e0b";
      ctx.textAlign = "center";
      ctx.fillText(`T${{ep}}`, mx, padTop - 8);
    }}
  }});

  // X axis labels
  ctx.fillStyle = "#64748b";
  ctx.textAlign = "center";
  for (let t = 0; t <= 5; t++) {{
    const tx = padLeft + plotW * (t / 5.0);
    const epVal = Math.round(maxEp * (t / 5.0));
    ctx.fillText(`Ep ${{epVal}}`, tx, H - padBottom + 20);
  }}
}}

// Section 6: Milestone Comparison Table
function renderMilestoneTable() {{
  const tbody = document.getElementById("milestoneTableBody");
  tbody.innerHTML = "";
  const milestones = FLIGHT_DATA.milestones || {{}};

  Object.keys(milestones).forEach((k, idx) => {{
    const m = milestones[k];
    const sc = m.showcase || {{}};
    const ev = m.eval_100 || {{}};
    const color = MILESTONE_COLORS[idx % MILESTONE_COLORS.length];
    const oc = ev.outcomes || {{}};

    // Stacked bar segments
    const perfectPct = (oc["Perfect landing"] || 0);
    const softPct = (oc["Soft landing"] || 0);
    const hardPct = (oc["Hard landing"] || 0);
    const crashPct = (oc["Crash"] || 0);
    const flippedPct = (oc["Flipped"] || 0);
    const flewPct = (oc["Flew away"] || 0);
    const timeoutPct = (oc["Timeout"] || 0);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-weight: 700; color: ${{color}};">● ${{m.label}}</td>
      <td><span class="outcome-pill ${{getOutcomePillClass(sc.outcome)}}">${{sc.outcome}}</span></td>
      <td style="font-weight: 800; font-size: 14px; color: ${{ev.success_rate_pct >= 90 ? '#34d399' : (ev.success_rate_pct >= 50 ? '#fbbf24' : '#f87171')}};">
        ${{ev.success_rate_pct != null ? ev.success_rate_pct.toFixed(1) + "%" : "--"}}
      </td>
      <td>${{ev.avg_vy != null ? Math.abs(ev.avg_vy).toFixed(2) + " m/s" : "--"}}</td>
      <td>${{ev.avg_tilt_deg != null ? ev.avg_tilt_deg.toFixed(1) + "°" : "--"}}</td>
      <td>${{ev.avg_x_error_m != null ? ev.avg_x_error_m.toFixed(2) + " m" : "--"}}</td>
      <td>${{ev.avg_fuel_left_kg != null ? ev.avg_fuel_left_kg.toFixed(1) + " kg" : "--"}}</td>
      <td>${{ev.avg_reward != null ? ev.avg_reward.toFixed(1) : "--"}}</td>
      <td class="stacked-bar-cell">
        <div class="stacked-bar" title="Perfect: ${{perfectPct}} | Soft: ${{softPct}} | Hard: ${{hardPct}} | Crash: ${{crashPct}} | Flip: ${{flippedPct}} | Flew: ${{flewPct}}">
          <div class="stacked-segment" style="width: ${{perfectPct}}%; background: #10b981;"></div>
          <div class="stacked-segment" style="width: ${{softPct}}%; background: #06b6d4;"></div>
          <div class="stacked-segment" style="width: ${{hardPct}}%; background: #f59e0b;"></div>
          <div class="stacked-segment" style="width: ${{crashPct}}%; background: #ef4444;"></div>
          <div class="stacked-segment" style="width: ${{flippedPct}}%; background: #a855f7;"></div>
          <div class="stacked-segment" style="width: ${{flewPct}}%; background: #f97316;"></div>
          <div class="stacked-segment" style="width: ${{timeoutPct}}%; background: #64748b;"></div>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  }});
}}

window.addEventListener("DOMContentLoaded", initDashboard);
window.addEventListener("resize", () => {{
  drawReplayFrame(currentStep);
  drawTelemetryChart();
  drawLearningCurve();
}});
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[Dashboard Generated] Self-contained dashboard saved -> {output_path}")


def build_all(replays_path="outputs/replays.json", log_path="outputs/training_log.csv"):
    if not os.path.exists(replays_path):
        print(f"Error: {replays_path} not found! Run record_replays.py first.")
        return

    with open(replays_path, "r", encoding="utf-8") as f:
        replays_data = json.load(f)

    training_episodes = load_training_log(log_path)
    print(f"Loaded {len(training_episodes)} training episodes from {log_path}")

    # Generate comparison.png
    generate_comparison_png(replays_data, training_episodes, output_path="outputs/comparison.png")

    # Generate dashboard.html
    generate_dashboard_html(replays_data, training_episodes, output_path="outputs/dashboard.html")

    print("\nAll deliverables generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline dashboard.html and comparison.png from replays and training logs")
    parser.add_argument("--replays", default="outputs/replays.json", help="Path to replays.json")
    parser.add_argument("--log", default="outputs/training_log.csv", help="Path to training_log.csv")
    args = parser.parse_args()

    build_all(replays_path=args.replays, log_path=args.log)
