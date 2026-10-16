import os
import json

def build_flight_simulator(replays_path="outputs/replays.json",
                           weights_path="outputs/model_weights.json",
                           output_path="outputs/flight_simulator.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    replays_data = {}
    if os.path.exists(replays_path):
        with open(replays_path, "r", encoding="utf-8") as f:
            replays_data = json.load(f)

    weights_data = {}
    if os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            weights_data = json.load(f)

    replays_json_str = json.dumps(replays_data)
    weights_json_str = json.dumps(weights_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RocketLander-RL | Cinema Flight Simulator & Comparison Arena</title>
  <style>
    :root {{
      --bg: #050811;
      --card-bg: rgba(13, 20, 38, 0.82);
      --card-border: rgba(56, 189, 248, 0.18);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.35);
      --green: #10b981;
      --green-glow: rgba(16, 185, 129, 0.4);
      --red: #ef4444;
      --red-glow: rgba(239, 68, 68, 0.4);
      --orange: #f97316;
      --yellow: #f59e0b;
      --purple: #c084fc;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      background: radial-gradient(circle at 50% 0%, #0d1a33 0%, var(--bg) 75%);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
      padding: 16px 20px 40px 20px;
    }}
    .container {{
      max-width: 1680px;
      margin: 0 auto;
    }}
    /* Top Header Bar */
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 22px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      backdrop-filter: blur(12px);
      margin-bottom: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-icon {{
      font-size: 26px;
      filter: drop-shadow(0 0 10px var(--accent));
    }}
    .brand-title {{
      font-size: 20px;
      font-weight: 800;
      letter-spacing: 0.5px;
      background: linear-gradient(135deg, #ffffff 40%, var(--accent) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .brand-sub {{
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    .nav-tabs {{
      display: flex;
      background: rgba(0,0,0,0.4);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.08);
      gap: 4px;
    }}
    .tab-btn {{
      padding: 8px 18px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .tab-btn:hover {{
      color: #ffffff;
      background: rgba(255,255,255,0.06);
    }}
    .tab-btn.active {{
      background: var(--accent);
      color: #030712;
      font-weight: 700;
      box-shadow: 0 0 16px var(--accent-glow);
    }}
    .header-links {{
      display: flex;
      gap: 12px;
    }}
    .link-btn {{
      padding: 8px 14px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 8px;
      color: var(--accent);
      text-decoration: none;
      font-size: 12px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}
    .link-btn:hover {{
      background: rgba(56, 189, 248, 0.15);
      border-color: var(--accent);
      transform: translateY(-1px);
    }}

    /* Global Control Bar */
    .control-deck {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 12px 20px;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 18px;
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .btn-group {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .play-btn {{
      background: linear-gradient(135deg, #0284c7, #38bdf8);
      color: #020617;
      border: none;
      font-weight: 800;
      font-size: 14px;
      padding: 9px 22px;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 0 16px var(--accent-glow);
      transition: all 0.2s;
    }}
    .play-btn:hover {{
      filter: brightness(1.15);
      transform: scale(1.02);
    }}
    .sec-btn {{
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .sec-btn:hover {{
      background: rgba(255,255,255,0.15);
    }}
    .scrub-container {{
      flex: 1;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .scrubber {{
      flex: 1;
      accent-color: var(--accent);
      height: 6px;
      cursor: pointer;
    }}
    .time-badge {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      font-weight: 700;
      color: var(--accent);
      min-width: 140px;
    }}
    .speed-btn-group {{
      display: flex;
      background: rgba(0,0,0,0.4);
      padding: 3px;
      border-radius: 8px;
      gap: 2px;
    }}
    .speed-pill {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 700;
      padding: 4px 8px;
      border-radius: 5px;
      cursor: pointer;
    }}
    .speed-pill.active {{
      background: var(--accent);
      color: #030712;
    }}

    /* View Modes */
    .view-section {{
      display: none;
    }}
    .view-section.active {{
      display: block;
    }}

    /* DUAL COMPARISON ARENA */
    .dual-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .sim-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 12px 36px rgba(0,0,0,0.45);
      position: relative;
    }}
    .card-top {{
      padding: 12px 18px;
      background: rgba(0,0,0,0.35);
      border-bottom: 1px solid rgba(255,255,255,0.06);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .selector-wrapper {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .slot-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      font-weight: 800;
      color: var(--accent);
    }}
    .milestone-select {{
      background: #0f172a;
      color: #ffffff;
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 13px;
      font-weight: 700;
      outline: none;
      cursor: pointer;
    }}
    .milestone-select:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
    }}
    .outcome-pill {{
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.4px;
      text-transform: uppercase;
    }}
    .outcome-success {{
      background: rgba(16, 185, 129, 0.2);
      border: 1px solid var(--green);
      color: #34d399;
      box-shadow: 0 0 12px var(--green-glow);
    }}
    .outcome-crash {{
      background: rgba(239, 68, 68, 0.2);
      border: 1px solid var(--red);
      color: #f87171;
      box-shadow: 0 0 12px var(--red-glow);
    }}
    .canvas-wrapper {{
      position: relative;
      width: 100%;
      height: 520px;
      background: #030712;
      overflow: hidden;
    }}
    canvas.sim-canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}

    /* Overlay Flight Cockpit HUD */
    .cockpit-hud {{
      position: absolute;
      top: 14px;
      left: 14px;
      background: rgba(10, 15, 30, 0.75);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px;
      padding: 8px 12px;
      backdrop-filter: blur(8px);
      display: flex;
      flex-direction: column;
      gap: 4px;
      pointer-events: none;
    }}
    .hud-row {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      font-size: 11px;
    }}
    .hud-label {{
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }}
    .hud-val {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-weight: 800;
      color: #ffffff;
    }}
    .hud-safe {{
      color: #34d399;
    }}
    .hud-warn {{
      color: #f87171;
    }}

    /* Bottom Telemetry Strip */
    .telemetry-strip {{
      padding: 12px 16px;
      background: rgba(0,0,0,0.4);
      border-top: 1px solid rgba(255,255,255,0.06);
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 8px;
      text-align: center;
    }}
    .stat-box {{
      background: rgba(255,255,255,0.03);
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.05);
    }}
    .stat-label {{
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 700;
    }}
    .stat-number {{
      font-size: 15px;
      font-weight: 800;
      color: var(--text);
      font-family: ui-monospace, monospace;
      margin-top: 2px;
    }}

    /* CINEMA SOLO VIEW */
    .cinema-layout {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .milestone-strip-bar {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .m-pill {{
      padding: 8px 16px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
      white-space: nowrap;
    }}
    .m-pill:hover {{
      color: #ffffff;
      border-color: var(--accent);
    }}
    .m-pill.active {{
      background: rgba(56, 189, 248, 0.15);
      border-color: var(--accent);
      color: var(--accent);
      box-shadow: 0 0 12px var(--accent-glow);
    }}
    .cinema-canvas-container {{
      position: relative;
      width: 100%;
      height: 640px;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--card-border);
      background: #030712;
      box-shadow: 0 16px 48px rgba(0,0,0,0.6);
    }}

    /* LIVE SANDBOX VIEW */
    .sandbox-grid {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
    }}
    .sandbox-panel {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      backdrop-filter: blur(12px);
    }}
    .panel-title {{
      font-size: 16px;
      font-weight: 800;
      color: var(--accent);
      border-bottom: 1px solid rgba(255,255,255,0.08);
      padding-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .slider-row {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .slider-header {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }}
    .slider-val {{
      color: #ffffff;
      font-family: ui-monospace, monospace;
      font-weight: 700;
    }}
    .slider-input {{
      accent-color: var(--accent);
      cursor: pointer;
    }}
    .launch-btn {{
      background: linear-gradient(135deg, #10b981, #059669);
      color: #ffffff;
      border: none;
      font-size: 15px;
      font-weight: 800;
      padding: 12px;
      border-radius: 10px;
      cursor: pointer;
      box-shadow: 0 0 20px var(--green-glow);
      transition: all 0.2s;
    }}
    .launch-btn:hover {{
      filter: brightness(1.15);
      transform: translateY(-1px);
    }}
    .manual-toggle {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
    }}
    .manual-toggle input {{
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }}
  </style>
</head>
<body>
<div class="container">

  <!-- Top Header Navigation -->
  <header>
    <div class="brand">
      <span class="brand-icon">🚀</span>
      <div>
        <div class="brand-title">RocketLander-RL Cinema & Arena</div>
        <div class="brand-sub">Autonomous Falcon-9 Guidance & Dual Milestone Comparison</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
      <button class="tab-btn active" id="tab-dual" onclick="switchView('dual')">
        <span>⚔️</span> Dual Comparison Arena
      </button>
      <button class="tab-btn" id="tab-cinema" onclick="switchView('cinema')">
        <span>🎬</span> Cinema Solo View
      </button>
      <button class="tab-btn" id="tab-sandbox" onclick="switchView('sandbox')">
        <span>🕹️</span> Live Interactive Sandbox
      </button>
    </div>

    <!-- Header Links -->
    <div class="header-links">
      <a href="dashboard.html" class="link-btn">
        <span>📊</span> Multi-Milestone Dashboard
      </a>
    </div>
  </header>

  <!-- Global Video / Flight Controls -->
  <div class="control-deck">
    <div class="btn-group">
      <button class="play-btn" id="master-play-btn" onclick="togglePlay()">
        <span id="play-icon">▶</span> <span id="play-text">PLAY</span>
      </button>
      <button class="sec-btn" onclick="restartReplay()">
        <span>↺</span> RESTART
      </button>
    </div>

    <div class="scrub-container">
      <input type="range" class="scrubber" id="master-scrubber" min="0" max="1000" value="0" oninput="onScrub(this.value)">
      <div class="time-badge" id="master-time-badge">T+0.00s | Step 0</div>
    </div>

    <div class="speed-btn-group">
      <button class="speed-pill" onclick="setSpeed(0.25)">0.25x</button>
      <button class="speed-pill" onclick="setSpeed(0.5)">0.5x</button>
      <button class="speed-pill active" onclick="setSpeed(1.0)">1.0x</button>
      <button class="speed-pill" onclick="setSpeed(2.0)">2.0x</button>
    </div>
  </div>

  <!-- VIEW 1: DUAL COMPARISON ARENA -->
  <div class="view-section active" id="view-dual">
    <div class="dual-grid">

      <!-- LEFT ROCKET (Default: Trial 1) -->
      <div class="sim-card">
        <div class="card-top">
          <div class="selector-wrapper">
            <span class="slot-label">Comparison Slot A:</span>
            <select class="milestone-select" id="select-a" onchange="changeSlot('a', this.value)">
              <!-- Options populated by JS -->
            </select>
          </div>
          <span class="outcome-pill" id="badge-a">CRASH</span>
        </div>

        <div class="canvas-wrapper">
          <canvas class="sim-canvas" id="canvas-a"></canvas>
          <div class="cockpit-hud">
            <div class="hud-row"><span class="hud-label">Altitude</span><span class="hud-val" id="hud-alt-a">0.0 m</span></div>
            <div class="hud-row"><span class="hud-label">Sink Rate (Vy)</span><span class="hud-val" id="hud-vy-a">0.0 m/s</span></div>
            <div class="hud-row"><span class="hud-label">Lateral (Vx)</span><span class="hud-val" id="hud-vx-a">0.0 m/s</span></div>
            <div class="hud-row"><span class="hud-label">Tilt Pitch</span><span class="hud-val" id="hud-tilt-a">0.0°</span></div>
          </div>
        </div>

        <div class="telemetry-strip">
          <div class="stat-box"><div class="stat-label">Throttle</div><div class="stat-number" id="strip-throt-a">0%</div></div>
          <div class="stat-box"><div class="stat-label">Thrust</div><div class="stat-number" id="strip-thrust-a">0 N</div></div>
          <div class="stat-box"><div class="stat-label">Gimbal</div><div class="stat-number" id="strip-gimbal-a">0.0°</div></div>
          <div class="stat-box"><div class="stat-label">Propellant</div><div class="stat-number" id="strip-fuel-a">0 kg</div></div>
          <div class="stat-box"><div class="stat-label">100-Seed Win</div><div class="stat-number" id="strip-sr-a">0%</div></div>
        </div>
      </div>

      <!-- RIGHT ROCKET (Default: Final Model) -->
      <div class="sim-card">
        <div class="card-top">
          <div class="selector-wrapper">
            <span class="slot-label">Comparison Slot B:</span>
            <select class="milestone-select" id="select-b" onchange="changeSlot('b', this.value)">
              <!-- Options populated by JS -->
            </select>
          </div>
          <span class="outcome-pill outcome-success" id="badge-b">SOFT LANDING</span>
        </div>

        <div class="canvas-wrapper">
          <canvas class="sim-canvas" id="canvas-b"></canvas>
          <div class="cockpit-hud">
            <div class="hud-row"><span class="hud-label">Altitude</span><span class="hud-val" id="hud-alt-b">0.0 m</span></div>
            <div class="hud-row"><span class="hud-label">Sink Rate (Vy)</span><span class="hud-val" id="hud-vy-b">0.0 m/s</span></div>
            <div class="hud-row"><span class="hud-label">Lateral (Vx)</span><span class="hud-val" id="hud-vx-b">0.0 m/s</span></div>
            <div class="hud-row"><span class="hud-label">Tilt Pitch</span><span class="hud-val" id="hud-tilt-b">0.0°</span></div>
          </div>
        </div>

        <div class="telemetry-strip">
          <div class="stat-box"><div class="stat-label">Throttle</div><div class="stat-number" id="strip-throt-b">0%</div></div>
          <div class="stat-box"><div class="stat-label">Thrust</div><div class="stat-number" id="strip-thrust-b">0 N</div></div>
          <div class="stat-box"><div class="stat-label">Gimbal</div><div class="stat-number" id="strip-gimbal-b">0.0°</div></div>
          <div class="stat-box"><div class="stat-label">Propellant</div><div class="stat-number" id="strip-fuel-b">0 kg</div></div>
          <div class="stat-box"><div class="stat-label">100-Seed Win</div><div class="stat-number" id="strip-sr-b">99%</div></div>
        </div>
      </div>

    </div>
  </div>

  <!-- VIEW 2: CINEMA SOLO VIEW (GIANT ROCKET) -->
  <div class="view-section" id="view-cinema">
    <div class="cinema-layout">
      <!-- Quick Milestone Switcher Strip -->
      <div class="milestone-strip-bar" id="cinema-milestones">
        <!-- Generated by JS -->
      </div>

      <div class="cinema-canvas-container">
        <canvas class="sim-canvas" id="canvas-cinema"></canvas>
        <div class="cockpit-hud" style="top: 20px; left: 20px; font-size: 13px;">
          <div class="hud-row"><span class="hud-label">Viewing</span><span class="hud-val" id="cinema-label">Final Model</span></div>
          <div class="hud-row"><span class="hud-label">Altitude</span><span class="hud-val" id="cinema-alt">0.0 m</span></div>
          <div class="hud-row"><span class="hud-label">Vertical Speed</span><span class="hud-val" id="cinema-vy">0.0 m/s</span></div>
          <div class="hud-row"><span class="hud-label">Tilt Angle</span><span class="hud-val" id="cinema-tilt">0.0°</span></div>
          <div class="hud-row"><span class="hud-label">Action Intent</span><span class="hud-val" id="cinema-intent" style="color:var(--accent);">Stabilizing</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- VIEW 3: LIVE INTERACTIVE SANDBOX -->
  <div class="view-section" id="view-sandbox">
    <div class="sandbox-grid">
      <!-- Control Panel -->
      <div class="sandbox-panel">
        <div class="panel-title"><span>⚙️</span> Mission Parameters</div>

        <div class="slider-row">
          <div class="slider-header"><span>Drop Altitude</span><span class="slider-val" id="val-alt">80 m</span></div>
          <input type="range" class="slider-input" id="in-alt" min="30" max="140" value="80" oninput="document.getElementById('val-alt').innerText = this.value + ' m'">
        </div>

        <div class="slider-row">
          <div class="slider-header"><span>Initial Tilt Angle</span><span class="slider-val" id="val-tilt">-8 deg</span></div>
          <input type="range" class="slider-input" id="in-tilt" min="-25" max="25" value="-8" oninput="document.getElementById('val-tilt').innerText = this.value + ' deg'">
        </div>

        <div class="slider-row">
          <div class="slider-header"><span>Horizontal Offset</span><span class="slider-val" id="val-x">8 m</span></div>
          <input type="range" class="slider-input" id="in-x" min="-30" max="30" value="8" oninput="document.getElementById('val-x').innerText = this.value + ' m'">
        </div>

        <div class="slider-row">
          <div class="slider-header"><span>Crosswind Speed</span><span class="slider-val" id="val-wind">0 m/s</span></div>
          <input type="range" class="slider-input" id="in-wind" min="-8" max="8" value="0" oninput="document.getElementById('val-wind').innerText = this.value + ' m/s'">
        </div>

        <label class="manual-toggle">
          <input type="checkbox" id="check-manual"> Manual Pilot Mode (Arrows: Up=Thrust, Left/Right=Gimbal)
        </label>

        <button class="launch-btn" onclick="startSandboxFlight()">🚀 IGNITE & LAUNCH</button>
        <button class="sec-btn" onclick="resetSandbox()">Reset to Default</button>
      </div>

      <!-- Live Simulation Canvas -->
      <div class="sim-card">
        <div class="card-top">
          <span class="slot-label">Live In-Browser Neural Network Engine (60 FPS)</span>
          <span class="outcome-pill" id="badge-sandbox">READY</span>
        </div>
        <div class="canvas-wrapper" style="height: 580px;">
          <canvas class="sim-canvas" id="canvas-sandbox"></canvas>
          <div class="cockpit-hud">
            <div class="hud-row"><span class="hud-label">Autopilot</span><span class="hud-val" id="sb-hud-pilot" style="color:var(--accent);">AI ACTIVE</span></div>
            <div class="hud-row"><span class="hud-label">Altitude</span><span class="hud-val" id="sb-hud-alt">0.0 m</span></div>
            <div class="hud-row"><span class="hud-label">Sink Rate</span><span class="hud-val" id="sb-hud-vy">0.0 m/s</span></div>
            <div class="hud-row"><span class="hud-label">Tilt</span><span class="hud-val" id="sb-hud-tilt">0.0°</span></div>
            <div class="hud-row"><span class="hud-label">Fuel</span><span class="hud-val" id="sb-hud-fuel">200 kg</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- EMBEDDED DATA -->
<script>
const REPLAYS_DATA = {replays_json_str};
const MODEL_WEIGHTS = {weights_json_str};

// State variables
let currentView = 'dual';
let isPlaying = false;
let playbackSpeed = 1.0;
let currentStep = 0;
let maxSteps = 0;
let playInterval = null;

// Slots
let slotA_Key = '1';
let slotB_Key = 'final';
let cinema_Key = 'final';

// Milestones list
const milestones = REPLAYS_DATA.milestones || {{}};
const milestoneKeys = Object.keys(milestones);

// Compute max steps
milestoneKeys.forEach(k => {{
  const steps = milestones[k]?.showcase?.steps || [];
  if (steps.length > maxSteps) maxSteps = steps.length;
}});

// Initialize
window.addEventListener('DOMContentLoaded', () => {{
  populateMilestoneSelectors();
  initCanvases();
  setupKeyboardControls();
  updateUI();
  drawAll();
}});

function switchView(viewName) {{
  currentView = viewName;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
  
  document.getElementById('tab-' + viewName).classList.add('active');
  const activeSec = document.getElementById('view-' + viewName);
  if (activeSec) activeSec.classList.add('active');
  
  // Wait for browser layout to complete before sizing visible canvas
  requestAnimationFrame(() => {{
    initCanvases();
    if (viewName === 'sandbox') {{
      resetSandbox();
    }} else {{
      updateUI();
      drawAll();
    }}
  }});
  setTimeout(() => {{
    initCanvases();
    drawAll();
  }}, 40);
}}

function populateMilestoneSelectors() {{
  const selA = document.getElementById('select-a');
  const selB = document.getElementById('select-b');
  const cinStrip = document.getElementById('cinema-milestones');

  selA.innerHTML = '';
  selB.innerHTML = '';
  cinStrip.innerHTML = '';

  milestoneKeys.forEach(k => {{
    const label = milestones[k]?.label || ('Trial ' + k);
    const optA = document.createElement('option');
    optA.value = k;
    optA.text = label;
    if (k === '1') optA.selected = true;
    selA.appendChild(optA);

    const optB = document.createElement('option');
    optB.value = k;
    optB.text = label;
    if (k === 'final') optB.selected = true;
    selB.appendChild(optB);

    // Cinema pill
    const pill = document.createElement('button');
    pill.className = 'm-pill' + (k === cinema_Key ? ' active' : '');
    pill.id = 'cin-pill-' + k;
    pill.innerText = label;
    pill.onclick = () => selectCinemaMilestone(k);
    cinStrip.appendChild(pill);
  }});
}}

function changeSlot(slot, key) {{
  if (slot === 'a') slotA_Key = key;
  if (slot === 'b') slotB_Key = key;
  updateUI();
  drawAll();
}}

function selectCinemaMilestone(key) {{
  cinema_Key = key;
  document.querySelectorAll('.m-pill').forEach(p => p.classList.remove('active'));
  const target = document.getElementById('cin-pill-' + key);
  if (target) target.classList.add('active');
  document.getElementById('cinema-label').innerText = milestones[key]?.label || key;
  drawAll();
}}

function setSpeed(sp) {{
  playbackSpeed = sp;
  document.querySelectorAll('.speed-pill').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  if (isPlaying) {{
    clearInterval(playInterval);
    startPlayLoop();
  }}
}}

function togglePlay() {{
  if (isPlaying) {{
    pauseReplay();
  }} else {{
    startPlay();
  }}
}}

function startPlay() {{
  if (currentStep >= maxSteps - 1) currentStep = 0;
  isPlaying = true;
  document.getElementById('play-icon').innerText = '⏸';
  document.getElementById('play-text').innerText = 'PAUSE';
  startPlayLoop();
}}

function pauseReplay() {{
  isPlaying = false;
  document.getElementById('play-icon').innerText = '▶';
  document.getElementById('play-text').innerText = 'PLAY';
  if (playInterval) clearInterval(playInterval);
}}

function restartReplay() {{
  currentStep = 0;
  drawAll();
  updateUI();
}}

function startPlayLoop() {{
  const intervalMs = (20.0 / playbackSpeed); // standard 50Hz = 20ms
  playInterval = setInterval(() => {{
    if (currentStep < maxSteps - 1) {{
      currentStep++;
      drawAll();
      updateUI();
    }} else {{
      pauseReplay();
    }}
  }}, intervalMs);
}}

function onScrub(val) {{
  currentStep = Math.floor(val / 1000 * (maxSteps - 1));
  drawAll();
  updateUI();
}}

function updateUI() {{
  // Update scrubber
  const pct = maxSteps > 1 ? (currentStep / (maxSteps - 1) * 1000) : 0;
  document.getElementById('master-scrubber').value = pct;
  const tSec = (currentStep * 0.02).toFixed(2);
  document.getElementById('master-time-badge').innerText = `T+${{tSec}}s | Step ${{currentStep}}`;

  // Update Slot A
  updateSlotHUD('a', slotA_Key);
  // Update Slot B
  updateSlotHUD('b', slotB_Key);
  // Update Cinema HUD
  if (currentView === 'cinema') {{
    updateCinemaHUD();
  }}
}}

function updateSlotHUD(slotId, key) {{
  const m = milestones[key];
  if (!m) return;
  const steps = m.showcase?.steps || [];
  const s = steps[Math.min(currentStep, steps.length - 1)] || {{}};
  const isFinalStep = currentStep >= steps.length - 1;

  // Badges
  const badge = document.getElementById(`badge-${{slotId}}`);
  const outcome = m.showcase?.outcome || 'Unknown';
  badge.innerText = isFinalStep ? outcome : (outcome.includes('landing') ? 'DESCENDING' : 'ACTIVE');
  if (outcome.includes('landing') && !outcome.includes('Hard')) {{
    badge.className = 'outcome-pill outcome-success';
  }} else {{
    badge.className = 'outcome-pill outcome-crash';
  }}

  // Cockpit values
  const alt = (s.y || 0).toFixed(1);
  const vy = (s.vy || 0).toFixed(1);
  const vx = (s.vx || 0).toFixed(1);
  const tilt = (s.theta_deg || 0).toFixed(1);

  const elAlt = document.getElementById(`hud-alt-${{slotId}}`);
  const elVy = document.getElementById(`hud-vy-${{slotId}}`);
  const elVx = document.getElementById(`hud-vx-${{slotId}}`);
  const elTilt = document.getElementById(`hud-tilt-${{slotId}}`);

  if (elAlt) elAlt.innerText = `${{alt}} m`;
  if (elVy) {{
    elVy.innerText = `${{vy}} m/s`;
    elVy.className = `hud-val ${{Math.abs(s.vy || 0) < 2.0 ? 'hud-safe' : 'hud-warn'}}`;
  }}
  if (elVx) elVx.innerText = `${{vx}} m/s`;
  if (elTilt) elTilt.innerText = `${{tilt}}°`;

  // Bottom strip
  document.getElementById(`strip-throt-${{slotId}}`).innerText = `${{Math.round(s.throttle_pct || 0)}}%`;
  document.getElementById(`strip-thrust-${{slotId}}`).innerText = `${{Math.round(s.thrust_N || 0)}} N`;
  document.getElementById(`strip-gimbal-${{slotId}}`).innerText = `${{(s.gimbal_deg || 0).toFixed(1)}}°`;
  document.getElementById(`strip-fuel-${{slotId}}`).innerText = `${{(s.fuel_kg || 0).toFixed(1)}} kg`;
  document.getElementById(`strip-sr-${{slotId}}`).innerText = `${{m.eval_100?.success_rate_pct || 0}}%`;
}}

function updateCinemaHUD() {{
  const m = milestones[cinema_Key];
  if (!m) return;
  const steps = m.showcase?.steps || [];
  const s = steps[Math.min(currentStep, steps.length - 1)] || {{}};
  document.getElementById('cinema-alt').innerText = `${{(s.y || 0).toFixed(1)}} m`;
  const elVy = document.getElementById('cinema-vy');
  elVy.innerText = `${{(s.vy || 0).toFixed(1)}} m/s`;
  elVy.className = `hud-val ${{Math.abs(s.vy || 0) < 2.0 ? 'hud-safe' : 'hud-warn'}}`;
  document.getElementById('cinema-tilt').innerText = `${{(s.theta_deg || 0).toFixed(1)}}°`;
  document.getElementById('cinema-intent').innerText = s.explanation || 'Descending';
}}

// Canvas resizing
function resizeCanvas(c) {{
  if (!c) return;
  const rect = c.getBoundingClientRect();
  let w = rect.width;
  let h = rect.height;
  if (!w || w < 20) {{
    w = c.parentElement ? c.parentElement.clientWidth : 0;
  }}
  if (!h || h < 20) {{
    h = c.parentElement ? c.parentElement.clientHeight : 0;
  }}
  if (!w || w < 20) w = (c.id === 'canvas-cinema' ? 1200 : 800);
  if (!h || h < 20) h = (c.id === 'canvas-cinema' ? 640 : 520);

  const dpr = window.devicePixelRatio || 1;
  const targetW = Math.floor(w * dpr);
  const targetH = Math.floor(h * dpr);

  if (c.width !== targetW || c.height !== targetH) {{
    c.width = targetW;
    c.height = targetH;
  }}
}}

function initCanvases() {{
  ['canvas-a', 'canvas-b', 'canvas-cinema', 'canvas-sandbox'].forEach(id => {{
    resizeCanvas(document.getElementById(id));
  }});
}}
window.addEventListener('resize', () => {{
  initCanvases();
  drawAll();
}});

// DRAWING ENGINE
function drawAll() {{
  if (currentView === 'dual') {{
    resizeCanvas(document.getElementById('canvas-a'));
    resizeCanvas(document.getElementById('canvas-b'));
    drawMilestoneCanvas('canvas-a', slotA_Key, false);
    drawMilestoneCanvas('canvas-b', slotB_Key, false);
  }} else if (currentView === 'cinema') {{
    resizeCanvas(document.getElementById('canvas-cinema'));
    drawMilestoneCanvas('canvas-cinema', cinema_Key, true);
  }} else if (currentView === 'sandbox') {{
    resizeCanvas(document.getElementById('canvas-sandbox'));
    drawSandboxCanvas(0, 0);
  }}
}}

function drawMilestoneCanvas(canvasId, key, isCinema=false) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  if (w <= 0 || h <= 0) return;

  const m = milestones[key];
  const steps = m?.showcase?.steps || [];
  const stepIdx = Math.min(currentStep, steps.length - 1);
  const state = steps[stepIdx] || {{ x: 0, y: 80, theta_rad: 0, throttle_pct: 0, gimbal_deg: 0, vy: 0 }};

  // Background sky
  ctx.save();
  ctx.fillStyle = '#050a16';
  ctx.fillRect(0, 0, w, h);

  // Stars
  drawStars(ctx, w, h);

  // World coordinates:
  // x in [-35, 35] m -> map to width
  // y in [0, 95] m -> map to height
  const padY = h - 60;
  const scale = h / 105.0; // pixels per meter

  // Ground Pad
  const padW = 20 * scale; // 20m wide landing ship/pad
  const padX = (w / 2);

  // Drone ship base
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(padX - padW/2 - 10, padY, padW + 20, 24);
  // Pad deck
  ctx.fillStyle = '#334155';
  ctx.fillRect(padX - padW/2, padY, padW, 8);
  // Center green target
  ctx.fillStyle = '#10b981';
  ctx.fillRect(padX - 8 * scale / 2, padY, 8 * scale, 8);
  // Pad Center line
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padX, padY - 200);
  ctx.lineTo(padX, padY);
  ctx.stroke();

  // Draw Trajectory Trail
  ctx.beginPath();
  ctx.strokeStyle = (key === '1') ? 'rgba(239, 68, 68, 0.4)' : 'rgba(56, 189, 248, 0.5)';
  ctx.lineWidth = 3;
  for (let i = 0; i <= stepIdx; i++) {{
    const st = steps[i];
    const px = padX + st.x * scale;
    const py = padY - st.y * scale;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }}
  ctx.stroke();

  // Draw the BIG Rocket!
  const rX = padX + state.x * scale;
  const rY = padY - state.y * scale;
  drawDetailedRocket(ctx, rX, rY, state.theta_rad, state.throttle_pct, state.gimbal_deg, state.y, isCinema);

  ctx.restore();
}}

// STAR GENERATION (seeded)
function drawStars(ctx, w, h) {{
  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  for (let i = 0; i < 45; i++) {{
    const sx = (Math.sin(i * 997) * 0.5 + 0.5) * w;
    const sy = (Math.cos(i * 613) * 0.5 + 0.5) * (h - 80);
    const sz = (i % 3 === 0) ? 2 : 1;
    ctx.fillRect(sx, sy, sz, sz);
  }}
}}

// DETAILED BIG ROCKET RENDERING
function drawDetailedRocket(ctx, x, y, theta_rad, throttle_pct, gimbal_deg, alt_m, isCinema=false) {{
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(theta_rad);

  // Scale of the rocket: 10 meters tall -> render at ~100-130 pixels
  const rHeight = isCinema ? 130 : 100;
  const rWidth = isCinema ? 20 : 16;
  const halfH = rHeight / 2;
  const halfW = rWidth / 2;

  // 1. GIMBALED ENGINE PLUME (Draw first so it appears behind/under engine)
  if (throttle_pct > 0.01) {{
    ctx.save();
    ctx.translate(0, halfH);
    const gimbalRad = (gimbal_deg * Math.PI / 180.0);
    ctx.rotate(gimbalRad);

    const flameLen = (throttle_pct / 100.0) * (isCinema ? 95 : 75) * (0.85 + Math.random() * 0.3);
    const flameW = rWidth * (0.7 + (throttle_pct / 100.0) * 0.4);

    // Outer plasma flame
    const flameGrad = ctx.createLinearGradient(0, 0, 0, flameLen);
    flameGrad.addColorStop(0, '#ffffff');
    flameGrad.addColorStop(0.2, '#38bdf8');
    flameGrad.addColorStop(0.5, '#f97316');
    flameGrad.addColorStop(0.9, '#ef4444');
    flameGrad.addColorStop(1, 'rgba(239, 68, 68, 0)');

    ctx.fillStyle = flameGrad;
    ctx.beginPath();
    ctx.moveTo(-flameW/2, 0);
    ctx.lineTo(flameW/2, 0);
    ctx.lineTo(0, flameLen);
    ctx.closePath();
    ctx.fill();

    // Supersonic inner Mach diamonds
    if (throttle_pct > 30) {{
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.moveTo(0, 8);
      ctx.lineTo(4, 16);
      ctx.lineTo(0, 24);
      ctx.lineTo(-4, 16);
      ctx.closePath();
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(0, 28);
      ctx.lineTo(3, 34);
      ctx.lineTo(0, 40);
      ctx.lineTo(-3, 34);
      ctx.closePath();
      ctx.fill();
    }}
    ctx.restore();
  }}

  // 2. ROCKET FUSELAGE (Metallic Falcon 9 Cylinder)
  const bodyGrad = ctx.createLinearGradient(-halfW, 0, halfW, 0);
  bodyGrad.addColorStop(0, '#cbd5e1');
  bodyGrad.addColorStop(0.4, '#ffffff');
  bodyGrad.addColorStop(0.8, '#94a3b8');
  bodyGrad.addColorStop(1, '#475569');

  ctx.fillStyle = bodyGrad;
  ctx.beginPath();
  if (ctx.roundRect) {{
    ctx.roundRect(-halfW, -halfH, rWidth, rHeight, 4);
  }} else {{
    ctx.rect(-halfW, -halfH, rWidth, rHeight);
  }}
  ctx.fill();
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Black Interstage band at top
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(-halfW, -halfH + 16, rWidth, 8);

  // Aerodynamic Grid Fins at top
  ctx.fillStyle = '#1e293b';
  // Left fin
  ctx.fillRect(-halfW - 7, -halfH + 14, 7, 10);
  // Right fin
  ctx.fillRect(halfW, -halfH + 14, 7, 10);

  // 3. ENGINE BELL NOZZLE
  ctx.fillStyle = '#334155';
  ctx.beginPath();
  ctx.moveTo(-halfW * 0.6, halfH);
  ctx.lineTo(halfW * 0.6, halfH);
  ctx.lineTo(halfW * 0.8, halfH + 8);
  ctx.lineTo(-halfW * 0.8, halfH + 8);
  ctx.closePath();
  ctx.fill();

  // 4. DEPLOYABLE LANDING LEGS
  // Legs deploy smoothly when altitude is under 15m!
  const legDeploy = Math.max(0.0, Math.min(1.0, (15.0 - alt_m) / 10.0));
  const legAngle = legDeploy * 0.55; // radians spread

  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 3;

  // Left leg
  ctx.save();
  ctx.translate(-halfW, halfH - 6);
  ctx.rotate(-legAngle);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-12 * (0.4 + legDeploy*0.6), 18);
  ctx.stroke();
  // Foot pad
  ctx.fillStyle = '#475569';
  ctx.fillRect(-15, 17, 6, 3);
  ctx.restore();

  // Right leg
  ctx.save();
  ctx.translate(halfW, halfH - 6);
  ctx.rotate(legAngle);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(12 * (0.4 + legDeploy*0.6), 18);
  ctx.stroke();
  // Foot pad
  ctx.fillStyle = '#475569';
  ctx.fillRect(9, 17, 6, 3);
  ctx.restore();

  ctx.restore();
}}

// LIVE SANDBOX SIMULATION LOGIC
let sbState = {{
  x: 0, y: 80, vx: 0, vy: -5, th: -0.1, w: 0, fuel: 200, mass: 1000
}};
let sbRunning = false;
let sbLoop = null;
let keysDown = {{}};

function setupKeyboardControls() {{
  window.addEventListener('keydown', e => {{ keysDown[e.key] = true; }});
  window.addEventListener('keyup', e => {{ keysDown[e.key] = false; }});
}}

function resetSandbox() {{
  if (sbLoop) cancelAnimationFrame(sbLoop);
  sbRunning = false;
  const inAlt = parseFloat(document.getElementById('in-alt').value);
  const inTilt = parseFloat(document.getElementById('in-tilt').value) * Math.PI / 180.0;
  const inX = parseFloat(document.getElementById('in-x').value);

  sbState = {{
    x: inX,
    y: inAlt,
    vx: 0.0,
    vy: -8.0,
    th: inTilt,
    w: 0.0,
    fuel: 200.0,
    mass: 1000.0
  }};
  document.getElementById('badge-sandbox').innerText = 'READY';
  document.getElementById('badge-sandbox').className = 'outcome-pill';
  drawSandboxCanvas(0, 0);
}}

function startSandboxFlight() {{
  resetSandbox();
  sbRunning = true;
  document.getElementById('badge-sandbox').innerText = 'FLYING';
  document.getElementById('badge-sandbox').className = 'outcome-pill outcome-success';
  lastTime = performance.now();
  requestAnimationFrame(sandboxStep);
}}

let lastTime = 0;
function sandboxStep(now) {{
  if (!sbRunning) return;
  const dt = 0.02; // 50 Hz physics

  // AI Policy or Manual Control
  let a0 = 0.0, a1 = 0.0;
  const isManual = document.getElementById('check-manual').checked;

  if (isManual) {{
    document.getElementById('sb-hud-pilot').innerText = 'MANUAL (PILOT)';
    if (keysDown['ArrowUp'] || keysDown['w']) a0 = 1.0;
    else a0 = -1.0;
    if (keysDown['ArrowLeft'] || keysDown['a']) a1 = -1.0;
    else if (keysDown['ArrowRight'] || keysDown['d']) a1 = 1.0;
  }} else {{
    document.getElementById('sb-hud-pilot').innerText = 'AI AUTOPILOT';
    // Run forward pass of 2x128 PPO MLP policy
    const obs = [
      sbState.x / 50.0,
      sbState.y / 100.0,
      sbState.vx / 10.0,
      sbState.vy / 10.0,
      Math.sin(sbState.th),
      Math.cos(sbState.th),
      sbState.w,
      sbState.fuel / 200.0
    ];
    const act = forwardPolicy(obs, MODEL_WEIGHTS);
    a0 = act[0];
    a1 = act[1];
  }}

  // Physics equations of motion (Semi-implicit Euler matching rocket_env.py)
  const u = (a0 + 1.0) / 2.0;
  const maxGimbal = 15.0 * Math.PI / 180.0;
  const phi = a1 * maxGimbal;
  const T = (sbState.fuel > 0) ? u * 20000.0 : 0.0;
  const m = 800.0 + sbState.fuel;
  const I = m * (10.0 * 10.0) / 12.0;

  const ax = -T / m * Math.sin(sbState.th + phi);
  const ay = T / m * Math.cos(sbState.th + phi) - 9.81;
  const alpha = -5.0 * T * Math.sin(phi) / I;

  const dm = T / (300.0 * 9.81) * dt;
  sbState.fuel = Math.max(0.0, sbState.fuel - dm);

  sbState.vx += ax * dt;
  sbState.vy += ay * dt;
  sbState.w += alpha * dt;
  sbState.x += sbState.vx * dt;
  sbState.y += sbState.vy * dt;
  sbState.th += sbState.w * dt;

  // Update HUD
  document.getElementById('sb-hud-alt').innerText = `${{sbState.y.toFixed(1)}} m`;
  document.getElementById('sb-hud-vy').innerText = `${{sbState.vy.toFixed(1)}} m/s`;
  document.getElementById('sb-hud-tilt').innerText = `${{(sbState.th * 180.0 / Math.PI).toFixed(1)}}°`;
  document.getElementById('sb-hud-fuel').innerText = `${{sbState.fuel.toFixed(1)}} kg`;

  drawSandboxCanvas(u * 100.0, phi * 180.0 / Math.PI);

  // Check Touchdown / Crash
  if (sbState.y <= 0.0 || Math.abs(sbState.th) > 1.0 || Math.abs(sbState.x) > 60.0) {{
    sbRunning = false;
    const isSoft = (Math.abs(sbState.vy) < 2.0 && Math.abs(sbState.vx) < 1.0 && Math.abs(sbState.th) < 0.1 && Math.abs(sbState.x) < 5.0);
    const badge = document.getElementById('badge-sandbox');
    if (isSoft) {{
      badge.innerText = 'TOUCHDOWN: SOFT LANDING!';
      badge.className = 'outcome-pill outcome-success';
    }} else {{
      badge.innerText = 'CRASHED!';
      badge.className = 'outcome-pill outcome-crash';
    }}
    return;
  }}

  requestAnimationFrame(sandboxStep);
}}

function forwardPolicy(obs, weights) {{
  if (!weights.w1) return [0.0, 0.0];
  // Layer 1: 8 -> 128 (Tanh)
  const h1 = new Float32Array(128);
  for (let i = 0; i < 128; i++) {{
    let sum = weights.b1[i];
    for (let j = 0; j < 8; j++) sum += weights.w1[i][j] * obs[j];
    h1[i] = Math.tanh(sum);
  }}
  // Layer 2: 128 -> 128 (Tanh)
  const h2 = new Float32Array(128);
  for (let i = 0; i < 128; i++) {{
    let sum = weights.b2[i];
    for (let j = 0; j < 128; j++) sum += weights.w2[i][j] * h1[j];
    h2[i] = Math.tanh(sum);
  }}
  // Action net: 128 -> 2 (Linear)
  const out = [weights.b_out[0], weights.b_out[1]];
  for (let j = 0; j < 128; j++) {{
    out[0] += weights.w_out[0][j] * h2[j];
    out[1] += weights.w_out[1][j] * h2[j];
  }}
  return [Math.max(-1.0, Math.min(1.0, out[0])), Math.max(-1.0, Math.min(1.0, out[1]))];
}}

function drawSandboxCanvas(throttle_pct, gimbal_deg) {{
  const canvas = document.getElementById('canvas-sandbox');
  if (!canvas) return;
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  if (w <= 0 || h <= 0) return;

  ctx.fillStyle = '#050a16';
  ctx.fillRect(0, 0, w, h);
  drawStars(ctx, w, h);

  const padY = h - 60;
  const scale = h / 110.0;
  const padX = w / 2;

  // Draw Pad
  ctx.fillStyle = '#334155';
  ctx.fillRect(padX - 100, padY, 200, 10);
  ctx.fillStyle = '#10b981';
  ctx.fillRect(padX - 40, padY, 80, 10);

  // Draw BIG Rocket
  const rx = padX + sbState.x * scale;
  const ry = padY - Math.max(0.0, sbState.y) * scale;
  drawDetailedRocket(ctx, rx, ry, sbState.th, throttle_pct, gimbal_deg, sbState.y, true);
}}
</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[Flight Simulator Generated] -> {output_path} ({os.path.getsize(output_path):,} bytes)")
    return output_path

if __name__ == "__main__":
    build_flight_simulator()
