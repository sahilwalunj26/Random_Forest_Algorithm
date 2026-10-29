"""
Market Pulse AI - Flask web app for random.pkl (RandomForestClassifier)

Predicts the market regime: Bull Market / Recession / Stable
from 21 company, layoff, hiring and AI-impact signals.

Run locally:   python app.py          ->  http://127.0.0.1:5000
Production:    gunicorn app:app --bind 0.0.0.0:$PORT
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "random.pkl"))

app = Flask(__name__)

# --------------------------------------------------------------------------
# Load model once at start-up
# --------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
CLASSES = [str(c) for c in model.classes_]
IMPORTANCES = dict(zip(FEATURES, [float(x) for x in model.feature_importances_]))

# --------------------------------------------------------------------------
# Categorical labels
# --------------------------------------------------------------------------
# The model was trained on LABEL-ENCODED categories (numbers), and the encoders
# were not saved inside random.pkl. Replace the lists below with your real
# names IN THE SAME ORDER as your LabelEncoder.classes_ (alphabetical order
# if you used sklearn's LabelEncoder). The position in the list = the code the
# model receives. Example:
#   "industry": ["Automotive", "Banking", "E-commerce", "Healthcare", ...]
CATEGORY_LABELS = {
    "company_name": [f"Company #{i}" for i in range(20)],
    "industry": [f"Industry #{i}" for i in range(7)],
    "country": [f"Country #{i}" for i in range(6)],
    "company_size": [f"Size class #{i}" for i in range(4)],
    "reason_for_layoffs": [f"Reason #{i}" for i in range(5)],
    "hiring_trend": [f"Hiring trend #{i}" for i in range(4)],
    "top_hiring_role": [f"Role #{i}" for i in range(8)],
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def cat(key):
    return [{"value": i, "label": lab} for i, lab in enumerate(CATEGORY_LABELS[key])]


# --------------------------------------------------------------------------
# Form schema (drives the UI, so UI and model never drift apart)
# --------------------------------------------------------------------------
GROUPS = [
    {"id": "profile", "title": "Company profile", "icon": "🏢",
     "blurb": "Who and when"},
    {"id": "layoffs", "title": "Layoffs & AI impact", "icon": "🤖",
     "blurb": "How much is changing"},
    {"id": "hiring", "title": "Hiring & growth", "icon": "📈",
     "blurb": "Where the company is heading"},
    {"id": "people", "title": "People & security", "icon": "🫶",
     "blurb": "How it feels inside"},
]

FIELDS = [
    # ---- profile
    {"key": "company_name", "group": "profile", "label": "Company",
     "kind": "select", "options": cat("company_name"), "default": 0},
    {"key": "industry", "group": "profile", "label": "Industry",
     "kind": "select", "options": cat("industry"), "default": 0},
    {"key": "country", "group": "profile", "label": "Country",
     "kind": "select", "options": cat("country"), "default": 0},
    {"key": "company_size", "group": "profile", "label": "Company size",
     "kind": "select", "options": cat("company_size"), "default": 1},
    {"key": "month", "group": "profile", "label": "Month",
     "kind": "select",
     "options": [{"value": i + 1, "label": m} for i, m in enumerate(MONTHS)],
     "default": 6},
    {"key": "year", "group": "profile", "label": "Year",
     "kind": "select",
     "options": [{"value": 2024, "label": "2024"}, {"value": 2025, "label": "2025"}],
     "default": 2025},
    # ---- layoffs
    {"key": "layoffs_count", "group": "layoffs", "label": "Layoffs count",
     "kind": "slider", "min": 0, "max": 15000, "step": 10, "default": 2400,
     "unit": "people"},
    {"key": "layoff_percentage", "group": "layoffs", "label": "Layoff percentage",
     "kind": "slider", "min": 0, "max": 30, "step": 0.1, "default": 9, "unit": "%"},
    {"key": "reason_for_layoffs", "group": "layoffs", "label": "Reason for layoffs",
     "kind": "select", "options": cat("reason_for_layoffs"), "default": 1},
    {"key": "ai_automation_impact", "group": "layoffs", "label": "AI automation impact",
     "kind": "slider", "min": 0, "max": 20, "step": 0.1, "default": 6.2, "unit": ""},
    {"key": "ai_replacement_risk", "group": "layoffs", "label": "AI replacement risk",
     "kind": "slider", "min": 1, "max": 10, "step": 0.1, "default": 6, "unit": "/10"},
    {"key": "ai_adoption_level", "group": "layoffs", "label": "AI adoption level",
     "kind": "slider", "min": 1, "max": 10, "step": 0.1, "default": 5.8, "unit": "/10"},
    # ---- hiring
    {"key": "open_roles", "group": "hiring", "label": "Open roles",
     "kind": "slider", "min": 0, "max": 10000, "step": 10, "default": 2300,
     "unit": "roles"},
    {"key": "hiring_trend", "group": "hiring", "label": "Hiring trend",
     "kind": "select", "options": cat("hiring_trend"), "default": 1},
    {"key": "top_hiring_role", "group": "hiring", "label": "Top hiring role",
     "kind": "select", "options": cat("top_hiring_role"), "default": 3},
    {"key": "remote_jobs_percentage", "group": "hiring", "label": "Remote jobs",
     "kind": "slider", "min": 0, "max": 100, "step": 1, "default": 50, "unit": "%"},
    {"key": "stock_growth_percent", "group": "hiring", "label": "Stock growth",
     "kind": "slider", "min": -50, "max": 100, "step": 0.5, "default": 20, "unit": "%"},
    {"key": "revenue_growth_percent", "group": "hiring", "label": "Revenue growth",
     "kind": "slider", "min": -30, "max": 65, "step": 0.5, "default": 16, "unit": "%"},
    {"key": "salary_budget_change", "group": "hiring", "label": "Salary budget change",
     "kind": "slider", "min": -25, "max": 40, "step": 0.5, "default": 6, "unit": "%"},
    # ---- people
    {"key": "employee_sentiment", "group": "people", "label": "Employee sentiment",
     "kind": "slider", "min": 1, "max": 10, "step": 0.1, "default": 6.8, "unit": "/10"},
    {"key": "job_security_score", "group": "people", "label": "Job security score",
     "kind": "slider", "min": 1, "max": 10, "step": 0.1, "default": 6.2, "unit": "/10"},
]

FIELD_MAP = {f["key"]: f for f in FIELDS}
missing = set(FEATURES) - set(FIELD_MAP)
assert not missing, f"Form is missing model features: {missing}"

# One-click demo scenarios (values are model inputs)
PRESETS = {
    "Bull Market": {"company_name": 9, "industry": 1, "country": 3, "company_size": 1, "month": 4, "year": 2025, "layoffs_count": 840, "layoff_percentage": 0.1, "reason_for_layoffs": 1, "ai_automation_impact": 4.2, "ai_replacement_risk": 3.1, "ai_adoption_level": 4.3, "open_roles": 5530, "hiring_trend": 0, "top_hiring_role": 2, "remote_jobs_percentage": 90, "stock_growth_percent": 86.5, "revenue_growth_percent": 5.5, "salary_budget_change": 10.5, "employee_sentiment": 7.7, "job_security_score": 8.0},
    "Recession": {"company_name": 9, "industry": 0, "country": 1, "company_size": 0, "month": 4, "year": 2024, "layoffs_count": 12520, "layoff_percentage": 28.6, "reason_for_layoffs": 2, "ai_automation_impact": 2.1, "ai_replacement_risk": 8.5, "ai_adoption_level": 1.1, "open_roles": 9890, "hiring_trend": 2, "top_hiring_role": 3, "remote_jobs_percentage": 25, "stock_growth_percent": 72.5, "revenue_growth_percent": -0.5, "salary_budget_change": -14.0, "employee_sentiment": 4.2, "job_security_score": 1.6},
    "Stable": {"company_name": 6, "industry": 6, "country": 2, "company_size": 0, "month": 9, "year": 2025, "layoffs_count": 3670, "layoff_percentage": 10.9, "reason_for_layoffs": 4, "ai_automation_impact": 2.9, "ai_replacement_risk": 8.8, "ai_adoption_level": 6.7, "open_roles": 7390, "hiring_trend": 3, "top_hiring_role": 1, "remote_jobs_percentage": 76, "stock_growth_percent": 38.5, "revenue_growth_percent": 57.0, "salary_budget_change": 2.0, "employee_sentiment": 8.8, "job_security_score": 5.9},
}

CLASS_STYLE = {
    "Bull Market": {"emoji": "🚀", "tagline": "Growth is leading. Hiring and confidence are strong."},
    "Recession": {"emoji": "🌧️", "tagline": "Headwinds ahead. Cuts and caution dominate."},
    "Stable": {"emoji": "⚖️", "tagline": "Balanced signals. No strong push in either direction."},
}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def home():
    return render_template_string(PAGE)


@app.get("/api/meta")
def meta():
    top = sorted(IMPORTANCES.items(), key=lambda kv: kv[1], reverse=True)[:8]
    return jsonify({
        "groups": GROUPS,
        "fields": FIELDS,
        "classes": CLASSES,
        "class_style": CLASS_STYLE,
        "presets": PRESETS,
        "importances": [{"key": k, "label": FIELD_MAP[k]["label"], "value": v} for k, v in top],
        "n_trees": int(model.n_estimators),
    })


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    row = {}
    for name in FEATURES:
        if name not in payload or payload[name] in ("", None):
            return jsonify({"error": f"Missing value for '{FIELD_MAP[name]['label']}'"}), 400
        try:
            row[name] = float(payload[name])
        except (TypeError, ValueError):
            return jsonify({"error": f"'{FIELD_MAP[name]['label']}' must be a number"}), 400

    X = pd.DataFrame([row], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = CLASSES[idx]

    return jsonify({
        "label": label,
        "confidence": float(proba[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, proba)},
        "emoji": CLASS_STYLE.get(label, {}).get("emoji", "✨"),
        "tagline": CLASS_STYLE.get(label, {}).get("tagline", ""),
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": type(model).__name__, "features": len(FEATURES)})


# --------------------------------------------------------------------------
# Frontend (single file: HTML + CSS + JS)
# --------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Pulse AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#07090f; --panel:rgba(255,255,255,.045); --line:rgba(255,255,255,.09);
  --text:#eaeefc; --muted:#8b93b0; --accent:#7c8cff; --accent2:#c06bff;
  --bull:#2ee59d; --rec:#ff5c7a; --stable:#5cb8ff;
  --r:20px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);
  min-height:100vh;overflow-x:hidden;line-height:1.5;
}
h1,h2,h3,.display{font-family:'Space Grotesk','Inter',sans-serif}

/* animated backdrop */
.bg{position:fixed;inset:0;z-index:-1;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(90px);opacity:.45;animation:float 18s ease-in-out infinite}
.o1{width:520px;height:520px;background:#4a3cff;top:-160px;left:-120px}
.o2{width:460px;height:460px;background:#c06bff;bottom:-160px;right:-100px;animation-delay:-6s}
.o3{width:360px;height:360px;background:#00c2ff;top:40%;left:55%;opacity:.22;animation-delay:-11s}
@keyframes float{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(40px,-50px) scale(1.12)}}
.grid-bg{position:absolute;inset:0;
  background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);
  background-size:56px 56px;mask-image:radial-gradient(ellipse at 50% 0%,#000 10%,transparent 70%);opacity:.5}

.wrap{max-width:1240px;margin:0 auto;padding:28px 22px 80px}

/* header */
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:34px}
.brand{display:flex;align-items:center;gap:12px;font-weight:700;font-size:1.15rem}
.logo{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;
  background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 8px 28px rgba(124,140,255,.45)}
.pill{font-size:.78rem;color:var(--muted);border:1px solid var(--line);padding:6px 12px;border-radius:99px;background:var(--panel);backdrop-filter:blur(8px)}
.pill b{color:var(--text)}

.hero{margin-bottom:34px;max-width:760px}
.hero h1{font-size:clamp(2.1rem,5vw,3.6rem);line-height:1.05;letter-spacing:-.03em;font-weight:700}
.grad{background:linear-gradient(100deg,#9aa6ff,#d28bff 45%,#5cd6ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{margin-top:14px;color:var(--muted);font-size:1.05rem;max-width:620px}

.presets{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px;align-items:center}
.presets span{font-size:.8rem;color:var(--muted);margin-right:4px}
.chip{cursor:pointer;border:1px solid var(--line);background:var(--panel);color:var(--text);
  padding:8px 14px;border-radius:99px;font-size:.85rem;font-family:inherit;transition:.2s;backdrop-filter:blur(8px)}
.chip:hover{transform:translateY(-2px);border-color:var(--accent);box-shadow:0 8px 22px rgba(124,140,255,.25)}

/* layout */
.layout{display:grid;grid-template-columns:1fr 400px;gap:26px;align-items:start}
@media(max-width:980px){.layout{grid-template-columns:1fr}}

.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:22px;
  backdrop-filter:blur(16px);margin-bottom:18px;animation:rise .6s both}
.card:nth-child(2){animation-delay:.07s}.card:nth-child(3){animation-delay:.14s}.card:nth-child(4){animation-delay:.21s}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
.card-head{display:flex;align-items:center;gap:12px;margin-bottom:18px}
.ico{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;font-size:1.2rem;
  background:linear-gradient(135deg,rgba(124,140,255,.25),rgba(192,107,255,.2));border:1px solid var(--line)}
.card-head h3{font-size:1.05rem}
.card-head small{color:var(--muted);font-size:.8rem}

.fields{display:grid;grid-template-columns:repeat(2,1fr);gap:16px 20px}
@media(max-width:640px){.fields{grid-template-columns:1fr}}
.field label{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);margin-bottom:7px;font-weight:500}
.field label output{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}

select{width:100%;padding:11px 12px;border-radius:12px;border:1px solid var(--line);background:rgba(8,10,20,.65);
  color:var(--text);font:inherit;font-size:.92rem;outline:none;transition:.2s;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='none' stroke='%238b93b0' stroke-width='2'%3E%3Cpath d='M1 1l5 5 5-5'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 14px center}
select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(124,140,255,.2)}
option{background:#10131f}

input[type=range]{-webkit-appearance:none;width:100%;height:6px;border-radius:99px;outline:none;cursor:pointer;
  background:linear-gradient(90deg,var(--accent) var(--p,50%),rgba(255,255,255,.1) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:#fff;
  border:3px solid var(--accent);box-shadow:0 2px 10px rgba(0,0,0,.5);transition:.15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2)}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#fff;border:3px solid var(--accent)}

/* result panel */
.side{position:sticky;top:20px}
@media(max-width:980px){.side{position:static}}
.result{position:relative;overflow:hidden;text-align:center;padding:30px 22px 26px;transition:border-color .5s,box-shadow .5s}
.result::before{content:"";position:absolute;inset:-40% -20% auto;height:70%;
  background:radial-gradient(circle,var(--glow,transparent),transparent 65%);opacity:.5;transition:.6s;pointer-events:none}
.result.bull{--glow:var(--bull);--ac:var(--bull);border-color:rgba(46,229,157,.4);box-shadow:0 0 60px rgba(46,229,157,.12)}
.result.rec{--glow:var(--rec);--ac:var(--rec);border-color:rgba(255,92,122,.4);box-shadow:0 0 60px rgba(255,92,122,.12)}
.result.stable{--glow:var(--stable);--ac:var(--stable);border-color:rgba(92,184,255,.4);box-shadow:0 0 60px rgba(92,184,255,.12)}
.eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);position:relative}

.gauge{position:relative;width:200px;height:200px;margin:18px auto 14px}
.gauge svg{transform:rotate(-90deg)}
.gauge .track{stroke:rgba(255,255,255,.08)}
.gauge .bar{stroke:var(--ac,var(--accent));stroke-linecap:round;transition:stroke-dashoffset 1.2s cubic-bezier(.2,.8,.2,1),stroke .5s;
  filter:drop-shadow(0 0 8px var(--ac,var(--accent)))}
.gauge .center{position:absolute;inset:0;display:grid;place-items:center;align-content:center}
.emoji{font-size:2.4rem;line-height:1;transition:transform .5s}
.pct{font-family:'Space Grotesk';font-size:2.1rem;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
.pct small{font-size:.9rem;color:var(--muted)}
.verdict{font-family:'Space Grotesk';font-size:1.8rem;font-weight:700;color:var(--ac,var(--text));position:relative}
.tag{color:var(--muted);font-size:.9rem;margin:6px auto 18px;max-width:300px;position:relative}
.idle .emoji{animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{50%{transform:scale(1.15)}}
.pop .emoji{animation:pop .6s}
@keyframes pop{0%{transform:scale(.3) rotate(-20deg)}60%{transform:scale(1.3) rotate(8deg)}100%{transform:scale(1)}}

.probs{text-align:left;position:relative}
.prob{margin-top:11px}
.prob .row{display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:5px}
.prob .row span:last-child{font-weight:600;font-variant-numeric:tabular-nums}
.meter{height:7px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden}
.meter i{display:block;height:100%;width:0;border-radius:99px;transition:width 1s cubic-bezier(.2,.8,.2,1)}
.c-bull i{background:var(--bull)}.c-rec i{background:var(--rec)}.c-stable i{background:var(--stable)}

button.go{width:100%;margin-top:22px;border:0;cursor:pointer;padding:15px;border-radius:14px;font:inherit;font-weight:600;font-size:1rem;
  color:#fff;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 10px 30px rgba(124,140,255,.4);
  transition:.2s;position:relative;overflow:hidden}
button.go:hover{transform:translateY(-2px);box-shadow:0 14px 38px rgba(124,140,255,.55)}
button.go:disabled{opacity:.7;cursor:wait;transform:none}
.row2{display:flex;gap:10px;margin-top:10px}
button.ghost{flex:1;cursor:pointer;padding:11px;border-radius:12px;border:1px solid var(--line);background:transparent;color:var(--text);font:inherit;font-size:.85rem;transition:.2s}
button.ghost:hover{background:var(--panel);border-color:var(--accent)}
.err{color:var(--rec);font-size:.85rem;margin-top:12px;min-height:1.2em}

.drivers h4{font-size:.9rem;margin-bottom:4px}
.drivers p{font-size:.76rem;color:var(--muted);margin-bottom:12px}
.drv{display:grid;grid-template-columns:130px 1fr 40px;gap:10px;align-items:center;font-size:.78rem;margin-top:9px}
.drv .meter i{background:linear-gradient(90deg,var(--accent),var(--accent2))}
.drv span:last-child{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}

footer{margin-top:40px;text-align:center;color:var(--muted);font-size:.78rem}
</style>
</head>
<body>
<div class="bg"><div class="grid-bg"></div><div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div></div>

<div class="wrap">
  <header>
    <div class="brand"><div class="logo">📡</div> Market Pulse AI</div>
    <div class="pill">Random Forest · <b id="trees">50</b> trees</div>
  </header>

  <section class="hero">
    <h1>Read the <span class="grad">market's mood</span> from the workforce signals.</h1>
    <p>Tune the layoff, hiring, AI and sentiment indicators below. The model weighs all 21 signals and tells you whether conditions look like a bull run, a recession or a steady market.</p>
    <div class="presets" id="presets"><span>Try a scenario:</span></div>
  </section>

  <div class="layout">
    <main id="form"></main>

    <aside class="side">
      <div class="card result idle" id="result">
        <div class="eyebrow">Market prediction</div>
        <div class="gauge">
          <svg width="200" height="200" viewBox="0 0 200 200">
            <circle class="track" cx="100" cy="100" r="86" fill="none" stroke-width="12"/>
            <circle class="bar" id="bar" cx="100" cy="100" r="86" fill="none" stroke-width="12"
                    stroke-dasharray="540.35" stroke-dashoffset="540.35"/>
          </svg>
          <div class="center">
            <div class="emoji" id="emoji">🔮</div>
            <div class="pct" id="pct">--<small>%</small></div>
          </div>
        </div>
        <div class="verdict" id="verdict">Awaiting input</div>
        <div class="tag" id="tag">Adjust the signals, then run the analysis.</div>
        <div class="probs" id="probs"></div>
        <button class="go" id="go">Analyze market ✨</button>
        <div class="row2">
          <button class="ghost" id="random">🎲 Surprise me</button>
          <button class="ghost" id="reset">↺ Reset</button>
        </div>
        <div class="err" id="err"></div>
      </div>

      <div class="card drivers">
        <h4>What the model cares about most</h4>
        <p>Global feature importance from the trained forest.</p>
        <div id="drivers"></div>
      </div>
    </aside>
  </div>

  <footer>Built with Flask + scikit-learn · Predictions are model outputs, not financial advice.</footer>
</div>

<script>
let META = null;
const $ = (s, el=document) => el.querySelector(s);
const fmt = (v, step) => {
  const d = step < 1 ? 1 : 0;
  return Number(v).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d});
};
const CLS = {"Bull Market":"bull", "Recession":"rec", "Stable":"stable"};

async function init(){
  META = await (await fetch('/api/meta')).json();
  $('#trees').textContent = META.n_trees;
  buildForm(); buildPresets(); buildDrivers(); buildProbs(null);
}

function buildForm(){
  const form = $('#form'); form.innerHTML = '';
  META.groups.forEach(g => {
    const card = document.createElement('section'); card.className = 'card';
    card.innerHTML = `<div class="card-head"><div class="ico">${g.icon}</div>
      <div><h3>${g.title}</h3><small>${g.blurb}</small></div></div><div class="fields"></div>`;
    const box = $('.fields', card);
    META.fields.filter(f => f.group === g.id).forEach(f => box.appendChild(makeField(f)));
    form.appendChild(card);
  });
}

function makeField(f){
  const w = document.createElement('div'); w.className = 'field';
  if(f.kind === 'select'){
    w.innerHTML = `<label for="${f.key}">${f.label}</label><select id="${f.key}">` +
      f.options.map(o => `<option value="${o.value}" ${o.value==f.default?'selected':''}>${o.label}</option>`).join('') + `</select>`;
  } else {
    w.innerHTML = `<label for="${f.key}">${f.label}<output id="o_${f.key}"></output></label>
      <input type="range" id="${f.key}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}">`;
    const r = $('input', w), o = $('output', w);
    const sync = () => {
      const p = (r.value - r.min) / (r.max - r.min) * 100;
      r.style.setProperty('--p', p + '%');
      o.textContent = fmt(r.value, f.step) + (f.unit ? ' ' + f.unit : '');
    };
    r.addEventListener('input', sync); sync();
  }
  return w;
}

function setValue(key, val){
  const el = document.getElementById(key); if(!el) return;
  el.value = val; el.dispatchEvent(new Event('input'));
}
function collect(){
  const o = {}; META.fields.forEach(f => o[f.key] = document.getElementById(f.key).value); return o;
}

function buildPresets(){
  const box = $('#presets');
  Object.keys(META.presets).forEach(name => {
    if(!Object.keys(META.presets[name]).length) return;
    const b = document.createElement('button'); b.className = 'chip'; b.textContent = name;
    b.onclick = () => { Object.entries(META.presets[name]).forEach(([k,v]) => setValue(k,v)); predict(); window.scrollTo({top:0,behavior:'smooth'}); };
    box.appendChild(b);
  });
}

function buildDrivers(){
  const max = Math.max(...META.importances.map(i => i.value));
  $('#drivers').innerHTML = META.importances.map(i => `
    <div class="drv"><span>${i.label}</span>
      <div class="meter"><i style="width:${i.value/max*100}%"></i></div>
      <span>${(i.value*100).toFixed(1)}%</span></div>`).join('');
}

function buildProbs(p){
  $('#probs').innerHTML = META.classes.map(c => {
    const v = p ? p[c] : 0;
    return `<div class="prob c-${CLS[c]||'stable'}"><div class="row"><span>${(META.class_style[c]||{}).emoji||''} ${c}</span>
      <span>${(v*100).toFixed(1)}%</span></div><div class="meter"><i style="width:${v*100}%"></i></div></div>`;
  }).join('');
}

async function predict(){
  const btn = $('#go'), err = $('#err'); err.textContent = '';
  btn.disabled = true; btn.textContent = 'Analyzing…';
  try{
    const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(collect())});
    const d = await res.json();
    if(!res.ok) throw new Error(d.error || 'Prediction failed');
    show(d);
  }catch(e){ err.textContent = e.message; }
  btn.disabled = false; btn.textContent = 'Analyze market ✨';
}

function show(d){
  const box = $('#result');
  box.className = 'card result ' + (CLS[d.label]||'stable');
  void box.offsetWidth; box.classList.add('pop');
  $('#emoji').textContent = d.emoji;
  $('#verdict').textContent = d.label;
  $('#tag').textContent = d.tagline;
  const C = 540.35;
  $('#bar').style.strokeDashoffset = C * (1 - d.confidence);
  animateNum($('#pct'), d.confidence * 100);
  setTimeout(() => buildProbs(d.probabilities), 60);
  requestAnimationFrame(() => requestAnimationFrame(() => buildProbs(d.probabilities)));
}
function animateNum(el, to){
  const t0 = performance.now();
  (function tick(t){
    const k = Math.min(1, (t - t0)/900), e = 1 - Math.pow(1-k, 3);
    el.innerHTML = (to*e).toFixed(1) + '<small>%</small>';
    if(k<1) requestAnimationFrame(tick);
  })(t0);
}

function randomize(){
  META.fields.forEach(f => {
    if(f.kind === 'select'){ setValue(f.key, f.options[Math.floor(Math.random()*f.options.length)].value); }
    else{
      const n = Math.round((f.min + Math.random()*(f.max-f.min))/f.step)*f.step;
      setValue(f.key, n.toFixed(f.step<1?1:0));
    }
  });
  predict();
}
function resetAll(){
  META.fields.forEach(f => setValue(f.key, f.default));
  const box = $('#result'); box.className = 'card result idle';
  $('#emoji').textContent = '🔮'; $('#pct').innerHTML = '--<small>%</small>';
  $('#verdict').textContent = 'Awaiting input'; $('#tag').textContent = 'Adjust the signals, then run the analysis.';
  $('#bar').style.strokeDashoffset = 540.35; buildProbs(null); $('#err').textContent = '';
}

$('#go').onclick = predict; $('#random').onclick = randomize; $('#reset').onclick = resetAll;
init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
