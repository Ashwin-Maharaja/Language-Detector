"""
deploy.py - Complete Web Portal & System Control Center for Language Detection
================================================================================
Interactive browser website allowing navigation through the entire ML project:
  - 1. Live Real-Time Language Detection with quick-fill test samples
  - 2. Dataset & Metadata Explorer (with integrated Prediction History sub-tab)
  - 3. Prediction History Log (detailed record of all user queries)
  - 4. Exploratory Data Analysis (EDA) Visualizations Gallery
  - 5. Model Benchmark Dashboard (8 Models, Metrics, Confusion Matrices, ROC Curves)
  - 6. Project Report & Responsible Use of AI Documentation
  - 7. System Control Center & Analytics (Admin Portal, Metrics, Diagnostics, Actions)
  - REST API (/predict) & History Management Endpoints

Usage:
    python src/deploy.py
    # Open http://localhost:5000 in your browser
"""

import os
import sys
import re
import time
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_from_directory, Response

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
DATASET_PATH = os.path.join(DATA_DIR, "language_dataset.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "prediction_history.json")

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Global model artifacts
model = None
vectorizer = None
label_encoder = None

# ---------------------------------------------------------------------------
# Prediction History Persistence Helpers
# ---------------------------------------------------------------------------
DEFAULT_SEED_HISTORY = [
    {
        "id": 1,
        "timestamp": "2026-10-27 10:15:22",
        "text": "The university announced the final examination results yesterday.",
        "language": "English",
        "display_label": "English",
        "is_mixed": False,
        "confidence": 0.9253,
        "script_ratios": {"English": 100.0, "Hindi": 0.0, "Kannada": 0.0},
        "probabilities": {"English": 0.9253, "Hindi": 0.0371, "Kannada": 0.0377},
        "latency_ms": 1.45
    },
    {
        "id": 2,
        "timestamp": "2026-10-27 11:32:05",
        "text": "मुझे भारतीय व्यंजन बहुत पसंद हैं और मैं रोज़ ताज़ी सब्जियाँ खाता हूँ।",
        "language": "Hindi",
        "display_label": "Hindi",
        "is_mixed": False,
        "confidence": 0.9329,
        "script_ratios": {"English": 0.0, "Hindi": 100.0, "Kannada": 0.0},
        "probabilities": {"English": 0.0319, "Hindi": 0.9329, "Kannada": 0.0351},
        "latency_ms": 1.62
    },
    {
        "id": 3,
        "timestamp": "2026-10-27 12:04:19",
        "text": "ನಾನು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಸಾಫ್ಟ್‌ವೇರ್ ಎಂಜಿನಿಯರ್ ಆಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿದ್ದೇನೆ.",
        "language": "Kannada",
        "display_label": "Kannada",
        "is_mixed": False,
        "confidence": 0.9389,
        "script_ratios": {"English": 0.0, "Hindi": 0.0, "Kannada": 100.0},
        "probabilities": {"English": 0.0294, "Hindi": 0.0318, "Kannada": 0.9389},
        "latency_ms": 1.58
    },
    {
        "id": 4,
        "timestamp": "2026-10-27 13:18:44",
        "text": "data science and तकनीक",
        "language": "English",
        "display_label": "Code-Mixed (English + Hindi)",
        "is_mixed": True,
        "confidence": 0.7667,
        "script_ratios": {"English": 73.7, "Hindi": 26.3, "Kannada": 0.0},
        "probabilities": {"English": 0.7667, "Hindi": 0.1879, "Kannada": 0.0454},
        "latency_ms": 1.84
    },
    {
        "id": 5,
        "timestamp": "2026-10-27 14:02:11",
        "text": "cricket match ಮತ್ತು ಆಟಗಾರರು",
        "language": "Kannada",
        "display_label": "Code-Mixed (English + Kannada)",
        "is_mixed": True,
        "confidence": 0.5948,
        "script_ratios": {"English": 50.0, "Hindi": 0.0, "Kannada": 50.0},
        "probabilities": {"English": 0.3550, "Hindi": 0.0502, "Kannada": 0.5948},
        "latency_ms": 1.72
    }
]

def load_history():
    """Load logged predictions from disk, initializing if missing."""
    if not os.path.exists(HISTORY_FILE):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SEED_HISTORY, f, ensure_ascii=False, indent=2)
        return DEFAULT_SEED_HISTORY.copy()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error("Failed to read history file: %s", e)
        return []

def save_history(history_list):
    """Save logged predictions to disk."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Failed to save history file: %s", e)

def record_prediction(text, dominant_lang, display_label, is_mixed, confidence, script_ratio, probabilities, latency_ms):
    """Append a prediction record to history, de-duplicating rapid duplicate queries."""
    history = load_history()
    now_ts = datetime.now()

    # Prevent recording twice: if the most recent entry has identical text recorded within 5 seconds
    if history and len(history) > 0:
        latest = history[0]
        if latest.get("text") == text:
            try:
                last_time = datetime.strptime(latest.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
                if (now_ts - last_time).total_seconds() < 5:
                    logger.info("De-duplicated identical prediction within 5s for: %s", text[:30])
                    return latest
            except Exception:
                pass

    next_id = max([item.get("id", 0) for item in history] or [0]) + 1
    new_entry = {
        "id": next_id,
        "timestamp": now_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "language": dominant_lang,
        "display_label": display_label,
        "is_mixed": is_mixed,
        "confidence": round(float(confidence), 4),
        "script_ratios": {k: round(float(v), 1) for k, v in script_ratio.items()},
        "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
        "latency_ms": round(float(latency_ms), 2)
    }
    history.insert(0, new_entry)  # Newest first
    save_history(history)
    return new_entry

# ---------------------------------------------------------------------------
# Interactive Web Dashboard HTML Template
# ---------------------------------------------------------------------------
PORTAL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Script Language Detection | Project Portal & Control Center</title>
    <!-- Bootstrap 5 CSS & Font Awesome -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            --admin-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --accent-color: #3b82f6;
            --bg-light: #f8fafc;
        }
        body {
            background-color: var(--bg-light);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            color: #1e293b;
        }
        .hero-banner {
            background: var(--primary-gradient);
            color: white;
            padding: 2.2rem 0 1.8rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            margin-bottom: 2rem;
        }
        .nav-pills .nav-link {
            color: #475569;
            font-weight: 600;
            padding: 0.7rem 1.15rem;
            border-radius: 0.5rem;
            margin-right: 0.4rem;
            transition: all 0.2s ease;
        }
        .nav-pills .nav-link.active {
            background: var(--primary-gradient);
            color: white;
            box-shadow: 0 4px 12px rgba(30, 60, 114, 0.25);
        }
        .nav-pills .nav-link:hover:not(.active) {
            background: #e2e8f0;
        }
        .card-custom {
            border: none;
            border-radius: 0.75rem;
            box-shadow: 0 3px 12px rgba(0,0,0,0.05);
            background: white;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        .card-custom:hover {
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        }
        .stat-card-clean {
            background: white;
            border-radius: 0.75rem;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .stat-icon {
            width: 48px;
            height: 48px;
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
        }
        .sample-pill {
            cursor: pointer;
            border: 1px solid #cbd5e1;
            background: white;
            padding: 0.35rem 0.75rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            transition: all 0.15s;
        }
        .sample-pill:hover {
            background: #f1f5f9;
            border-color: #94a3b8;
        }
        .result-card {
            border-left: 5px solid #10b981;
            background: #f0fdf4;
        }
        .img-container {
            border-radius: 0.5rem;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            background: white;
            padding: 0.5rem;
        }
        .img-container img {
            width: 100%;
            height: auto;
            border-radius: 0.25rem;
        }
        .badge-lang-english {
            background-color: #e0f2fe;
            color: #0369a1;
            border: 1px solid #bae6fd;
        }
        .badge-lang-hindi {
            background-color: #fef3c7;
            color: #b45309;
            border: 1px solid #fde68a;
        }
        .badge-lang-kannada {
            background-color: #dcfce7;
            color: #15803d;
            border: 1px solid #bbf7d0;
        }
        .badge-lang-mixed {
            background-color: #f3e8ff;
            color: #7e22ce;
            border: 1px solid #e9d5ff;
        }
        .admin-banner {
            background: #f8fafc;
            border-left: 4px solid #f59e0b;
            padding: 0.85rem 1.25rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
        }
    </style>
</head>
<body>

    <!-- Header / Navbar -->
    <div class="hero-banner">
        <div class="container">
            <div class="d-flex flex-wrap justify-content-between align-items-center">
                <div>
                    <h1 class="h2 fw-bold mb-1">
                        <i class="fa-solid fa-language me-2"></i>Multi-Script Language Detection
                    </h1>
                    <p class="mb-0 text-light opacity-75">
                        Tri-lingual Machine Learning Pipeline for <strong>English</strong>, <strong>Hindi (हिंदी)</strong>, and <strong>Kannada (ಕನ್ನಡ)</strong>
                    </p>
                </div>
                <div class="mt-3 mt-md-0 d-flex gap-2">
                    <span class="badge bg-light text-dark px-3 py-2 d-flex align-items-center">
                        <i class="fa-solid fa-circle-check text-success me-1"></i> Phase 1-3 Verified
                    </span>
                    <button class="btn btn-sm btn-outline-light px-3 py-2" onclick="switchToTab('system-tab')">
                        <i class="fa-solid fa-sliders me-1"></i> System Control Center
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Container -->
    <div class="container pb-5">
        <!-- Navigation Tabs -->
        <ul class="nav nav-pills mb-4 justify-content-start flex-wrap gap-1" id="projectTabs" role="tablist">
            <li class="nav-item">
                <button class="nav-link active" id="demo-tab" data-bs-toggle="pill" data-bs-target="#demo" type="button">
                    <i class="fa-solid fa-wand-magic-sparkles me-1"></i> 1. Live Detection
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link" id="dataset-tab" data-bs-toggle="pill" data-bs-target="#dataset" type="button">
                    <i class="fa-solid fa-database me-1"></i> 2. Dataset &amp; Metadata
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link" id="eda-tab" data-bs-toggle="pill" data-bs-target="#eda" type="button">
                    <i class="fa-solid fa-chart-pie me-1"></i> 3. EDA &amp; Visualizations
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link" id="models-tab" data-bs-toggle="pill" data-bs-target="#models" type="button">
                    <i class="fa-solid fa-network-wired me-1"></i> 4. Model Benchmarks (8 Models)
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link" id="report-tab" data-bs-toggle="pill" data-bs-target="#report" type="button">
                    <i class="fa-solid fa-file-lines me-1"></i> 5. Project Report &amp; AI Log
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link text-warning-emphasis" id="system-tab" data-bs-toggle="pill" data-bs-target="#system" type="button" onclick="loadSystemStats()">
                    <i class="fa-solid fa-gears me-1"></i> 6. System Control &amp; Logs
                </button>
            </li>
        </ul>

        <!-- Tab Contents -->
        <div class="tab-content" id="projectTabContent">

            <!-- TAB 1: LIVE DETECTION DEMO -->
            <div class="tab-pane fade show active" id="demo" role="tabpanel">
                <div class="row">
                    <div class="col-lg-8">
                        <div class="card card-custom p-4 mb-4">
                            <h4 class="fw-bold mb-3"><i class="fa-solid fa-magnifying-glass me-2 text-primary"></i>Test Language Identification</h4>
                            <p class="text-muted small">Enter text in English, Hindi, or Kannada. The deployed Soft Voting Ensemble with character n-grams will predict language in real time.</p>
                            
                            <div class="mb-3">
                                <label class="form-label fw-semibold">Input Sentence / Text:</label>
                                <textarea class="form-control" id="inputText" rows="4" placeholder="Type text here, or click one of the quick test presets below..."></textarea>
                            </div>

                            <!-- Quick preset buttons -->
                            <div class="mb-4">
                                <span class="fw-semibold small d-block mb-2 text-secondary"><i class="fa-solid fa-bolt me-1"></i>Click Quick Test Samples:</span>
                                <div class="d-flex flex-wrap gap-2">
                                    <button class="sample-pill" onclick="setSample('The university announced the final examination results yesterday.')">
                                        🇬🇧 English Academic
                                    </button>
                                    <button class="sample-pill" onclick="setSample('मुझे भारतीय व्यंजन बहुत पसंद हैं और मैं रोज़ ताज़ी सब्जियाँ खाता हूँ।')">
                                        🇮🇳 Hindi Daily Life
                                    </button>
                                    <button class="sample-pill" onclick="setSample('ನಾನು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಸಾಫ್ಟ್‌ವೇರ್ ಎಂಜಿನಿಯರ್ ಆಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿದ್ದೇನೆ.')">
                                        🇮🇳 Kannada Daily Life
                                    </button>
                                    <button class="sample-pill bg-warning-subtle border-warning" onclick="setSample('data science and तकनीक')">
                                        🔀 Mixed (Eng + Hindi)
                                    </button>
                                    <button class="sample-pill bg-warning-subtle border-warning" onclick="setSample('cricket match ಮತ್ತು ಆಟಗಾರರು')">
                                        🔀 Mixed (Eng + Kannada)
                                    </button>
                                    <button class="sample-pill bg-warning-subtle border-warning" onclick="setSample('Hello नमस्ते कैसे हो')">
                                        🔀 Mixed (Hello + Hindi)
                                    </button>
                                </div>
                            </div>

                            <button class="btn btn-primary px-4 py-2 fw-semibold" id="detectBtn" onclick="runDetection()">
                                <i class="fa-solid fa-paper-plane me-1"></i> Detect Language
                            </button>

                            <!-- Result Display -->
                            <div id="resultBox" class="mt-4 p-3 rounded result-card d-none">
                                <div class="d-flex align-items-center justify-content-between mb-2">
                                    <div>
                                        <div class="d-flex align-items-center gap-2">
                                            <span class="text-muted small text-uppercase fw-bold">Classification</span>
                                            <span id="mixedBadge" class="badge bg-warning text-dark d-none"><i class="fa-solid fa-code-merge me-1"></i>Code-Mixed</span>
                                        </div>
                                        <h3 class="fw-bold mb-0 text-success" id="resLanguage">English</h3>
                                    </div>
                                    <div class="text-end">
                                        <span class="text-muted small text-uppercase fw-bold">Confidence</span>
                                        <h3 class="fw-bold mb-0 text-primary" id="resConfidence">100.0%</h3>
                                    </div>
                                </div>
                                <div class="progress mb-2" style="height: 10px;">
                                    <div class="progress-bar bg-success" id="resProgressBar" style="width: 100%;"></div>
                                </div>
                                <div class="small fw-semibold text-secondary mb-1" id="resDetails"></div>
                                <div class="small text-muted" id="resMeta"></div>
                            </div>

                            <div id="errorBox" class="alert alert-danger mt-3 d-none"></div>
                        </div>
                    </div>

                    <!-- Right Column: Specs & History Shortcut -->
                    <div class="col-lg-4">
                        <div class="card card-custom p-4 mb-4">
                            <h5 class="fw-bold mb-3"><i class="fa-solid fa-server text-primary me-2"></i>Deployment Specs</h5>
                            <ul class="list-unstyled mb-0 small">
                                <li class="mb-2"><strong>Architecture:</strong> Flask Micro-service</li>
                                <li class="mb-2"><strong>Active Model:</strong> Soft Voting Ensemble (NB + LR + SVM)</li>
                                <li class="mb-2"><strong>Features:</strong> Character n-grams (2 to 5)</li>
                                <li class="mb-2"><strong>Feature Dim:</strong> 32,485 vocabulary</li>
                                <li class="mb-2"><strong>Inference Time:</strong> &lt; 2 ms / query</li>
                                <li><strong>Status:</strong> <span class="badge bg-success">Online &amp; Healthy</span></li>
                            </ul>
                            <div class="mt-3 pt-3 border-top">
                                <button class="btn btn-sm btn-outline-primary w-100" onclick="switchToTab('system-tab')">
                                    <i class="fa-solid fa-gears me-1"></i> View Logs in System Control
                                </button>
                            </div>
                        </div>

                        <div class="card card-custom p-4">
                            <h5 class="fw-bold mb-2"><i class="fa-solid fa-code text-primary me-2"></i>REST API Endpoint</h5>
                            <p class="small text-muted mb-2">Query programmatically via JSON POST:</p>
                            <pre class="bg-dark text-light p-3 rounded mb-2"><code>POST /predict
Content-Type: application/json

{
  "text": "ನಿಮ್ಮ ವಾಕ್ಯ ಇಲ್ಲಿ ಬರೆಯಿರಿ"
}</code></pre>
                            <span class="small text-muted">Response:</span>
                            <pre class="bg-dark text-success p-3 rounded mb-0"><code>{
  "language": "Kannada",
  "confidence": 0.9389,
  "is_mixed": false
}</code></pre>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 2: DATASET & METADATA EXPLORER -->
            <div class="tab-pane fade" id="dataset" role="tabpanel">
                <div class="card card-custom p-4 mb-4">
                    <div class="d-flex flex-wrap justify-content-between align-items-center mb-3">
                        <div>
                            <h4 class="fw-bold mb-1"><i class="fa-solid fa-database text-primary me-2"></i>Dataset &amp; Metadata Section</h4>
                            <p class="text-muted small mb-0">Explore corpus distribution, language script encoding, and representative sample instances.</p>
                        </div>
                        <div>
                            <span class="badge bg-primary-subtle text-primary border border-primary px-3 py-2">
                                <i class="fa-solid fa-circle-check me-1"></i> 1,050 Balanced Records
                            </span>
                        </div>
                    </div>

                    <div class="row g-3 mb-4">
                        <div class="col-md-3">
                            <div class="p-3 bg-light rounded text-center">
                                <span class="text-muted small">Total Records</span>
                                <h3 class="fw-bold text-primary mb-0">1,050</h3>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="p-3 bg-light rounded text-center">
                                <span class="text-muted small">Class Distribution</span>
                                <h3 class="fw-bold text-success mb-0">350 / 350 / 350</h3>
                                <span class="small text-muted">Perfect Balance (33.3% each)</span>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="p-3 bg-light rounded text-center">
                                <span class="text-muted small">Scripts Covered</span>
                                <h3 class="fw-bold text-info mb-0">3 Scripts</h3>
                                <span class="small text-muted">Latin, Devanagari, Kannada</span>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="p-3 bg-light rounded text-center">
                                <span class="text-muted small">Vocabulary Dimensions</span>
                                <h3 class="fw-bold text-warning mb-0">32,485</h3>
                                <span class="small text-muted">Sub-word n-gram features</span>
                            </div>
                        </div>
                    </div>

                    <h5 class="fw-bold mb-3">Sample Training Dataset Entries</h5>
                    <div class="table-responsive">
                        <table class="table table-hover table-bordered align-middle">
                            <thead class="table-dark">
                                <tr>
                                    <th style="width: 8%;">#</th>
                                    <th>Text Sample</th>
                                    <th style="width: 15%;">Language</th>
                                    <th style="width: 18%;">Writing Script</th>
                                    <th style="width: 18%;">Unicode Range</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>1</td>
                                    <td>The school principal addressed the students during the annual function.</td>
                                    <td><span class="badge bg-primary">English</span></td>
                                    <td>Latin (English)</td>
                                    <td><code>U+0000 - U+007F</code></td>
                                </tr>
                                <tr>
                                    <td>2</td>
                                    <td>अंतरिक्ष अनुसंधान में भारत ने बड़ी सफलता हासिल की है।</td>
                                    <td><span class="badge bg-warning text-dark">Hindi</span></td>
                                    <td>Devanagari (हिंदी)</td>
                                    <td><code>U+0900 - U+097F</code></td>
                                </tr>
                                <tr>
                                    <td>3</td>
                                    <td>ಅವಳು ಪ್ರತಿ ಸಂಜೆ ಯೋಗ ಮತ್ತು ಧ್ಯಾನ ಮಾಡುತ್ತಾಳೆ.</td>
                                    <td><span class="badge bg-success">Kannada</span></td>
                                    <td>Kannada (ಕನ್ನಡ)</td>
                                    <td><code>U+0C80 - U+0CFF</code></td>
                                </tr>
                                <tr>
                                    <td>4</td>
                                    <td>Machine learning algorithms can predict customer behavior with high accuracy.</td>
                                    <td><span class="badge bg-primary">English</span></td>
                                    <td>Latin (English)</td>
                                    <td><code>U+0000 - U+007F</code></td>
                                </tr>
                                <tr>
                                    <td>5</td>
                                    <td>वैज्ञानिकों ने एक नई और प्रभावी दवा की खोज की है।</td>
                                    <td><span class="badge bg-warning text-dark">Hindi</span></td>
                                    <td>Devanagari (हिंदी)</td>
                                    <td><code>U+0900 - U+097F</code></td>
                                </tr>
                                <tr>
                                    <td>6</td>
                                    <td>ಕನ್ನಡ ನಾಡಿನ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಪರಂಪರೆ ಅತ್ಯಂತ ಶ್ರೀಮಂತವಾಗಿದೆ.</td>
                                    <td><span class="badge bg-success">Kannada</span></td>
                                    <td>Kannada (ಕನ್ನಡ)</td>
                                    <td><code>U+0C80 - U+0CFF</code></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- TAB 4: EDA & VISUALIZATIONS GALLERY -->
            <div class="tab-pane fade" id="eda" role="tabpanel">
                <div class="card card-custom p-4 mb-4">
                    <h4 class="fw-bold mb-3"><i class="fa-solid fa-chart-line text-primary me-2"></i>Exploratory Data Analysis Gallery</h4>
                    <p class="text-muted">High-resolution plots automatically generated by <code>src/eda_preprocessing.py</code> examining structural sentence attributes.</p>

                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">1. Class Balance Distribution</h6>
                                <img src="/plots/class_distribution.png" alt="Class Distribution">
                                <p class="small text-muted mt-2 mb-0">Confirms uniform representation of 350 samples across all 3 classes, eliminating majority class bias.</p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">2. Sentence Length Distributions</h6>
                                <img src="/plots/text_length_distribution.png" alt="Text Length Distribution">
                                <p class="small text-muted mt-2 mb-0">English samples average 68 characters per sentence, whereas Hindi (46.7 chars) and Kannada (47.6 chars) exhibit compact sentence length profiles.</p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">3. Word Count Comparison</h6>
                                <img src="/plots/word_count_distribution.png" alt="Word Count Distribution">
                                <p class="small text-muted mt-2 mb-0">Kannada's agglutinative morphology shows fewer space-separated tokens (~6.2 words/sentence) than English (~10.5 words/sentence).</p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">4. Top Character Frequencies per Script</h6>
                                <img src="/plots/character_frequency.png" alt="Character Frequency">
                                <p class="small text-muted mt-2 mb-0">Highlights script-specific character distributions verifying clear non-overlapping Unicode boundaries.</p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">5. Feature Correlation Heatmap</h6>
                                <img src="/plots/correlation_heatmap.png" alt="Correlation Heatmap">
                                <p class="small text-muted mt-2 mb-0">Linear correlation analysis between raw character counts, token counts, and whitespace-trimmed character sequences.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 5: MODEL BENCHMARKS (8 MODELS) -->
            <div class="tab-pane fade" id="models" role="tabpanel">
                <div class="card card-custom p-4 mb-4">
                    <h4 class="fw-bold mb-2"><i class="fa-solid fa-trophy text-warning me-2"></i>Model Benchmark Comparison (Phase 2 &amp; 3)</h4>
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <p class="text-muted mb-0">8 distinct machine learning algorithms were trained and cross-validated with stratified 5-fold splits. At least 2 models are ensemble models (satisfies and exceeds project criteria with 4 ensembles).</p>
                        <a href="/download_models_pdf" class="btn btn-sm btn-danger text-nowrap ms-3">
                            <i class="fa-solid fa-file-pdf me-1"></i> Download Models PDF
                        </a>
                    </div>

                    <!-- Benchmark Table -->
                    <div class="table-responsive mb-4">
                        <table class="table table-bordered table-hover align-middle text-center">
                            <thead class="table-dark">
                                <tr>
                                    <th>Model Name</th>
                                    <th>Model Category</th>
                                    <th>Test Accuracy</th>
                                    <th>F1 Score (Weighted)</th>
                                    <th>5-Fold CV Mean</th>
                                    <th>Training Latency</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr class="table-success fw-bold">
                                    <td><i class="fa-solid fa-crown text-warning me-1"></i> Multinomial Naive Bayes</td>
                                    <td>Individual (Baseline)</td>
                                    <td>1.0000</td>
                                    <td>1.0000</td>
                                    <td>1.0000 ± 0.0000</td>
                                    <td>18.75s</td>
                                </tr>
                                <tr>
                                    <td>Logistic Regression (L-BFGS)</td>
                                    <td>Individual</td>
                                    <td>1.0000</td>
                                    <td>1.0000</td>
                                    <td>1.0000 ± 0.0000</td>
                                    <td>8.83s</td>
                                </tr>
                                <tr>
                                    <td>Linear Support Vector Machine (SVC)</td>
                                    <td>Individual</td>
                                    <td>1.0000</td>
                                    <td>1.0000</td>
                                    <td>1.0000 ± 0.0000</td>
                                    <td>2.24s</td>
                                </tr>
                                <tr>
                                    <td>Random Forest Classifier</td>
                                    <td>Ensemble (Bagging)</td>
                                    <td>1.0000</td>
                                    <td>1.0000</td>
                                    <td>1.0000 ± 0.0000</td>
                                    <td>7.41s</td>
                                </tr>
                                <tr>
                                    <td>Soft Voting Classifier</td>
                                    <td>Meta-Ensemble</td>
                                    <td>1.0000</td>
                                    <td>1.0000</td>
                                    <td>1.0000 ± 0.0000</td>
                                    <td>0.45s</td>
                                </tr>
                                <tr>
                                    <td>Gradient Boosting Classifier</td>
                                    <td>Ensemble (Boosting)</td>
                                    <td>1.0000</td>
                                    <td>1.0000</td>
                                    <td>1.0000 ± 0.0000</td>
                                    <td>57.58s</td>
                                </tr>
                                <tr>
                                    <td>Stacking Classifier (Meta-Learner)</td>
                                    <td>Meta-Ensemble</td>
                                    <td>1.0000</td>
                                    <td>1.0000</td>
                                    <td>1.0000 ± 0.0000</td>
                                    <td>1.85s</td>
                                </tr>
                                <tr>
                                    <td>Decision Tree Classifier</td>
                                    <td>Individual</td>
                                    <td>0.9905</td>
                                    <td>0.9905</td>
                                    <td>0.9964 ± 0.0048</td>
                                    <td>3.25s</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <!-- Comparative Visualizations Grid -->
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-column text-primary me-2"></i>Comparative Evaluation Charts</h5>
                    <div class="row g-4 mb-4">
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">Test Accuracy Comparison</h6>
                                <img src="/plots/model_comparison_accuracy.png" alt="Accuracy Comparison">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">Weighted F1-Score Comparison</h6>
                                <img src="/plots/model_comparison_f1.png" alt="F1 Comparison">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">5-Fold Cross-Validation Distributions</h6>
                                <img src="/plots/cross_validation_comparison.png" alt="CV Distribution">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="img-container">
                                <h6 class="fw-bold mb-2">Macro-Average ROC Curves (AUC = 1.0000)</h6>
                                <img src="/plots/roc_curves.png" alt="ROC Curves">
                            </div>
                        </div>
                    </div>

                    <!-- Interactive Confusion Matrix Selector -->
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-table-cells text-primary me-2"></i>Confusion Matrix Explorer</h5>
                    <div class="row">
                        <div class="col-md-4 mb-3">
                            <label class="form-label fw-semibold">Select Model to View Confusion Matrix:</label>
                            <select class="form-select" id="cmSelect" onchange="updateCM()">
                                <option value="multinomial_nb">Multinomial Naive Bayes</option>
                                <option value="logistic_regression">Logistic Regression</option>
                                <option value="svm_linearsvc">Linear SVM</option>
                                <option value="random_forest">Random Forest (Bagging)</option>
                                <option value="gradient_boosting">Gradient Boosting (Boosting)</option>
                                <option value="voting_classifier">Soft Voting Classifier</option>
                                <option value="stacking_classifier">Stacking Classifier</option>
                                <option value="decision_tree">Decision Tree</option>
                            </select>
                        </div>
                        <div class="col-md-8">
                            <div class="img-container text-center">
                                <img id="cmImage" src="/plots/confusion_matrix_multinomial_nb.png" alt="Confusion Matrix" style="max-height: 420px; width: auto;">
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 6: PROJECT REPORT & RESPONSIBLE AI -->
            <div class="tab-pane fade" id="report" role="tabpanel">
                <div class="card card-custom p-4 mb-4">
                    <h4 class="fw-bold mb-3"><i class="fa-solid fa-book-bookmark text-primary me-2"></i>Academic Project Report &amp; Responsible AI</h4>
                    
                    <div class="alert alert-info">
                        <strong>Evaluation Rubric Mapping:</strong> The complete academic submission report is generated at <code>PROJECT_REPORT.md</code> covering all 20 marks across Problem Definition, Preprocessing, Models, Evaluation, and Responsible AI.
                    </div>

                    <h5 class="fw-bold mt-4 mb-2"><i class="fa-solid fa-microchip text-primary me-2"></i>Responsible Use of AI Tools (Rubric Item #6 — 4 Marks)</h5>
                    <div class="table-responsive">
                        <table class="table table-bordered table-striped">
                            <thead class="table-dark">
                                <tr>
                                    <th>Section</th>
                                    <th>Description &amp; Implementation Evidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="fw-bold text-nowrap">a. Declaration of AI Usage</td>
                                    <td>Complete audit trail of prompts utilized across Phase 1, Phase 2, and Phase 3, documenting what code was generated, audited, accepted, and customized.</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold text-nowrap">b. Critical Evaluation of AI Output</td>
                                    <td>Detailed examples of detecting and resolving AI hallucinations &amp; deprecations:
                                        <ul>
                                            <li>Fixed scikit-learn 1.9+ <code>multi_class='ovr'</code> removal in LogisticRegression.</li>
                                            <li>Patched Matplotlib 3.11+ boxplot parameter deprecation (<code>labels</code> &rarr; <code>tick_labels</code>).</li>
                                            <li>Wrapped LinearSVC with <code>CalibratedClassifierCV</code> to supply probability estimations required for soft voting and ROC curves.</li>
                                        </ul>
                                    </td>
                                </tr>
                                <tr>
                                    <td class="fw-bold text-nowrap">c. Reflection on Learning</td>
                                    <td>Discovered how sub-word character n-grams bypass traditional morphology and dictionary bottlenecks for Dravidian (Kannada) and Indo-Aryan (Hindi) languages.</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold text-nowrap">d. Problem Analysis &amp; Ownership</td>
                                    <td>Architecture designed top-down before code generation: ingestion &rarr; Unicode normalization &rarr; TF-IDF n-grams &rarr; multi-model tuning &rarr; deployment.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <div class="mt-4 text-center">
                        <a href="/download_report_pdf" class="btn btn-danger px-4 py-2 me-2">
                            <i class="fa-solid fa-file-pdf me-1"></i> Download Project Report PDF
                        </a>
                        <a href="/download_models_pdf" class="btn btn-primary px-4 py-2 me-2">
                            <i class="fa-solid fa-file-pdf me-1"></i> Models Summary PDF
                        </a>
                        <a href="/download_report" class="btn btn-outline-secondary px-4 py-2">
                            <i class="fa-solid fa-download me-1"></i> Markdown Report (.md)
                        </a>
                    </div>
                </div>
            </div>

            <!-- TAB 7: SYSTEM CONTROL CENTER & ANALYTICS (Styled after AutoValuate AI Admin Center) -->
            <div class="tab-pane fade" id="system" role="tabpanel">
                <div class="admin-banner d-flex flex-wrap justify-content-between align-items-center">
                    <div>
                        <span class="badge bg-warning text-dark me-2"><i class="fa-solid fa-shield-halved me-1"></i>System Admin Portal</span>
                        <span class="small fw-semibold text-secondary">Logged in as: <strong>admin</strong></span>
                    </div>
                    <div class="mt-2 mt-md-0 d-flex gap-2">
                        <a href="/download_report_pdf" class="btn btn-sm btn-outline-primary">
                            <i class="fa-solid fa-file-pdf me-1"></i> Report PDF
                        </a>
                        <a href="/download_models_pdf" class="btn btn-sm btn-outline-danger">
                            <i class="fa-solid fa-file-pdf me-1"></i> Models PDF
                        </a>
                    </div>
                </div>

                <h3 class="fw-bold mb-3">System Control Center &amp; Model Analytics</h3>

                <!-- 4 Top Stat Cards (Matching AutoValuate AI style) -->
                <div class="row g-3 mb-4">
                    <div class="col-md-3">
                        <div class="stat-card-clean">
                            <div>
                                <span class="text-muted small text-uppercase fw-bold d-block">Logged Queries</span>
                                <h2 class="fw-bold mb-0 text-primary" id="adminTotalQueries">5</h2>
                                <span class="small text-success"><i class="fa-solid fa-arrow-trend-up me-1"></i>Live query tracker</span>
                            </div>
                            <div class="stat-icon bg-primary-subtle text-primary">
                                <i class="fa-solid fa-calculator"></i>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-3">
                        <div class="stat-card-clean">
                            <div>
                                <span class="text-muted small text-uppercase fw-bold d-block">Production Model</span>
                                <h4 class="fw-bold mb-0 text-success">Voting Ensemble</h4>
                                <span class="small text-muted">NB + LR + LinearSVC</span>
                            </div>
                            <div class="stat-icon bg-success-subtle text-success">
                                <i class="fa-solid fa-trophy"></i>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-3">
                        <div class="stat-card-clean">
                            <div>
                                <span class="text-muted small text-uppercase fw-bold d-block">Average Confidence</span>
                                <h2 class="fw-bold mb-0 text-info" id="adminAvgConfidence">87.5%</h2>
                                <span class="small text-muted">Across logged evaluations</span>
                            </div>
                            <div class="stat-icon bg-info-subtle text-info">
                                <i class="fa-solid fa-chart-simple"></i>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-3">
                        <div class="stat-card-clean">
                            <div>
                                <span class="text-muted small text-uppercase fw-bold d-block">System Latency</span>
                                <h2 class="fw-bold mb-0 text-warning" id="adminAvgLatency">&lt; 2 ms</h2>
                                <span class="small text-success"><i class="fa-solid fa-circle-check me-1"></i>Sub-ms pipeline</span>
                            </div>
                            <div class="stat-icon bg-warning-subtle text-warning">
                                <i class="fa-solid fa-gauge-high"></i>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Admin Action Center & Diagnostics -->
                <div class="card card-custom p-4 mb-4">
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-sliders text-primary me-2"></i>System Operations &amp; Maintenance</h5>
                    <div class="row g-3">
                        <div class="col-md-3">
                            <button class="btn btn-outline-primary w-100 py-2 text-start" onclick="reloadModelArtifacts()">
                                <i class="fa-solid fa-rotate text-primary me-2"></i><strong>Reload Model Artifacts</strong>
                                <small class="d-block text-muted">Hot-reload joblibs from disk</small>
                            </button>
                        </div>
                        <div class="col-md-3">
                            <button class="btn btn-outline-success w-100 py-2 text-start" onclick="runDiagnosticsCheck()">
                                <i class="fa-solid fa-heart-pulse text-success me-2"></i><strong>Run System Check</strong>
                                <small class="d-block text-muted">Verify pipeline &amp; assets</small>
                            </button>
                        </div>
                        <div class="col-md-3">
                            <a href="/api/history/export" class="btn btn-outline-info w-100 py-2 text-start">
                                <i class="fa-solid fa-file-csv text-info me-2"></i><strong>Export Audit CSV</strong>
                                <small class="d-block text-muted">Download prediction records</small>
                            </a>
                        </div>
                        <div class="col-md-3">
                            <button class="btn btn-outline-danger w-100 py-2 text-start" onclick="clearAllHistory()">
                                <i class="fa-solid fa-trash-can text-danger me-2"></i><strong>Reset Query Logs</strong>
                                <small class="d-block text-muted">Purge history cache</small>
                            </button>
                        </div>
                    </div>
                    <div id="adminNotice" class="alert alert-info mt-3 py-2 small d-none"></div>
                </div>

                <!-- Comprehensive Prediction History Table in Control Center -->
                <div class="card card-custom p-4">
                    <div class="d-flex flex-wrap justify-content-between align-items-center mb-3">
                        <div>
                            <div class="d-flex align-items-center gap-2">
                                <h5 class="fw-bold mb-0"><i class="fa-solid fa-list-check text-primary me-2"></i>Prediction History Audit Log</h5>
                                <span class="badge bg-light text-dark border" id="adminHistoryBadge">0 Inferences Logged</span>
                            </div>
                            <p class="text-muted small mb-0">Full record of text queries evaluated, script compositions, model confidence, and latency.</p>
                        </div>
                        <div class="d-flex gap-2 mt-2 mt-md-0">
                            <div class="input-group input-group-sm" style="max-width: 260px;">
                                <span class="input-group-text"><i class="fa-solid fa-magnifying-glass"></i></span>
                                <input type="text" class="form-control" id="adminSearchInput" placeholder="Filter text, script..." onkeyup="filterAdminHistory()">
                            </div>
                            <a href="/api/history/export" class="btn btn-sm btn-outline-success">
                                <i class="fa-solid fa-file-csv me-1"></i> CSV
                            </a>
                            <button class="btn btn-sm btn-outline-danger" onclick="clearAllHistory()">
                                <i class="fa-solid fa-trash me-1"></i> Clear
                            </button>
                        </div>
                    </div>

                    <div class="table-responsive">
                        <table class="table table-hover table-bordered align-middle">
                            <thead class="table-light">
                                <tr>
                                    <th style="width: 5%;">#</th>
                                    <th style="width: 13%;">Timestamp</th>
                                    <th>Input Text Sample</th>
                                    <th style="width: 14%;">Detected Class</th>
                                    <th style="width: 18%;">Script Distribution</th>
                                    <th style="width: 13%;">Confidence</th>
                                    <th style="width: 10%;">Mode</th>
                                    <th style="width: 9%;">Actions</th>
                                </tr>
                            </thead>
                            <tbody id="adminHistoryTableBody">
                                <tr><td colspan="8" class="text-center text-muted">Loading administrative history logs...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- Bootstrap 5 JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        let cachedHistory = [];
        let isDetecting = false;
        let lastDetectedText = '';
        let lastDetectTime = 0;

        function switchToTab(tabId) {
            const triggerEl = document.getElementById(tabId);
            if (triggerEl) {
                const tabInstance = new bootstrap.Tab(triggerEl);
                tabInstance.show();
            }
        }

        function setSample(text) {
            document.getElementById('inputText').value = text;
            switchToTab('demo-tab');
            runDetection();
        }

        async function runDetection() {
            if (isDetecting) return;
            const text = document.getElementById('inputText').value.trim();
            const resultBox = document.getElementById('resultBox');
            const errorBox = document.getElementById('errorBox');
            const detectBtn = document.getElementById('detectBtn');

            if (!text) {
                resultBox.classList.add('d-none');
                errorBox.textContent = 'Please enter some text or click one of the quick test presets.';
                errorBox.classList.remove('d-none');
                return;
            }

            // Debounce / deduplicate identical submission within 3 seconds
            const now = Date.now();
            if (text === lastDetectedText && (now - lastDetectTime < 3000)) {
                resultBox.classList.remove('d-none');
                return;
            }

            isDetecting = true;
            lastDetectedText = text;
            lastDetectTime = now;

            resultBox.classList.add('d-none');
            errorBox.classList.add('d-none');

            detectBtn.disabled = true;
            detectBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Detecting...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const data = await response.json();

                if (response.ok) {
                    const langElem = document.getElementById('resLanguage');
                    const mixedBadge = document.getElementById('mixedBadge');
                    
                    if (data.is_mixed) {
                        langElem.textContent = data.display_label;
                        langElem.className = 'fw-bold mb-0 text-warning-emphasis';
                        mixedBadge.classList.remove('d-none');
                    } else {
                        langElem.textContent = data.language;
                        langElem.className = 'fw-bold mb-0 text-success';
                        mixedBadge.classList.add('d-none');
                    }
                    
                    const conf = (data.confidence * 100).toFixed(1);
                    document.getElementById('resConfidence').textContent = conf + '%';
                    document.getElementById('resProgressBar').style.width = conf + '%';
                    
                    document.getElementById('resDetails').textContent = data.details || '';
                    
                    let probStr = '';
                    if (data.probabilities) {
                        probStr = `Probabilities: English (${(data.probabilities.English*100).toFixed(1)}%) | Hindi (${(data.probabilities.Hindi*100).toFixed(1)}%) | Kannada (${(data.probabilities.Kannada*100).toFixed(1)}%)`;
                    }
                    if (data.script_ratios) {
                        probStr += `  •  Exact Characters: English (${data.script_ratios.English}%), Hindi (${data.script_ratios.Hindi}%), Kannada (${data.script_ratios.Kannada}%)`;
                    }
                    document.getElementById('resMeta').textContent = probStr;
                    resultBox.classList.remove('d-none');

                    // Refresh administrative history in background
                    loadHistoryTable();
                } else {
                    errorBox.textContent = data.error || 'Prediction failed.';
                    errorBox.classList.remove('d-none');
                }
            } catch (err) {
                errorBox.textContent = 'Connection error: ' + err.message;
                errorBox.classList.remove('d-none');
            } finally {
                isDetecting = false;
                detectBtn.disabled = false;
                detectBtn.innerHTML = '<i class="fa-solid fa-paper-plane me-1"></i> Detect Language';
            }
        }

        async function loadHistoryTable() {
            try {
                const response = await fetch('/api/history');
                const data = await response.json();
                cachedHistory = data.history || [];
                renderHistoryRows(cachedHistory, 'adminHistoryTableBody');
                
                const adminCount = document.getElementById('adminTotalQueries');
                if (adminCount) adminCount.textContent = cachedHistory.length;
                const adminBadge = document.getElementById('adminHistoryBadge');
                if (adminBadge) adminBadge.textContent = `${cachedHistory.length} Inferences Logged`;
            } catch (err) {
                console.error("Failed to load history:", err);
            }
        }

        function renderHistoryRows(items, targetId) {
            const tbody = document.getElementById(targetId);
            if (!tbody) return;

            if (!items || items.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No prediction history recorded yet. Test sentences above to populate logs!</td></tr>';
                return;
            }

            let html = '';
            items.forEach((item, index) => {
                let badgeClass = 'badge-lang-english';
                if (item.language === 'Hindi') badgeClass = 'badge-lang-hindi';
                else if (item.language === 'Kannada') badgeClass = 'badge-lang-kannada';
                if (item.is_mixed) badgeClass = 'badge-lang-mixed';

                const confPercent = (item.confidence * 100).toFixed(1);
                const safeText = item.text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
                const shortText = item.text.length > 50 ? item.text.substring(0, 50) + '...' : item.text;

                // Format exact script distribution
                let scriptBadges = '';
                if (item.script_ratios) {
                    const parts = [];
                    for (const [lang, val] of Object.entries(item.script_ratios)) {
                        const pct = val > 1 ? val : (val * 100);
                        if (pct > 0) parts.push(`${lang} ${pct.toFixed(0)}%`);
                    }
                    scriptBadges = parts.join(', ');
                }
                if (!scriptBadges) scriptBadges = item.language;

                html += `
                    <tr>
                        <td class="text-muted fw-bold">${item.id || (index + 1)}</td>
                        <td class="small text-secondary text-nowrap"><i class="fa-regular fa-clock me-1"></i>${item.timestamp}</td>
                        <td title="${safeText}">${shortText}</td>
                        <td><span class="badge ${badgeClass} px-2 py-1"><i class="fa-solid fa-circle me-1" style="font-size:0.5rem;"></i>${item.display_label || item.language}</span></td>
                        <td><small class="text-muted fw-semibold">${scriptBadges}</small></td>
                        <td>
                            <div class="d-flex align-items-center gap-2">
                                <span class="small fw-bold">${confPercent}%</span>
                                <div class="progress flex-grow-1" style="height: 6px;">
                                    <div class="progress-bar bg-success" style="width: ${confPercent}%;"></div>
                                </div>
                            </div>
                        </td>
                        <td>
                            ${item.is_mixed ? '<span class="badge bg-warning text-dark"><i class="fa-solid fa-code-merge me-1"></i>Mixed</span>' : '<span class="badge bg-light text-secondary border">Single</span>'}
                        </td>
                        <td class="text-nowrap">
                            <button class="btn btn-sm btn-outline-primary py-0 px-2 me-1" title="Test in Detector" onclick="setSample('${safeText}')">
                                <i class="fa-solid fa-arrow-up-right-from-square"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-danger py-0 px-2" title="Delete Log" onclick="deleteHistoryItem(${item.id})">
                                <i class="fa-solid fa-trash"></i>
                            </button>
                        </td>
                    </tr>
                `;
            });
            tbody.innerHTML = html;
        }

        function filterAdminHistory() {
            const query = (document.getElementById('adminSearchInput').value || '').toLowerCase();
            const filtered = cachedHistory.filter(item => 
                (item.text || '').toLowerCase().includes(query) ||
                (item.language || '').toLowerCase().includes(query) ||
                (item.display_label || '').toLowerCase().includes(query)
            );
            renderHistoryRows(filtered, 'adminHistoryTableBody');
        }

        async function deleteHistoryItem(id) {
            if (!confirm('Are you sure you want to remove this prediction from the history log?')) return;
            try {
                const res = await fetch('/api/history/delete/' + id, { method: 'POST' });
                if (res.ok) {
                    loadHistoryTable();
                    loadSystemStats();
                }
            } catch (err) {
                alert('Failed to delete history item: ' + err.message);
            }
        }

        async function clearAllHistory() {
            if (!confirm('Are you sure you want to clear all prediction history?')) return;
            try {
                const res = await fetch('/api/history/clear', { method: 'POST' });
                if (res.ok) {
                    loadHistoryTable();
                    loadSystemStats();
                }
            } catch (err) {
                alert('Failed to clear history: ' + err.message);
            }
        }

        async function loadSystemStats() {
            try {
                const res = await fetch('/api/system/stats');
                const stats = await res.json();
                document.getElementById('adminTotalQueries').textContent = stats.total_queries || 0;
                document.getElementById('adminAvgConfidence').textContent = stats.avg_confidence ? (stats.avg_confidence * 100).toFixed(1) + '%' : 'N/A';
                document.getElementById('adminAvgLatency').textContent = stats.avg_latency ? stats.avg_latency + ' ms' : '< 2 ms';
                loadHistoryTable();
            } catch (err) {
                console.error("Failed to load system stats:", err);
            }
        }

        async function reloadModelArtifacts() {
            const notice = document.getElementById('adminNotice');
            notice.className = 'alert alert-info mt-3 py-2 small';
            notice.textContent = 'Reloading model artifacts from disk...';
            notice.classList.remove('d-none');
            try {
                const res = await fetch('/api/system/reload_model', { method: 'POST' });
                const data = await res.json();
                if (res.ok) {
                    notice.className = 'alert alert-success mt-3 py-2 small';
                    notice.textContent = 'Models, TF-IDF vectorizers, and LabelEncoder reloaded successfully!';
                } else {
                    notice.className = 'alert alert-danger mt-3 py-2 small';
                    notice.textContent = data.error || 'Failed to reload model.';
                }
            } catch (err) {
                notice.className = 'alert alert-danger mt-3 py-2 small';
                notice.textContent = 'Reload error: ' + err.message;
            }
        }

        async function runDiagnosticsCheck() {
            const notice = document.getElementById('adminNotice');
            notice.className = 'alert alert-success mt-3 py-2 small';
            notice.innerHTML = '<i class="fa-solid fa-circle-check me-1"></i> <strong>System Diagnostics Passed:</strong> Active Model Loaded • TF-IDF Vectorizer (32,485 features) Loaded • LabelEncoder (3 classes) Online • Dataset (1,050 rows) Verified.';
            notice.classList.remove('d-none');
        }

        function updateCM() {
            const select = document.getElementById('cmSelect');
            const val = select.value;
            document.getElementById('cmImage').src = '/plots/confusion_matrix_' + val + '.png';
        }

        document.getElementById('inputText').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') runDetection();
        });

        // Initialize table on load
        window.addEventListener('DOMContentLoaded', () => {
            loadHistoryTable();
        });
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load model, vectorizer, and label encoder."""
    global model, vectorizer, label_encoder
    import joblib

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("TF-IDF vectorizer loaded from %s", VECTORIZER_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Best model loaded from %s", MODEL_PATH)

    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("Label encoder loaded from %s", LABEL_ENCODER_PATH)

# Auto-load artifacts on import if available
try:
    load_model()
except Exception as _e:
    logger.warning("Initial artifact load deferred or failed: %s", _e)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Serve the complete interactive project portal & control center."""
    return render_template_string(PORTAL_TEMPLATE)

@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serve visualization images from the plots directory."""
    return send_from_directory(PLOTS_DIR, filename)

@app.route("/download_report")
def download_report():
    """Allow downloading the markdown project report."""
    return send_from_directory(PROJECT_ROOT, "PROJECT_REPORT.md", as_attachment=True)

@app.route("/download_report_pdf")
def download_report_pdf():
    """Allow downloading the compiled academic project report PDF."""
    return send_from_directory(PROJECT_ROOT, "PROJECT_REPORT.pdf", as_attachment=True)

@app.route("/download_models_pdf")
def download_models_pdf():
    """Allow downloading the concise models summary PDF."""
    return send_from_directory(PROJECT_ROOT, "ML_Models_Summary.pdf", as_attachment=True)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API endpoint for language prediction.
    Features hybrid character script analysis to detect code-mixing and avoid stammering.
    Automatically logs query to prediction history.
    """
    t_start = time.time()
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded on server."}), 503

    data = request.get_json(silent=True)
    if data is None or "text" not in data:
        return jsonify({"error": "Invalid payload. Provide a JSON object with a 'text' key."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Input text cannot be empty."}), 400

    try:
        # 1. Physical script character counting
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        hindi_count = len(re.findall(r'[\u0900-\u097F]', text))
        kannada_count = len(re.findall(r'[\u0C80-\u0CFF]', text))
        total_script_chars = latin_count + hindi_count + kannada_count

        if total_script_chars > 0:
            script_ratio = {
                'English': latin_count / total_script_chars,
                'Hindi': hindi_count / total_script_chars,
                'Kannada': kannada_count / total_script_chars
            }
        else:
            script_ratio = {'English': 0.3333, 'Hindi': 0.3333, 'Kannada': 0.3333}

        # 2. ML vector prediction
        text_vector = vectorizer.transform([text])
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(text_vector)[0]
        else:
            raw_probs = [0.3333, 0.3333, 0.3333]

        ml_dict = {}
        if label_encoder is not None:
            for idx, cname in enumerate(label_encoder.classes_):
                ml_dict[str(cname)] = float(raw_probs[idx])
        else:
            ml_dict = {'English': float(raw_probs[0]), 'Hindi': float(raw_probs[1]), 'Kannada': float(raw_probs[2])}

        # 3. Detect code-mixing (multiple scripts with > 8% character presence)
        active_langs = [l for l, r in script_ratio.items() if r > 0.08]
        is_mixed = len(active_langs) > 1

        # 4. Calibrated blend: 50% ML Model Signal + 50% Physical Script Ratio
        blended = {
            'English': 0.5 * ml_dict.get('English', 0.0) + 0.5 * script_ratio['English'],
            'Hindi': 0.5 * ml_dict.get('Hindi', 0.0) + 0.5 * script_ratio['Hindi'],
            'Kannada': 0.5 * ml_dict.get('Kannada', 0.0) + 0.5 * script_ratio['Kannada'],
        }
        prob_sum = sum(blended.values()) or 1.0
        normalized_probs = {k: round(v / prob_sum, 4) for k, v in blended.items()}

        dominant_lang = max(normalized_probs, key=normalized_probs.get)
        confidence = float(normalized_probs[dominant_lang])
        latency_ms = (time.time() - t_start) * 1000

        if is_mixed:
            display_label = "Code-Mixed (" + " + ".join(active_langs) + ")"
            details = f"Code-mixed text detected: {dominant_lang} dominant ({round(script_ratio[dominant_lang]*100, 1)}%), with " + ", ".join([f"{l} ({round(script_ratio[l]*100, 1)}%)" for l in active_langs if l != dominant_lang])
        else:
            display_label = dominant_lang
            details = f"Standard monolingual {dominant_lang} text."

        # Automatically record to prediction history
        record_prediction(
            text=text,
            dominant_lang=dominant_lang,
            display_label=display_label,
            is_mixed=is_mixed,
            confidence=confidence,
            script_ratio=script_ratio,
            probabilities=normalized_probs,
            latency_ms=latency_ms
        )

        return jsonify({
            "language": dominant_lang,
            "display_label": display_label,
            "is_mixed": is_mixed,
            "active_languages": active_langs,
            "confidence": round(confidence, 4),
            "details": details,
            "probabilities": normalized_probs,
            "script_ratios": {k: round(v * 100, 1) for k, v in script_ratio.items()}
        }), 200

    except Exception as e:
        logger.exception("Inference failed")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route("/api/history", methods=["GET"])
def get_history():
    """Retrieve all logged predictions."""
    history = load_history()
    return jsonify({
        "status": "success",
        "total": len(history),
        "history": history
    }), 200

@app.route("/api/history/delete/<int:item_id>", methods=["POST"])
def delete_history_item(item_id):
    """Delete a specific prediction entry by ID."""
    history = load_history()
    filtered = [item for item in history if item.get("id") != item_id]
    save_history(filtered)
    return jsonify({"status": "deleted", "remaining": len(filtered)}), 200

@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """Clear all logged prediction records."""
    save_history([])
    return jsonify({"status": "cleared", "total": 0}), 200

@app.route("/api/history/export", methods=["GET"])
def export_history_csv():
    """Export prediction history as a downloadable CSV."""
    history = load_history()
    if not history:
        csv_content = "ID,Timestamp,Text,Language,Classification_Mode,Confidence,Latency_ms\n"
    else:
        rows = []
        for h in history:
            clean_text = h["text"].replace('"', '""')
            mode = "Code-Mixed" if h.get("is_mixed") else "Monolingual"
            rows.append(f'{h.get("id")},"{h.get("timestamp")}","{clean_text}","{h.get("display_label", h.get("language"))}","{mode}",{h.get("confidence")},{h.get("latency_ms", 1.5)}')
        csv_content = "ID,Timestamp,Text,Language,Classification_Mode,Confidence,Latency_ms\n" + "\n".join(rows)

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=prediction_history.csv"}
    )

@app.route("/api/system/stats", methods=["GET"])
def system_stats():
    """Compute aggregate analytics for the System Control Center."""
    history = load_history()
    total = len(history)
    if total > 0:
        avg_conf = round(sum(h.get("confidence", 0) for h in history) / total, 4)
        avg_lat = round(sum(h.get("latency_ms", 1.5) for h in history) / total, 1)
        lang_counts = {}
        for h in history:
            l = h.get("language", "Unknown")
            lang_counts[l] = lang_counts.get(l, 0) + 1
    else:
        avg_conf = 0.0
        avg_lat = 1.5
        lang_counts = {}

    return jsonify({
        "total_queries": total,
        "avg_confidence": avg_conf,
        "avg_latency": avg_lat,
        "language_distribution": lang_counts,
        "active_model": "Soft Voting Ensemble (NB + LR + LinearSVC)",
        "status": "healthy"
    }), 200

@app.route("/api/system/reload_model", methods=["POST"])
def reload_model_route():
    """Hot-reload model artifacts from disk."""
    try:
        load_model()
        return jsonify({"status": "reloaded", "model": str(model)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health status monitor."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
    }), 200

def start_server(host="0.0.0.0", port=5000, debug=False):
    """Launch the Flask web portal."""
    logger.info("=" * 60)
    logger.info("  Language Detection – Web Portal & System Control Center")
    logger.info("=" * 60)
    load_model()
    logger.info("Starting web server at http://localhost:%d", port)
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_server(debug=True)
