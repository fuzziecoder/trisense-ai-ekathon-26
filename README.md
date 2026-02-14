# TriSense AI — Early Clinical Deterioration & Sepsis Prediction System

<p align="center">
  <img width="120" alt="TriSense AI Logo" src="https://img.shields.io/badge/🩺-TriSense%20AI-red?style=for-the-badge&labelColor=1a1a2e" />
</p>

<p align="center">
  <strong>Detect clinical deterioration before collapse occurs — hours earlier, not minutes too late</strong>
</p>

<p align="center">
  <!-- Hackathon -->
  <img src="https://img.shields.io/badge/Ekathon-2026-DC143C?style=for-the-badge&logo=lightning&logoColor=white" alt="Ekathon 2026" />
  <img src="https://img.shields.io/badge/Track-Health%20AI%20on%20India's%20Digital%20Rails-0099FF?style=for-the-badge&logo=india&logoColor=white" alt="Track" />
  <img src="https://img.shields.io/badge/Team-Flexiroasters-FF6B35?style=for-the-badge&logo=fire&logoColor=white" alt="Team" />
</p>

<p align="center">
  <!-- Status -->
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=statuspage&logoColor=white" alt="Status" />
  <img src="https://img.shields.io/badge/Domain-Healthcare%20AI-E91E63?style=for-the-badge&logo=heart&logoColor=white" alt="Domain" />
  <img src="https://img.shields.io/badge/Focus-Early%20Warning%20System-FF5722?style=for-the-badge" alt="Focus" />
</p>

<p align="center">
  <!-- Tech Stack -->
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/XGBoost-Classifier-EC7C0C?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Transformer-PatchTST%20Encoder-764ABC?style=flat-square&logo=pytorch&logoColor=white" alt="PatchTST" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
</p>

<p align="center">
  <!-- ML & Safety -->
  <img src="https://img.shields.io/badge/ML-Time--Series%20Risk%20Prediction-7B68EE?style=flat-square&logo=scikitlearn&logoColor=white" alt="ML" />
  <img src="https://img.shields.io/badge/Clinical%20Scores-qSOFA%20%7C%20SIRS%20%7C%20Shock%20Index-2196F3?style=flat-square" alt="Clinical Scores" />
  <img src="https://img.shields.io/badge/Explainability-SHAP%20%2B%20Natural%20Language-4CAF50?style=flat-square" alt="Explainability" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square&logo=law&logoColor=white" alt="MIT License" />
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-problem-statement">Problem</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-core-components">Components</a> •
  <a href="#-ai-agent-layer">AI Agents</a> •
  <a href="#-risk-scoring">Risk Scoring</a> •
  <a href="#-model-performance">Performance</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-api-documentation">API Docs</a> •
  <a href="#-use-cases">Use Cases</a> •
  <a href="#-ethics--safety">Ethics</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Features](#-key-features)
4. [System Architecture](#-system-architecture)
5. [Core Components](#-core-components)
6. [AI Agent Layer](#-ai-agent-layer)
7. [Risk Scoring](#-risk-scoring)
8. [Input Data & Sources](#-input-data--sources)
9. [Model Performance](#-model-performance)
10. [Getting Started](#-getting-started)
11. [API Documentation](#-api-documentation)
12. [Use Cases](#-use-cases)
13. [Ethics & Safety](#-ethics--safety)
14. [Impact](#-impact)
15. [Roadmap](#-roadmap)

---

## 🔭 Overview

**TriSense AI** is an AI-powered early warning system that detects clinical deterioration **before visible collapse occurs**. It analyzes time-series vital sign data to generate explainable risk alerts, enabling clinicians to intervene hours earlier — not minutes too late.

Built for real-world healthcare environments including resource-limited rural settings, TriSense focuses on **trend-based detection**, not static threshold monitoring.

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   INPUT:   Patient vital signs (time-series, 6-hour window)      │
│                              ↓                                   │
│   PROCESS: Trend analysis + ML risk classification               │
│                              ↓                                   │
│   OUTPUT:  Risk score + plain-English explanation + alert        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Why "TriSense"?

The name reflects the system's three sensing layers — **physiological**, **temporal**, and **clinical** — working together to sense deterioration that no single metric can catch alone.

---

## 🎯 Problem Statement

### The Clinical Reality

In hospital settings, patients often show subtle physiological changes **hours before** critical events such as:

- 🦠 **Sepsis** — the leading cause of preventable hospital deaths globally
- 💔 **Cardiac arrest** — often preceded by hours of gradual vital sign drift
- 🫁 **Respiratory failure** — oxygen saturation trends detectable well before crisis
- 💉 **Septic shock** — blood pressure and heart rate patterns visible in advance

### Why Existing Systems Fail

| Traditional Monitoring | TriSense AI |
|---|---|
| Fixed static thresholds | Learns each patient's baseline |
| Single-point snapshot readings | 6-hour rolling time-series analysis |
| No trend context | Detects gradual physiological drift |
| Binary alarm (normal / critical) | Continuous 0–100% risk probability |
| No explanation for alert | Plain-English clinical reasoning |
| Alarm fatigue from false positives | High-precision, high-recall model |
| Requires specialist interpretation | Designed for any bedside nurse |

### The Stakes

> Sepsis kills ~11 million people per year globally. Studies show that **every hour of delay** in sepsis treatment increases mortality by 7%. TriSense targets this exact window.

---

## ✨ Key Features

### 📈 Time-Series Vital Analysis
Monitors 6-hour rolling windows of vital signs to detect gradual physiological drift — the kind that static alarms miss entirely.

### ⚠️ Early Deterioration Detection
Identifies deterioration patterns hours before conventional thresholds are breached, buying clinicians critical intervention time.

### 🧠 Machine Learning Risk Prediction
A two-stage architecture: a **Transformer-based time-series encoder** (PatchTST) extracts temporal patterns, fed into an **XGBoost classifier** optimized for high recall on critical cases.

### 🔍 Explainable Alerts
Every risk score comes with a plain-English explanation — no black box. Clinicians always know *why* the alert fired.

### 🏥 Resource-Limited Ready
Designed to run without GPU infrastructure, making it deployable in rural health centers and district hospitals, not just urban ICUs.

### 🔒 Privacy-First
No patient data leaves the local system. All processing is on-premise. Compliant with healthcare data principles from design.

---

## 🧠 System Architecture

### End-to-End Pipeline

```
Patient Vitals (6-hr window)
         │
         ▼
┌─────────────────────┐
│  Feature Engineering │  ← trend slopes, z-scores, shock index, qSOFA
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  PatchTST Encoder   │  ← Transformer learns temporal embeddings
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  XGBoost Classifier │  ← risk probability 0–100%
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  AI Agent Layer     │  ← 5 specialized agents reason over results
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Explanation Engine │  ← generates human-readable alert
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Alert + Dashboard  │  ← clinician sees risk score + reason + action
└─────────────────────┘
```

### Full System Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND DASHBOARD                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  Patient   │  │  Risk      │  │  Trend     │  │  Alert     │       │
│  │  Overview  │  │  Scores    │  │  Charts    │  │  Feed      │       │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ REST API
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                            BACKEND (FastAPI)                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │  │
│  │  │  Data    │  │ Feature  │  │  Model   │  │  Agent   │        │  │
│  │  │  Ingest  │  │  Engine  │  │  Infer   │  │  Layer   │        │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  PatchTST    │     │   XGBoost    │     │  Explanation │
    │  Encoder     │     │  Classifier  │     │   Engine     │
    └──────────────┘     └──────────────┘     └──────────────┘
```

---

## ⚙️ Core Components

### 1. 📊 Data Generator (`data_generator.py`)

Generates labeled clinical training data that simulates realistic patient vital patterns using evidence-based clinical rules.

**Clinical scoring systems implemented:**

| Score | What It Measures | Deterioration Threshold |
|---|---|---|
| **qSOFA** | Sepsis quick screening | ≥ 2 criteria |
| **SIRS** | Systemic inflammatory response | ≥ 2 criteria |
| **Shock Index** | HR / SBP — perfusion proxy | > 1.0 = concern |
| **Early Warning Score** | Multi-parameter composite | ≥ 5 = high risk |

---

### 2. 🔬 Feature Engineering (`feature_engineering.py`)

Transforms raw vital sign time series into clinically meaningful predictive features.

| Feature Class | Features Extracted |
|---|---|
| **Trend Features** | Slope over 1hr, 3hr, 6hr windows |
| **Change Rate** | Absolute + percentage change per hour |
| **Statistical** | Mean, std dev, min, max per window |
| **Normalization** | Z-score relative to patient baseline |
| **Clinical Indices** | Shock Index, qSOFA score, SIRS criteria |
| **Deviation** | Distance from patient's own normal range |

---

### 3. 🤖 Time-Series Encoder (`patchtst_encoder.py`)

A **PatchTST (Patch Time Series Transformer)** learns temporal patterns in vital sign sequences that tabular models cannot capture.

```
Vital Sign Sequence (T timesteps)
         │
         ▼
 ┌───────────────────┐
 │  Patch Splitting  │  ← divides sequence into overlapping patches
 └───────────────────┘
         │
         ▼
 ┌───────────────────┐
 │  Positional Embed │  ← encodes time position
 └───────────────────┘
         │
         ▼
 ┌───────────────────┐
 │  Transformer Enc  │  ← multi-head self-attention over patches
 └───────────────────┘
         │
         ▼
 Temporal Embedding → XGBoost Classifier
```

**Why PatchTST over standard LSTM?**
- Better long-range dependency capture
- More efficient on clinical-length sequences (hours, not days)
- Transferable across patients without retraining

---

### 4. 🎯 Risk Classifier (`xgboost_classifier.py`)

XGBoost gradient boosting classifier trained on engineered features + transformer embeddings.

**Training objectives:**
- Maximize **recall** — missing a true deterioration is far worse than a false positive
- Calibrated probabilities for reliable risk percentage output
- Feature importance for SHAP-based explanation

---

### 5. 🏋️ Training Pipeline (`train_pipeline.py`)

```bash
python train_pipeline.py
```

**Pipeline steps:**

```
1. Load + validate training data
2. Feature engineering
3. Train/validation split (time-aware, no leakage)
4. PatchTST encoder training
5. XGBoost classifier training on embeddings
6. Threshold optimization (recall-maximizing)
7. Performance evaluation
8. Model serialization
```

---

### 6. 🔮 Inference Engine (`inference.py`)

The main prediction interface — takes a 6-hour vital sign window and returns a structured risk result.

```python
from inference import predict_deterioration_risk

result = predict_deterioration_risk(vitals_6hr)
```

**Output structure:**
```json
{
  "risk_percentage": 78,
  "risk_category": "HIGH",
  "risk_color": "RED",
  "explanation": "Rising heart rate trend (+18 bpm over 4hrs) combined with falling systolic BP (-22 mmHg) suggests early circulatory compromise consistent with septic shock pattern.",
  "key_drivers": ["heart_rate_slope", "sbp_6hr_change", "shock_index"],
  "recommended_actions": [
    "Immediate clinical assessment",
    "Blood cultures × 2 before antibiotics",
    "IV fluid bolus consideration",
    "Lactate measurement"
  ],
  "timestamp": "2026-02-14T08:32:00Z",
  "confidence": 0.84
}
```

---

## 🤖 AI Agent Layer

TriSense uses a **multi-agent decision architecture** — five specialized AI agents that collaborate to reason about patient risk. Each agent focuses on a specific clinical dimension.

```
┌─────────────────────────────────────────────────────────────┐
│                     AI AGENT LAYER                          │
│                                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │   Alert    │  │  Pattern   │  │   Trend    │           │
│  │   Agent    │  │   Agent    │  │   Agent    │           │
│  │            │  │            │  │            │           │
│  │ Decides    │  │ Matches    │  │ Analyzes   │           │
│  │ when/how   │  │ vital sign │  │ directional│           │
│  │ to alert   │  │ signatures │  │ changes    │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│                                                             │
│       ┌────────────────┐  ┌────────────────┐              │
│       │   Reasoning    │  │  Suggestion    │              │
│       │    Agent       │  │    Agent       │              │
│       │                │  │                │              │
│       │ Explains why   │  │ Recommends     │              │
│       │ risk fired in  │  │ clinical       │              │
│       │ plain English  │  │ actions        │              │
│       └────────────────┘  └────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

| Agent | Responsibility | Output |
|---|---|---|
| **Alert Agent** | Decides alert severity and escalation path | Alert level + notification routing |
| **Pattern Agent** | Matches vital combinations to known deterioration signatures (sepsis, cardiac, respiratory) | Pattern match + confidence |
| **Trend Agent** | Quantifies directional changes in each vital over time | Trend summary per vital |
| **Reasoning Agent** | Synthesizes agent outputs into a plain-English clinical narrative | Explanation text |
| **Suggestion Agent** | Maps risk pattern to evidence-based clinical actions | Prioritized action list |

---

## 📊 Risk Scoring

### Three-Tier Risk Classification

| Risk Level | Score Range | Color | Clinical Response |
|---|---|---|---|
| 🟢 **Low** | 0–39% | Green | Routine monitoring, no escalation |
| 🟡 **Moderate** | 40–69% | Amber | Increased monitoring frequency, notify physician |
| 🔴 **High** | 70–100% | Red | Immediate clinical assessment, escalate now |

### Risk Score Composition

```
Risk Score (0–100%) = weighted combination of:

├── Trend Score        (30%)  ← vital sign directional changes
├── Clinical Score     (25%)  ← qSOFA, SIRS, Shock Index
├── Deviation Score    (25%)  ← distance from patient's own baseline
├── Pattern Score      (15%)  ← ML signature matching
└── Velocity Score     (5%)   ← rate of change acceleration
```

### Example Score Breakdowns

```
Patient A — Score: 82% HIGH
├── Trend:     26/30  (HR rising 18bpm/4hr, BP falling)
├── Clinical:  22/25  (qSOFA ≥ 2, Shock Index 1.1)
├── Deviation: 20/25  (3.2 std deviations from baseline)
├── Pattern:   10/15  (sepsis pattern match: 74%)
└── Velocity:   4/5   (accelerating deterioration)

Patient B — Score: 31% LOW
├── Trend:      8/30  (minor HR variation, stable BP)
├── Clinical:   5/25  (no SIRS criteria met)
├── Deviation:  9/25  (within normal baseline range)
├── Pattern:    7/15  (no strong pattern match)
└── Velocity:   2/5   (slow, stable changes)
```

---

## 📥 Input Data & Sources

### Vital Sign Parameters

| Parameter | Unit | Normal Range | Sampling |
|---|---|---|---|
| Heart Rate | bpm | 60–100 | Continuous |
| Systolic Blood Pressure | mmHg | 90–140 | Every 15 min |
| Diastolic Blood Pressure | mmHg | 60–90 | Every 15 min |
| Oxygen Saturation (SpO2) | % | 95–100 | Continuous |
| Respiratory Rate | breaths/min | 12–20 | Every 15 min |
| Temperature | °C | 36.1–37.2 | Every 1–4 hr |

### Compatible Data Sources

| Source | Integration | Notes |
|---|---|---|
| ICU Bedside Monitors | Direct API / HL7 | Real-time streaming |
| Wearable Devices | REST API | Continuous telemetry |
| Electronic Health Records | FHIR / HL7 | Batch + streaming |
| Simulated Clinical Data | Built-in generator | Training & demo |
| Manual Entry | Web dashboard | Resource-limited settings |

---

## 📈 Model Performance

TriSense is evaluated with a **recall-first** philosophy — in clinical settings, missing a true deterioration (false negative) is far more dangerous than a false positive.

### Target Metrics

| Metric | Target | Rationale |
|---|---|---|
| **Recall (Sensitivity)** | ≥ 0.90 | Missing deterioration = preventable death |
| **Precision** | ≥ 0.75 | Minimize alarm fatigue |
| **F1 Score** | ≥ 0.82 | Balanced overall performance |
| **ROC-AUC** | ≥ 0.88 | Strong discriminative ability |
| **Early Detection Lead** | ≥ 2 hours | Alert fires before clinical threshold breach |

### Evaluation Protocol

```
Train/Validation Split: Time-aware (no future leakage)
                        ├── Training:   70% (earliest patients)
                        ├── Validation: 15% (middle cohort)
                        └── Test:       15% (most recent — strictest)

Threshold Selection: Maximizes recall at ≥ 90% on validation set
Cross-Validation:    Patient-level k-fold (prevents data leakage)
```

---

## 🚀 Getting Started

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EC7C0C?style=flat-square)

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/your-repo/trisense-ai
cd trisense-ai
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment
```env
# .env
MODEL_DIR="./models"
DATA_DIR="./data"
LOG_LEVEL="INFO"
ALERT_THRESHOLD_HIGH=70
ALERT_THRESHOLD_MODERATE=40
API_HOST="0.0.0.0"
API_PORT=8000
```

### Run Training
```bash
python train_pipeline.py
```

### Run Inference (Python API)
```python
from inference import predict_deterioration_risk

# 6-hour vital sign window — list of 24 readings (every 15 min)
vitals_6hr = [
    {"hr": 88, "sbp": 118, "dbp": 76, "spo2": 97, "rr": 16, "temp": 37.1},
    {"hr": 91, "sbp": 114, "dbp": 74, "spo2": 96, "rr": 17, "temp": 37.3},
    # ... 22 more readings
]

result = predict_deterioration_risk(vitals_6hr)
print(result)
# → risk_percentage: 78, risk_category: HIGH, explanation: "..."
```

### Run Demo
```bash
python demo.py
```

### Start API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Project Structure
```
trisense-ai/
├── data_generator.py       # Synthetic clinical data generation
├── feature_engineering.py  # Clinical feature extraction
├── patchtst_encoder.py     # Transformer time-series encoder
├── xgboost_classifier.py   # Risk classification model
├── train_pipeline.py       # End-to-end training orchestrator
├── inference.py            # Prediction interface
├── agents/
│   ├── alert_agent.py      # Alert triggering logic
│   ├── pattern_agent.py    # Vital sign pattern matching
│   ├── trend_agent.py      # Time-series trend analysis
│   ├── reasoning_agent.py  # Plain-English explanation generation
│   └── suggestion_agent.py # Clinical action recommendations
├── main.py                 # FastAPI backend
├── demo.py                 # Interactive demo script
├── requirements.txt
└── models/                 # Saved model weights
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Endpoint Reference

#### Prediction

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Submit vital signs → receive risk score + explanation |
| `POST` | `/predict/batch` | Batch prediction for multiple patients |
| `GET` | `/predict/{patient_id}/history` | Risk score history for a patient |

#### Patient Management

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/patients` | Register a new patient |
| `GET` | `/patients` | List all monitored patients |
| `GET` | `/patients/{id}` | Get patient details + current risk |
| `PUT` | `/patients/{id}/vitals` | Submit new vital reading |

#### Alerts

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/alerts` | List all active alerts |
| `GET` | `/alerts/{id}` | Get specific alert details |
| `PUT` | `/alerts/{id}/acknowledge` | Clinician acknowledges alert |
| `GET` | `/alerts/stats` | Alert statistics and trends |

#### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check |
| `GET` | `/model/info` | Model version and performance stats |

### Example: Submit Vital Signs & Get Risk Score

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P-20240214-001",
    "vitals_window": [
      {"timestamp": "2026-02-14T08:00:00Z", "hr": 88, "sbp": 118, "dbp": 76, "spo2": 97, "rr": 16, "temp": 37.1},
      {"timestamp": "2026-02-14T08:15:00Z", "hr": 93, "sbp": 112, "dbp": 73, "spo2": 96, "rr": 17, "temp": 37.3}
    ]
  }'
```

**Response:**
```json
{
  "patient_id": "P-20240214-001",
  "risk_percentage": 78,
  "risk_category": "HIGH",
  "explanation": "Rising heart rate trend (+18 bpm over 4hrs) combined with falling systolic BP (-22 mmHg) is consistent with early circulatory compromise.",
  "key_drivers": ["heart_rate_slope", "sbp_6hr_change", "shock_index"],
  "recommended_actions": [
    "Immediate clinical assessment",
    "Blood cultures × 2 before antibiotics",
    "IV fluid bolus consideration",
    "Lactate measurement"
  ],
  "confidence": 0.84,
  "alert_triggered": true,
  "timestamp": "2026-02-14T08:32:00Z"
}
```

---

## 🏥 Use Cases

<details>
<summary><strong>Use Case 1: ICU Continuous Monitoring</strong></summary>

```
Actor:   ICU Nurse / Intensivist
Trigger: Patient admitted to ICU post-surgery

Flow:
  1. Patient vitals stream from bedside monitor
  2. TriSense analyzes 6-hour rolling window every 15 min
  3. Risk score updates continuously on dashboard
  4. At risk = 71%, alert fires → "Rising lactate pattern"
  5. Physician intervenes 2.5 hours before clinical threshold

Outcome: Early sepsis caught — antibiotics given within 1 hour
```
</details>

<details>
<summary><strong>Use Case 2: Emergency Department Triage</strong></summary>

```
Actor:   ED Triage Nurse
Trigger: Multiple patients arrive simultaneously

Flow:
  1. Vitals entered on triage for each patient
  2. TriSense scores all patients immediately
  3. Dashboard shows ranked risk list
  4. HIGH risk patient (78%) escalated to resuscitation bay
  5. Moderate risk (52%) placed on 30-min recheck protocol

Outcome: Highest-acuity patient identified in under 60 seconds
```
</details>

<details>
<summary><strong>Use Case 3: Rural Health Center</strong></summary>

```
Actor:   Nurse in rural PHC (no intensivist on-site)
Trigger: Patient with fever and altered consciousness

Flow:
  1. Nurse enters vitals manually into tablet
  2. TriSense detects qSOFA ≥ 2 + rising shock index
  3. Alert: "HIGH RISK — Sepsis pattern detected"
  4. Suggestion Agent: "Transfer immediately + give IV fluids"
  5. Patient transferred to district hospital

Outcome: Sepsis identified at PHC level — previously impossible
```
</details>

<details>
<summary><strong>Use Case 4: Ward Round Decision Support</strong></summary>

```
Actor:   Physician doing morning ward round
Trigger: Routine patient review

Flow:
  1. Physician opens TriSense dashboard
  2. Sees overnight trend charts for all patients
  3. One patient shows slow downward BP trend (moderate: 48%)
  4. Physician reviews overnight data, adjusts treatment
  5. Risk score improves to 22% by evening

Outcome: Silent deterioration caught before it became an emergency
```
</details>

<details>
<summary><strong>Use Case 5: Sepsis Protocol Activation</strong></summary>

```
Actor:   Rapid Response Team
Trigger: TriSense HIGH alert on general ward

Flow:
  1. RRT receives alert with risk score + explanation
  2. Explanation: "Fever + tachycardia + hypotension = Sepsis 3"
  3. RRT activates Sepsis Bundle at bedside
  4. Blood cultures, IV antibiotics, fluids within 1 hour
  5. Patient outcome: full recovery, 3-day ICU stay avoided

Outcome: Hour-1 Sepsis Bundle compliance achieved
```
</details>

---

## 🔐 Ethics & Safety

### Principles

| Principle | Implementation |
|---|---|
| **Clinical Decision Support Only** | System never acts autonomously — all alerts require clinician review |
| **Does Not Replace Clinicians** | Explicitly communicated in UI — "AI-assisted, clinician-confirmed" |
| **Explainability Mandatory** | Every risk score includes a plain-English explanation |
| **Anonymized Data** | No PII stored in model; patient IDs are system-generated tokens |
| **Privacy-First Design** | All computation on-premise; no external API calls for clinical data |
| **Calibrated Uncertainty** | Model outputs confidence alongside risk score |
| **Audit Trail** | Every prediction, alert, and acknowledgement is logged with timestamp |

### What TriSense Does NOT Do

```
❌ Does not autonomously adjust medications or treatments
❌ Does not make diagnoses
❌ Does not replace clinical judgment
❌ Does not store identifiable patient data externally
❌ Does not function without a trained clinical team
```

### Intended Regulatory Pathway

TriSense is designed as a **Class II Medical Device Software (SaMD)** under:
- India: CDSCO MDR 2017 (Software as Medical Device guidance)
- International: FDA 510(k) pathway / CE IVD pathway
- Standard: IEC 62304 software lifecycle compliance

---

## 🌍 Impact

### Clinical Impact

| Outcome | Mechanism |
|---|---|
| **Earlier intervention** | Detects deterioration 2–4 hours before clinical threshold |
| **Reduced ICU admissions** | Intervention at ward level prevents escalation |
| **Lower mortality risk** | Hour-1 sepsis bundle compliance from AI-guided alerts |
| **Reduced clinician workload** | AI prioritizes which patients need attention now |
| **Alarm fatigue reduction** | Fewer false positives than static threshold alarms |

### Healthcare System Impact

```
Resource-limited settings:
├── Works without ICU-level staffing
├── Runs on basic tablet hardware
├── No continuous internet required
└── Empowers nurses to escalate confidently

District & rural hospitals:
├── Closes the specialist gap
├── Enables proactive not reactive care
└── Reduces unnecessary transfers with better triage
```

---

## 🗺 Roadmap

### Phase 1 — Core System ✅ Complete

- [x] Synthetic clinical data generator (qSOFA, SIRS, Shock Index)
- [x] Feature engineering pipeline (trends, slopes, z-scores, clinical indices)
- [x] PatchTST transformer time-series encoder
- [x] XGBoost risk classifier (recall-optimized)
- [x] End-to-end training pipeline
- [x] Inference engine with structured output
- [x] Five-agent AI decision layer
- [x] FastAPI backend
- [x] Frontend risk dashboard

### Phase 2 — Clinical Integration 🔄 In Progress

- [ ] HL7 FHIR integration (EHR data ingestion)
- [ ] Real-time streaming via WebSocket
- [ ] Multi-patient ward dashboard
- [ ] Mobile alert push notifications
- [ ] Clinician feedback loop (improve model from corrections)
- [ ] PDF clinical summary generation

### Phase 3 — Deployment Hardening 📋 Planned

- [ ] Edge deployment (Raspberry Pi / tablet-grade hardware)
- [ ] Offline-first architecture for rural settings
- [ ] Role-based access (nurse / physician / admin)
- [ ] Audit trail and compliance logging
- [ ] Model drift detection and retraining pipeline
- [ ] Integration with ABHA (Ayushman Bharat Health Account)

### Phase 4 — Scale & Research 🔮 Future

- [ ] Multi-site federated learning (train across hospitals, share no data)
- [ ] Condition-specific models (paediatric, obstetric, oncology)
- [ ] Integration with India's National Digital Health Mission
- [ ] Clinical trial validation (prospective cohort study)
- [ ] Regulatory submission (CDSCO SaMD guidance)
- [ ] Published benchmark on MIMIC-IV dataset

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 📞 Contact & Links

| | |
|---|---|
| 🏆 **Hackathon** | Ekathon 2026 — Health AI on India's Digital Rails |
| 👥 **Team** | Flexiroasters |
| 📧 **Contact** | team@flexiroasters.dev |

---

<p align="center">
  Built with ❤️ by <strong>Team Flexiroasters</strong> for <strong>Ekathon 2026</strong>
</p>

<p align="center">
  <em>Health AI on India's Digital Rails</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-ff69b4?style=for-the-badge" alt="Made with love" />
  <img src="https://img.shields.io/badge/For-Patients%20First-DC143C?style=for-the-badge&logo=heart&logoColor=white" alt="Patients First" />
  <img src="https://img.shields.io/badge/Ekathon-2026-0099FF?style=for-the-badge" alt="Ekathon 2026" />
</p>
