# EventCorrelationEngineUsingML

A machine learning project that detects unusual network slowdowns and pinpoints the likely root cause.  
It combines anomaly detection, hop-level modeling, trend analysis, and event correlation to explain not just **what failed — but why**.

---

# Network Failure Root Cause Analysis using Isolation Forest

## Overview

This project implements an end-to-end **network failure detection and root cause analysis (RCA) pipeline** using unsupervised anomaly detection and per-hop modeling.

It simulates DMM (Delay Measurement Message) telemetry, decomposes end-to-end latency into individual IGP hops, and applies layered analytics to identify:

-  Which DMM failures are truly abnormal  
-  Which IGP hop is most likely responsible  
-  Whether a hop shows long-term degradation  
-  Which recent network events may explain the failure  

Although the dataset is synthetic, the architecture mirrors real-world telemetry and observability systems.

---

##  Architecture

The pipeline follows a layered design:

---

### 1️. Synthetic Data Generation

- Generates large-scale DMM test data with configurable failure ratios  
- Expands end-to-end delay into per-IGP hop delays  
- Generates time-aligned synthetic network events  

---

### 2️. Global Anomaly Detection (DMM Level)

- Uses **Isolation Forest**
- Trained on end-to-end delay (unsupervised)
- Identifies anomalous DMM failures
- Provides severity ranking via anomaly score

**Purpose:**  
Triage and prioritize failures worth investigating.

---

### 3️. Per-IGP Hop Modeling

- Trains one Isolation Forest per hop (PASS-only data)
- Learns hop-specific normal delay behavior
- Scores each hop measurement for abnormality

**Purpose:**  
Determine whether a delay is abnormal relative to that hop’s historical behavior.

---

### 4️. Baseline & Statistical Deviation

- Computes PASS-only mean and standard deviation per hop
- Calculates z-score deviation for failed tests

**Purpose:**  
Quantify structural deviation from normal operating conditions.

---

### 5️. Root Cause Attribution

For each failed DMM test:

- Merge hop-level data with DMM anomaly score  
- Compute a combined score using:
  - Hop z-score  
  - Hop anomaly score  
- Select the hop with the highest combined deviation  

**Output includes:**

- Predicted root IGP hop  
- Severity score  
- Deviation metrics  

---

### 6️. Trend Analysis

- Uses Linear Regression per hop (PASS-only data)
- Computes delay slope (ms/hour)
- Labels hop as:
  - `INCREASING`
  - `DECREASING`
  - `STABLE`

**Purpose:**  
Provide directional context for degradation patterns.

---

### 7️. Event Correlation

- Associates failures with recent network events  
- Uses time-window alignment  
- Attaches likely event metadata  

**Purpose:**  
Add operational context and improve explainability of RCA results.

---

## Dashboard

A Streamlit dashboard provides:

- Filtering by failed timestamps (epoch-based)  
- Visualization of anomaly clusters  
- Inspection of predicted root causes  
- Exploration of DMM, IGP, and event datasets  

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn (Isolation Forest, Linear Regression)  
- Joblib  
- Streamlit  
- Plotly  

---

## Key Design Principles

-  Unsupervised anomaly detection  
-  PASS-only baseline modeling  
-  Layered RCA (triage → localization → context)  
-  Explainability-first design  
-  Synthetic data with production-style architecture  

---

## Why This Project

This project demonstrates:

- How to build a scalable anomaly detection pipeline  
- How to localize root causes using hop-level modeling  
- How to combine statistical deviation with unsupervised scoring  
- How to integrate telemetry with event correlation  
- How to design explainable ML systems for network operations  

---

## How to Run

```bash
# Generate synthetic data
python generate_dmm_data.py
python generate_lsp_data.py
python generate_network_events.py

# Run RCA pipeline
python trainAndPredictIsolationForest.py

# Launch dashboard
streamlit run dashboard.py
