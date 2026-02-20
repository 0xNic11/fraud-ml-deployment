# Credit Card Fraud Detection — End-to-End ML Project

An end-to-end machine learning project that detects fraudulent transactions using a production-oriented workflow.
The project goes beyond model training by implementing threshold tuning, robustness validation, monitoring design, and governance standards.

---

## 🎯 Project Purpose

Fraud detection systems must prioritize **recall** to avoid missing fraudulent transactions while maintaining acceptable precision.
This project demonstrates how to:

* Train a baseline fraud detection model
* Optimize decision thresholds using Precision-Recall analysis
* Validate model stability across transaction segments
* Simulate data drift and define monitoring signals
* Establish governance and retraining policies

---

## 🧠 What This Project Demonstrates

✔ Practical machine learning pipeline design
✔ Handling extreme class imbalance
✔ Threshold-based decision strategy
✔ Model robustness validation
✔ Monitoring & governance planning
✔ Production-ready project organization

---

## 📊 Model Overview

| Component       | Description                               |
| --------------- | ----------------------------------------- |
| Model           | Logistic Regression Pipeline              |
| Target Metric   | Recall (Fraud Class)                      |
| Decision Policy | Recall-prioritized                        |
| Threshold       | Stored in `metadata.json`                 |
| Evaluation      | Precision-Recall curve & confusion matrix |
| Validation      | Segment performance + drift simulation    |

---

## 📁 Project Structure

```
fraud-ml-deployment/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       └── creditcard_processed.csv
│
├── models/
│   ├── baseline_pipeline.pkl
│   └── metadata.json
│
├── notebooks/
│   ├── eda.ipynb
│   ├── baseline_model.ipynb
│   ├── threshold_strategy.ipynb
│   ├── robustness_analysis.ipynb
│   └── monitoring_plan.ipynb
│
├── reports/metrics/
│   ├── baseline_metrics.json
│   └── threshold_analysis.json
│
├── src/
│   ├── data/
│   │   ├── checks.py
│   │   └── make_dataset.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── threshold.py
│   │
│   └── monitoring/
│       ├── drift.py
│       ├── logging_utils.py
│       └── log/prediction_logs.jsonl
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone <https://github.com/0xNic11/fraud-ml-deployment>
cd fraud-ml-deployment

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 1️⃣ Train the Model

```bash
python src/models/train.py
```

### 2️⃣ Evaluate Performance

```bash
python src/models/evaluate.py
```

### 3️⃣ Run Predictions

```bash
python src/models/predict.py
```

---

## 📈 Evaluation Strategy

Because fraud detection is highly imbalanced, accuracy alone is misleading.
This project focuses on:

* Precision-Recall Curve
* PR-AUC
* Recall at chosen threshold
* Confusion Matrix analysis

The decision threshold is selected to ensure strong fraud recall.

---

## 🧪 Robustness Validation

Model performance is evaluated across transaction amount segments to ensure consistent behavior.

Checks performed:

* PR-AUC by segment
* Recall stability
* Weak zone identification

---

## 📉 Drift Simulation

The project simulates distribution shifts in transaction amounts to illustrate how model performance can degrade over time.

Population Stability Index (PSI) is used to quantify drift risk.

---

## 🛰 Monitoring Plan

The system should track:

### Input Monitoring

* Transaction amount distribution
* Fraud rate trends
* Probability distribution

### Performance Monitoring

* Fraud recall
* Precision
* PR-AUC

### Operational Metrics

* Volume of flagged transactions
* False positive feedback

---

## 🔁 Retraining Policy

Retraining should be triggered when:

* Recall drops below target threshold
* PR-AUC decreases significantly
* Data distribution shift detected
* Fraud rate changes materially

Recommended review cadence: **Quarterly**

---

## 🧾 Model Governance

Key artifacts stored in `/models`:

* `baseline_pipeline.pkl` — trained pipeline
* `metadata.json` — threshold & decision policy

The prediction service must always read threshold values from metadata to ensure reproducibility.

---

## 📌 Key Learnings

This project illustrates the difference between:

**Training a model** vs **operating a model**

It emphasizes:

* Decision policies
* Monitoring signals
* Reliability considerations
* Lifecycle thinking

---

## 🏁 Project Status

✅ Model trained
✅ Threshold optimized
✅ Performance validated
✅ Robustness analyzed
✅ Monitoring defined
✅ Governance documented

The model is deployment-ready from a lifecycle perspective.


---


## 👤 Author

Abdullah Ashraf
Data Scientist / Machine Learning Engineer

---

## 📜 License

This project is for educational and portfolio purposes.

```
```
