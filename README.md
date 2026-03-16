# 🎓 Academic Risk & Engagement Prediction System (AREPS)

> **Identifying at-risk students early and delivering personalised study recommendations — before it's too late**
>
> A learning analytics system that analyses five behavioural signals from a student's VLE (Virtual Learning Environment) interactions and predicts their academic risk level — High, Medium, or Low. Critically, the model output is never shown to users. Instead, it is translated into plain-English observations and actionable next steps via a Streamlit interface.

---

<div align="center">

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Model-DecisionTree-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Data-OULAD%2032%2C593%20students-4B0082)](https://analyse.kmi.open.ac.uk/open_dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

> **Want the visual overview first?** The deck covers the research problem, 4-notebook pipeline, feature engineering rationale, and system design in 12 slides.

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/119_kAFPmuEVT1xKVpVZZ34KprUcbIN3_/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Problem Statement](#1-problem-statement) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Dataset — OULAD](#6-dataset--oulad) |
| 7 | [4-Notebook Research Pipeline](#7-4-notebook-research-pipeline) |
| 8 | [Feature Engineering Rationale](#8-feature-engineering-rationale) |
| 9 | [Model Design — Interpretable by Choice](#9-model-design--interpretable-by-choice) |
| 10 | [Prediction Output & Recommendation Logic](#10-prediction-output--recommendation-logic) |
| 11 | [Streamlit Interface & CLI](#11-streamlit-interface--cli) |
| 12 | [How to Replicate — Full Setup Guide](#12-how-to-replicate--full-setup-guide) |
| 13 | [Business Applications & Other Domains](#13-business-applications--other-domains) |
| 14 | [How to Improve This Project](#14-how-to-improve-this-project) |
| 15 | [Troubleshooting](#15-troubleshooting) |
| 16 | [Glossary](#16-glossary) |

---

## 1. Problem Statement

### The challenge in education

Student attrition and academic failure carry severe consequences — for students personally, for institutions financially, and for the social mission of education. The critical window for intervention is **early in the course**: once a student has disengaged, re-engagement is exponentially harder.

Traditional academic warning systems are reactive — they trigger after a student fails an assessment or misses a deadline. By then, the pattern of disengagement is well-established and difficult to reverse.

**The insight:** Behavioural signals from Virtual Learning Environments — how much a student clicks, how early they engage, whether they accessed course materials before the start date — are available *weeks* before any grade exists. These signals are powerful early predictors of academic outcomes.

Core pain points:

- 📉 **Late intervention** — conventional early warning systems flag risk after poor grades, not before
- 🔢 **Numbers without meaning** — raw probability scores from ML models are not actionable for tutors or students
- 👁️ **Engagement blindness** — institutions collect rich VLE clickstream data but lack tools to translate it into student-facing guidance
- 🏛️ **Scale** — a single tutor cannot monitor engagement patterns for hundreds of students; automated systems are essential

### What AREPS answers

> *"Based on this student's VLE engagement behaviour, what is their academic risk level — and what specific actions should they take right now?"*

The system deliberately hides the ML prediction from the end user. Instead, it surfaces **observations** (what concerning patterns were detected) and **recommendations** (what the student should do about it) — making the system counsellor-friendly rather than technically opaque.

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Dataset** | OULAD (Open University Learning Analytics Dataset) — 32,593 student enrolments |
| **Target variable** | Academic risk: `1` = at-risk (Fail/Withdrawn), `0` = successful (Pass/Distinction) |
| **Class distribution** | At-risk: 17,208 (52.8%) · Successful: 15,385 (47.2%) |
| **5 input features** | total_click, early_click, early_active_days, first_activity_day, pre_course_engaged |
| **Model** | `DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=10)` |
| **Feature importance** | total_click (0.836) · early_click (0.112) · first_activity_day (0.052) |
| **Risk levels** | High ≥ 0.7 · Medium ≥ 0.4 · Low < 0.4 (from `config.yaml`) |
| **Model explainability** | `decision_path()` — surfaces the exact tree splits that led to the prediction |
| **Output format** | Risk level + probability + observations + recommendations + model explanation |
| **UI** | Streamlit (sliders + number input + radio) |
| **CLI** | `python main.py` with hardcoded test input |
| **Python version** | 3.12 |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.12 | Core language |
| **ML Model** | Scikit-learn `DecisionTreeClassifier` | Interpretable risk classifier — `decision_path()` enables rule extraction |
| **Data Processing** | Pandas 2.3.3 · NumPy 2.3.5 | OULAD data loading, aggregation, feature engineering |
| **Memory Management** | Python `gc` + dtype optimisation | Reduces `studentVle.csv` from 1,402 MB to 142 MB (10× reduction) |
| **Serialisation** | joblib 1.5.3 | Saves/loads `finalized_model.joblib` |
| **Config** | PyYAML 6.0.3 | Reads `config/config.yaml` for risk thresholds and engagement cutoffs |
| **Web UI** | Streamlit 1.52.1 | Interactive sliders + explainer expanders |
| **Visualisation** | Matplotlib, Seaborn (notebooks) | EDA plots across 4 research notebooks |
| **Path management** | `pathlib.Path` | Portable absolute path resolution in `utils/path_utils.py` |

---

## 4. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA RESEARCH PIPELINE                        │
│                                                                      │
│  OULAD raw CSVs (10.6M VLE rows + 32,593 student records)           │
│         │                                                            │
│  N1: Memory-safe loading + student_master_v1.csv                    │
│  N2: Target design (success=1/0) + student_master_v2.csv            │
│  N3: Temporal features + student_master_v3_temporal.csv             │
│  N4: Feature selection + DecisionTree + finalized_model.joblib       │
└──────────────────────────────────────────────────────────────────────┘
                               │
                     models/artefacts/
                     finalized_model.joblib
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                       INFERENCE LAYER                                │
│                                                                      │
│  5 engagement features (student_dict)                               │
│         │                                                            │
│  utils/validators.py  → validate_input()                            │
│  core/feature_builder.py → build_features()                         │
│         │  (sentinel 999 for never-engaged first_activity_day)       │
│  core/predictor.py → predict_and_recommend()                        │
│         │                                                            │
│    model.predict_proba() → risk_probability                          │
│    config.yaml thresholds → risk_level (High/Medium/Low)            │
│    core/recommender.py → observations + actions                     │
│    explain_tree_decision() → decision_path rules                     │
│         │                                                            │
│    {risk_probability, risk_level, observations,                      │
│     recommendations, model_explanation}                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   streamlit_app.py                         main.py
   Interactive sliders                    CLI test entry
   + risk display                         point
   + recommendations
   + "Why was this predicted?" expander
```

---

## 5. Repository Structure

```
Academic-Risk-Engagement-Prediction-System/
│
├── core/                              # Business logic
│   ├── __init__.py
│   ├── feature_builder.py             # build_features() — dict → DataFrame, sentinel fill
│   ├── predictor.py                   # predict_and_recommend() + explain_tree_decision()
│   └── recommender.py                 # generate_recommendations() — observation + action rules
│
├── utils/                             # Infrastructure
│   ├── __init__.py
│   ├── model_loader.py                # load_model() — joblib.load via MODEL_PATH
│   ├── path_utils.py                  # PROJECT_ROOT, MODEL_PATH, CONFIG_PATH constants
│   └── validators.py                 # validate_input() — required field check
│
├── models/artefacts/
│   └── finalized_model.joblib         # Trained DecisionTreeClassifier
│
├── config/
│   └── config.yaml                    # Risk thresholds + early engagement cutoff
│
├── test/experiment/notebook/         # 4-notebook research pipeline
│   ├── N1_Data_Understanding_and_Memory-Safe_Loading.ipynb
│   ├── N2_Engagement_Understanding_and_Target_Design.ipynb
│   ├── N3_Temporal_Engagement_and_Early_Signals.ipynb
│   └── N4_Engagement-Aware_Recommendation_System.ipynb
│
├── streamlit_app.py                   # Streamlit UI
├── main.py                            # CLI entry point
├── requirements.txt                   # Core dependencies
├── requirements_new.txt               # Full frozen dependency set
└── .python-version                    # 3.12
```

---

## 6. Dataset — OULAD

### Open University Learning Analytics Dataset

The **OULAD** (Open University Learning Analytics Dataset) is a real-world anonymised dataset from the UK Open University, covering 22 modules and 32,593 student enrolments from 2013–2014.

| File | Description | Size |
|------|-------------|------|
| `studentInfo.csv` | Student demographics + final result | 32,593 rows |
| `studentRegistration.csv` | Enrolment and withdrawal dates | 32,593 rows |
| `studentVle.csv` | VLE clickstream events per student per day | **10,655,280 rows** |
| `vle.csv` | VLE resource metadata (activity types) | 6,364 rows |
| `assessments.csv` | Assessment metadata | 206 rows |
| `studentAssessment.csv` | Per-student assessment scores | 173,912 rows |
| `courses.csv` | Module metadata | 22 rows |

### Memory Optimisation (Notebook N1)

`studentVle.csv` at default dtypes consumes **1,402 MB**. By downcasting to appropriate types, memory is reduced to **142 MB** — a 10× reduction:

```python
student_vle_dtypes = {
    "code_module":       "category",   # object → category
    "code_presentation": "category",   # object → category
    "id_student":        "int32",      # int64 → int32
    "id_site":           "int32",      # int64 → int32
    "date":              "int16",      # int64 → int16
    "sum_click":         "int16",      # int64 → int16
}
```

### Target Variable Design (Notebook N2)

Raw `final_result` has 4 classes. These are collapsed into a binary risk label:

| Original value | Count | Risk label |
|---------------|-------|-----------|
| Pass | 12,361 | 0 — Successful |
| Distinction | 3,024 | 0 — Successful |
| Withdrawn | 10,156 | 1 — At Risk |
| Fail | 7,052 | 1 — At Risk |

The model predicts `P(at-risk)` — i.e., probability that the student will Withdraw or Fail.

### Key Engagement Statistics

From N2 analysis, the difference in engagement between groups is dramatic:

| Metric | At-Risk students | Successful students | Difference |
|--------|-----------------|-------------------|-----------|
| Mean total clicks | 696 | 2,355 | **3.4× more** |
| Mean active days | 33 | 98 | **3× more** |

From N3 temporal analysis:

| Metric | At-Risk | Successful | Implication |
|--------|---------|-----------|-------------|
| Mean early clicks (first 14 days) | 105 | 207 | Successful engage **2× more early** |
| Mean first activity day | −8.2 | −10.0 | Successful start **1.8 days earlier** |
| Pre-course engagement rate | 66.7% | 86.7% | Motivated students access before start |

---

## 7. 4-Notebook Research Pipeline

The system was built through four sequential research notebooks, each adding a new layer of understanding before any production code was written.

### Notebook N1 — Data Understanding & Memory-Safe Loading

**Goal:** Load the OULAD dataset safely without crashing memory.

Key outcomes:
- Identified `studentVle.csv` as the primary engagement source (10.6M rows)
- Applied dtype optimisation to reduce memory 10× (1,402 MB → 142 MB)
- Computed student-level engagement aggregates: `total_click`, `avg_clicks_per_day`, `max_clicks_day`, `active_days`
- Merged engagement signals with student demographics → `student_master_v1.csv`
- Found that 8.75% of students never interacted with VLE at all

### Notebook N2 — Engagement Understanding & Target Design

**Goal:** Define what "risk" means and validate that engagement predicts outcomes.

Key outcomes:
- Collapsed 4-class `final_result` into binary `success` target (Pass/Distinction=1, Fail/Withdrawn=0)
- Confirmed ML will work: at-risk students have 3.4× fewer clicks and 3× fewer active days than successful students
- Created `engagement_level` (Low/Medium/High via `pd.qcut`) for EDA
- Added `no_engagement_flag` for completely disengaged students
- Output: `student_master_v2.csv`

### Notebook N3 — Temporal Engagement & Early Signals

**Goal:** Answer — does *when* a student engages matter as much as *how much* they engage?

Key outcomes:
- Reloaded `studentVle.csv` (memory-optimised) to extract temporal signals
- `first_activity_day`: minimum date per student — negative values mean pre-course engagement
- `pre_course_engaged`: binary flag — did the student interact before day 0?
- `early_click` and `early_active_days`: aggregated from days 0–14 (first two weeks)
- Confirmed: successful students start 1.8 days earlier, engage 2× more in first 14 days
- Sentinel value 999 used for `first_activity_day` of students who never engaged (meaningful absence of behaviour — not missing data)
- Output: `student_master_v3_temporal.csv`

### Notebook N4 — Engagement-Aware Recommendation System

**Goal:** Build the final model and translate predictions into human-readable guidance.

Key outcomes:
- Selected 5 final features based on research findings and interpretability
- Fitted `DecisionTreeClassifier(max_depth=4, min_samples_leaf=200)` — deliberately shallow for explainability
- Feature importances: total_click (0.836) · early_click (0.112) · first_activity_day (0.052)
- Designed observation and recommendation logic layered on top of model output
- Saved model as `finalized_model.joblib`
- Core design principle: **"We do NOT expose ML to the user. We expose recommendations."**

---

## 8. Feature Engineering Rationale

All 5 features are derived from VLE clickstream data and chosen for both predictive power and early availability:

| Feature | Source | Rationale |
|---------|--------|-----------|
| `total_click` | Sum of all VLE interactions | Strongest overall signal (feature importance 0.836) — reflects cumulative learning effort |
| `early_click` | Sum of VLE clicks in days 0–14 | Second strongest (0.112) — early engagement predicts final outcomes; 2× difference between groups |
| `early_active_days` | Distinct days active in days 0–14 | Measures **consistency** vs intensity — daily habits matter more than cramming |
| `first_activity_day` | Minimum date in student's VLE records | Measures **timing** — students who start before course start (negative values) show motivation; sentinel 999 for never-engaged |
| `pre_course_engaged` | Binary: any VLE activity before day 0 | Measures **intrinsic motivation** — 86.7% of successful students vs 66.7% of at-risk |

### The Sentinel Value Design

Students who **never** engaged with the VLE have no rows in `studentVle.csv`, meaning `first_activity_day` is `NaN`. This is not missing data — it is **informative absence** (the student never logged in). Replacing with the median or dropping these students would hide the most at-risk group. Instead:

```python
X["first_activity_day"] = X["first_activity_day"].fillna(999)
```

A sentinel value of 999 places never-engaged students far beyond any legitimate activity day, correctly predicting high risk for this group.

---

## 9. Model Design — Interpretable by Choice

### Why a Decision Tree?

The model is `DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=10)` — deliberately simple. This is a **design decision, not a limitation**.

| Property | Value | Reasoning |
|----------|-------|-----------|
| `max_depth=4` | Maximum 4 decision splits | Prevents overfitting on 32,593 rows; keeps tree explainable |
| `min_samples_leaf=200` | Each leaf needs 200+ students | Ensures every prediction is backed by statistically significant evidence |
| `random_state=10` | Reproducible | Same tree structure every time the model is retrained |

A more complex ensemble (Random Forest, XGBoost) would produce marginally better classification metrics but would lose the single most valuable property for this use case: **the ability to explain exactly why a student was flagged**.

### Decision Path Explainability

`explain_tree_decision()` uses `model.decision_path()` — a scikit-learn method that returns the Compressed Sparse Row matrix of nodes traversed for a given input:

```python
node_indicators = model.decision_path(input_df)  # CSR matrix

for node_id in node_indicators.indices:
    if feature[node_id] != -2:                    # -2 = leaf node
        fname = feature_names[feature[node_id]]
        thresh = threshold[node_id]
        val = input_df.iloc[0][fname]

        if val <= thresh:
            explanation.append(f"{fname} <= {int(thresh)}")
        else:
            explanation.append(f"{fname} > {int(thresh)}")
```

This produces human-readable rules like:
- `Total Click <= 377` — student's total interactions are very low
- `Early Click <= 139` — first-14-day activity is below the at-risk threshold
- `First Activity Day > 20` — student started significantly later than peers

These rules are shown in the Streamlit UI under **"🧠 Why was this risk predicted?"**.

### Feature Importance

From the trained model:

| Feature | Importance | Interpretation |
|---------|-----------|---------------|
| `total_click` | **0.836** | Dominant signal — overall VLE engagement volume |
| `early_click` | **0.112** | Strong secondary signal — early engagement quality |
| `first_activity_day` | **0.052** | Timing signal — when the student first showed up |
| `early_active_days` | 0.000 | Not used by this tree (correlated with early_click) |
| `pre_course_engaged` | 0.000 | Not used by this tree (correlated with first_activity_day) |

---

## 10. Prediction Output & Recommendation Logic

### Full Output Structure

`predict_and_recommend()` returns a dictionary with five keys:

```python
{
    "risk_probability":  0.847,                              # float 0–1
    "risk_level":        "High",                             # "High" / "Medium" / "Low"
    "observations":      ["No engagement in early course period",
                          "Low early interaction with course content"],
    "recommendations":   ["Start with introductory materials immediately.",
                          "Watch intro videos and explore course structure",
                          "Schedule academic support or mentoring session."],
    "model_explanation": ["Total Click <= 377",
                          "Early Click <= 139"]
}
```

### Risk Level Thresholds (`config/config.yaml`)

```yaml
risk_levels:
  high:   0.7      # P(at-risk) >= 0.7 → High risk
  medium: 0.4      # P(at-risk) >= 0.4 → Medium risk
                   # P(at-risk) < 0.4  → Low risk

early_engagement:
  min_early_click: 50    # Below this → "Low early interaction" observation
```

### Recommendation Rules (`core/recommender.py`)

Observations and actions are rule-based — they fire based on input values, independent of the model probability:

| Trigger condition | Observation | Recommended action |
|------------------|-------------|-------------------|
| `early_active_days == 0` | No engagement in early course period | Start with introductory materials immediately |
| `early_click < 50` | Low early interaction with course content | Watch intro videos and explore course structure |
| `first_activity_day > 0` | Late course start compared to peers | Follow a structured 7-day recovery plan |
| `risk_probability >= 0.7` | (High risk flag) | Schedule academic support or mentoring session |

### The Human-First Design Principle

The system deliberately does not show the ML model's probability or class prediction to the student. Instead:

1. **Observations** explain *what* was detected in their behaviour
2. **Recommendations** give *specific, actionable* next steps
3. **"Why was this predicted?"** expander provides the technical explanation for interested tutors

This approach follows the research finding: *"Binary labels are not actionable."* A student seeing "High Risk: 0.847" has no idea what to do. A student seeing "No activity in first 14 days → Watch introductory videos" knows exactly what to do.

---

## 11. Streamlit Interface & CLI

### Streamlit App (`streamlit_app.py`)

The UI uses student-friendly language throughout, avoiding technical ML terminology:

| Input | Widget | Range | Description shown |
|-------|--------|-------|-------------------|
| Total interactions | Slider | 0–30,000 | "Number of learning actions such as opening materials, watching videos…" |
| Early interactions | Slider | 0–5,000 | "Number of learning actions in the first few days" |
| Early active days | Slider | 0–15 | "Number of distinct days the student was active early in the course" |
| First activity day | Number input | −25 to 60 | Negative values = pre-course engagement (explained in help text) |
| Pre-course engaged | Radio | 0 / 1 | "Did the student interact before the official start?" |

The `ℹ️ How do we measure engagement?` expander explains to non-technical users what VLE interactions mean, avoiding confusion between "clicks" and "time spent".

### Output Display

On clicking **"Analyze Engagement"**:
1. `## ⚠️ Engagement Risk: **{risk_level}**` — prominently displayed
2. Risk probability (raw number, for tutor context)
3. **Key Observations** — bulleted list of concerning patterns detected
4. **Recommended Next Steps** — bulleted list of specific actions
5. Expandable **"🧠 Why was this risk predicted?"** — decision tree path rules

### CLI (`main.py`)

```python
student_input = {
    "total_click":        800,
    "early_click":        0,
    "early_active_days":  0,
    "first_activity_day": 15,
    "pre_course_engaged": 0
}
result = predict_and_recommend(student_input)
print(result)
```

This represents a high-risk profile: 800 total clicks (below average), no early engagement, started 15 days after course start, no pre-course activity.

---

## 12. How to Replicate — Full Setup Guide

### Prerequisites

- Python 3.12
- OULAD dataset (for running notebooks; not required for inference)

---

### Step 1 — Clone & Install

```bash
git clone https://github.com/sahatanmoyofficial/Academic-Risk-Engagement-Prediction-System.git
cd Academic-Risk-Engagement-Prediction-System

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

### Step 2 — Verify Model File

The trained model should already be present:

```bash
ls models/artefacts/finalized_model.joblib
```

If missing, download OULAD from the [Open University](https://analyse.kmi.open.ac.uk/open_dataset), place CSVs in `dataset/unzipped/anonymisedData/`, and run notebooks N1 through N4 in order.

---

### Step 3 — Run the Streamlit App

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Step 4 — Run the CLI

```bash
python main.py
# Prints prediction dict for hardcoded high-risk profile
```

### Step 5 — Explore the Research Notebooks

```bash
pip install notebook matplotlib seaborn
jupyter notebook test/experiment/notebook/
```

Run N1 → N2 → N3 → N4 in order to reproduce the full data pipeline.

---

## 13. Business Applications & Other Domains

### Primary Use Case — Higher Education Early Warning

| Stakeholder | Value Delivered |
|-------------|----------------|
| **Students** | Personalised, actionable early-warning with specific recovery steps |
| **Tutors / academic advisors** | Prioritised caseload — focus support on high-risk students flagged early |
| **Course designers** | Identify courses where low early engagement is systemic → design intervention |
| **Institution leadership** | Reduce withdrawal rates and improve completion metrics at scale |
| **Accessibility teams** | Flag students from lower socioeconomic backgrounds for targeted outreach |

### Broader Applications of the Pattern

The architecture — behavioural signals → risk probability → human-readable recommendations — generalises beyond education:

| Domain | Analogous Application | Adaptation |
|--------|----------------------|-----------|
| **Corporate L&D** | Employee training completion risk | Replace VLE clicks with LMS interactions |
| **Customer success** | SaaS churn prediction + recommended interventions | Replace clicks with product usage events |
| **Healthcare** | Medication adherence monitoring | Replace VLE activity with app check-in signals |
| **Financial inclusion** | Credit risk with behavioural features | Replace engagement with mobile banking patterns |
| **HR / onboarding** | New employee performance risk | Replace VLE with onboarding task completion signals |

---

## 14. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Add model evaluation metrics** | 🔴 High | The notebook trains on all 32,593 rows with no train/test split — add `classification_report`, AUC-ROC, and confusion matrix on a held-out test set |
| **Train/test split** | 🔴 High | Currently no validation split — add `train_test_split(test_size=0.2, random_state=10, stratify=y)` before fitting |
| **Compare ensemble models** | 🟡 Medium | Benchmark against RandomForest and GradientBoosting — the interpretability argument only holds if the accuracy trade-off is documented |
| **SHAP values** | 🟡 Medium | Add SHAP force plots in Streamlit for richer per-prediction explanations beyond decision path rules |
| **Cross-validation** | 🟡 Medium | Add `StratifiedKFold` CV for more reliable performance estimates across demographic subgroups |
| **Demographic fairness** | 🟡 Medium | Check model performance by `imd_band` (deprivation) and `disability` — at-risk students in deprived bands should not be systematically mis-predicted |

### 🏗️ Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **Fix typo in recommender.py** | `"ate course start compared to peers"` → `"Late course start compared to peers"` |
| **Move risk thresholds to predictor** | `predict_and_recommend()` hardcodes `0.7` and `0.4` but `config.yaml` has these values — load from config consistently |
| **Unit tests** | Test `validate_input()` (missing fields), `build_features()` (sentinel fill), `explain_tree_decision()` (correct rule extraction) |
| **Add model metadata** | Record training date, dataset version, and feature list alongside `finalized_model.joblib` |
| **Notebook reproducibility** | Add a single `run_all_notebooks.sh` script to reproduce the full pipeline from OULAD raw files |

### 📦 Product Improvements

- Add a **cohort comparison** mode — "How does this student compare to the average successful student?"
- Display a **"What would change my risk?"** panel — show the threshold values that would move the student from High to Medium risk
- Support **CSV batch upload** — input a spreadsheet of students, get risk assessments for the whole cohort
- Add **progress tracking** — allow tutors to update engagement metrics weekly and see how risk level changes over time

---

## 15. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `FileNotFoundError: finalized_model.joblib` | Confirm `models/artefacts/finalized_model.joblib` exists; retrain from N4 notebook if missing |
| `ValueError: Missing required fields` | Check that all 5 keys are present in the `student_dict` passed to `predict_and_recommend()` |
| `AttributeError: decision_path` | Ensure `finalized_model.joblib` is a single `DecisionTreeClassifier`, not an ensemble — `decision_path()` is not available on RandomForest |
| Streamlit runs but shows no output | Click **"Analyze Engagement"** button — output only displays after button press |
| Observations list is empty | Expected when all engagement signals are strong — `"No concerning behavior detected so far."` is displayed |
| `first_activity_day` gives unexpected risk | Check that negative values (pre-course engagement) are being entered correctly — negative days are valid and lower risk |
| Python version mismatch | Use exactly Python 3.12 (see `.python-version`); joblib models are pickle-based and version-sensitive |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **OULAD** | Open University Learning Analytics Dataset — anonymised real-world dataset of 32,593 UK Open University student enrolments |
| **VLE** | Virtual Learning Environment — online platform where students interact with course materials (analogous to Moodle, Blackboard, Canvas) |
| **sum_click** | The number of times a student clicked on a VLE resource on a given day — proxy for learning engagement |
| **total_click** | Aggregate of all `sum_click` events for a student across the full course |
| **early_click** | Total clicks in the first 14 days of the course (days 0–14) |
| **early_active_days** | Number of distinct calendar days the student was active in the first 14 days |
| **first_activity_day** | The earliest date (relative to course start) at which the student accessed the VLE — negative values indicate pre-course engagement |
| **pre_course_engaged** | Binary flag: 1 if the student accessed the VLE before the course officially started (day 0) |
| **Sentinel value** | A deliberately chosen out-of-range value (999) used to represent informative absence of data — students who never engaged have no `first_activity_day`, and 999 correctly places them at maximum risk |
| **DecisionTreeClassifier** | Sklearn tree model that learns axis-aligned splits on features — interpretable by design |
| **decision_path()** | Sklearn method that returns the nodes traversed for a given input — used to extract human-readable split rules |
| **max_depth** | Maximum number of splits in the decision tree — limiting to 4 prevents overfitting and keeps explanations concise |
| **min_samples_leaf** | Minimum training samples required in a leaf node — 200 ensures every prediction is backed by sufficient evidence |
| **feature_importances_** | Attribute on a fitted tree: proportion of information gain attributable to each feature |
| **Risk probability** | `model.predict_proba(df)[0][1]` — P(at-risk), the model's confidence that a student will Fail or Withdraw |
| **Dtype optimisation** | Downcasting column types (e.g. int64 → int16) to reduce DataFrame memory usage — critical for the 10.6M-row VLE file |
| **IMD band** | Index of Multiple Deprivation — UK socioeconomic deprivation measure; present in OULAD as `imd_band` |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

---
