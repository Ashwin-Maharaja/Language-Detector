# 🌐 Multi-Script Language Detection System
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.2+-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-v2.3+-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline and production-ready web application for detecting languages across distinct scripts: **English** (Latin), **Hindi** (Devanagari), and **Kannada** (Kannada script). The system features character/word n-gram TF-IDF engineering, 8 benchmarked ML models (individual, bagging, boosting, and meta-ensembles), script-ratio code-mixing analysis, administrative audit logging, and automated PDF report generators.

---

## 📑 Table of Contents
- [Problem Formulation](#-problem-formulation)
- [Repository Structure](#-repository-structure)
- [Dataset Specifications](#-dataset-specifications)
- [Model Architecture & Benchmarks](#-model-architecture--benchmarks)
- [Installation & Quickstart](#-installation--quickstart)
- [Execution Modes](#-execution-modes)
- [Interactive Web Application](#-interactive-web-application)
- [Documentation & Reports](#-documentation--reports)
- [License](#-license)

---

## 📋 Problem Formulation

In multilingual Natural Language Processing (NLP) pipelines, accurately routing queries to downstream services (such as machine translation, sentiment analysis, or search indexing) depends on rapid, low-latency language identification.

This project addresses tri-lingual classification across disjoint scripts:
1. **English (Latin):** Unicode `U+0000`–`U+007F`
2. **Hindi (Devanagari):** Unicode `U+0900`–`U+097F`
3. **Kannada (Dravidian script):** Unicode `U+0C80`–`U+0CFF`

### Key Highlights
- **Tri-lingual Dataset:** 1,050 balanced samples (350 per language) across 8 everyday thematic domains.
- **Robust Feature Engineering:** Multi-level TF-IDF extraction (character n-grams $n \in [2, 5]$ and word n-grams $n \in [1, 2]$).
- **8 Benchmarked Classifiers:** 4 standalone models + 4 ensemble architectures (Random Forest, Gradient Boosting, Soft Voting Meta-Ensemble, Stacking Meta-Ensemble).
- **Code-Mixing & Robustness Engine:** Unicode script-ratio verification to prevent Naive Bayes length-bias on mixed text.
- **Enterprise Web Interface:** Flask web app with interactive testing, prediction audit logging, confusion matrices, and PDF exports.

---

## 🗂️ Repository Structure

```
Language Detection/
├── data/                                      # Datasets & Metadata
│   ├── Language Detection.csv                 # Kaggle reference source dataset (17 languages)
│   ├── language_dataset.csv                   # Balanced curated tri-lingual dataset (1,050 rows)
│   ├── metadata.json                          # Machine-readable schema & Unicode distributions
│   ├── prediction_history.json                # Live query audit log store
│   └── README.md                              # Dataset documentation & provenance
│
├── docs/                                      # Academic Reports & Shortlisting
│   ├── model_shortlisting.md                  # Mathematical rationale for classifier selection
│   ├── PHASE_1_REPORT.md                      # Phase 1: EDA & Preprocessing documentation
│   └── PHASE_2_REPORT.md                      # Phase 2: Training & Benchmarking documentation
│
├── models/                                    # Serialized Model Artifacts (.joblib)
│   ├── best_model.joblib                      # Soft Voting Meta-Ensemble (production model)
│   ├── tfidf_vectorizer.joblib                # Primary character TF-IDF vectorizer
│   ├── label_encoder.joblib                   # Target label encoder
│   ├── multinomial_nb_model.joblib            # Multinomial Naive Bayes
│   ├── logistic_regression_model.joblib       # Regularized Logistic Regression
│   ├── svm_linearsvc_model.joblib             # Calibrated Linear Support Vector Classifier
│   ├── decision_tree_model.joblib             # Decision Tree Classifier
│   ├── random_forest_model.joblib             # Random Forest (Bagging)
│   ├── gradient_boosting_model.joblib         # Gradient Boosting
│   ├── voting_classifier_model.joblib         # Soft Voting Meta-Ensemble
│   └── stacking_classifier_model.joblib       # Stacking Meta-Ensemble
│
├── plots/                                     # High-Resolution Evaluation Figures
│   ├── class_distribution.png                 # Class balance bar chart
│   ├── text_length_distribution.png           # Sentence length distributions
│   ├── character_frequency.png                # Script character frequency profiles
│   ├── correlation_heatmap.png                # Feature correlation matrix
│   ├── confusion_matrix_*.png                 # Confusion matrices for all 8 models
│   ├── model_comparison_accuracy.png          # Test accuracy benchmark chart
│   ├── model_comparison_f1.png                # Macro & Weighted F1 benchmark chart
│   ├── training_time_comparison.png           # Training duration & latency profile
│   ├── cross_validation_comparison.png        # 5-fold CV score boxplot
│   └── roc_curves.png                         # Multi-class One-vs-Rest ROC curves
│
├── preprocessed_artifacts/                    # Train/Test TF-IDF matrices & encoders
│   ├── tfidf_vectorizer.joblib
│   ├── label_encoder.joblib
│   ├── X_train_tfidf.joblib
│   ├── X_test_tfidf.joblib
│   ├── y_train.joblib
│   └── y_test.joblib
│
├── results/                                   # Evaluation Metrics & Benchmarks
│   └── model_comparison.csv                   # Complete metrics table across all 8 models
│
├── src/                                       # Core Application Source Modules
│   ├── __init__.py                            # Package initialization
│   ├── eda_preprocessing.py                   # Data cleaning, EDA plots, & feature pipelines
│   ├── model_training.py                      # Multi-model training, 5-fold CV, & evaluation
│   └── deploy.py                              # Flask server, REST APIs, & interactive UI
│
├── create_dataset.py                          # Synthetic & balanced dataset generation script
├── generate_models_pdf.py                     # ReportLab PDF compiler for concise model summaries
├── generate_report_pdf.py                     # Publication-grade academic report PDF compiler
├── main.py                                    # Master CLI orchestration runner
├── ML_Models_Summary.pdf                      # Formatted 3-page ML models reference PDF
├── Phase1_Data_Preprocessing.ipynb            # Condensed Phase 1 Jupyter Notebook
├── Phase2_Model_Training_and_Evaluation.ipynb # Condensed Phase 2 Jupyter Notebook
├── PROJECT_REPORT.md                          # Full academic report (markdown)
├── PROJECT_REPORT.pdf                         # Full publication-grade 8-page report (PDF)
├── requirements.txt                           # Production Python dependencies
├── .gitignore                                 # Git ignore configuration
└── README.md                                  # Repository documentation (this file)
```

---

## 📊 Dataset Specifications

| Metric / Attribute | Value |
| :--- | :--- |
| **Languages Supported** | English, Hindi (हिंदी), Kannada (ಕನ್ನಡ) |
| **Total Observations** | 1,050 samples (350 per language — perfectly balanced) |
| **Feature Columns** | `Text` (input string), `Language` (ground truth label) |
| **Thematic Coverage** | Daily Life, Education, Technology, Sports, News, Science, Food & Culture, Weather |
| **Character Encoding** | UTF-8 |
| **Benchmark Baseline Source**| [Kaggle: Language Detection Dataset (Basil Saji)](https://www.kaggle.com/datasets/basilb2s/language-detection) |

---

## 🤖 Model Architecture & Benchmarks

All models were evaluated on an 80/20 stratified train-test split (840 training samples, 210 test samples) and verified using **Stratified 5-Fold Cross-Validation**.

| # | Classifier Architecture | Type | Test Accuracy | Macro F1 | 5-Fold CV Mean | Inference Latency |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| 1 | **Multinomial Naive Bayes** | Probabilistic | 100.0% | 1.0000 | 1.0000 ± 0.000 | 0.05 ms |
| 2 | **Logistic Regression** (L2) | Linear Classifier | 100.0% | 1.0000 | 1.0000 ± 0.000 | 0.00 ms |
| 3 | **Linear Support Vector Classifier** | Margin-based | 100.0% | 1.0000 | 1.0000 ± 0.000 | 0.02 ms |
| 4 | **Decision Tree Classifier** | Rule Induction | 99.05% | 0.9905 | 0.9964 ± 0.005 | 0.01 ms |
| 5 | **Random Forest Classifier** | Bagging Ensemble | 100.0% | 1.0000 | 1.0000 ± 0.000 | 0.20 ms |
| 6 | **Gradient Boosting Classifier** | Boosting Ensemble | 100.0% | 1.0000 | 1.0000 ± 0.000 | 0.02 ms |
| 7 | **Soft Voting Meta-Ensemble** *(Selected)* | Meta-Ensemble | **100.0%** | **1.0000** | **1.0000 ± 0.000** | **0.03 ms** |
| 8 | **Stacking Classifier** | Meta-Ensemble | 100.0% | 1.0000 | 1.0000 ± 0.000 | 0.03 ms |

*Selected Production Model:* **Soft Voting Meta-Ensemble** combining Multinomial Naive Bayes, Logistic Regression, and Calibrated LinearSVC. It produces well-calibrated class posterior probabilities while retaining sub-millisecond inference speed.

---

## ⚙️ Installation & Quickstart

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/<your-username>/language-detection.git
cd "Language Detection"

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Modes

### Master Runner (`main.py`)
The master runner provides unified CLI commands to reproduce the entire pipeline:

```bash
# Run the full pipeline sequentially (dataset generation -> EDA -> model training)
python main.py all

# Run individual pipeline stages:
python main.py dataset    # Generate balanced tri-lingual dataset
python main.py eda        # Execute EDA and generate preprocessed feature matrices
python main.py train      # Train all 8 models, run 5-fold CV, and export plots
python main.py deploy     # Start the Flask web application
```

### Direct Script Invocations
```bash
# Generate documentation PDFs
python generate_report_pdf.py     # Generates PROJECT_REPORT.pdf
python generate_models_pdf.py     # Generates ML_Models_Summary.pdf

# Launch Web Application directly
python src/deploy.py
```

### Interactive Jupyter Notebooks
For step-by-step experimentation and visualization:
- [`Phase1_Data_Preprocessing.ipynb`](file:///Phase1_Data_Preprocessing.ipynb) — In-depth data analysis, script distribution histograms, and n-gram extraction.
- [`Phase2_Model_Training_and_Evaluation.ipynb`](file:///Phase2_Model_Training_and_Evaluation.ipynb) — Model training, hyperparameter configurations, cross-validation, and ROC curve generation.

---

## 🌐 Interactive Web Application

Start the server using `python src/deploy.py` or `python main.py deploy`, then navigate to `http://localhost:5000`.

### UI Highlights
1. **Live Detection Console:** Real-time text classification with per-class posterior probabilities, inference latency timers, and one-click presets.
2. **Script Breakdown & Code-Mixing Analysis:** Visual breakdown of Latin, Devanagari, and Kannada character ratios.
3. **Dataset & Metadata Explorer:** Interactive data table with script filters and metadata viewer.
4. **Model Benchmark Dashboard:** Interactive model selector with confusion matrix previews and metric comparisons.
5. **System Control & Prediction Audit Logs:** Admin dashboard with query logs, debounced logging, and downloadable CSV exports.
6. **One-Click Academic PDF Downloads:** Direct download endpoints for `PROJECT_REPORT.pdf` and `ML_Models_Summary.pdf`.

---

## 📄 Documentation & Reports

- **Full Project Report (PDF):** [`PROJECT_REPORT.pdf`](file:///PROJECT_REPORT.pdf) (8-page comprehensive publication-style report).
- **Concise Models Summary (PDF):** [`ML_Models_Summary.pdf`](file:///ML_Models_Summary.pdf) (3-page executive summary).
- **Markdown Documentation:** [`PROJECT_REPORT.md`](file:///PROJECT_REPORT.md) and [`docs/`](file:///docs/).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
