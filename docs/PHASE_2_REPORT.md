# Phase 2 Academic Project Report: Model Implementation, Tuning & Evaluation

**Course/Evaluation:** Machine Learning Mini Project – Phase 2 Submission  
**Domain:** Natural Language Processing (NLP) & Multi-Class Text Classification  
**Target Languages:** English, Hindi (Devanagari), and Kannada (Kannada script)  
**Date:** October 2026  
**Status:** Completed, Tuned & Validated across 8 Models  

---

## 1. Executive Summary & Phase 2 Objectives

Phase 2 transitions from the exploratory data preparation of Phase 1 into the empirical core of the machine learning pipeline: **model implementation, hyperparameter optimization, cross-validation, and comparative benchmarking**.

### Rubric Requirements Addressed:
1. **Model Implementation ($\ge 6$ models with $\ge 2$ ensembles):** Implemented **8 models** (4 individual + 4 ensembles), exceeding all rubric criteria.
2. **Hyperparameter Tuning:** Conducted grid search optimization (`GridSearchCV`) across parameter spaces for regularization ($C$), smoothing ($\alpha$), tree depths, and estimator quantities.
3. **Cross-Validation Protocol:** Applied Stratified 5-Fold Cross-Validation (`cv=5`) to ensure balanced fold evaluation without data leakage.
4. **Performance Metrics:** Computed classification accuracy, precision, recall, macro-averaged F1, weighted F1, training latency, and per-sample inference latency.
5. **Visual Evaluation:** Generated 13 high-resolution visualization figures (8 confusion matrices, multi-class One-vs-Rest ROC curves, cross-validation variance box plots, and comparative metric bar charts).
6. **Production Selection & Error Analysis:** Evaluated model robustness, addressing high-confidence stammering on code-mixed inputs and selecting the Soft Voting Meta-Ensemble.

---

## 2. Summary Benchmark Results Across All 8 Models

Evaluated on the held-out test set ($N_{\text{test}} = 210$ samples, 70 English, 70 Hindi, 70 Kannada) and Stratified 5-Fold Cross-Validation on the training set ($N_{\text{train}} = 840$ samples):

| # | Model Name | Architecture Category | Test Acc | Macro F1 | Weighted F1 | 5-Fold CV Mean | CV Std Dev | Train Time (s) | Best Hyperparameters |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Multinomial Naive Bayes** | Individual / Generative | **1.0000** | **1.0000** | **1.0000** | **1.0000** | $\pm 0.0000$ | 651.52s | `alpha: 0.01`, `fit_prior: True` |
| **2** | **Logistic Regression** | Individual / Linear Margin | **1.0000** | **1.0000** | **1.0000** | **1.0000** | $\pm 0.0000$ | 8.83s | `C: 0.1`, `penalty: 'l2'`, `solver: 'lbfgs'` |
| **3** | **Linear Support Vector Classifier** | Individual / Maximum Margin | **1.0000** | **1.0000** | **1.0000** | **1.0000** | $\pm 0.0000$ | 2.24s | `C: 0.1`, `CalibratedClassifierCV (sigmoid)` |
| **4** | **Decision Tree Classifier** | Individual / Recursive Split | 0.9905 | 0.9905 | 0.9905 | 0.9964 | $\pm 0.0048$ | 3.25s | `criterion: 'gini'`, `max_depth: 20`, `min_samples_split: 2` |
| **5** | **Random Forest Classifier** | Ensemble / Bagging | **1.0000** | **1.0000** | **1.0000** | **1.0000** | $\pm 0.0000$ | 7.41s | `n_estimators: 50`, `max_depth: 30`, `max_features: 'sqrt'` |
| **6** | **Gradient Boosting Classifier** | Ensemble / Sequential Boosting | **1.0000** | **1.0000** | **1.0000** | **1.0000** | $\pm 0.0000$ | 57.58s | `n_estimators: 50`, `learning_rate: 0.1`, `max_depth: 3` |
| **7** | **Soft Voting Meta-Ensemble** | Ensemble / Probability Consensus | **1.0000** | **1.0000** | **1.0000** | **1.0000** | $\pm 0.0000$ | 0.45s | Consensual averaging of NB, LR, and Calibrated LinearSVC |
| **8** | **Stacking Meta-Ensemble** | Ensemble / Two-Tier Meta-Learner | **1.0000** | **1.0000** | **1.0000** | **1.0000** | $\pm 0.0000$ | 1.85s | Base: [NB, LR, SVC], Meta-Learner: `LogisticRegression(C=1.0)` |

---

## 3. Detailed Model Implementations & Tuning Profiles

### 3.1 Individual Base Models (4 Models)

#### Model 1: Multinomial Naive Bayes (MNB)
- **Role:** Canonical NLP baseline operating on discrete n-gram term frequencies.
- **Tuning Grid:** $\alpha \in \{0.001, 0.01, 0.1, 0.5, 1.0, 2.0\}$, `fit_prior` $\in \{\text{True}, \text{False}\}$.
- **Optimal Setting:** $\alpha = 0.01$, `fit_prior = True`.
- **Performance:** 100% test accuracy, zero CV variance across all 5 folds.
- **Critical Observation:** While achieving 100% on monolingual inputs, Naive Bayes suffers from length-induced overconfidence on code-mixed utterances due to the conditional feature independence assumption.

#### Model 2: Multiclass Logistic Regression
- **Role:** Discriminative soft-margin linear classifier optimizing cross-entropy loss.
- **Tuning Grid:** $C \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$, penalty $= 'l2'$.
- **Optimal Setting:** $C = 0.1$ with `lbfgs` solver.
- **Performance:** 100% test accuracy, 8.83s training time. Well-regularized linear margins cleanly separate character n-grams across Latin, Devanagari, and Kannada.

#### Model 3: Linear Support Vector Classifier (LinearSVC + Platt Calibration)
- **Role:** Maximum-margin hyperplane separator.
- **Tuning Grid:** $C \in \{0.01, 0.1, 1.0, 10.0\}$, calibrated via `CalibratedClassifierCV` (sigmoid Platt scaling).
- **Optimal Setting:** $C = 0.1$.
- **Performance:** 100% test accuracy, extremely rapid training (2.24s). Calibrated outputs yield probabilistic confidence scores suitable for user-facing deployments.

#### Model 4: Decision Tree Classifier (CART)
- **Role:** Non-parametric recursive binary splitting baseline.
- **Tuning Grid:** `max_depth` $\in \{10, 20, 30, \text{None}\}$, `criterion` $\in \{\text{"gini"}, \text{"entropy"}\}$, `min_samples_split` $\in \{2, 5, 10\}$.
- **Optimal Setting:** `criterion = "gini"`, `max_depth = 20`, `min_samples_split = 2`.
- **Performance:** 99.05% test accuracy (2 errors out of 210), CV mean = 99.64% ($\pm 0.0048$).
- **Linguistic Insight:** Trees evaluate one character feature at a time without aggregating multi-feature context, occasionally confusing sentences containing solitary loanwords or punctuation.

---

### 3.2 Ensemble Architectures (4 Models)

#### Model 5: Random Forest Classifier (Bagging Ensemble)
- **Ensemble Paradigm:** Bootstrap aggregation of 50 decorrelated decision trees with random feature subsampling ($m = \sqrt{D} \approx 180$ features per split).
- **Tuning Grid:** `n_estimators` $\in \{50, 100\}$, `max_depth` $\in \{20, 30, \text{None}\}$.
- **Optimal Setting:** `n_estimators = 50`, `max_depth = 30`, `max_features = 'sqrt'`.
- **Performance:** 100% test accuracy, resolving the 0.95% variance error of individual decision trees.

#### Model 6: Gradient Boosting Classifier (Boosting Ensemble)
- **Ensemble Paradigm:** Sequential stagewise additive tree modeling minimizing multinomial deviance loss.
- **Tuning Grid:** `n_estimators` $\in \{50, 100\}$, `learning_rate` $\in \{0.05, 0.1, 0.2\}$, `max_depth` $\in \{3, 5\}$.
- **Optimal Setting:** `n_estimators = 50`, `learning_rate = 0.1`, `max_depth = 3`.
- **Performance:** 100% test accuracy, CV mean = 100%. Demonstrates robust convergence, albeit with higher training latency (57.58s).

#### Model 7: Soft Voting Meta-Ensemble (Selected Production Model)
- **Ensemble Paradigm:** Consensus probability averaging across heterogeneous paradigms:
  $$P_{\text{Voting}}(y = k \mid \mathbf{x}) = \frac{1}{3} P_{\text{NB}}(y = k \mid \mathbf{x}) + \frac{1}{3} P_{\text{LR}}(y = k \mid \mathbf{x}) + \frac{1}{3} P_{\text{SVC}}(y = k \mid \mathbf{x})$$
- **Performance:** 100% test accuracy, 100% 5-fold CV score, ultra-fast construction (0.45s).
- **Core Advantage:** Dampens extreme probabilistic outliers. When an input contains both English and Hindi/Kannada tokens, the soft vote produces smooth, calibrated posterior distributions rather than binary 100% / 0% polarization.

#### Model 8: Stacking Meta-Ensemble
- **Ensemble Paradigm:** Two-tier stacked generalization.
  - **Tier-1 Base Estimators:** Multinomial Naive Bayes, Logistic Regression, LinearSVC.
  - **Tier-2 Meta-Learner:** Multiclass Logistic Regression ($C = 1.0$).
- **Performance:** 100% test accuracy, 100% 5-fold CV score. Meta-features ($3 \times 3 = 9$ probabilities) are fed into the meta-classifier to learn optimal hypothesis weighting.

---

## 4. Cross-Validation & Statistical Evaluation

### 4.1 Stratified 5-Fold Cross-Validation Analysis
To verify that models did not overfit the 840 training samples:

```
Fold 1 (N=168): 56 English | 56 Hindi | 56 Kannada -> Fold Accuracy = 1.0000
Fold 2 (N=168): 56 English | 56 Hindi | 56 Kannada -> Fold Accuracy = 1.0000
Fold 3 (N=168): 56 English | 56 Hindi | 56 Kannada -> Fold Accuracy = 1.0000
Fold 4 (N=168): 56 English | 56 Hindi | 56 Kannada -> Fold Accuracy = 1.0000
Fold 5 (N=168): 56 English | 56 Hindi | 56 Kannada -> Fold Accuracy = 1.0000

Mean Cross-Validation Accuracy = 1.0000 (0.0000 std dev) across 7 of 8 models.
Decision Tree achieved 0.9964 (0.0048 std dev).
```

### 4.2 Why is Cross-Validation Accuracy 100%?
The Unicode character sets for Latin (`[a-zA-Z]`), Devanagari (`[\u0900-\u097F]`), and Kannada (`[\u0C80-\u0CFF]`) share **zero overlapping character code points**. Sub-word character $n$-grams ($n \in [2, 5]$) guarantee that every monolingual sentence contains unambiguous script-specific tokens, yielding linearly separable hyperplanes.

---

## 5. Visual Artifacts in `Phase-2/plots/`

1. **Confusion Matrices (8 figures):**
   - `confusion_matrix_multinomial_nb.png`
   - `confusion_matrix_logistic_regression.png`
   - `confusion_matrix_svm_linearsvc.png`
   - `confusion_matrix_decision_tree.png`
   - `confusion_matrix_random_forest.png`
   - `confusion_matrix_gradient_boosting.png`
   - `confusion_matrix_voting_classifier.png`
   - `confusion_matrix_stacking_classifier.png`
2. **ROC Curves:** `roc_curves.png` (Multi-class One-vs-Rest AUC = 1.0000 across all classes).
3. **Metric Comparisons:**
   - `model_comparison_accuracy.png`
   - `model_comparison_f1.png`
   - `training_time_comparison.png`
   - `cross_validation_comparison.png` (Box plot showing fold variance).

---

## 6. Production Model Selection & Code-Mixed Handling

### 6.1 Why Soft Voting Ensemble is the Production Winner
While 7 of the 8 models scored 100% on clean monolingual test data, real-world deployment presents **code-mixed sentences** (e.g., `"data science and तकनीक"` or `"cricket match ಮತ್ತು ಆಟಗಾರರು"`).
- **Naive Bayes Flaw:** Independent feature probability multiplication causes longer tokens in one script to drown out shorter tokens in another, leading to 100% overconfident misclassification.
- **Soft Voting Advantage:** Probability averaging prevents single-model overconfidence, providing calibrated continuous probabilities that accurately reflect script mixtures.

---

## 7. Phase 2 Deliverables Checklist (5/5 Complete)

- [x] **8 Algorithms Implemented & Serialized in `Phase-2/models/`**
- [x] **Hyperparameter Optimization via GridSearchCV documented in `Phase-2/results/model_comparison.csv`**
- [x] **Stratified 5-Fold Cross-Validation executed across all models**
- [x] **13 High-Resolution Evaluation Figures generated in `Phase-2/plots/`**
- [x] **Automated Execution & Benchmark Runner created: `Phase-2/run_phase2.py`**
