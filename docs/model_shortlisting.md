# Phase 1: Model Shortlisting & Mathematical Rationale

**Project Title:** Multi-Script Language Detection (English, Hindi, Kannada)  
**Evaluation Rubric Objective:** Shortlist at least 6 models, including at least 2 ensemble models, with detailed theoretical justification.

---

## 1. Selection Strategy & Rationale

In natural language processing with high-dimensional character $n$-gram representations ($D = 32,485$ features), the feature matrix is sparse, non-negative, and linearly separable across disjoint Unicode writing scripts. 

To ensure algorithmic diversity and evaluate distinct learning paradigms, **8 models** (4 individual base learners and 4 ensemble architectures) have been shortlisted for empirical evaluation in Phase 2:

```
                                  [Shortlisted Algorithms (8 Models)]
                                                   │
                  ┌────────────────────────────────┴────────────────────────────────┐
                  ▼                                                                 ▼
      [Individual Base Learners (4)]                                    [Ensemble Architectures (4)]
      ├── 1. Multinomial Naive Bayes (Probabilistic)                    ├── 5. Random Forest (Bagging)
      ├── 2. Logistic Regression (Linear Margin)                       ├── 6. Gradient Boosting (Boosting)
      ├── 3. Linear Support Vector Classifier (Max Margin)              ├── 7. Soft Voting Classifier (Meta-Ensemble)
      └── 4. Decision Tree Classifier (Hierarchical Partitioning)       └── 8. Stacking Classifier (Two-Tier Meta)
```

---

## 2. Individual Model Profiles & Mathematical Formulation

### 2.1 Model 1: Multinomial Naive Bayes (MNB)
- **Family:** Generative Probabilistic Classifier.
- **Suitability for NLP:** MNB is the canonical baseline for text categorization. It operates directly on word/character frequency counts and TF-IDF representations.
- **Mathematical Principle:**
  Under the conditional independence assumption of features given class $y$:
  $$P(y \mid \mathbf{x}) \propto P(y) \prod_{i=1}^{D} P(x_i \mid y)^{x_i}$$
  With Laplace smoothing ($\alpha > 0$) to eliminate zero-frequency penalties:
  $$\hat{\theta}_{y,i} = \frac{\sum_{j \in \text{class } y} x_{j,i} + \alpha}{\sum_{j \in \text{class } y} \sum_{k=1}^D x_{j,k} + \alpha D}$$
- **Hyperparameter Space for Phase 2:**
  - Smoothing parameter: $\alpha \in \{0.01, 0.1, 0.5, 1.0, 2.0\}$.
  - Class prior fitting: `fit_prior` $\in \{\text{True}, \text{False}\}$.

---

### 2.2 Model 2: Multiclass Logistic Regression
- **Family:** Discriminative Generalized Linear Model.
- **Suitability for NLP:** Highly effective in sparse, high-dimensional spaces where decision boundaries between disjoint scripts are linear.
- **Mathematical Principle:**
  Predicts class posterior probabilities using the Softmax link function:
  $$P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$
  Optimized via Cross-Entropy loss with $L_2$ Tikhonov regularization:
  $$\mathcal{L}(\mathbf{W}) = -\sum_{n=1}^N \sum_{k=1}^K y_{n,k} \log P(y = k \mid \mathbf{x}_n) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$
- **Hyperparameter Space for Phase 2:**
  - Inverse regularization strength: $C \in \{0.1, 1.0, 10.0, 50.0\}$.
  - Solver: `lbfgs` with `max_iter` $\in \{500, 1000\}$.

---

### 2.3 Model 3: Linear Support Vector Classifier (LinearSVC)
- **Family:** Maximum-Margin Linear Separator.
- **Suitability for NLP:** Joachims (1998) established SVMs as ideal for text classification because text induces sparse, orthogonal vector spaces where overfitting is prevented by maximizing the functional geometric margin.
- **Mathematical Principle:**
  Finds the optimal separating hyperplane by minimizing the squared hinge loss:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))^2$$
  *CalibratedClassifierCV* is applied with Platt scaling / Isotonic regression to yield well-calibrated class posterior probabilities $P(y \mid \mathbf{x})$.
- **Hyperparameter Space for Phase 2:**
  - Penalty parameter: $C \in \{0.1, 1.0, 5.0, 10.0\}$.
  - Calibration: `method='sigmoid'`, 5-fold stratified cross-validation.

---

### 2.4 Model 4: Decision Tree Classifier (CART)
- **Family:** Non-Parametric Greedy Recursive Partitioning.
- **Suitability for NLP:** Serves as a transparent, white-box non-linear baseline. It identifies critical character sub-sequences (e.g., Unicode threshold tests) that instantly partition writing systems.
- **Mathematical Principle:**
  Greedily selects split feature $j$ and threshold $t$ minimizing Gini Impurity:
  $$I_G(S) = 1 - \sum_{k=1}^K p_k^2$$
  $$\Delta I_G(S, j, t) = I_G(S) - \frac{|S_L|}{|S|} I_G(S_L) - \frac{|S_R|}{|S|} I_G(S_R)$$
- **Hyperparameter Space for Phase 2:**
  - Maximum tree depth: `max_depth` $\in \{10, 20, 30, \text{None}\}$.
  - Split criterion: `criterion` $\in \{\text{"gini"}, \text{"entropy"}\}$.

---

## 3. Ensemble Model Profiles & Mathematical Formulation

### 3.1 Model 5: Random Forest Classifier (Bagging Ensemble)
- **Ensemble Paradigm:** Bootstrap Aggregation (Bagging) + Feature Random Subspace.
- **Suitability for NLP:** Alleviates the high variance and tendency of individual decision trees to overfit sparse n-gram features by averaging across an ensemble of decorrelated trees.
- **Mathematical Principle:**
  Draws $B$ bootstrap samples $\mathcal{D}_b \subset \mathcal{D}$. At each node of each tree $T_b$, only a random subset of $m = \sqrt{D}$ features are evaluated for splitting.
  The ensemble probability is the average over all trees:
  $$P_{\text{RF}}(y = k \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_{T_b}(y = k \mid \mathbf{x})$$
- **Hyperparameter Space for Phase 2:**
  - Number of estimators: $n\_estimators \in \{50, 100, 200\}$.
  - Max features: `max_features` $\in \{\text{"sqrt"}, \text{"log2"}\}$.

---

### 3.2 Model 6: Gradient Boosting Classifier (Boosting Ensemble)
- **Ensemble Paradigm:** Sequential Gradient Error Boosting.
- **Suitability for NLP:** Sequentially trains shallow decision trees (pseudo-residuals), progressively focusing capacity on hard-to-classify or short utterances.
- **Mathematical Principle:**
  For iteration $m = 1, \dots, M$, computes pseudo-residuals representing the negative gradient of the multinomial deviance loss:
  $$r_{i,k,m} = -\left[\frac{\partial \mathcal{L}(y_i, f(\mathbf{x}_i))}{\partial f_k(\mathbf{x}_i)}\right]_{f = f_{m-1}} = \mathbb{I}(y_i = k) - p_{k,m-1}(\mathbf{x}_i)$$
  Fits a regression tree $h_{k,m}(\mathbf{x})$ to the residuals and updates the ensemble model:
  $$f_{k,m}(\mathbf{x}) = f_{k,m-1}(\mathbf{x}) + \eta \sum_{j=1}^{J} \gamma_{j,k,m} \mathbb{I}(\mathbf{x} \in R_{j,k,m})$$
  where $\eta \in (0, 1]$ represents shrinkage (learning rate).
- **Hyperparameter Space for Phase 2:**
  - Learning rate: $\eta \in \{0.05, 0.1, 0.2\}$.
  - Tree estimators: $n\_estimators \in \{50, 100\}$.
  - Tree depth: `max_depth` $\in \{3, 5\}$.

---

### 3.3 Model 7: Soft Voting Classifier (Heterogeneous Meta-Ensemble)
- **Ensemble Paradigm:** Probability Averaging across Heterogeneous Hypotheses.
- **Suitability for NLP:** Combines three fundamentally distinct learning paradigms (Generative Bayesian, Maximum Margin, and Discriminative Linear Margin) to eliminate individual model blindspots.
- **Base Estimators:**
  1. Multinomial Naive Bayes ($P_{\text{NB}}$)
  2. Multiclass Logistic Regression ($P_{\text{LR}}$)
  3. Calibrated Linear Support Vector Classifier ($P_{\text{SVC}}$)
- **Mathematical Principle:**
  Computes the weighted consensus posterior probability distribution:
  $$P_{\text{Voting}}(y = k \mid \mathbf{x}) = \sum_{m=1}^M w_m P_m(y = k \mid \mathbf{x}), \quad \sum_{m=1}^M w_m = 1$$
  Final decision rule:
  $$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} P_{\text{Voting}}(y = k \mid \mathbf{x})$$
- **Weight Configuration:** Equal weighting ($w_{\text{NB}} = w_{\text{LR}} = w_{\text{SVC}} = \frac{1}{3}$) or optimized weights based on Phase 2 CV F1 scores.

---

### 3.4 Model 8: Stacking Classifier (Two-Tier Meta-Ensemble)
- **Ensemble Paradigm:** Stacked Generalization (Wolpert, 1992).
- **Suitability for NLP:** Instead of static linear averaging, Stacking trains a meta-classifier that learns the characteristic biases and strengths of each base learner from out-of-fold validation predictions.
- **Architecture:**
  - **Tier-1 Base Learners:** Multinomial Naive Bayes, Logistic Regression, LinearSVC.
  - **Tier-2 Meta-Learner:** Multiclass Logistic Regression with $L_2$ regularization.
- **Mathematical Principle:**
  Constructs a meta-feature vector $\mathbf{z}_i$ from 5-fold cross-validated out-of-fold predictions:
  $$\mathbf{z}_i = \left[ P_{\text{NB}}(y \mid \mathbf{x}_i), P_{\text{LR}}(y \mid \mathbf{x}_i), P_{\text{SVC}}(y \mid \mathbf{x}_i) \right] \in \mathbb{R}^{3 \times 3} = \mathbb{R}^9$$
  The meta-learner $g(\mathbf{z})$ minimizes regularized empirical cross-entropy on $\mathbf{z}$:
  $$\hat{y} = g(\mathbf{z}) = \text{Softmax}(\mathbf{W}_{\text{meta}} \mathbf{z} + \mathbf{b}_{\text{meta}})$$

---

## 4. Phase 2 Cross-Validation & Metric Evaluation Protocol

| Evaluation Component | Protocol Specification |
| :--- | :--- |
| **Validation Scheme** | Stratified 5-Fold Cross-Validation (`cv=5`, maintaining 33.3% per class) |
| **Primary Metric** | Macro-Averaged F1-Score ($F1_{\text{macro}} = \frac{1}{K}\sum_{k=1}^K F1_k$) |
| **Secondary Metrics** | Classification Accuracy, Weighted F1, Precision, Recall, Confusion Matrices |
| **ROC Analysis** | Multi-class One-vs-Rest (OvR) Area Under Curve (AUC-ROC) |
| **Latency Benchmark** | Mean wall-clock training time (seconds) and per-query inference latency (ms) |
| **Overfitting Guard** | Cross-validation score variance analysis ($|\text{Score}_{\text{train}} - \text{Score}_{\text{test}}| \le 0.02$) |