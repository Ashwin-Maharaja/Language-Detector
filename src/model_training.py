"""
model_training.py - Comprehensive Model Training & Evaluation for Language Detection
=====================================================================================

This script trains, evaluates, and compares 8 machine learning models (including
ensemble methods) for language detection using TF-IDF character n-gram features.

Models Implemented:
    Individual:
        1. Multinomial Naive Bayes
        2. Logistic Regression (One-vs-Rest)
        3. Support Vector Machine (LinearSVC + CalibratedClassifierCV)
        4. Decision Tree Classifier (with pruning)
    Ensemble:
        5. Random Forest Classifier (bagging)
        6. Gradient Boosting Classifier (boosting)
    Meta-Ensemble:
        7. Voting Classifier (soft voting, top-3 models)
        8. Stacking Classifier (Logistic Regression meta-learner)

Usage:
    python src/model_training.py

Outputs:
    - Trained models saved to   models/
    - Evaluation plots saved to plots/
    - Comparison CSV saved to   results/model_comparison.csv
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – safe for scripts / servers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.calibration import CalibratedClassifierCV

# ---------- Individual classifiers ----------
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# ---------- Ensemble classifiers ----------
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)

# Suppress convergence / future warnings during grid search
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # Language Detection/
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
PLOTS_DIR    = PROJECT_ROOT / "plots"
RESULTS_DIR  = PROJECT_ROOT / "results"

DATASET_PATH = DATA_DIR / "language_dataset.csv"

# Preprocessed artefact paths (produced by preprocessing script)
PREPROCESSED_XTRAIN = MODELS_DIR / "X_train.joblib"
PREPROCESSED_XTEST  = MODELS_DIR / "X_test.joblib"
PREPROCESSED_YTRAIN = MODELS_DIR / "y_train.joblib"
PREPROCESSED_YTEST  = MODELS_DIR / "y_test.joblib"
PREPROCESSED_TFIDF  = MODELS_DIR / "tfidf_vectorizer.joblib"
PREPROCESSED_LE     = MODELS_DIR / "label_encoder.joblib"

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Plotting Style
# ──────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLOT_DPI = 150
FIGSIZE_WIDE  = (14, 7)
FIGSIZE_SQUARE = (10, 8)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. DATA LOADING & INLINE PREPROCESSING                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_preprocessed_data():
    """
    Attempt to load preprocessed data saved by the preprocessing pipeline.

    Returns
    -------
    tuple or None
        (X_train, X_test, y_train, y_test, tfidf_vectorizer, label_encoder)
        Returns None if any required file is missing.
    """
    required_files = [
        PREPROCESSED_XTRAIN, PREPROCESSED_XTEST,
        PREPROCESSED_YTRAIN, PREPROCESSED_YTEST,
        PREPROCESSED_TFIDF,  PREPROCESSED_LE,
    ]
    if all(f.exists() for f in required_files):
        logger.info("Loading preprocessed data from '%s' ...", MODELS_DIR)
        X_train = joblib.load(PREPROCESSED_XTRAIN)
        X_test  = joblib.load(PREPROCESSED_XTEST)
        y_train = joblib.load(PREPROCESSED_YTRAIN)
        y_test  = joblib.load(PREPROCESSED_YTEST)
        tfidf   = joblib.load(PREPROCESSED_TFIDF)
        le      = joblib.load(PREPROCESSED_LE)
        logger.info(
            "Preprocessed data loaded  –  train=%d, test=%d, classes=%d",
            X_train.shape[0], X_test.shape[0], len(le.classes_),
        )
        return X_train, X_test, y_train, y_test, tfidf, le
    return None


def preprocess_inline(csv_path: Path, test_size: float = 0.2, random_state: int = 42):
    """
    Self-contained preprocessing when pre-saved artefacts are unavailable.

    Steps:
        1. Load the CSV dataset.
        2. Clean text (drop nulls, strip whitespace, remove short entries).
        3. Encode labels.
        4. TF-IDF character n-gram vectorisation  (analyzer='char', ngram_range=(2,5)).
        5. Train/test split (stratified).
        6. Save artefacts to disk for future reuse.

    Parameters
    ----------
    csv_path : Path
        Path to ``language_dataset.csv``.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, tfidf_vectorizer, label_encoder)
    """
    logger.info("Preprocessed files not found – running inline preprocessing …")

    # --- Load ------------------------------------------------------------------
    if not csv_path.exists():
        logger.error("Dataset not found at '%s'. Aborting.", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    logger.info("Raw dataset shape: %s", df.shape)

    # --- Identify columns (flexible) ------------------------------------------
    # Try common column names for text and label
    text_col = None
    label_col = None
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in ("text", "content", "sentence", "data"):
            text_col = col
        elif col_lower in ("language", "label", "lang", "class", "target"):
            label_col = col
    # Fallback: assume first column is text, second is label (or vice-versa)
    if text_col is None or label_col is None:
        if df.shape[1] >= 2:
            text_col = df.columns[0] if text_col is None else text_col
            label_col = df.columns[1] if label_col is None else label_col
        else:
            logger.error("Cannot determine text / label columns from: %s", list(df.columns))
            sys.exit(1)
    logger.info("Using columns  –  text='%s', label='%s'", text_col, label_col)

    # --- Clean -----------------------------------------------------------------
    df = df[[text_col, label_col]].copy()
    df.dropna(inplace=True)
    df[text_col]  = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    # Remove very short texts (likely noise)
    df = df[df[text_col].str.len() >= 3].reset_index(drop=True)
    logger.info("Cleaned dataset shape: %s", df.shape)

    # --- Encode labels ---------------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    logger.info("Classes (%d): %s", len(le.classes_), list(le.classes_))

    # --- TF-IDF character n-grams (2–5) ----------------------------------------
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=50_000,
        sublinear_tf=True,
        strip_accents=None,  # Keep accents – valuable signal for language ID
    )
    X = tfidf.fit_transform(df[text_col])
    logger.info("TF-IDF matrix shape: %s  (nnz=%d)", X.shape, X.nnz)

    # --- Train / Test split (stratified) ----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    logger.info("Train size: %d  |  Test size: %d", X_train.shape[0], X_test.shape[0])

    # --- Persist artefacts ------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(X_train, PREPROCESSED_XTRAIN)
    joblib.dump(X_test,  PREPROCESSED_XTEST)
    joblib.dump(y_train, PREPROCESSED_YTRAIN)
    joblib.dump(y_test,  PREPROCESSED_YTEST)
    joblib.dump(tfidf,   PREPROCESSED_TFIDF)
    joblib.dump(le,      PREPROCESSED_LE)
    logger.info("Preprocessed artefacts saved to '%s'.", MODELS_DIR)

    return X_train, X_test, y_train, y_test, tfidf, le


def get_data():
    """
    Master loader: tries preprocessed data first, falls back to inline.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, tfidf_vectorizer, label_encoder)
    """
    result = load_preprocessed_data()
    if result is not None:
        return result
    return preprocess_inline(DATASET_PATH)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. MODEL DEFINITIONS  (each returns name, model, param_grid)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_multinomial_nb():
    """Multinomial Naive Bayes – classic text classification baseline."""
    model = MultinomialNB()
    param_grid = {
        "alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
        "fit_prior": [True, False],
    }
    return "Multinomial_NB", model, param_grid


def build_logistic_regression():
    """Logistic Regression classifier."""
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
    }
    return "Logistic_Regression", model, param_grid


def build_svm():
    """
    Linear SVM wrapped in CalibratedClassifierCV so we get probability estimates
    (needed for soft voting and ROC curves).
    """
    base_svc = LinearSVC(max_iter=3000, random_state=42)
    model = CalibratedClassifierCV(estimator=base_svc, cv=3)
    # Grid search over the inner estimator's hyperparameters
    param_grid = {
        "estimator__C": [0.1, 1.0, 5.0],
    }
    return "SVM_LinearSVC", model, param_grid


def build_decision_tree():
    """Decision Tree with pre-pruning to avoid overfitting."""
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        "max_depth": [20, 40, None],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"],
    }
    return "Decision_Tree", model, param_grid


def build_random_forest():
    """Random Forest – bagging ensemble of decision trees."""
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [30, None],
        "min_samples_split": [2, 5],
    }
    return "Random_Forest", model, param_grid


def build_gradient_boosting():
    """Gradient Boosting Classifier – sequential boosting ensemble."""
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.1],
        "max_depth": [3],
    }
    return "Gradient_Boosting", model, param_grid


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. TRAINING PIPELINE                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def train_model_with_tuning(name, model, param_grid, X_train, y_train, cv_folds=5):
    """
    Train a single model with GridSearchCV hyperparameter tuning.

    Parameters
    ----------
    name : str
        Human-readable model name.
    model : estimator
        Scikit-learn compatible estimator.
    param_grid : dict
        Hyperparameter search space.
    X_train : sparse matrix
        Training features.
    y_train : array-like
        Training labels.
    cv_folds : int
        Number of cross-validation folds for grid search.

    Returns
    -------
    dict
        {
            "name": str,
            "best_model": fitted estimator,
            "best_params": dict,
            "cv_scores": array of CV accuracy scores,
            "train_time": float (seconds),
        }
    """
    logger.info("=" * 70)
    logger.info("Training: %s", name)
    logger.info("=" * 70)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    start = time.time()

    if param_grid:
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=skf,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best_model  = grid.best_estimator_
        best_params = grid.best_params_
        logger.info("Best params : %s", best_params)
        logger.info("Best CV acc : %.4f", grid.best_score_)
    else:
        # No grid search (e.g. meta-ensembles) – just fit directly
        model.fit(X_train, y_train)
        best_model  = model
        best_params = {}

    elapsed = time.time() - start
    logger.info("Training time: %.2f s", elapsed)

    # 5-fold cross-validation scores on best model
    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1,
    )
    logger.info("CV accuracy  : %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    return {
        "name": name,
        "best_model": best_model,
        "best_params": best_params,
        "cv_scores": cv_scores,
        "train_time": elapsed,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. EVALUATION                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def evaluate_model(trained_info, X_test, y_test, class_names):
    """
    Evaluate a trained model on the held-out test set.

    Parameters
    ----------
    trained_info : dict
        Output from ``train_model_with_tuning``.
    X_test : sparse matrix
        Test features.
    y_test : array-like
        True test labels.
    class_names : list[str]
        Ordered class names for display.

    Returns
    -------
    dict
        Evaluation results including accuracy, precision, recall, F1, confusion
        matrix, classification report, and predicted probabilities (if available).
    """
    name  = trained_info["name"]
    model = trained_info["best_model"]

    logger.info("-" * 50)
    logger.info("Evaluating: %s", name)
    logger.info("-" * 50)

    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    logger.info("Accuracy  : %.4f", acc)
    logger.info("Precision : %.4f (weighted)", precision)
    logger.info("Recall    : %.4f (weighted)", recall)
    logger.info("F1        : %.4f (weighted)", f1)
    logger.info("\n%s", report)

    # Predicted probabilities (for ROC curves) – not all models support this
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    return {
        "name": name,
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cv_scores": trained_info["cv_scores"],
        "train_time": trained_info["train_time"],
        "best_params": trained_info["best_params"],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. VISUALISATION FUNCTIONS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def plot_confusion_matrix(cm, class_names, model_name, save_dir):
    """
    Plot and save a heatmap confusion matrix for a single model.

    Parameters
    ----------
    cm : ndarray
        Confusion matrix (n_classes × n_classes).
    class_names : list[str]
        Class labels.
    model_name : str
        Model identifier (used in title & filename).
    save_dir : Path
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6),
                                    max(6, len(class_names) * 0.5)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    fpath = save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fpath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", fpath)


def plot_model_comparison_accuracy(results, save_dir):
    """Bar chart comparing test-set accuracy across all models."""
    names = [r["name"] for r in results]
    accs  = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(names, accs, color=sns.color_palette("viridis", len(names)), edgecolor="black")
    # Annotate each bar with its value
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison – Test Accuracy")
    ax.set_ylim(0, min(1.05, max(accs) + 0.08))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fpath = save_dir / "model_comparison_accuracy.png"
    fig.savefig(fpath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", fpath)


def plot_model_comparison_f1(results, save_dir):
    """Bar chart comparing weighted F1 score across all models."""
    names = [r["name"] for r in results]
    f1s   = [r["f1_weighted"] for r in results]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(names, f1s, color=sns.color_palette("magma", len(names)), edgecolor="black")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1 Score (weighted)")
    ax.set_title("Model Comparison – Weighted F1 Score")
    ax.set_ylim(0, min(1.05, max(f1s) + 0.08))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fpath = save_dir / "model_comparison_f1.png"
    fig.savefig(fpath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", fpath)


def plot_training_time_comparison(results, save_dir):
    """Horizontal bar chart comparing training times."""
    names = [r["name"] for r in results]
    times = [r["train_time"] for r in results]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.barh(names, times, color=sns.color_palette("coolwarm", len(names)), edgecolor="black")
    for bar, val in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=10)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Comparison – Training Time")
    plt.tight_layout()
    fpath = save_dir / "training_time_comparison.png"
    fig.savefig(fpath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", fpath)


def plot_cross_validation_comparison(results, save_dir):
    """Box plot of 5-fold cross-validation accuracy distributions."""
    data, labels = [], []
    for r in results:
        data.append(r["cv_scores"])
        labels.append(r["name"])

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)
    colors = sns.color_palette("Set2", len(labels))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Model Comparison – 5-Fold Cross-Validation Scores")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fpath = save_dir / "cross_validation_comparison.png"
    fig.savefig(fpath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", fpath)


def plot_roc_curves(results, y_test, class_names, save_dir):
    """
    One-vs-Rest ROC curves for every model that supports predict_proba.

    For readability with many classes, we plot the macro-average ROC per model.
    """
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    colors = sns.color_palette("tab10", len(results))

    plotted = False
    for idx, r in enumerate(results):
        y_proba = r.get("y_proba")
        if y_proba is None:
            continue

        # Compute macro-average ROC
        fpr_grid = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(fpr_grid)

        valid_classes = 0
        for c in range(n_classes):
            if y_test_bin.shape[1] <= c:
                continue
            fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, c], y_proba[:, c])
            mean_tpr += np.interp(fpr_grid, fpr_c, tpr_c)
            valid_classes += 1

        if valid_classes == 0:
            continue

        mean_tpr /= valid_classes
        macro_auc = auc(fpr_grid, mean_tpr)
        ax.plot(fpr_grid, mean_tpr, color=colors[idx], lw=2,
                label=f"{r['name']}  (macro AUC = {macro_auc:.4f})")
        plotted = True

    if plotted:
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Macro-Average ROC Curves (One-vs-Rest)")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        fpath = save_dir / "roc_curves.png"
        fig.savefig(fpath, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info("Saved → %s", fpath)
    else:
        logger.warning("No model supports predict_proba – skipping ROC plot.")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. RESULTS SUMMARY & EXPORT                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_comparison_csv(results, save_dir):
    """
    Compile all model metrics into a DataFrame and save to CSV.

    Parameters
    ----------
    results : list[dict]
        Evaluation results for each model.
    save_dir : Path
        Directory to save the CSV.

    Returns
    -------
    pd.DataFrame
        The comparison table.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r["name"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision (weighted)": round(r["precision_weighted"], 4),
            "Recall (weighted)": round(r["recall_weighted"], 4),
            "F1 (weighted)": round(r["f1_weighted"], 4),
            "F1 (macro)": round(r["f1_macro"], 4),
            "CV Mean Accuracy": round(r["cv_scores"].mean(), 4),
            "CV Std": round(r["cv_scores"].std(), 4),
            "Training Time (s)": round(r["train_time"], 2),
            "Best Params": str(r["best_params"]),
        })
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    save_dir.mkdir(parents=True, exist_ok=True)
    fpath = save_dir / "model_comparison.csv"
    df.to_csv(fpath, index=False)
    logger.info("Saved comparison CSV → %s", fpath)
    return df


def print_summary(comparison_df):
    """Print a formatted summary table and announce the best model."""
    logger.info("\n" + "=" * 90)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 90)
    print(comparison_df.to_string(index=False))
    best = comparison_df.iloc[0]
    logger.info("\n🏆  Best model: %s  (Accuracy=%.4f, F1=%.4f)",
                best["Model"], best["Accuracy"], best["F1 (weighted)"])
    logger.info("=" * 90)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. MODEL PERSISTENCE                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_model(model, name, save_dir):
    """
    Persist a trained model to disk via joblib.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn model.
    name : str
        Model name (used for the filename).
    save_dir : Path
        Target directory.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    fpath = save_dir / f"{name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, fpath)
    logger.info("Saved model → %s", fpath)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. META-ENSEMBLE BUILDERS  (Voting & Stacking)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_voting_classifier(trained_models, results):
    """
    Soft Voting Classifier combining the top-3 models by test accuracy.

    Parameters
    ----------
    trained_models : list[dict]
        Output dicts from ``train_model_with_tuning``.
    results : list[dict]
        Output dicts from ``evaluate_model`` (already sorted or unsorted).

    Returns
    -------
    tuple
        (name, voting_clf, param_grid={})
    """
    # Sort by accuracy descending and pick top 3
    ranked = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    top3_names = [r["name"] for r in ranked[:3]]
    logger.info("Voting Classifier – top-3 models: %s", top3_names)

    estimators = []
    for tm in trained_models:
        if tm["name"] in top3_names:
            estimators.append((tm["name"], tm["best_model"]))

    voting_clf = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return "Voting_Classifier", voting_clf, {}


def build_stacking_classifier(trained_models, results):
    """
    Stacking Classifier with Logistic Regression meta-learner using top-3 base models.

    Parameters
    ----------
    trained_models : list[dict]
        Output dicts from ``train_model_with_tuning``.
    results : list[dict]
        Output dicts from ``evaluate_model``.

    Returns
    -------
    tuple
        (name, stacking_clf, param_grid={})
    """
    ranked = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    top3_names = [r["name"] for r in ranked[:3]]
    logger.info("Stacking Classifier – base models: %s", top3_names)

    estimators = []
    for tm in trained_models:
        if tm["name"] in top3_names:
            estimators.append((tm["name"], tm["best_model"]))

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return "Stacking_Classifier", stacking_clf, {}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  9. MAIN PIPELINE                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    End-to-end pipeline:
        1.  Load / preprocess data
        2.  Train 4 individual models (with grid search)
        3.  Evaluate each on the test set
        4.  Train 2 primary ensemble models (Random Forest, Gradient Boosting)
        5.  Build & train meta-ensembles (Voting, Stacking) from top-3 models
        6.  Generate all comparison plots and confusion matrices
        7.  Save models, results CSV, and print summary
    """
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("LANGUAGE DETECTION – MODEL TRAINING & EVALUATION PIPELINE")
    logger.info("Started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # ── Create output directories ────────────────────────────────────────────
    for d in [MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, tfidf, le = get_data()
    class_names = list(le.classes_)

    # ── 2. Define individual + primary ensemble models ───────────────────────
    model_builders = [
        build_multinomial_nb,
        build_logistic_regression,
        build_svm,
        build_decision_tree,
        build_random_forest,
        build_gradient_boosting,
    ]

    trained_models = []   # list of dicts from train_model_with_tuning
    eval_results   = []   # list of dicts from evaluate_model

    for builder in model_builders:
        name, model, param_grid = builder()
        try:
            trained = train_model_with_tuning(name, model, param_grid, X_train, y_train)
            trained_models.append(trained)

            result = evaluate_model(trained, X_test, y_test, class_names)
            eval_results.append(result)

            # Save the trained model
            save_model(trained["best_model"], name, MODELS_DIR)

            # Confusion matrix plot for this model
            plot_confusion_matrix(result["confusion_matrix"], class_names, name, PLOTS_DIR)
        except Exception as exc:
            logger.error("Failed on model '%s': %s", name, exc, exc_info=True)

    # ── 3. Meta-ensemble models (Voting & Stacking) ─────────────────────────
    if len(eval_results) >= 3:
        # --- Voting Classifier ------------------------------------------------
        try:
            v_name, v_model, v_grid = build_voting_classifier(trained_models, eval_results)
            v_trained = train_model_with_tuning(v_name, v_model, v_grid, X_train, y_train)
            trained_models.append(v_trained)
            v_result = evaluate_model(v_trained, X_test, y_test, class_names)
            eval_results.append(v_result)
            save_model(v_trained["best_model"], v_name, MODELS_DIR)
            plot_confusion_matrix(v_result["confusion_matrix"], class_names, v_name, PLOTS_DIR)
        except Exception as exc:
            logger.error("Voting Classifier failed: %s", exc, exc_info=True)

        # --- Stacking Classifier -----------------------------------------------
        try:
            s_name, s_model, s_grid = build_stacking_classifier(trained_models, eval_results)
            s_trained = train_model_with_tuning(s_name, s_model, s_grid, X_train, y_train)
            trained_models.append(s_trained)
            s_result = evaluate_model(s_trained, X_test, y_test, class_names)
            eval_results.append(s_result)
            save_model(s_trained["best_model"], s_name, MODELS_DIR)
            plot_confusion_matrix(s_result["confusion_matrix"], class_names, s_name, PLOTS_DIR)
        except Exception as exc:
            logger.error("Stacking Classifier failed: %s", exc, exc_info=True)
    else:
        logger.warning(
            "Fewer than 3 models succeeded – skipping Voting & Stacking ensembles."
        )

    # ── 4. Comparison Visualisations ─────────────────────────────────────────
    if eval_results:
        plot_model_comparison_accuracy(eval_results, PLOTS_DIR)
        plot_model_comparison_f1(eval_results, PLOTS_DIR)
        plot_training_time_comparison(eval_results, PLOTS_DIR)
        plot_cross_validation_comparison(eval_results, PLOTS_DIR)
        plot_roc_curves(eval_results, y_test, class_names, PLOTS_DIR)

    # ── 5. Save the best model as best_model.joblib (for deployment) ────────
    if eval_results:
        best_result = max(eval_results, key=lambda r: r["accuracy"])
        best_trained = None
        for tm in trained_models:
            if tm["name"] == best_result["name"]:
                best_trained = tm
                break
        if best_trained is not None:
            best_model_path = MODELS_DIR / "best_model.joblib"
            joblib.dump(best_trained["best_model"], best_model_path)
            logger.info("Best model '%s' saved to %s", best_result["name"], best_model_path)
            # Also save the TF-IDF vectorizer used for deployment
            tfidf_deploy_path = MODELS_DIR / "tfidf_vectorizer.joblib"
            joblib.dump(tfidf, tfidf_deploy_path)
            logger.info("TF-IDF vectorizer saved to %s", tfidf_deploy_path)
            # Save label encoder for deployment predictions
            le_deploy_path = MODELS_DIR / "label_encoder.joblib"
            joblib.dump(le, le_deploy_path)
            logger.info("Label encoder saved to %s", le_deploy_path)

    # ── 6. Results CSV & Summary ─────────────────────────────────────────────
    if eval_results:
        comparison_df = save_comparison_csv(eval_results, RESULTS_DIR)
        print_summary(comparison_df)

    total_time = time.time() - pipeline_start
    logger.info("\n✅  Pipeline completed in %.1f seconds.", total_time)
    logger.info("    Models  → %s", MODELS_DIR)
    logger.info("    Plots   → %s", PLOTS_DIR)
    logger.info("    Results → %s", RESULTS_DIR / "model_comparison.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
