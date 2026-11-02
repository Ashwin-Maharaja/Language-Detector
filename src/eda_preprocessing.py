"""
EDA and Preprocessing Pipeline for Language Detection
======================================================

This script performs comprehensive Exploratory Data Analysis (EDA) and builds
a reusable preprocessing pipeline for a tri-lingual (English, Hindi, Kannada)
language detection project.

Dataset: language_dataset.csv with columns "Text" and "Language"
Output:  Plots saved to ../plots/, preprocessed artifacts saved to ../models/

Usage:
    python src/eda_preprocessing.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import re
import sys
import logging
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – must be set before pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = DATA_DIR / "language_dataset.csv"

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib / Seaborn global style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.max_open_warning": 0})


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                     SECTION 1 – DATA LOADING                          ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the language dataset CSV and return a DataFrame.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with at least 'Text' and 'Language' columns.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the given path.
    """
    if not path.exists():
        logger.error(
            "Dataset not found at '%s'. Please place the CSV file there before running.",
            path,
        )
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Make sure 'language_dataset.csv' exists in the data/ directory."
        )

    logger.info("Loading dataset from '%s' ...", path)
    df = pd.read_csv(path)
    logger.info("Dataset loaded successfully – %d rows, %d columns.", *df.shape)
    return df


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                SECTION 2 – EXPLORATORY DATA ANALYSIS                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def basic_info(df: pd.DataFrame) -> None:
    """Print basic dataset information: shape, dtypes, head, tail."""
    logger.info("=" * 60)
    logger.info("BASIC DATASET INFO")
    logger.info("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    print(f"\nDescribe:\n{df.describe(include='all')}")


def check_missing_and_duplicates(df: pd.DataFrame) -> None:
    """Report missing values and duplicate rows."""
    logger.info("-" * 60)
    logger.info("MISSING VALUES & DUPLICATES")
    logger.info("-" * 60)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_report = pd.DataFrame({"Count": missing, "Percentage": missing_pct})
    print(f"\nMissing values:\n{missing_report}")

    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}  ({dup_count / len(df) * 100:.2f}%)")
    if dup_count > 0:
        logger.warning("%d duplicate rows detected.", dup_count)


def class_distribution(df: pd.DataFrame) -> None:
    """Display class distribution (counts and percentages)."""
    logger.info("-" * 60)
    logger.info("CLASS DISTRIBUTION")
    logger.info("-" * 60)

    counts = df["Language"].value_counts()
    percentages = df["Language"].value_counts(normalize=True) * 100
    dist = pd.DataFrame({"Count": counts, "Percentage (%)": percentages.round(2)})
    print(f"\n{dist}")


def text_length_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute text-length statistics per language.

    Adds 'text_length', 'word_count', and 'char_count' columns to *df*
    (in-place) and prints a summary table.

    Returns
    -------
    pd.DataFrame
        Summary statistics grouped by Language.
    """
    logger.info("-" * 60)
    logger.info("TEXT LENGTH & WORD COUNT ANALYSIS")
    logger.info("-" * 60)

    # Derived features
    df["text_length"] = df["Text"].astype(str).apply(len)
    df["word_count"] = df["Text"].astype(str).apply(lambda t: len(t.split()))
    df["char_count"] = df["Text"].astype(str).apply(lambda t: len(t.replace(" ", "")))

    # Per-language summary
    stats = (
        df.groupby("Language")[["text_length", "word_count", "char_count"]]
        .agg(["mean", "median", "min", "max", "std"])
        .round(2)
    )
    print(f"\nPer-language text statistics:\n{stats}")
    return stats


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart of language distribution saved to plots/class_distribution.png."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["Language"].value_counts()
    colors = sns.color_palette("viridis", n_colors=len(counts))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black")

    # Annotate bars with counts
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center", va="bottom", fontweight="bold",
        )

    ax.set_title("Language Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Language")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    save_path = PLOTS_DIR / "class_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_text_length_distribution(df: pd.DataFrame) -> None:
    """Histogram + box plot of text lengths per language."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    for lang in df["Language"].unique():
        subset = df[df["Language"] == lang]
        axes[0].hist(subset["text_length"], bins=40, alpha=0.6, label=lang)
    axes[0].set_title("Text Length Histogram")
    axes[0].set_xlabel("Text Length (characters)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Box plot
    sns.boxplot(data=df, x="Language", y="text_length", ax=axes[1], palette="viridis")
    axes[1].set_title("Text Length Box Plot")
    axes[1].set_ylabel("Text Length (characters)")

    plt.suptitle("Text Length Distribution by Language", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = PLOTS_DIR / "text_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_word_count_distribution(df: pd.DataFrame) -> None:
    """Box plot of word counts per language."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Language", y="word_count", palette="Set2", ax=ax)
    ax.set_title("Word Count Distribution by Language", fontsize=14, fontweight="bold")
    ax.set_xlabel("Language")
    ax.set_ylabel("Word Count")
    plt.tight_layout()
    save_path = PLOTS_DIR / "word_count_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_wordclouds(df: pd.DataFrame) -> None:
    """Generate and save word clouds for each language.

    Gracefully skips if the `wordcloud` package is not installed.
    """
    try:
        from wordcloud import WordCloud  # type: ignore
    except ImportError:
        logger.warning(
            "wordcloud package not installed – skipping word cloud generation. "
            "Install with: pip install wordcloud"
        )
        return

    languages = df["Language"].unique()
    for lang in languages:
        text_corpus = " ".join(df[df["Language"] == lang]["Text"].astype(str).tolist())
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=200,
            colormap="viridis",
            collocations=False,
        ).generate(text_corpus)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud – {lang}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Sanitise language name for filename (lowercase, underscores)
        fname = f"wordcloud_{lang.lower().replace(' ', '_')}.png"
        save_path = PLOTS_DIR / fname
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", save_path)


def plot_character_frequency(df: pd.DataFrame, top_n: int = 25) -> None:
    """Bar chart of top character frequencies for each language."""
    languages = df["Language"].unique()
    n_langs = len(languages)
    fig, axes = plt.subplots(1, n_langs, figsize=(7 * n_langs, 5), sharey=False)

    # Handle single-language edge case
    if n_langs == 1:
        axes = [axes]

    for ax, lang in zip(axes, languages):
        corpus = "".join(df[df["Language"] == lang]["Text"].astype(str).tolist())
        # Count characters (exclude whitespace)
        char_counts = Counter(c for c in corpus if not c.isspace())
        most_common = char_counts.most_common(top_n)
        chars, freqs = zip(*most_common) if most_common else ([], [])

        ax.barh(list(reversed(chars)), list(reversed(freqs)), color=sns.color_palette("coolwarm", top_n))
        ax.set_title(f"Top {top_n} Characters – {lang}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency")

    plt.suptitle("Character Frequency Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = PLOTS_DIR / "character_frequency.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of correlations between numerical features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        logger.info("Not enough numerical features for a correlation heatmap – skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = PLOTS_DIR / "correlation_heatmap.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full EDA pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Dataset enriched with text_length, word_count, and char_count columns.
    """
    logger.info("=" * 60)
    logger.info("STARTING EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    basic_info(df)
    check_missing_and_duplicates(df)
    class_distribution(df)
    text_length_analysis(df)  # adds derived columns in-place

    # --- Plots ---
    logger.info("Generating plots ...")
    plot_class_distribution(df)
    plot_text_length_distribution(df)
    plot_word_count_distribution(df)
    plot_wordclouds(df)
    plot_character_frequency(df)
    plot_correlation_heatmap(df)

    logger.info("EDA complete. All plots saved to '%s'.", PLOTS_DIR)
    return df


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║              SECTION 3 – TEXT PREPROCESSING PIPELINE                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝

# Precompiled regex patterns for performance
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_HTML = re.compile(r"<[^>]+>")
_RE_DIGITS = re.compile(r"\d+")
_RE_EXTRA_SPACES = re.compile(r"\s{2,}")

# Unicode ranges for scripts we want to PRESERVE
# Devanagari: U+0900–U+097F, U+A8E0–U+A8FF (extended)
# Kannada:    U+0C80–U+0CFF
_RE_SPECIAL_CHARS = re.compile(
    r"[^\w\s"                         # keep word chars & whitespace
    r"\u0900-\u097F"                  # Devanagari
    r"\uA8E0-\uA8FF"                 # Devanagari Extended
    r"\u0C80-\u0CFF"                 # Kannada
    r"]",
    re.UNICODE,
)


def remove_urls(text: str) -> str:
    """Strip HTTP/HTTPS URLs and bare www links."""
    return _RE_URL.sub("", text)


def remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags."""
    return _RE_HTML.sub("", text)


def remove_special_characters(text: str) -> str:
    """Remove special characters while preserving Devanagari & Kannada scripts."""
    return _RE_SPECIAL_CHARS.sub("", text)


def remove_digits(text: str) -> str:
    """Remove all numeric digits."""
    return _RE_DIGITS.sub("", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip."""
    return _RE_EXTRA_SPACES.sub(" ", text).strip()


def smart_lowercase(text: str) -> str:
    """Lowercase only Latin-script characters; leave Devanagari/Kannada unchanged.

    This avoids unintentional mutations on non-Latin Unicode codepoints.
    """
    result = []
    for ch in text:
        if ch.isascii() or unicodedata.category(ch).startswith("L"):
            # Only lowercase if the character is Latin
            if "LATIN" in unicodedata.name(ch, ""):
                result.append(ch.lower())
            else:
                result.append(ch)
        else:
            result.append(ch)
    return "".join(result)


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization for consistent representation."""
    return unicodedata.normalize("NFC", text)


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for a single text string.

    Steps (in order):
        1. Unicode normalization (NFC)
        2. Remove URLs
        3. Remove HTML tags
        4. Remove digits
        5. Remove special characters (preserve Devanagari & Kannada)
        6. Smart lowercase (Latin only)
        7. Collapse extra whitespace

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_digits(text)
    text = remove_special_characters(text)
    text = smart_lowercase(text)
    text = remove_extra_whitespace(text)
    return text


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║          SECTION 4 – FEATURE EXTRACTION & SERIALISATION               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def build_features_and_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Preprocess text, extract features, encode labels, and split data.

    Feature extractors created:
        - TF-IDF (word-level unigrams + bigrams, max 10 000 features)
        - TF-IDF (character-level 2–4-grams, max 30 000 features)
        - CountVectorizer (word-level unigrams, max 10 000 features)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'Text' and 'Language' columns.
    test_size : float
        Fraction of data reserved for testing (default 0.20).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing all artifacts (splits, vectorizers, encoder).
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING & FEATURE EXTRACTION")
    logger.info("=" * 60)

    # --- 4a. Clean text ---
    logger.info("Applying text preprocessing pipeline ...")
    df = df.copy()
    df["cleaned_text"] = df["Text"].astype(str).apply(preprocess_text)

    # Drop rows where cleaning left an empty string
    empty_mask = df["cleaned_text"].str.strip() == ""
    n_empty = empty_mask.sum()
    if n_empty > 0:
        logger.warning(
            "Dropping %d rows with empty text after preprocessing.", n_empty
        )
        df = df[~empty_mask].reset_index(drop=True)

    # --- 4b. Label encoding ---
    logger.info("Encoding target labels ...")
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Language"])
    logger.info("Label mapping: %s", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # --- 4c. Train / Test split ---
    logger.info("Splitting data (train=%.0f%%, test=%.0f%%) ...", (1 - test_size) * 100, test_size * 100)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    logger.info("Train size: %d | Test size: %d", len(X_train_raw), len(X_test_raw))

    # --- 4d. TF-IDF word-level ---
    logger.info("Fitting TF-IDF (word-level) vectorizer ...")
    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True,
    )
    X_train_tfidf_word = tfidf_word.fit_transform(X_train_raw)
    X_test_tfidf_word = tfidf_word.transform(X_test_raw)
    logger.info("TF-IDF word matrix shape: train=%s, test=%s", X_train_tfidf_word.shape, X_test_tfidf_word.shape)

    # --- 4e. TF-IDF character-level ---
    logger.info("Fitting TF-IDF (character-level) vectorizer ...")
    tfidf_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=30_000,
        sublinear_tf=True,
    )
    X_train_tfidf_char = tfidf_char.fit_transform(X_train_raw)
    X_test_tfidf_char = tfidf_char.transform(X_test_raw)
    logger.info("TF-IDF char matrix shape: train=%s, test=%s", X_train_tfidf_char.shape, X_test_tfidf_char.shape)

    # --- 4f. CountVectorizer (word-level) ---
    logger.info("Fitting CountVectorizer (word-level) ...")
    count_vec = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=10_000,
    )
    X_train_count = count_vec.fit_transform(X_train_raw)
    X_test_count = count_vec.transform(X_test_raw)
    logger.info("CountVec matrix shape: train=%s, test=%s", X_train_count.shape, X_test_count.shape)

    # --- Bundle everything ---
    artifacts = {
        # Raw splits
        "X_train_raw": X_train_raw.reset_index(drop=True),
        "X_test_raw": X_test_raw.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        # TF-IDF word
        "X_train_tfidf_word": X_train_tfidf_word,
        "X_test_tfidf_word": X_test_tfidf_word,
        "tfidf_word_vectorizer": tfidf_word,
        # TF-IDF char
        "X_train_tfidf_char": X_train_tfidf_char,
        "X_test_tfidf_char": X_test_tfidf_char,
        "tfidf_char_vectorizer": tfidf_char,
        # CountVectorizer
        "X_train_count": X_train_count,
        "X_test_count": X_test_count,
        "count_vectorizer": count_vec,
        # Encoder
        "label_encoder": label_encoder,
        # Metadata
        "preprocessed_df": df,
    }
    return artifacts


def save_artifacts(artifacts: dict) -> None:
    """Persist preprocessed data and fitted transformers to disk.

    Files are saved under ``models/`` using joblib for efficient
    serialisation of scipy sparse matrices and sklearn objects.

    Parameters
    ----------
    artifacts : dict
        Dictionary returned by :func:`build_features_and_split`.
    """
    logger.info("Saving artifacts to '%s' ...", MODELS_DIR)

    # Vectorizers & encoder
    joblib.dump(artifacts["tfidf_word_vectorizer"], MODELS_DIR / "tfidf_word_vectorizer.joblib")
    joblib.dump(artifacts["tfidf_char_vectorizer"], MODELS_DIR / "tfidf_char_vectorizer.joblib")
    joblib.dump(artifacts["count_vectorizer"], MODELS_DIR / "count_vectorizer.joblib")
    joblib.dump(artifacts["label_encoder"], MODELS_DIR / "label_encoder.joblib")

    # Train / test matrices
    joblib.dump(artifacts["X_train_tfidf_word"], MODELS_DIR / "X_train_tfidf_word.joblib")
    joblib.dump(artifacts["X_test_tfidf_word"], MODELS_DIR / "X_test_tfidf_word.joblib")
    joblib.dump(artifacts["X_train_tfidf_char"], MODELS_DIR / "X_train_tfidf_char.joblib")
    joblib.dump(artifacts["X_test_tfidf_char"], MODELS_DIR / "X_test_tfidf_char.joblib")
    joblib.dump(artifacts["X_train_count"], MODELS_DIR / "X_train_count.joblib")
    joblib.dump(artifacts["X_test_count"], MODELS_DIR / "X_test_count.joblib")

    # Labels
    joblib.dump(artifacts["y_train"], MODELS_DIR / "y_train.joblib")
    joblib.dump(artifacts["y_test"], MODELS_DIR / "y_test.joblib")

    # Raw text splits (useful for debugging / re-vectorising)
    joblib.dump(artifacts["X_train_raw"], MODELS_DIR / "X_train_raw.joblib")
    joblib.dump(artifacts["X_test_raw"], MODELS_DIR / "X_test_raw.joblib")

    # Full preprocessed DataFrame
    artifacts["preprocessed_df"].to_csv(MODELS_DIR / "preprocessed_dataset.csv", index=False)

    logger.info("All artifacts saved successfully.")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                        SECTION 5 – MAIN                              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """Entry point: load data → EDA → preprocess → save."""
    logger.info("=" * 60)
    logger.info("LANGUAGE DETECTION – EDA & PREPROCESSING")
    logger.info("=" * 60)

    # 1. Load
    df = load_dataset()

    # 2. EDA (adds text_length, word_count, char_count columns)
    df = run_eda(df)

    # 3. Preprocess, extract features, split
    artifacts = build_features_and_split(df)

    # 4. Save everything
    save_artifacts(artifacts)

    logger.info("=" * 60)
    logger.info("PIPELINE FINISHED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
