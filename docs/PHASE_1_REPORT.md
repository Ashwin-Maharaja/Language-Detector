# Phase 1 Academic Project Report: Multi-Script Language Detection

**Course/Evaluation:** Machine Learning Mini Project – Phase 1 Submission  
**Domain:** Natural Language Processing (NLP) & Multi-Class Text Classification  
**Target Languages:** English, Hindi (Devanagari), and Kannada (Kannada script)  
**Date:** October 2026  
**Status:** Completed & Validated  

---

## 1. Problem Statement & Objectives

### 1.1 Problem Statement
In multilingual societies such as India, digital applications (government citizen portals, customer support bots, educational platforms, and news aggregators) receive inputs written across diverse languages and scripts. Automatically determining the language of user-submitted text is the fundamental prerequisite of every downstream natural language processing (NLP) application. 

Failure or inaccuracy at this initial detection stage causes catastrophic failures in downstream machine translation, sentiment analysis, named entity recognition, and text-to-speech engines. Traditional dictionary-based lookups and static token matching break down when processing short texts, domain-specific terminology, conversational idioms, or multilingual inputs. 

This project solves this challenge by developing a robust, script-aware machine learning pipeline capable of accurately identifying text among three major Indian linguistic ecosystems representing distinct families and writing scripts:
1. **English** (Indo-European / Germanic family; Latin script: `U+0000 - U+007F`)
2. **Hindi** (Indo-European / Indo-Aryan family; Devanagari script: `U+0900 - U+097F`)
3. **Kannada** (Dravidian family; Kannada script: `U+0C80 - U+0CFF`)

### 1.2 Core Project Objectives
- **Data Engineering:** Construct and validate a curated, balanced dataset of 1,050 sentences spanning 8 distinct daily and technical domains across English, Hindi, and Kannada.
- **Exploratory Data Analysis (EDA):** Perform granular empirical profiling of character lengths, word counts, and script frequency distributions with high-resolution visual analytics.
- **Script-Preserving Preprocessing:** Architect an NLP pipeline that normalizes text (Unicode NFC), filters noise, and extracts sublinear character-level TF-IDF $n$-grams while strictly preserving non-ASCII Indic characters.
- **Model Shortlisting:** Technically and mathematically shortlist **8 machine learning algorithms** (exceeding the requirement of 6), including **4 ensemble architectures** (Random Forest, Gradient Boosting, Soft Voting Classifier, and Stacking Classifier).
- **Validation Protocol:** Design a Stratified 5-Fold Cross-Validation scheme to ensure generalizability and eliminate class bias.

---

## 2. Dataset Identification & Metadata Specification

### 2.1 Dataset Profile
The project dataset mirrors the schema of established international benchmarks (such as the Kaggle 17-Language Identification Dataset) tailored specifically to the tri-lingual focus:

| Metadata Attribute | Specification / Value |
| :--- | :--- |
| **Dataset File** | `language_dataset.csv` (located in `Phase-1/data/`) |
| **Total Sample Size ($N$)** | 1,050 records |
| **Class Distribution** | Exactly balanced: 350 English, 350 Hindi, 350 Kannada |
| **Class Proportions** | $33.33\%$ per language class |
| **File Format & Encoding**| CSV (Comma-Separated Values), UTF-8 Unicode |
| **Feature Columns** | `Text` (string) and `Language` (categorical target) |
| **Missing / Null Values** | 0 records (100% complete data) |
| **Duplicate Entries** | 0 records (verified unique across classes) |

### 2.2 Class Distribution & Unicode Range Mapping

| Language | Sample Count | Writing Script | Unicode Block Range | Example Token |
| :--- | :---: | :--- | :--- | :--- |
| **English** | 350 | Latin | `U+0000` to `U+007F` | `university`, `algorithms` |
| **Hindi** | 350 | Devanagari (हिंदी) | `U+0900` to `U+097F` | `वैज्ञानिकों`, `अनुसंधान` |
| **Kannada** | 350 | Kannada (ಕನ್ನಡ) | `U+0C80` to `U+0CFF` | `ಸಂಸ್ಕೃತಿ`, `ವಿದ್ಯಾರ್ಥಿಗಳು` |

### 2.3 Domain Taxonomy
The 1,050 sentences span 8 representative topical domains:
1. Daily Life & Household
2. Education & Academics
3. Technology & Computer Science
4. Sports & Games
5. News, Politics & Current Affairs
6. Food, Cuisine & Culture
7. Weather, Nature & Geography
8. Science & Medicine

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Statistical Profiling Summary

| Metric | English (Latin) | Hindi (Devanagari) | Kannada (Kannada) |
| :--- | :---: | :---: | :---: |
| **Sample Count** | 350 | 350 | 350 |
| **Mean Character Length** | **68.06** chars | **46.76** chars | **47.60** chars |
| **Character Std Dev** | 7.17 | 7.19 | 8.26 |
| **Min / Max Chars** | 52 / 87 chars | 29 / 64 chars | 29 / 70 chars |
| **Mean Word Count** | **10.5** words | **8.8** words | **6.2** words |
| **Word Count Std Dev** | 1.8 words | 1.6 words | 1.4 words |
| **Min / Max Words** | 6 / 15 words | 5 / 13 words | 3 / 11 words |

### 3.2 Key Linguistic Findings from EDA
1. **Agglutinative Morphology of Kannada:** Kannada exhibits the lowest mean word count (6.2 words) but maintains a high mean character length (47.6 chars). This is direct evidence of Dravidian agglutination, where grammatical suffixes attach directly to root stems, resulting in fewer words that are longer in length.
2. **Class Balance Verification:** All classes possess exactly 350 samples ($p = 0.333$), ensuring evaluation metrics (Accuracy, Macro-F1) will not suffer from majority-class distortion.
3. **Absence of Overlapping Scripts:** The Unicode code point ranges for Latin (`0x00-0x7F`), Devanagari (`0x0900-0x097F`), and Kannada (`0x0C80-0x0CFF`) are completely disjoint. This confirms that character $n$-gram representations will yield sharp linear boundaries.

### 3.3 Visual Artifacts Generated in Phase 1
All generated EDA visualizations are stored in `Phase-1/plots/`:
- `class_distribution.png`: Bar chart verifying uniform 350-sample balance.
- `text_length_distribution.png`: KDE and histogram plots illustrating sentence length distributions across the three languages.
- `word_count_distribution.png`: Box plot comparing word distributions.
- `character_frequency.png`: Top character n-gram frequencies across scripts.
- `correlation_heatmap.png`: Feature correlation matrix across text structural metrics.

---

## 4. Script-Aware Preprocessing Pipeline Architecture

### 4.1 Preprocessing Workflow
Standard English NLP packages (e.g. NLTK word tokenizers, WordNet lemmatizers, Porter Stemmers) strip or corrupt non-ASCII characters, which destroys Devanagari and Kannada text. We constructed a custom script-preserving pipeline:

```
[Raw Multilingual String]
          │
          ▼
[1. Unicode NFC Normalization]  (unicodedata.normalize('NFC', text))
          │
          ▼
[2. Regex Cleansing]            (Strips URLs, HTML markup, and isolated numeric digits)
          │
          ▼
[3. Script Preservation Filter] (Retains [a-zA-Z], U+0900-U+097F, U+0C80-U+0CFF, whitespace)
          │
          ▼
[4. Selective Lowercasing]      (Lowercases Latin without mutating Indic code points)
          │
          ▼
[5. Sublinear TF-IDF N-Grams]   (Character analyzer, n-gram range (2, 5), sublinear_tf=True)
          │
          ▼
[6. Stratified 80/20 Splitting] -> [Training: 840 instances | Test: 210 instances]
```

### 4.2 Why Sublinear Character N-Grams ($n \in [2, 5]$)?
- **Sub-word Script Signatures:** Character $n$-grams capture root morphological fragments (e.g., `ಕನ್ನಡ`, `ಹೊಂದಿದೆ` in Kannada or `भारत`, `अनुसंधान` in Hindi).
- **Out-of-Vocabulary Robustness:** Word-level tokenizers fail when encountering novel or misspelled words. Character $n$-grams match overlapping substrings even on unseen vocabulary.
- **Sublinear Term Frequency Scaling:** Applies $\text{TF}_{\text{sublinear}} = 1 + \log(\text{TF})$ for $\text{TF} > 0$, preventing repetitive characters from dominating feature weights.
- **Vocabulary Size:** Generates **32,485 features** across the corpus.

---

## 5. Model Shortlisting for Phase 2 Implementation

To satisfy and surpass the academic requirement of **at least 6 models with at least 2 ensemble models**, **8 distinct algorithms** across 6 learning paradigms were shortlisted:

| # | Shortlisted Algorithm | Architecture Type | Learning Paradigm | Primary Justification for Text Classification |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Multinomial Naive Bayes** | Individual Base Model | Generative Probabilistic | Canonical baseline for discrete text frequency data; high training speed. |
| **2** | **Logistic Regression** | Individual Base Model | Discriminative Linear Margin | Convex optimization; highly effective for linearly separable sparse matrices. |
| **3** | **Linear Support Vector Classifier** | Individual Base Model | Maximum-Margin Separation | Maximizes geometric margin; robust against curse of dimensionality in 32k features. |
| **4** | **Decision Tree Classifier (CART)** | Individual Base Model | Non-Parametric Partitioning | White-box interpretability; reveals critical single character decision boundaries. |
| **5** | **Random Forest Classifier** | Ensemble (Bagging) | Bootstrap Aggregation | Reduces variance and prevents overfitting via randomized feature subspace selection. |
| **6** | **Gradient Boosting Classifier** | Ensemble (Boosting) | Sequential Residual Reduction | Sequentially corrects residual classification errors on hard or short text utterances. |
| **7** | **Soft Voting Classifier** | Meta-Ensemble | Consensus Probability Averaging | Blends Bayesian, Margin, and Linear models to eliminate individual blindspots. |
| **8** | **Stacking Classifier** | Meta-Ensemble | Two-Tier Meta-Learning | Trains a meta-learner (Logistic Regression) to optimize blending weights dynamically. |

---

## 6. Phase 2 Evaluation & Cross-Validation Strategy

1. **Stratified 5-Fold Cross-Validation:** To prevent sampling bias, every fold maintains exactly $33.33\%$ English, $33.33\%$ Hindi, and $33.33\%$ Kannada.
2. **Hyperparameter Optimization:** GridSearchCV will tune regularization $C$, smoothing $\alpha$, and tree estimators $n\_estimators$.
3. **Comprehensive Metrics:**
   - Classification Accuracy
   - Macro-Averaged F1-Score ($F1_{\text{macro}}$)
   - Weighted Precision and Recall
   - Confusion Matrices for all 8 models
   - Multi-Class OvR ROC curves with AUC scores
   - Training latency (seconds) and inference latency (ms / query)

---

## 7. Phase 1 Deliverables Summary

All Phase 1 objective assets are organized in the `Phase-1/` directory:
- `data/language_dataset.csv`: 1,050 balanced samples.
- `data/metadata.json`: Machine-readable metadata specification.
- `scripts/create_dataset.py`: Reproducible synthetic data generation script.
- `scripts/eda_preprocessing.py`: Standalone EDA and preprocessing pipeline.
- `plots/`: 5 high-resolution EDA plots.
- `model_shortlisting.md`: Detailed mathematical and theoretical profiles for the 8 models.
- `run_phase1.py`: Automated one-click execution and audit script.
