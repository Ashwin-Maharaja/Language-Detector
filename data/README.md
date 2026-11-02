# Language Dataset - Metadata

## Source
Synthetically generated dataset simulating the structure of the popular
[Language Detection (17 Languages) Kaggle dataset](https://www.kaggle.com/datasets/basilb2s/language-detection),
filtered to three languages: **English**, **Hindi**, and **Kannada**.

## Dataset File
- **Filename**: `language_dataset.csv`
- **Encoding**: UTF-8
- **Format**: Comma-Separated Values (CSV)

## Schema

| Column   | Data Type | Description                                       |
|----------|-----------|---------------------------------------------------|
| `Text`   | `object`  | A sentence or phrase written in one of 3 languages |
| `Language`| `object` | Target label – "English", "Hindi", or "Kannada"    |

## Statistics

| Language | Count | Script      |
|----------|-------|-------------|
| English  | 350   | Latin       |
| Hindi    | 350   | Devanagari  |
| Kannada  | 350   | Kannada     |
| **Total**| **1050** |          |

## Topics Covered
Sentences span multiple domains:
- Daily Life & Family
- Education & Academics
- Technology & Science
- Sports & Entertainment
- News & Politics
- Food & Culture
- Weather & Travel
- General Knowledge

## Notes
- Hindi sentences use **Devanagari script** (हिंदी)
- Kannada sentences use **Kannada script** (ಕನ್ನಡ)
- English sentences use **Latin script**
- The dataset is shuffled with `random_state=42` for reproducibility
