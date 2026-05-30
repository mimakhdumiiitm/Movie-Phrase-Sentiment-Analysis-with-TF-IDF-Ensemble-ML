# Movie Phrase Sentiment Analysis with TF-IDF & Ensemble ML

This repository contains my machine learning work on sentiment analysis using text data. This project was originally designed as **Kaggle Assignment 3** for the *Machine Learning Practice* course, part of the **IIT Madras BS Degree in Data Science and Applications**. 

> ⚠️ **Note:** The original competition was hosted on Kaggle but has since been set to private by the course authorities. I am archiving and uploading my complete solution here to showcase my approach, data exploration, and model evaluation pipeline.

---

## 📌 Project Overview
The objective of this project is to accurately analyze and classify the underlying sentiment expressed in movie phrases. The dataset presents text samples coupled with predefined metadata features to predict sentiment classes:
* `0`: Negative
* `1`: Neutral
* `2`: Positive

## 📊 Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA)
* Identified and categorized multi-modal data types (Numerical features vs. Text structures).
* Generated descriptive statistics to map class distributions and variance benchmarks.

### 2. Data Preprocessing & Cleaning
* **Missing Value Imputation:** Handled systematic `NaN` values across auxiliary numerical features by mapping them using strategic placeholder markers (`-1`) along with dynamic boolean flags (`_nan`).
* **Feature Transformation:** Applied natural log transformations (`log1p`) to skewed mathematical distributions to stabilize variance before scaling.
* **Text Normalization:** Cleaned phrase indices by removing formatting anomalies and expanding contracted words (e.g., *n't* ➔ *not*, *'re* ➔ *are*).
* **Outlier Mitigation:** Implemented an Interquartile Range (IQR) clipping routine (`clip_outliers`) to anchor extreme boundary weights.

### 3. Feature Extraction
* Leveraged **TF-IDF Vectorization** (`TfidfVectorizer`) to model text sequences into optimized n-gram matrices capturing essential token relevance.
* Merged engineered textual representations with normalized numerical anchors using sparse matrices (`hstack`).

### 4. Model Training & Ensemble Architectures
Explored a diverse suite of machine learning classifiers using `scikit-learn`, `xgboost`, and `lightgbm`, optimizing performance benchmarks across:
* Logistic Regression & SGD Classifiers
* Multinomial Naive Bayes (NB)
* Linear Support Vector Classifiers (SVC)
* Tree-Based Ensembles (Random Forests)
* Gradient Boosting (XGBoost & LightGBM)

## 📈 Key Visualizations Produced
The pipeline includes script setups to generate three descriptive exploratory graphs:
1. **Distribution of Sentiment Classes:** Identifies structural target imbalances.
2. **KDE Density Word Counts:** Maps length distribution of phrases across individual target sentiments.
3. **Log-Scaled Boxplots:** Visualizes out-of-boundary distributions for implicit metadata features.

---
*All implementations and code blocks can be accessed in full inside the attached `.ipynb` notebook file.*
