# 社交媒体情感分析系统Social Media Sentiment Analysis with NLP Pipeline

[![bilibili](https://img.shields.io/badge/🎥-Video%20on%20Bilibili-red)](https://www.bilibili.com/video/BV1M9qwB1EBc/?share_source=copy_web&vd_source=56cdc7ef44ed1ee2c9b9515febf8e9ce&t=223)


[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/indeedlove/Social-Media-Sentiment-Analysis)
[![GitHub](https://img.shields.io/badge/📂-GitHub-black)](https://github.com/inneedloveBu/social_media_sentiment)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Web%20Framework-Streamlit-red)



An end-to-end Natural Language Processing (NLP) project for sentiment classification on social media text.

This project demonstrates:

Text preprocessing pipeline

Feature extraction techniques

Classical ML vs Deep Learning comparison

Model evaluation and error analysis

Reproducible NLP workflow

Project Overview

The objective is to classify social media posts into sentiment categories (e.g., positive / negative / neutral).

The project is structured as a complete NLP pipeline:

Text cleaning

Tokenization

Feature extraction

Model training

Hyperparameter tuning

Performance evaluation

Error analysis

The focus is on systematic modeling rather than single-model experimentation.

Dataset

The dataset consists of labeled social media posts.

Target Classes

Positive

Negative

(Optional) Neutral

Data Characteristics

Short informal text

Slang and abbreviations

Noisy punctuation

Imbalanced class distribution

NLP Preprocessing Pipeline
Text Cleaning

Lowercasing

URL removal

Username/mention removal

Emoji normalization

Punctuation filtering

Tokenization

Word-level tokenization

Optional stopword removal

Optional stemming / lemmatization

Vectorization Methods

Two feature extraction strategies were implemented:

1️⃣ TF-IDF Representation

N-gram range: (1,2)

Max features control

Sparse high-dimensional representation

2️⃣ Word Embedding-Based Representation

Pre-trained embeddings (if used)

Or trainable embedding layer (deep learning model)

Models Implemented
Classical Machine Learning Models

Logistic Regression

Linear SVM

Naive Bayes

Random Forest

These models operate on TF-IDF features.

Deep Learning Models

LSTM-based classifier

(Optional) CNN for text classification

Architecture example (LSTM):

Embedding → LSTM → Dropout → Fully Connected → Softmax

Training Strategy

Train / validation split (80/20)

Stratified sampling for class balance

Cross-validation (for classical models)

Early stopping (for deep learning)

Learning rate scheduling

Loss function:

Cross-Entropy Loss

Optimizer:

Adam

Evaluation Metrics

Because sentiment classification can be imbalanced, multiple metrics are used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC (binary case)

Primary metric:

Macro F1-score

Results

(请填入你的真实数值)

Example structure:

Classical Model (Best: Linear SVM)

Accuracy: XX%

Macro F1-score: XX

Precision: XX

Recall: XX

Deep Learning Model (LSTM)

Accuracy: XX%

Macro F1-score: XX

Validation loss convergence after XX epochs

Model Comparison

Key findings:

Linear models perform strongly with TF-IDF features.

Deep learning models require more data but capture sequential dependencies.

TF-IDF + Linear SVM provides strong baseline performance.

Class imbalance affects recall for minority class.

Error Analysis

Misclassified sarcastic sentences

Ambiguous sentiment expressions

Mixed sentiment posts

Slang and informal abbreviations

Example:

"Great, another Monday 🙃"
Model confusion due to sarcasm.

Technical Stack

Python 3.8+

scikit-learn

NLTK / spaCy

PyTorch (for deep models)

NumPy

Pandas

Matplotlib / Seaborn

Project Structure
social_media_sentiment/
├── data/
├── notebooks/
├── models/
├── train_ml.py
├── train_dl.py
├── utils.py
├── requirements.txt
└── README.md
How to Run
Classical ML
python train_ml.py
Deep Learning Model
python train_dl.py
Key Contributions

Designed full NLP preprocessing pipeline

Compared classical ML vs deep learning models

Implemented multi-metric evaluation

Conducted structured error analysis

Demonstrated understanding of text feature representations

Future Improvements

Transformer-based model (BERT)

Data augmentation for minority class

Hyperparameter optimization with Optuna

Deploy as web inference app

Add attention visualization
