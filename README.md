# 社交媒体情感分析系统Social Media Sentiment Analysis System

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/indeedlove/Social-Media-Sentiment-Analysis)
[![GitHub](https://img.shields.io/badge/📂-View%20on%20GitHub-black)](https://github.com/inneedloveBu/social_media_sentiment)
[![bilibili](https://img.shields.io/badge/🎥-View%20on%20Bilibili-yellow)](https://www.bilibili.com/video/BV1M9qwB1EBc/?share_source=copy_web&vd_source=56cdc7ef44ed1ee2c9b9515febf8e9ce&t=223)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Web%20Framework-Streamlit-red)


## 🎯 Project Overview

A complete Chinese social media (Weibo) sentiment analysis system, covering the entire process from data acquisition, cleaning, feature engineering to model training, evaluation, and interactive demonstration. The project aims to showcase the application of Natural Language Processing (NLP) and Machine Learning (ML) techniques in real-world scenarios.

**Core Highlights:**
- Complete implementation of a data science workflow
- Optimized processing for Chinese social media text
- Comparison and in-depth analysis of multiple machine learning models
- Interactive web application for real-time demonstration
- Exploration of big data processing frameworks (Spark)

## 📊 Project Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Best F1 Score** | 0.70 | Performance of Logistic Regression model on test set |
| **Accuracy** | 0.70 | Overall classification accuracy of the model |
| **Data Size** | 119,988 | Balanced positive and negative sentiment Weibo posts |
| **Feature Dimension** | 2,000 | Size of TF-IDF feature vector |

## 🏗️ Project Structure

```
social_media_sentiment/
│
├── data/                      # Dataset
│   ├── raw/                   # Original data
│   └── processed/             # Cleaned data
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Data exploration
│   ├── 02_feature_engineering.ipynb # Feature engineering
│   └── 03_model_training.ipynb      # Model training & evaluation
│
├── src/                        # Source code
│   ├── data/                   # Data processing scripts
│   │   ├── clean.py
│   │   └── preprocess.py
│   ├── features/               # Feature engineering
│   │   └── build_features.py
│   ├── models/                  # Model training & prediction
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/           # Visualization
│       └── visualize.py
│
├── models/                      # Trained models
│   └── logistic_regression.pkl
│
├── app/                         # Web application
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔧 Technology Stack

| Category | Tools/Technologies |
|----------|---------------------|
| **Programming Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Chinese NLP** | Jieba (segmentation) |
| **Feature Engineering** | Scikit-learn (TF-IDF) |
| **Machine Learning** | Scikit-learn (LR, RF, XGBoost) |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Big Data Processing** | PySpark (exploratory attempt) |

## 📈 Core Findings & Insights

### 1. Model Performance Comparison
By comparing the performance of Logistic Regression, Random Forest, and XGBoost on the same features (TF-IDF), we found:

- **Logistic Regression performs best** (F1=0.70): For high-dimensional sparse text features, linear models have a better inductive bias
- **Tree models underperform**: Random Forest (F1=0.67) and XGBoost (F1=0.61) failed to outperform the simple linear model
- **Key Insight**: Under the "bag-of-words + TF-IDF" feature representation, linear models are near the performance ceiling. Breaking through requires more advanced feature representations (e.g., word embeddings, BERT)

### 2. Technical Challenges & Solutions
| Challenge | Solution | Learning Outcome |
|-----------|----------|------------------|
| High noise in Weibo text | Regex cleaning, removal of emojis/URLs/@mentions | Domain-specific data cleaning techniques |
| Specifics of Chinese word segmentation | Using Jieba tokenizer with stopwords list | Chinese NLP processing pipeline |
| High-dimensional sparse features | TF-IDF feature hashing (2000 dimensions) | Best practices in text feature engineering |
| Spark environment issues on Windows | Multiple attempts, finally using Pandas UDF | Differences between production and experimental environments |

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the project
git clone <your-repo-url>
cd social_media_sentiment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```


## 🎯 项目概述

一个完整的中文社交媒体（微博）情感分析系统，涵盖从数据获取、清洗、特征工程到模型训练、评估与交互式展示的全流程。项目旨在展示自然语言处理(NLP)与机器学习(ML)技术在真实场景中的应用。

**核心亮点：**
- 完整的数据科学工作流实现
- 针对中文社交媒体文本的优化处理
- 多种机器学习模型对比与深入分析
- 交互式Web应用实时演示
- 大数据处理框架(Spark)的技术探索

## 📊 项目性能

| 指标 | 分数 | 说明 |
|------|------|------|
| **最佳F1分数** | 0.70 | 逻辑回归模型在测试集上的表现 |
| **准确率** | 0.70 | 模型整体分类准确率 |
| **数据规模** | 119,988条 | 平衡的正负情感微博数据 |
| **特征维度** | 2,000维 | TF-IDF特征向量大小 |

## 🏗️ 项目结构



## 🔧 技术栈

| 类别 | 工具/技术 |
|------|-----------|
| **编程语言** | Python 3.x |
| **数据处理** | Pandas, NumPy |
| **中文NLP** | Jieba (分词) |
| **特征工程** | Scikit-learn (TF-IDF) |
| **机器学习** | Scikit-learn (LR, RF, XGBoost) |
| **可视化** | Matplotlib, Seaborn |
| **Web框架** | Streamlit |
| **大数据处理** | PySpark (探索性尝试) |

## 📈 核心发现与洞察

### 1. 模型性能对比
通过对比逻辑回归、随机森林和XGBoost在相同特征(TF-IDF)上的表现，我们发现：

- **逻辑回归表现最佳** (F1=0.70)：对于高维稀疏的文本特征，线性模型具有更好的归纳偏置
- **树模型表现欠佳**：随机森林(F1=0.67)和XGBoost(F1=0.61)未能超越简单线性模型
- **关键洞见**：在"词袋+TF-IDF"特征表示下，线性模型已接近性能天花板，要突破需要更高级的特征表示(如词向量、BERT)

### 2. 技术挑战与解决方案
| 挑战 | 解决方案 | 学习收获 |
|------|----------|----------|
| 微博文本噪声多 | 正则表达式清洗、表情/URL/@用户去除 | 领域特定的数据清洗技巧 |
| 中文分词特殊性 | 使用Jieba分词，加载停用词表 | 中文NLP处理流程 |
| 特征维度高且稀疏 | TF-IDF特征哈希(2000维) | 文本特征工程的最佳实践 |
| Windows Spark环境问题 | 多方案尝试，最终采用Pandas UDF | 生产环境与实验环境的差异 |

## 🚀 快速开始

### 1. 环境配置
```bash
# 克隆项目
git clone <your-repo-url>
cd social_media_sentiment

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
