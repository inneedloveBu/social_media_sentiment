# streamlit_app_en.py
import streamlit as st
import pandas as pd
import pickle
import jieba
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Configure matplotlib for consistent display
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analysis System",
    page_icon="😊",
    layout="wide"
)

# Sidebar - Project Information
with st.sidebar:
    st.header("ℹ️ Project Information")
    st.markdown("""
    **Project Overview**
    - Machine learning based Chinese social media text sentiment analysis
    - Using logistic regression model trained on Weibo dataset
    - F1 Score: ~0.70
    
    **Technology Stack**
    - Data Processing: Pandas, NumPy
    - NLP Tool: Jieba (Chinese word segmentation)
    - Feature Engineering: TF-IDF
    - Machine Learning: Scikit-learn
    - Visualization: Matplotlib, Seaborn
    - Deployment: Streamlit
    """)
    
    st.markdown("---")
    st.caption("Project Duration: 5 days")
    st.caption("Data Scale: 119,988 labeled Weibo posts")

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load trained model and vectorizer"""
    try:
        model_path = './data/pandas_processed/best_sentiment_model.pkl'
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['vectorizer']
    except FileNotFoundError:
        st.error("Model file not found. Please ensure you have run the model training script.")
        return None, None

model, vectorizer = load_model()

# Text cleaning function
def clean_weibo_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)  # Remove @mentions
    text = re.sub(r'\[.*?\]', '', text)  # Remove emoticons
    text = re.sub(r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '', text)  # Keep Chinese/English characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Word segmentation function
def jieba_cut(text):
    return ' '.join(jieba.cut(text))

# Application title
st.title("😊 Social Media Sentiment Analysis System")
st.markdown("---")

# Main interface divided into two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔍 Real-time Sentiment Analysis")
    
    # Text input area
    user_input = st.text_area(
        "Enter Chinese text to analyze:",
        height=150,
        placeholder="e.g., This movie is really great, the acting is superb, and the plot is captivating!",
        help="Supports Chinese content like Weibo, comments, short texts"
    )
    
    # Analysis button
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                # 1. Clean text
                cleaned_text = clean_weibo_text(user_input)
                
                # 2. Word segmentation
                segmented_text = jieba_cut(cleaned_text)
                
                # 3. Convert to TF-IDF features
                if vectorizer and model:
                    features = vectorizer.transform([segmented_text])
                    
                    # 4. Predict
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0]
                    
                    # 5. Display results
                    sentiment = "Positive 😊" if prediction == 1 else "Negative 😔"
                    confidence = probability[1] if prediction == 1 else probability[0]
                    
                    # Result display cards
                    st.markdown("### Analysis Results")
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric("Sentiment", sentiment)
                    
                    with result_col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Probability visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sentiments = ['Negative', 'Positive']
                    colors = ['#FF6B6B', '#4ECDC4']
                    bars = ax.bar(sentiments, probability, color=colors, alpha=0.8)
                    
                    # Add value labels
                    for bar, prob in zip(bars, probability):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.2%}', ha='center', va='bottom')
                    
                    ax.set_ylabel('Probability')
                    ax.set_ylim([0, 1.1])
                    ax.set_title('Sentiment Prediction Probability Distribution')
                    st.pyplot(fig)
                    
                    # Display processed text
                    with st.expander("View Text Processing Details"):
                        st.write("**Original Text:**", user_input)
                        st.write("**Cleaned Text:**", cleaned_text)
                        st.write("**Segmented Text:**", segmented_text)
                else:
                    st.error("Model failed to load, cannot perform analysis.")
        else:
            st.warning("Please enter text to analyze.")

with col2:
    st.header("📊 Project Overview & Performance")
    
    # Display model performance
    st.subheader("Model Performance Metrics")
    
    # Create metric cards
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("F1 Score", "0.70")
    
    with metric_col2:
        st.metric("Accuracy", "0.70")
    
    with metric_col3:
        st.metric("Data Size", "119,988")
    
    # Try to load and display generated charts
    st.subheader("Model Comparison Analysis")
    
    try:
        # Try to load previously generated chart
        chart_path = './data/pandas_processed/advanced_model_comparison.png'
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Different ML Model Performance Comparison")
        else:
            # If chart doesn't exist, create a simple example
            st.info("Generating performance chart...")
            fig, ax = plt.subplots(figsize=(10, 6))
            models = ['Logistic Regression', 'Random Forest', 'XGBoost']
            f1_scores = [0.70, 0.67, 0.61]
            colors = ['#4ECDC4', '#FF6B6B', '#FFE66D']
            
            bars = ax.bar(models, f1_scores, color=colors, alpha=0.8)
            ax.set_ylabel('F1 Score')
            ax.set_ylim([0.5, 0.75])
            ax.set_title('Model Performance Comparison (F1 Score)')
            
            # Add value labels
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{score:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Chart loading failed: {str(e)}")
    
    # Project highlights
    st.subheader("✨ Project Highlights")
    st.markdown("""
    - ✅ **Complete Pipeline**: Full implementation from data cleaning to model deployment
    - ✅ **Chinese Optimization**: Specialized cleaning and segmentation for Weibo text
    - ✅ **In-depth Analysis**: Comparison of multiple models with key insights
    - ✅ **Interactive Display**: Real-time sentiment analysis with intuitive results
    - ✅ **Technical Exploration**: Attempted Spark big data processing (environment issues documented)
    """)

# Bottom project summary
st.markdown("---")
st.header("📋 Technical Summary & Future Work")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.subheader("Technical Implementation")
    st.markdown("""
    1. **Data Acquisition**: WeiboSenti100K public dataset
    2. **Data Preprocessing**: Text cleaning, Chinese word segmentation, stop word filtering
    3. **Feature Engineering**: TF-IDF text vectorization (2000 features)
    4. **Model Training**: Comparison of Logistic Regression, Random Forest, XGBoost
    5. **Model Tuning**: Hyperparameter optimization via grid search
    6. **Evaluation**: Multi-dimensional evaluation with accuracy, F1 score, etc.
    """)

with summary_col2:
    st.subheader("Future Optimization Directions")
    st.markdown("""
    1. **Feature Optimization**: Try word embeddings, BERT, and deep learning features
    2. **Model Upgrade**: Use Transformer models to improve accuracy
    3. **Data Expansion**: Crawl real-time Weibo data for diversity
    4. **Deployment Optimization**: Docker containerization and cloud server deployment
    5. **Feature Expansion**: Sentiment reason analysis, topic mining, etc.
    """)

# Running instructions
with st.expander("🏃 How to Run This Application"):
    st.markdown("""
    **Install Dependencies:**
    ```bash
    pip install streamlit pandas scikit-learn jieba matplotlib seaborn
    ```
    
    **Run Application:**
    ```bash
    streamlit run streamlit_app_en.py
    ```
    
    **Ensure these files exist:**
    - `./data/pandas_processed/best_sentiment_model.pkl` (Trained model)
    - `./data/pandas_processed/advanced_model_comparison.png` (Performance comparison chart, optional)
    """)

st.markdown("---")
st.caption("© 2024 Social Media Sentiment Analysis Project | Python Data Science Portfolio")