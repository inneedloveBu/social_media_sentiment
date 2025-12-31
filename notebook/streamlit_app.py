# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
import jieba
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys

# 导入中文字体配置
sys.path.append('.')
from matplotlib_config import configure_matplotlib

# 配置matplotlib中文字体
configure_matplotlib()

# 多语言文本配置
TEXTS = {
    "zh": {
        "title": "😊 社交媒体情感分析系统",
        "project_info": "ℹ️ 项目信息",
        "project_overview": "**项目概述**\n- 基于机器学习的中文社交媒体文本情感分析\n- 使用逻辑回归模型，在微博数据集上训练\n- F1分数：约 0.70",
        "tech_stack": "**技术栈**\n- 数据处理：Pandas, NumPy\n- NLP工具：Jieba (中文分词)\n- 特征工程：TF-IDF\n- 机器学习：Scikit-learn\n- 可视化：Matplotlib, Seaborn\n- 部署展示：Streamlit",
        "analysis_title": "🔍 实时情感分析",
        "input_placeholder": "例如：这部电影真的很棒，演员演技在线，剧情扣人心弦！",
        "analyze_button": "开始分析",
        "analyzing": "正在分析情感...",
        "result_title": "分析结果",
        "sentiment": "情感倾向",
        "confidence": "置信度",
        "probability_dist": "情感预测概率分布",
        "view_details": "查看文本处理过程",
        "original_text": "**原始文本：**",
        "cleaned_text": "**清洗后文本：**",
        "segmented_text": "**分词结果：**",
        "overview_title": "📊 项目概览与性能",
        "performance": "模型性能指标",
        "data_size": "数据量",
        "model_comparison": "模型对比分析",
        "highlights_title": "✨ 项目亮点",
        "highlights": "- ✅ **完整流程**：从数据清洗到模型部署的全流程实现\n- ✅ **中文优化**：针对微博文本的特殊清洗和分词处理\n- ✅ **深入分析**：对比多种模型，得出关键技术洞见\n- ✅ **交互展示**：实时情感分析，直观呈现结果\n- ✅ **技术探索**：尝试Spark大数据处理框架",
        "summary_title": "📋 技术总结与展望",
        "implementation": "技术实现",
        "implementation_details": "1. **数据获取**：使用WeiboSenti100K公开数据集\n2. **数据预处理**：文本清洗、中文分词、停用词过滤\n3. **特征工程**：TF-IDF文本向量化（2000维特征）\n4. **模型训练**：逻辑回归、随机森林、XGBoost对比\n5. **模型调优**：网格搜索优化超参数\n6. **结果评估**：准确率、F1分数等多维度评估",
        "future_work": "未来优化方向",
        "future_details": "1. **特征优化**：尝试词向量、BERT等深度学习特征\n2. **模型升级**：使用Transformer模型提升准确率\n3. **数据扩展**：爬取实时微博数据，增加数据多样性\n4. **部署优化**：使用Docker容器化，云服务器部署\n5. **功能扩展**：情感原因分析、主题挖掘等",
        "run_instructions": "🏃 如何运行此应用",
        "installation": "**安装依赖：**",
        "run_command": "**运行应用：**",
        "requirements": "**确保以下文件存在：**",
        "model_not_found": "未找到模型文件。请确保已运行模型训练脚本。",
        "no_text_warning": "请输入要分析的文本。",
        "positive": "正面 😊",
        "negative": "负面 😔",
        "chart_generating": "项目性能图表生成中...",
        "chart_failed": "图表加载失败"
    },
    "en": {
        "title": "😊 Social Media Sentiment Analysis System",
        "project_info": "ℹ️ Project Information",
        "project_overview": "**Project Overview**\n- Machine learning based Chinese social media text sentiment analysis\n- Using logistic regression model trained on Weibo dataset\n- F1 Score: ~0.70",
        "tech_stack": "**Technology Stack**\n- Data Processing: Pandas, NumPy\n- NLP Tool: Jieba (Chinese word segmentation)\n- Feature Engineering: TF-IDF\n- Machine Learning: Scikit-learn\n- Visualization: Matplotlib, Seaborn\n- Deployment: Streamlit",
        "analysis_title": "🔍 Real-time Sentiment Analysis",
        "input_placeholder": "e.g., This movie is really great, the acting is superb, and the plot is captivating!",
        "analyze_button": "Analyze Sentiment",
        "analyzing": "Analyzing sentiment...",
        "result_title": "Analysis Results",
        "sentiment": "Sentiment",
        "confidence": "Confidence",
        "probability_dist": "Sentiment Prediction Probability Distribution",
        "view_details": "View Text Processing Details",
        "original_text": "**Original Text:**",
        "cleaned_text": "**Cleaned Text:**",
        "segmented_text": "**Segmented Text:**",
        "overview_title": "📊 Project Overview & Performance",
        "performance": "Model Performance Metrics",
        "data_size": "Data Size",
        "model_comparison": "Model Comparison Analysis",
        "highlights_title": "✨ Project Highlights",
        "highlights": "- ✅ **Complete Pipeline**: Full implementation from data cleaning to model deployment\n- ✅ **Chinese Optimization**: Specialized cleaning and segmentation for Weibo text\n- ✅ **In-depth Analysis**: Comparison of multiple models with key insights\n- ✅ **Interactive Display**: Real-time sentiment analysis with intuitive results\n- ✅ **Technical Exploration**: Attempted Spark big data processing (environment issues documented)",
        "summary_title": "📋 Technical Summary & Future Work",
        "implementation": "Technical Implementation",
        "implementation_details": "1. **Data Acquisition**: WeiboSenti100K public dataset\n2. **Data Preprocessing**: Text cleaning, Chinese word segmentation, stop word filtering\n3. **Feature Engineering**: TF-IDF text vectorization (2000 features)\n4. **Model Training**: Comparison of Logistic Regression, Random Forest, XGBoost\n5. **Model Tuning**: Hyperparameter optimization via grid search\n6. **Evaluation**: Multi-dimensional evaluation with accuracy, F1 score, etc.",
        "future_work": "Future Optimization Directions",
        "future_details": "1. **Feature Optimization**: Try word embeddings, BERT, and deep learning features\n2. **Model Upgrade**: Use Transformer models to improve accuracy\n3. **Data Expansion**: Crawl real-time Weibo data for diversity\n4. **Deployment Optimization**: Docker containerization and cloud server deployment\n5. **Feature Expansion**: Sentiment reason analysis, topic mining, etc.",
        "run_instructions": "🏃 How to Run This Application",
        "installation": "**Install Dependencies:**",
        "run_command": "**Run Application:**",
        "requirements": "**Ensure these files exist:**",
        "model_not_found": "Model file not found. Please ensure you have run the model training script.",
        "no_text_warning": "Please enter text to analyze.",
        "positive": "Positive 😊",
        "negative": "Negative 😔",
        "chart_generating": "Generating performance chart...",
        "chart_failed": "Chart loading failed"
    }
}

# 页面设置
st.set_page_config(
    page_title="Social Media Sentiment Analysis System",
    page_icon="😊",
    layout="wide"
)

# 初始化语言选择
if 'language' not in st.session_state:
    st.session_state.language = 'zh'

# 侧边栏 - 语言选择
with st.sidebar:
    # 语言切换按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("中文", use_container_width=True):
            st.session_state.language = 'zh'
            st.rerun()
    with col2:
        if st.button("English", use_container_width=True):
            st.session_state.language = 'en'
            st.rerun()
    
    # 使用当前语言获取文本
    t = TEXTS[st.session_state.language]
    
    st.header(t["project_info"])
    st.markdown(t["project_overview"])
    st.markdown(t["tech_stack"])
    
    st.markdown("---")
    st.caption("Project Duration: 5 days")
    st.caption("Data Scale: 119,988 labeled Weibo posts")

# 加载模型和向量化器
@st.cache_resource
def load_model():
    """加载训练好的模型和向量化器"""
    try:
        model_path = './data/pandas_processed/best_sentiment_model.pkl'
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['vectorizer']
    except FileNotFoundError:
        st.error(TEXTS[st.session_state.language]["model_not_found"])
        return None, None

model, vectorizer = load_model()

# 文本清洗函数
def clean_weibo_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 分词函数
def jieba_cut(text):
    return ' '.join(jieba.cut(text))

# 获取当前语言的文本
t = TEXTS[st.session_state.language]

# 应用标题
st.title(t["title"])
st.markdown("---")

# 主界面分为两列
col1, col2 = st.columns([1, 1])

with col1:
    st.header(t["analysis_title"])
    
    # 文本输入区
    user_input = st.text_area(
        "请输入要分析的中文文本：" if st.session_state.language == 'zh' else "Enter Chinese text to analyze:",
        height=150,
        placeholder=t["input_placeholder"],
        help="支持微博、评论、短文本等中文内容" if st.session_state.language == 'zh' else "Supports Chinese content like Weibo, comments, short texts"
    )
    
    # 分析按钮
    if st.button(t["analyze_button"], type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner(t["analyzing"]):
                # 1. 清洗文本
                cleaned_text = clean_weibo_text(user_input)
                
                # 2. 分词
                segmented_text = jieba_cut(cleaned_text)
                
                # 3. 转换为TF-IDF特征
                if vectorizer and model:
                    features = vectorizer.transform([segmented_text])
                    
                    # 4. 预测
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0]
                    
                    # 5. 显示结果
                    sentiment = t["positive"] if prediction == 1 else t["negative"]
                    confidence = probability[1] if prediction == 1 else probability[0]
                    
                    # 结果展示卡片
                    st.markdown(f"### {t['result_title']}")
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric(t["sentiment"], sentiment)
                    
                    with result_col2:
                        st.metric(t["confidence"], f"{confidence:.2%}")
                    
                    # 概率可视化
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # 根据语言设置标签
                    if st.session_state.language == 'zh':
                        sentiments = ['负面', '正面']
                    else:
                        sentiments = ['Negative', 'Positive']
                        
                    colors = ['#FF6B6B', '#4ECDC4']
                    bars = ax.bar(sentiments, probability, color=colors, alpha=0.8)
                    
                    # 添加数值标签
                    for bar, prob in zip(bars, probability):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.2%}', ha='center', va='bottom')
                    
                    ax.set_ylabel('Probability' if st.session_state.language == 'en' else '概率')
                    ax.set_ylim([0, 1.1])
                    ax.set_title(t["probability_dist"])
                    st.pyplot(fig)
                    
                    # 显示处理后的文本
                    with st.expander(t["view_details"]):
                        st.write(t["original_text"], user_input)
                        st.write(t["cleaned_text"], cleaned_text)
                        st.write(t["segmented_text"], segmented_text)
                else:
                    st.error("Model failed to load, cannot perform analysis." if st.session_state.language == 'en' else "模型加载失败，无法进行分析。")
        else:
            st.warning(t["no_text_warning"])

with col2:
    st.header(t["overview_title"])
    
    # 显示模型性能
    st.subheader(t["performance"])
    
    # 创建指标卡片
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("F1 Score" if st.session_state.language == 'en' else "F1分数", "0.70")
    
    with metric_col2:
        st.metric("Accuracy" if st.session_state.language == 'en' else "准确率", "0.70")
    
    with metric_col3:
        st.metric(t["data_size"], "119,988")
    
    # 尝试加载并显示生成的图表
    st.subheader(t["model_comparison"])
    
    try:
        # 尝试加载之前生成的图表
        chart_path = './data/pandas_processed/advanced_model_comparison.png'
        if os.path.exists(chart_path):
            st.image(chart_path, 
                    caption="Different ML Model Performance Comparison" if st.session_state.language == 'en' 
                    else "不同机器学习模型性能对比")
        else:
            # 如果图表不存在，创建一个简单的示例图
            st.info(t["chart_generating"])
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 根据语言设置模型名称
            if st.session_state.language == 'zh':
                models = ['逻辑回归', '随机森林', 'XGBoost']
            else:
                models = ['Logistic Regression', 'Random Forest', 'XGBoost']
                
            f1_scores = [0.70, 0.67, 0.61]
            colors = ['#4ECDC4', '#FF6B6B', '#FFE66D']
            
            bars = ax.bar(models, f1_scores, color=colors, alpha=0.8)
            ax.set_ylabel('F1 Score' if st.session_state.language == 'en' else 'F1分数')
            ax.set_ylim([0.5, 0.75])
            
            title = 'Model Performance Comparison (F1 Score)' if st.session_state.language == 'en' else '模型性能对比 (F1分数)'
            ax.set_title(title)
            
            # 添加数值标签
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{score:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"{t['chart_failed']}: {str(e)}")
    
    # 项目亮点
    st.subheader(t["highlights_title"])
    st.markdown(t["highlights"])

# 底部的项目总结
st.markdown("---")
st.header(t["summary_title"])

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.subheader(t["implementation"])
    st.markdown(t["implementation_details"])

with summary_col2:
    st.subheader(t["future_work"])
    st.markdown(t["future_details"])

# 运行说明
with st.expander(t["run_instructions"]):
    st.markdown(f"""
    **{t['installation']}**
    ```bash
    pip install streamlit pandas scikit-learn jieba matplotlib seaborn
    ```
    
    **{t['run_command']}**
    ```bash
    streamlit run streamlit_app.py
    ```
    
    **{t['requirements']}**
    - `./data/pandas_processed/best_sentiment_model.pkl` (Trained model)
    - `./data/pandas_processed/advanced_model_comparison.png` (Performance comparison chart, optional)
    """)

st.markdown("---")
if st.session_state.language == 'zh':
    st.caption("© 2024 社交媒体情感分析项目 | 基于Python的数据科学作品集")
else:
    st.caption("© 2024 Social Media Sentiment Analysis Project | Python Data Science Portfolio")