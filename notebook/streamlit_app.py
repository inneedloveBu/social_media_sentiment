# streamlit_app.py
from pyspark.sql.functions import col
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
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import numpy as np

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
        "project_overview": "**项目概述**\n- 基于机器学习的中文社交媒体文本情感分析\n- 使用多种模型在微博数据集上训练\n- 最佳F1分数：约 0.703",
        "tech_stack": "**技术栈**\n- 数据处理：Pandas, NumPy, Spark\n- NLP工具：Jieba (中文分词)\n- 特征工程：TF-IDF\n- 机器学习：Scikit-learn, XGBoost, Spark ML\n- 可视化：Matplotlib, Seaborn\n- 部署展示：Streamlit",
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
        "project_overview": "**Project Overview**\n- Machine learning based Chinese social media text sentiment analysis\n- Using multiple models trained on Weibo dataset\n- Best F1 Score: ~0.703",
        "tech_stack": "**Technology Stack**\n- Data Processing: Pandas, NumPy, Spark\n- NLP Tool: Jieba (Chinese word segmentation)\n- Feature Engineering: TF-IDF\n  Machine Learning: Scikit-learn, XGBoost, Spark ML\n- Visualization: Matplotlib, Seaborn\n- Deployment: Streamlit",
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

# 文本处理函数
def clean_weibo_text(text):
    """清洗微博文本，去除特殊字符和噪声"""
    if not text:
        return ""
    
    # 去URL
    text = re.sub(r'https?://\S+', '', text)
    # 去@用户
    text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)
    # 去表情符号 [表情]
    text = re.sub(r'\[.*?\]', '', text)
    # 去特殊字符，保留中文、英文、数字和常见标点
    text = re.sub(r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '', text)
    # 去多余空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def jieba_cut(text):
    """使用jieba进行中文分词"""
    if not text:
        return ""
    # 分词并返回空格分隔的字符串
    return ' '.join(jieba.cut(text))

def load_stopwords(stopwords_path='./data/cn_stopwords.txt'):
    """加载停用词"""
    stopwords = set()
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    except FileNotFoundError:
        # 如果文件不存在，使用默认的停用词
        stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都"}
    return stopwords

def remove_stopwords(text, stopwords):
    """去除停用词"""
    if not text:
        return ""
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

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
        # 尝试加载调优后的模型
        try:
            model_path = './data/pandas_processed/tuned_best_model.pkl'
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data['model'], model_data['vectorizer']
        except:
            st.error(TEXTS[st.session_state.language]["model_not_found"])
            return None, None

model, vectorizer = load_model()

@st.cache_resource
def load_spark_model():
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import ArrayType, StringType
    import jieba
    import re
    
    spark = SparkSession.builder \
        .appName("StreamlitApp") \
        .master("local[*]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    # 加载模型
    spark_pipeline_model = PipelineModel.load("./spark_sentiment_model")
    
    # 定义清洗UDF
    def clean_text_for_spark(text):
        if not text:
            return ""
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    clean_udf = udf(clean_text_for_spark, StringType())
    
    # 定义分词UDF
    def jieba_cut_for_spark(text):
        if text and isinstance(text, str):
            return list(jieba.cut(text))
        return []
    
    segment_udf = udf(jieba_cut_for_spark, ArrayType(StringType()))
    
    # 将UDF注册到Spark会话
    spark.udf.register("clean_udf", clean_text_for_spark, StringType())
    spark.udf.register("segment_udf", jieba_cut_for_spark, ArrayType(StringType()))
    
    return spark, spark_pipeline_model, clean_udf, segment_udf

# 获取真实的模型性能数据
def get_real_model_performance():
    """从配置文件中获取真实的性能数据"""
    config_path = "./data/pandas_processed/model_performance_config.json"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            models_data = config['models']
            
            performance_data = {
                'models': list(models_data.keys()),
                'accuracy': [models_data[m]['accuracy'] for m in models_data],
                'f1_scores': [models_data[m]['f1_score'] for m in models_data],
                'descriptions': [models_data[m]['description'] for m in models_data]
            }
            
            return performance_data
            
        except Exception as e:
            st.warning(f"读取配置文件时出错: {e}")
    
    # 如果配置文件不存在，使用默认数据
    return {
        'models': ['逻辑回归', '随机森林', 'XGBoost', 'Spark ML', 'Pandas基础版'],
        'accuracy': [0.701, 0.672, 0.614, 0.685, 0.698],
        'f1_scores': [0.703, 0.674, 0.612, 0.682, 0.700],
        'descriptions': ['经过调优的逻辑回归', '100棵决策树', '梯度提升树', 'Spark分布式模型', '基础逻辑回归']
    }

# 获取当前语言的文本
t = TEXTS[st.session_state.language]

# 应用标题
st.title(t["title"])
st.markdown("---")

# 主界面分为两列
main_col1, main_col2 = st.columns([1, 1])

# 存储分析结果的session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

with main_col1:
    st.header(t["analysis_title"])
    
    # 文本输入区
    user_input = st.text_area(
        "请输入要分析的中文文本：" if st.session_state.language == 'zh' else "Enter Chinese text to analyze:",
        height=150,
        placeholder=t["input_placeholder"],
        help="支持微博、评论、短文本等中文内容" if st.session_state.language == 'zh' else "Supports Chinese content like Weibo, comments, short texts",
        key="user_input"
    )
    
    # 分析按钮
    analyze_clicked = st.button(t["analyze_button"], type="primary", use_container_width=True)
    
    if analyze_clicked:
        if user_input.strip():
            with st.spinner(t["analyzing"]):
                # 加载停用词
                stopwords = load_stopwords()
                
                # Scikit-learn模型预测
                skl_sentiment = None
                skl_confidence = None
                skl_probability = None
                cleaned_text = clean_weibo_text(user_input)
                segmented_text = jieba_cut(cleaned_text)
                filtered_text = remove_stopwords(segmented_text, stopwords)
                
                if vectorizer and model:
                    features = vectorizer.transform([segmented_text])
                    skl_prediction = model.predict(features)[0]
                    skl_probability = model.predict_proba(features)[0]
                    skl_sentiment = t["positive"] if skl_prediction == 1 else t["negative"]
                    skl_confidence = skl_probability[1] if skl_prediction == 1 else skl_probability[0]
                else:
                    st.error("Scikit-learn模型加载失败。")
                
                # Spark模型预测
                spark_sentiment = None
                spark_confidence = None
                spark_probability = None
                try:
                    spark_session, spark_pipeline_model, clean_udf, segment_udf = load_spark_model()
                    input_df = spark_session.createDataFrame([(user_input,)], ["text"])
                    input_df = input_df.withColumn("cleaned", clean_udf(col("text")))
                    input_df = input_df.withColumn("words", segment_udf(col("cleaned")))
                    spark_prediction_row = spark_pipeline_model.transform(input_df).collect()[0]
                    
                    # 获取Spark模型的预测概率
                    # Spark ML的probability是一个DenseVector
                    spark_probability_list = spark_prediction_row.probability
                    # 转换为numpy数组
                    spark_probability = np.array(spark_probability_list.toArray())
                    
                    spark_sentiment = "正面 😊" if spark_prediction_row.prediction == 1 else "负面 😔"
                    spark_confidence = spark_probability[1] if spark_prediction_row.prediction == 1 else spark_probability[0]
                    
                except Exception as e:
                    st.error(f"Spark模型预测失败: {e}")
                
                # 保存结果到session state
                st.session_state.analysis_results = {
                    'user_input': user_input,
                    'cleaned_text': cleaned_text,
                    'segmented_text': segmented_text,
                    'skl_sentiment': skl_sentiment,
                    'skl_confidence': skl_confidence,
                    'skl_probability': skl_probability,
                    'spark_sentiment': spark_sentiment,
                    'spark_confidence': spark_confidence,
                    'spark_probability': spark_probability
                }
        else:
            st.warning(t["no_text_warning"])

# 如果有分析结果，显示在两个列中
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # 在左栏显示实时分析结果
    with main_col1:
        st.markdown(f"### {t['result_title']}")
        
        # 使用两列并排显示两个模型的结果
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.subheader("🤖 Scikit-learn 模型")
            if results['skl_sentiment']:
                # 情感倾向卡片
                sentiment_color = "🟢" if "正面" in results['skl_sentiment'] else "🔴"
                st.metric(t["sentiment"], f"{sentiment_color} {results['skl_sentiment']}")
                
                # 置信度卡片
                confidence_color = "🟢" if results['skl_confidence'] > 0.7 else "🟡" if results['skl_confidence'] > 0.5 else "🔴"
                st.metric(t["confidence"], f"{confidence_color} {results['skl_confidence']:.2%}")
                
                # 预测概率
                if results['skl_probability'] is not None:
                    with st.expander("📊 详细概率", expanded=True):
                        negative_prob = results['skl_probability'][0]
                        positive_prob = results['skl_probability'][1]
                        
                        st.progress(positive_prob, text=f"正面概率: {positive_prob:.2%}")
                        st.progress(negative_prob, text=f"负面概率: {negative_prob:.2%}")
            else:
                st.error("Scikit-learn模型预测失败")
        
        with result_col2:
            st.subheader("🚀 Spark ML 模型")
            if results['spark_sentiment']:
                # 情感倾向卡片
                sentiment_color = "🟢" if "正面" in results['spark_sentiment'] else "🔴"
                st.metric("情感倾向", f"{sentiment_color} {results['spark_sentiment']}")
                
                # 置信度卡片
                confidence_color = "🟢" if results['spark_confidence'] > 0.7 else "🟡" if results['spark_confidence'] > 0.5 else "🔴"
                st.metric("置信度", f"{confidence_color} {results['spark_confidence']:.2%}")
                
                # 预测概率
                if results['spark_probability'] is not None:
                    with st.expander("📊 详细概率", expanded=True):
                        negative_prob = results['spark_probability'][0]
                        positive_prob = results['spark_probability'][1]
                        
                        st.progress(positive_prob, text=f"正面概率: {positive_prob:.2%}")
                        st.progress(negative_prob, text=f"负面概率: {negative_prob:.2%}")
            else:
                st.error("Spark ML模型预测失败")
        
        st.markdown("---")
        
        # 两个模型的概率分布对比图表
        if results['skl_probability'] is not None and results['spark_probability'] is not None:
            st.subheader("📊 双模型概率分布对比")
            
            # 创建对比图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Scikit-learn模型概率
            sentiments_skl = ['负面', '正面'] if st.session_state.language == 'zh' else ['Negative', 'Positive']
            colors_skl = ['#FF6B6B', '#4ECDC4']
            bars_skl = ax1.bar(sentiments_skl, results['skl_probability'], color=colors_skl, alpha=0.8)
            ax1.set_title('Scikit-learn 模型')
            ax1.set_ylabel('概率')
            ax1.set_ylim([0, 1.1])
            
            for bar, prob in zip(bars_skl, results['skl_probability']):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.2%}', ha='center', va='bottom', fontsize=10)
            
            # Spark ML模型概率
            sentiments_spark = ['负面', '正面'] if st.session_state.language == 'zh' else ['Negative', 'Positive']
            colors_spark = ['#FF6B6B', '#4ECDC4']
            bars_spark = ax2.bar(sentiments_spark, results['spark_probability'], color=colors_spark, alpha=0.8)
            ax2.set_title('Spark ML 模型')
            ax2.set_ylabel('概率')
            ax2.set_ylim([0, 1.1])
            
            for bar, prob in zip(bars_spark, results['spark_probability']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.2%}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 添加对比分析
            if results['skl_sentiment'] == results['spark_sentiment']:
                st.success("✅ 两个模型预测结果一致！")
            else:
                st.warning("⚠️ 两个模型预测结果不一致，建议结合上下文判断。")
        
        # 文本处理详情
        with st.expander(t["view_details"], expanded=False):
            st.write(t["original_text"])
            st.info(results['user_input'])
            
            st.write(t["cleaned_text"])
            st.success(results['cleaned_text'])
            
            st.write(t["segmented_text"])
            st.code(results['segmented_text'])

# 右栏：项目概览和性能对比
with main_col2:
    st.header(t["overview_title"])
    
    # 显示模型性能
    st.subheader(t["performance"])
    
    # 获取真实性能数据
    performance_data = get_real_model_performance()
    
    # 找到最佳F1分数和准确率
    best_f1_index = performance_data['f1_scores'].index(max(performance_data['f1_scores']))
    best_model = performance_data['models'][best_f1_index]
    best_f1 = max(performance_data['f1_scores'])
    
    best_acc_index = performance_data['accuracy'].index(max(performance_data['accuracy']))
    best_acc_model = performance_data['models'][best_acc_index]
    best_acc = max(performance_data['accuracy'])
    
    # 创建指标卡片
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("最佳F1分数", f"{best_f1:.3f}", f"{best_model}")
    
    with metric_col2:
        st.metric("最佳准确率", f"{best_acc:.3f}", f"{best_acc_model}")
    
    with metric_col3:
        st.metric(t["data_size"], "119,988", "微博数据")
    
    # 模型对比分析图表
    st.subheader(t["model_comparison"])
    
    try:
        # 重新配置中文字体
        configure_matplotlib()
        
        # 根据语言设置模型名称
        if st.session_state.language == 'zh':
            models = performance_data['models']
        else:
            # 英文名称映射
            model_mapping = {
                '逻辑回归': 'Logistic Regression',
                '随机森林': 'Random Forest',
                'XGBoost': 'XGBoost',
                'Spark ML': 'Spark ML',
                'Pandas基础版': 'Pandas Baseline'
            }
            models = [model_mapping.get(m, m) for m in performance_data['models']]
        
        # 准备数据
        accuracy_scores = performance_data['accuracy']
        f1_scores = performance_data['f1_scores']
        
        # 颜色设置
        colors = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#95E1D3', '#FF9A8B']
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准确率图表
        bars1 = ax1.bar(models, accuracy_scores, color=colors, alpha=0.8)
        ax1.set_ylabel('准确率' if st.session_state.language == 'zh' else 'Accuracy')
        ax1.set_ylim([0.5, 0.75])
        ax1.set_title('模型准确率对比' if st.session_state.language == 'zh' else 'Model Accuracy Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=15)
        
        # 添加数值标签
        for bar, acc in zip(bars1, accuracy_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # F1分数图表
        bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.8)
        ax2.set_ylabel('F1分数' if st.session_state.language == 'zh' else 'F1 Score')
        ax2.set_ylim([0.5, 0.75])
        ax2.set_title('模型F1分数对比' if st.session_state.language == 'zh' else 'Model F1 Score Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=15)
        
        # 添加数值标签
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 添加详细数据表格
        with st.expander("📋 查看详细性能数据", expanded=False):
            if st.session_state.language == 'zh':
                df_performance = pd.DataFrame({
                    '模型': performance_data['models'],
                    '准确率': [f"{acc:.4f}" for acc in accuracy_scores],
                    'F1分数': [f"{f1:.4f}" for f1 in f1_scores],
                    '排名': [sorted(f1_scores, reverse=True).index(f1) + 1 for f1 in f1_scores]
                })
            else:
                df_performance = pd.DataFrame({
                    'Model': models,
                    'Accuracy': [f"{acc:.4f}" for acc in accuracy_scores],
                    'F1 Score': [f"{f1:.4f}" for f1 in f1_scores],
                    'Rank': [sorted(f1_scores, reverse=True).index(f1) + 1 for f1 in f1_scores]
                })
            df_performance = df_performance.sort_values('排名' if st.session_state.language == 'zh' else 'Rank')
            st.dataframe(df_performance, use_container_width=True, hide_index=True)
        
        # 添加分析总结
        st.info(f"""
        **分析总结**: 
        - 在测试的{len(models)}个模型中，**{best_model}** 表现最佳
        - 最佳F1分数：**{best_f1:.3f}**，最佳准确率：**{best_acc:.3f}**
        - 逻辑回归模型在性能和训练速度上取得了最佳平衡
        - Spark ML模型展示了分布式计算在情感分析中的潜力
        """)
        
    except Exception as e:
        st.warning(f"{t['chart_failed']}: {str(e)}")

# 底部的项目总结
st.markdown("---")
st.header(t["summary_title"])

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.subheader(t["highlights_title"])
    st.markdown(t["highlights"])

with summary_col2:
    st.subheader(t["implementation"])
    st.markdown(t["implementation_details"])

with summary_col3:
    st.subheader(t["future_work"])
    st.markdown(t["future_details"])

# 运行说明
with st.expander(t["run_instructions"], expanded=False):
    st.markdown(f"""
    **{t['installation']}**
    ```bash
    pip install streamlit pandas scikit-learn jieba matplotlib seaborn pyspark xgboost
    ```
    
    **{t['run_command']}**
    ```bash
    streamlit run streamlit_app.py
    ```
    
    **{t['requirements']}**
    - `./data/pandas_processed/best_sentiment_model.pkl` (训练好的模型)
    - `./spark_sentiment_model` (Spark模型目录)
    - `./data/cn_stopwords.txt` (停用词文件，可选)
    """)

st.markdown("---")
if st.session_state.language == 'zh':
    st.caption("© 2025 社交媒体情感分析项目 | 基于Python的数据科学作品集")
else:
    st.caption("© 2025 Social Media Sentiment Analysis Project | Python Data Science Portfolio")