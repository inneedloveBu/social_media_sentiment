# pandas_feature_engineering.py
import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

print("开始使用Pandas进行特征工程...")

# 1. 加载数据
df = pd.read_csv('./data/raw_data_backup.csv', encoding='utf-8')
print(f"数据加载成功，形状: {df.shape}")
print(df.head())

# 2. 文本清洗函数 (与之前逻辑一致)
def clean_weibo_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("正在进行文本清洗...")
df['cleaned_review'] = df['review'].apply(clean_weibo_text)

# 3. 中文分词
print("正在进行中文分词...")
def jieba_cut(text):
    return ' '.join(jieba.cut(text))  # 用空格连接，以便后续TF-IDF
df['words'] = df['cleaned_review'].apply(jieba_cut)

# 4. 加载外部停用词
stopwords_path = './data/cn_stopwords.txt'
stop_words = []
if os.path.exists(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stop_words = [line.strip() for line in f if line.strip()]
    except:
        with open(stopwords_path, 'r', encoding='gbk') as f:
            stop_words = [line.strip() for line in f if line.strip()]
    print(f"已加载 {len(stop_words)} 个外部停用词。")
else:
    stop_words = ["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "也"]
    print("未找到外部停用词文件，使用示例停用词列表。")

# 5. TF-IDF特征提取
print("正在进行TF-IDF特征提取...")
# 初始化向量化器，传入停用词
vectorizer = TfidfVectorizer(max_features=2000, stop_words=stop_words)
# 学习词汇并转换文本为TF-IDF矩阵
X_tfidf = vectorizer.fit_transform(df['words'])
print(f"TF-IDF特征矩阵形状: {X_tfidf.shape}")  # 应为 (119988, 2000)

# 6. 准备标签
y = df['label']

# 7. 保存处理后的数据供后续建模使用
output_dir = './data/pandas_processed'
os.makedirs(output_dir, exist_ok=True)

# 保存特征矩阵和标签 (使用稀疏矩阵格式节省空间)
import scipy.sparse
scipy.sparse.save_npz(os.path.join(output_dir, 'X_tfidf_features.npz'), X_tfidf)
# 保存标签
y.to_pickle(os.path.join(output_dir, 'y_labels.pkl'))
# 保存向量化器，以便后续预测时使用相同的转换
with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)
# 保存一份带分词文本的DataFrame以便查看
df[['label', 'cleaned_review', 'words']].to_csv(os.path.join(output_dir, 'processed_data_sample.csv'), index=False, encoding='utf-8-sig')

print("="*50)
print("特征工程完成！")
print(f"1. TF-IDF特征矩阵已保存至: {output_dir}/X_tfidf_features.npz")
print(f"2. 标签已保存至: {output_dir}/y_labels.pkl")
print(f"3. 向量化器已保存至: {output_dir}/tfidf_vectorizer.pkl")
print(f"4. 样本数据已保存至: {output_dir}/processed_data_sample.csv")
print("="*50)
print("\n下一步可以直接进行模型训练！")