# spark_feature_engineering.py - 修复版

# ！！！！！！ 最重要的第一步：先运行补丁 ！！！！！！
import spark_windows_patch  # 这行必须在所有PySpark导入之前！

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover
from pyspark.ml import Pipeline
import jieba
import re
# 在 spark = SparkSession.builder... 之前添加这些行
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
# 设置Spark和本地临时目录，使用绝对路径，避免中文和空格
os.environ['SPARK_LOCAL_DIRS'] = 'C:\\tmp\\spark'  # 或任何其他磁盘上的空文件夹
os.environ['TEMP'] = 'C:\\tmp'

# 确保这些目录真实存在，如果不存在，手动创建它们

# 设置 Hadoop 环境变量（指向你已配置的 winutils）
os.environ['HADOOP_HOME'] = 'C:\\hadoop'  # 请确认路径与你放置 winutils.exe 的目录一致

# 初始化 SparkSession (现在应该可以成功了)
spark = SparkSession.builder \
    .appName("Weibo_Sentiment_Analysis") \
    .config("spark.driver.memory", "2g") \
    .config("spark.python.worker.timeout", "60")\
    .config("spark.python.worker.reuse", "false") \
    .config("spark.executorEnv.PYTHONHASHSEED", "0") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

print("=" * 50)
print("Spark 会话创建成功！开始处理数据...")
print("=" * 50)

# --- 以下是你原来的数据清洗和特征工程代码，保持不变 ---
# 2. 加载数据
df_raw = spark.read.csv(
    "./data/raw_data_backup.csv",
    header=True,
    inferSchema=True,
    encoding="utf-8"
)
print(f"原始数据行数：{df_raw.count()}")
df_raw.show(5, truncate=False)

# 3. 定义并应用清洗UDF
def clean_weibo_text(text):
    if not text:
        return ""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

clean_text_udf = udf(clean_weibo_text, StringType())
df_cleaned = df_raw.withColumn("cleaned_review", clean_text_udf(col("review")))
print("\n清洗后的文本示例：")
df_cleaned.select("review", "cleaned_review").show(3, truncate=False)

# 4. 中文分词
def jieba_cut(text):
    return list(jieba.cut(text)) if text else []
seg_udf = udf(jieba_cut, ArrayType(StringType()))
df_segmented = df_cleaned.withColumn("words", seg_udf(col("cleaned_review")))
print("\n分词结果示例：")
df_segmented.select("cleaned_review", "words").show(3, truncate=False)

# 5. 去除停用词（读取外部停用词文件）
stopwords_file_path = "./data/cn_stopwords.txt"  # 请修改为你的实际文件路径
stop_words = []

try:
    # 尝试用utf-8编码读取，如果失败则尝试gbk
    with open(stopwords_file_path, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f if line.strip()]
except UnicodeDecodeError:
    try:
        with open(stopwords_file_path, 'r', encoding='gbk') as f:
            stop_words = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取停用词文件失败，将使用示例列表。错误: {e}")
        # 保留原有的示例列表作为备选
        stop_words = ["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "也"]

print(f"已加载 {len(stop_words)} 个停用词。")

# 接下来的 StopWordsRemover 部分保持不变
stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=stop_words)
df_no_stopwords = stop_words_remover.transform(df_segmented)

# 6. 特征提取：TF-IDF
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=2000)
idf = IDF(inputCol="raw_features", outputCol="features")
pipeline = Pipeline(stages=[hashing_tf, idf])
tfidf_model = pipeline.fit(df_no_stopwords)
df_features = tfidf_model.transform(df_no_stopwords)

print("\n特征工程完成！最终数据结构：")
df_features.select("label", "filtered_words", "features").show(3, truncate=False)
print(f"特征向量维度：{len(df_features.first()['features'])}")

# 7. 保存处理后的数据
output_path = "./data/weibo_processed_spark"
df_features.write.mode("overwrite").parquet(output_path)
print(f"\n>>> 处理后的数据已保存至：{output_path}")

# 8. 停止Spark
spark.stop()
print("Spark会话已停止。所有任务完成！")