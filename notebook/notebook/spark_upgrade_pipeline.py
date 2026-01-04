# spark_upgrade_pipeline.py (修正版)

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, regexp_replace
from pyspark.sql.types import ArrayType, StringType
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import jieba
import time
# 在文件顶部添加导入（在已有的导入部分添加）
from pyspark.sql.functions import col

start_time = time.time()

# 1. 初始化SparkSession
spark = SparkSession.builder \
    .appName("Weibo_Sentiment_Spark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 2. 加载数据
df = spark.read.csv("/home/concentriccircler/projects/social_media_sentiment/windows_project/data/raw_data_backup.csv", header=True, inferSchema=True, encoding='utf-8')
print(f"原始数据量: {df.count()} 条")

# 3. 数据清洗
df_cleaned = df.withColumn(
    "cleaned_review",
    regexp_replace(
        regexp_replace(
            regexp_replace(col("review"), r'https?://\S+', ''),
            r'@[\w\u4e00-\u9fa5]+', ''
        ),
        r'\[.*?\]', ''
    )
)
df_cleaned = df_cleaned.withColumn("cleaned_review",
    regexp_replace(col("cleaned_review"), r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '')
)

# 4. 中文分词UDF
def jieba_cut(text):
    if text and isinstance(text, str):
        return list(jieba.cut(text))
    return []
segment_udf = udf(jieba_cut, ArrayType(StringType()))
df_words = df_cleaned.withColumn("words", segment_udf(col("cleaned_review")))
df_words = df_words.filter(F.size(col("words")) > 0)

# 5. 加载停用词并定义管道阶段


from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover

# 5. 定义所有Pipeline阶段（注意新增的两个阶段）
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col

# 5.1 文本清洗转换器（使用SQL函数）
# 我们需要一个自定义Transformer来包装清洗逻辑，这里简化处理，将清洗逻辑也写入UDF
# 但更规范的做法是继承Spark的Transformer类，为简化我们先采用一种变通方法：
# 直接在训练前将清洗列加入DataFrame，并确保Pipeline包含此逻辑。
# 为了清晰，我们定义一个“清洗”UDF，并注册为Spark SQL函数。

clean_udf = udf(lambda x: re.sub(r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '', x) if x else '', StringType())
spark.udf.register("clean_udf", clean_udf)

# 5.2 定义各阶段（注意：现在我们假设输入DataFrame已有“review”列）
# 阶段1: 从“review”列生成“cleaned_review”列
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark import keyword_only
import re





# 5.1 移除停用词（可以加载你的cn_stopwords.txt）
stopwords_path = "./cn_stopwords.txt"  # 或 windows_project/data/cn_stopwords.txt
stop_words = []
try:
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f if line.strip()]
    print(f"已从文件加载 {len(stop_words)} 个停用词")
except FileNotFoundError:
    print("未找到停用词文件，使用基础停用词列表")
    stop_words = ["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都"]  # 备用的基础列表
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=stop_words)

df_filtered = remover.transform(df_words)


hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=2000)
idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=2)
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)

# 6. 构建并训练完整的单一Pipeline
full_pipeline = Pipeline(stages=[remover, hashing_tf, idf, lr])

# 划分数据集
train_df, test_df = df_words.randomSplit([0.8, 0.2], seed=42)

# 训练
full_model = full_pipeline.fit(train_df)

# 预测与评估
predictions = full_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Spark模型准确率: {accuracy:.4f}")
predictions.select("label", "prediction", "probability", "cleaned_review").show(10, truncate=30)

# 7. 保存这个唯一的完整模型
model_save_path = "./spark_sentiment_model"
full_model.write().overwrite().save(model_save_path)
print(f"✅ 模型已成功保存至: {model_save_path}")

spark.stop()
print(f"总运行时间: {time.time() - start_time:.2f}秒")


