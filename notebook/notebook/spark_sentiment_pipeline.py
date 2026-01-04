# spark_sentiment_pipeline.py - 修复版
import os
import sys

# ====== 1. 应用Windows修复补丁 ======
# 导入修复补丁
import spark_windows_patch  # 确保这个文件在相同目录

print("="*60)
print("社交媒体情感分析 - Spark ML 全流程实现 (修复版)")
print("="*60)

# ====== 2. 延迟导入Spark相关模块 ======
# 先导入基础库
import pandas as pd
import jieba

# ====== 3. 初始化SparkSession ======
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, col, regexp_replace, when
    from pyspark.sql.types import ArrayType, StringType, FloatType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    
    print("✅ PySpark模块导入成功")
except ImportError as e:
    print(f"❌ PySpark导入失败: {e}")
    print("请运行: pip install pyspark")
    sys.exit(1)

# ====== 4. 创建SparkSession ======
def create_spark_session():
    """创建并配置SparkSession"""
    try:
        spark = SparkSession.builder \
            .appName("Weibo_Sentiment_Analysis_Spark") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.cores", "2") \
            .config("spark.python.worker.timeout", "300") \
            .config("spark.python.worker.reuse", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "50") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.driver.port", "9999") \
            .master("local[*]") \
            .getOrCreate()
        
        print("✅ Spark会话创建成功！")
        print(f"Spark版本: {spark.version}")
        return spark
    except Exception as e:
        print(f"❌ Spark会话创建失败: {e}")
        return None

# ====== 5. 主程序逻辑 ======
def main():
    # 创建Spark会话
    spark = create_spark_session()
    if spark is None:
        return
    
    try:
        # ====== 加载数据 ======
        data_path = "./data/raw_data_backup.csv"
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            # 创建示例数据
            print("创建示例数据...")
            sample_data = pd.DataFrame({
                'review': [
                    '这部电影真的很好看！演员演技在线，剧情也很吸引人！',
                    '太失望了，浪费时间和金钱，完全不值得看。',
                    '中规中矩吧，没什么特别出彩的地方。',
                    '强烈推荐！这是我今年看过最好的电影！',
                    '非常糟糕的体验，导演到底在想什么？'
                ],
                'label': [1, 0, 0, 1, 0]
            })
            sample_data.to_csv(data_path, index=False, encoding='utf-8-sig')
            print(f"✅ 已创建示例数据: {data_path}")
        
        # 读取数据
        print(f"📊 正在读取数据: {data_path}")
        df_raw = spark.read.csv(data_path, header=True, inferSchema=True, encoding="utf-8")
        print(f"原始数据行数：{df_raw.count()}")
        df_raw.show(5, truncate=50)
        
        # ====== 数据清洗 ======
        print("\n🔧 开始数据清洗...")
        df_cleaned = df_raw.withColumn(
            "cleaned_review",
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        col("review"),
                        r'https?://\S+', ''  # 去除URL
                    ),
                    r'@[\w\u4e00-\u9fa5]+', ''  # 去除@用户
                ),
                r'\[.*?\]', ''  # 去除[表情]
            )
        )
        
        df_cleaned = df_cleaned.withColumn(
            "cleaned_review",
            regexp_replace(col("cleaned_review"), r'[^\w\u4e00-\u9fa5，。！？、；：\"\'\s]', '')
        ).withColumn(
            "cleaned_review",
            regexp_replace(col("cleaned_review"), r'\s+', ' ')
        )
        
        print("清洗后样本示例：")
        df_cleaned.select("review", "cleaned_review").show(3, truncate=50)
        
        # ====== 中文分词 ======
        print("\n🔪 开始中文分词...")
        
        # 定义普通UDF（更稳定）
        from pyspark.sql.functions import udf
        from pyspark.sql.types import ArrayType, StringType
        
        def jieba_segment(text):
            """使用jieba进行中文分词"""
            if not text or not isinstance(text, str):
                return []
            try:
                return list(jieba.cut(text.strip()))
            except Exception:
                return []
        
        # 注册UDF
        segment_udf = udf(jieba_segment, ArrayType(StringType()))
        
        # 应用UDF
        df_segmented = df_cleaned.withColumn("words", segment_udf(col("cleaned_review")))
        
        # 过滤空分词结果
        from pyspark.sql.functions import size
        df_segmented = df_segmented.filter(size(col("words")) > 0)
        
        print(f"分词后有效数据行数：{df_segmented.count()}")
        df_segmented.select("cleaned_review", "words").show(3, truncate=False)
        
        # ====== 加载停用词 ======
        stopwords_path = "./data/cn_stopwords.txt"
        stop_words_list = []
        
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stop_words_list = [line.strip() for line in f if line.strip()]
            print(f"📖 已从文件加载 {len(stop_words_list)} 个停用词")
        else:
            # 基础停用词表
            stop_words_list = ["的", "了", "在", "是", "我", "有", "和", "就", 
                             "不", "人", "都", "也", "而", "及", "与", "着", 
                             "或", "个", "没有", "这", "那", "就", "也"]
            print(f"⚠️  未找到外部停用词文件，使用内置 {len(stop_words_list)} 个停用词")
        
        # ====== 特征工程 ======
        print("\n⚙️  开始特征工程（TF-IDF）...")
        
        # 移除停用词
        stopwords_remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words",
            stopWords=stop_words_list
        )
        
        # TF-IDF
        hashing_tf = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=1000  # 降低特征维度以提高速度
        )
        
        idf = IDF(
            inputCol="raw_features",
            outputCol="features",
            minDocFreq=1
        )
        
        # ====== 构建并训练模型 ======
        print("\n🤖 构建机器学习Pipeline并训练...")
        
        # 逻辑回归分类器
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=50,  # 减少迭代次数
            regParam=0.1,
            elasticNetParam=0
        )
        
        # 构建Pipeline
        pipeline = Pipeline(stages=[
            stopwords_remover,
            hashing_tf,
            idf,
            lr
        ])
        
        # 划分数据集
        train_df, test_df = df_segmented.randomSplit([0.7, 0.3], seed=42)
        print(f"训练集样本数: {train_df.count()}")
        print(f"测试集样本数: {test_df.count()}")
        
        # 训练模型
        print("开始训练模型...")
        pipeline_model = pipeline.fit(train_df)
        print("✅ 模型训练完成！")
        
        # ====== 模型评估 ======
        print("\n📈 模型评估...")
        predictions = pipeline_model.transform(test_df)
        
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        
        evaluator_accuracy = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        f1_score = evaluator_f1.evaluate(predictions)
        accuracy = evaluator_accuracy.evaluate(predictions)
        
        print(f"测试集 F1 分数: {f1_score:.4f}")
        print(f"测试集 准确率: {accuracy:.4f}")
        
        print("\n预测结果示例:")
        predictions.select("label", "prediction", "cleaned_review").show(10, truncate=30)
        
        # ====== 保存模型 ======
        print("\n💾 保存模型...")
        model_save_path = "./data/spark_models/weibo_sentiment_model"
        
        # 确保目录存在
        os.makedirs("./data/spark_models", exist_ok=True)
        
        # 保存模型
        pipeline_model.write().overwrite().save(model_save_path)
        print(f"✅ 模型已保存至: {model_save_path}")
        
        # 保存预测结果
        predictions_sample = predictions.select("label", "prediction", "cleaned_review").limit(50)
        output_csv_path = "./data/pandas_processed/spark_predictions_sample.csv"
        
        os.makedirs("./data/pandas_processed", exist_ok=True)
        predictions_sample.toPandas().to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 预测结果示例已保存至: {output_csv_path}")
        
        # ====== 创建预测函数 ======
        print("\n🔮 创建预测函数示例...")
        
        def predict_sentiment(text):
            """使用Spark模型预测单条文本情感"""
            # 创建测试数据
            test_data = spark.createDataFrame([(text,)], ["review"])
            
            # 应用相同的清洗步骤
            test_cleaned = test_data.withColumn(
                "cleaned_review",
                regexp_replace(
                    regexp_replace(
                        regexp_replace(col("review"), r'https?://\S+', ''),
                        r'@[\w\u4e00-\u9fa5]+', ''
                    ),
                    r'\[.*?\]', ''
                )
            )
            
            # 分词
            test_segmented = test_cleaned.withColumn("words", segment_udf(col("cleaned_review")))
            
            # 预测
            result = pipeline_model.transform(test_segmented)
            
            # 提取结果
            if result.count() > 0:
                pred = result.first()
                sentiment = "正面" if pred["prediction"] == 1 else "负面"
                
                # 获取概率
                probability_vector = pred["probability"]
                if probability_vector:
                    prob = float(probability_vector[1]) if pred["prediction"] == 1 else float(probability_vector[0])
                else:
                    prob = 0.5
                    
                return sentiment, prob
            return "未知", 0.0
        
        # 测试预测函数
        test_texts = [
            "这部电影真的太好看了，演员演技在线，剧情也很吸引人！",
            "太失望了，完全浪费时间",
            "一般般，没什么特别的"
        ]
        
        print("\n测试预测函数:")
        for text in test_texts:
            sentiment, prob = predict_sentiment(text)
            print(f"  文本: {text[:30]}...")
            print(f"  情感: {sentiment} (置信度: {prob:.2%})")
            print()
        
        print("\n" + "="*60)
        print("🎉 Spark ML 全流程实现完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 停止Spark会话
        if spark:
            spark.stop()
            print("\n🛑 Spark会话已停止。")

if __name__ == "__main__":
    main()