# spark_sentiment_simple.py
import os
import sys

print("="*60)
print("社交媒体情感分析 - Spark ML 简化版（避免UDF问题）")
print("="*60)

# ====== 1. 设置环境变量 ======
os.environ['HADOOP_HOME'] = 'C:\\hadoop'  # 如果存在，否则设为空
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ====== 2. 导入Spark模块 ======
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, regexp_replace
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    print("✅ PySpark模块导入成功")
except ImportError as e:
    print(f"❌ PySpark导入失败: {e}")
    print("请安装pyspark: pip install pyspark")
    sys.exit(1)

# ====== 3. 创建SparkSession ======
try:
    spark = SparkSession.builder \
        .appName("Weibo_Sentiment_Simple") \
        .master("local[*]") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.python.worker.timeout", "600") \
        .getOrCreate()
    
    print("✅ Spark会话创建成功！")
    print(f"Spark版本: {spark.version}")
except Exception as e:
    print(f"❌ Spark会话创建失败: {e}")
    sys.exit(1)

# ====== 4. 主函数 ======
def main():
    try:
        # 创建示例数据（避免文件读取问题）
        print("\n📊 创建示例数据...")
        data = [
            (1, "这部电影真的很好看，演员演技在线，剧情也很吸引人！"),
            (0, "太失望了，浪费时间和金钱，完全不值得看。"),
            (1, "强烈推荐！这是我今年看过最好的电影！"),
            (0, "非常糟糕的体验，导演到底在想什么？"),
            (1, "音乐很棒，画面很美，值得一看。"),
            (0, "剧情拖沓，毫无新意，浪费时间。"),
            (1, "演员表演出色，故事情节感人。"),
            (0, "特效太假，剧情漏洞百出。"),
            (1, "导演功力深厚，每个细节都很到位。"),
            (0, "看了十分钟就想离开，太无聊了。"),
            (1, "剧情反转出人意料，非常精彩。"),
            (0, "角色塑造失败，无法引起共鸣。"),
            (1, "视觉效果震撼，值得去电影院观看。"),
            (0, "台词生硬，演员表演做作。"),
            (1, "情感真挚，让人感动落泪。"),
        ]
        
        columns = ["label", "review"]
        df = spark.createDataFrame(data, columns)
        print(f"创建了 {df.count()} 条示例数据")
        df.show(5)
        
        # ====== 5. 数据清洗 ======
        print("\n🔧 数据清洗...")
        
        # 使用正则表达式清洗
        df_cleaned = df.withColumn(
            "cleaned_review",
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        col("review"),
                        r'https?://\S+', ''
                    ),
                    r'@\w+', ''
                ),
                r'\[.*?\]', ''
            )
        )
        
        df_cleaned = df_cleaned.withColumn(
            "cleaned_review",
            regexp_replace(col("cleaned_review"), r'[^\w\u4e00-\u9fa5\s]', '')
        )
        
        print("清洗后数据示例:")
        df_cleaned.select("review", "cleaned_review").show(3, truncate=False)
        
        # ====== 6. 使用Spark内置Tokenizer（避免UDF） ======
        print("\n🔪 文本分词（使用Spark内置Tokenizer）...")
        
        # Spark内置的Tokenizer是按空格分词，对于中文效果有限
        # 但这是最简单的解决方案，不会触发Python worker问题
        tokenizer = Tokenizer(inputCol="cleaned_review", outputCol="words")
        
        # 应用分词
        df_tokenized = tokenizer.transform(df_cleaned)
        print("分词结果示例:")
        df_tokenized.select("cleaned_review", "words").show(3, truncate=False)
        
        # ====== 7. 停用词处理 ======
        print("\n📖 停用词过滤...")
        # 简单的停用词列表
        stop_words = ["的", "了", "在", "是", "我", "有", "和", "就", 
                     "不", "人", "都", "也", "而", "及", "与", "着", 
                     "或", "个", "没有", "这", "那", "就", "也", "很"]
        
        remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words",
            stopWords=stop_words
        )
        
        df_filtered = remover.transform(df_tokenized)
        
        # ====== 8. 特征工程 ======
        print("\n⚙️ 特征工程...")
        
        hashing_tf = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=100
        )
        
        idf = IDF(
            inputCol="raw_features",
            outputCol="features",
            minDocFreq=1
        )
        
        # ====== 9. 模型训练 ======
        print("\n🤖 训练逻辑回归模型...")
        
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=10,
            regParam=0.1
        )
        
        # 构建Pipeline
        pipeline = Pipeline(stages=[
            tokenizer,
            remover,
            hashing_tf,
            idf,
            lr
        ])
        
        # 划分数据集
        train_df, test_df = df_filtered.randomSplit([0.7, 0.3], seed=42)
        print(f"训练集: {train_df.count()} 条")
        print(f"测试集: {test_df.count()} 条")
        
        # 训练模型
        model = pipeline.fit(train_df)
        print("✅ 模型训练完成！")
        
        # ====== 10. 模型评估 ======
        print("\n📈 模型评估...")
        predictions = model.transform(test_df)
        
        # 评估指标
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        accuracy = evaluator.evaluate(predictions)
        print(f"模型准确率: {accuracy:.4f}")
        
        print("\n预测结果:")
        predictions.select("label", "prediction", "review").show(10, truncate=False)
        
        # ====== 11. 保存结果 ======
        print("\n💾 保存结果...")
        
        # 确保目录存在
        os.makedirs("./data/spark_results", exist_ok=True)
        
        # 保存预测结果
        predictions.select("label", "prediction", "review").toPandas().to_csv(
            "./data/spark_results/simple_predictions.csv", 
            index=False, 
            encoding='utf-8-sig'
        )
        
        # 保存模型报告
        with open("./data/spark_results/simple_model_report.txt", "w", encoding="utf-8") as f:
            f.write("Spark简化版情感分析模型报告\n")
            f.write("="*50 + "\n")
            f.write(f"数据量: {df.count()} 条\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write("模型: 逻辑回归\n")
            f.write("特征: HashingTF + IDF\n")
        
        print("✅ 结果已保存到 ./data/spark_results/")
        
        # ====== 12. 演示预测功能 ======
        print("\n🔮 演示预测功能...")
        
        # 创建测试数据
        test_samples = [
            ("这部电影真的很棒",),
            ("太糟糕了，完全浪费时间",),
            ("一般般，没什么特别",)
        ]
        
        test_df_demo = spark.createDataFrame(test_samples, ["review"])
        
        # 对测试数据应用相同的预处理
        test_cleaned = test_df_demo.withColumn(
            "cleaned_review",
            regexp_replace(col("review"), r'[^\w\u4e00-\u9fa5\s]', '')
        )
        
        # 进行预测
        test_predictions = model.transform(test_cleaned)
        
        print("测试预测结果:")
        for row in test_predictions.collect():
            sentiment = "正面" if row.prediction == 1 else "负面"
            print(f"  文本: {row.review}")
            print(f"  预测: {sentiment}")
            print()
        
        print("\n" + "="*60)
        print("✅ Spark简化版运行成功！")
        print("="*60)
        print("\n注意：由于Windows上的Spark限制，此版本:")
        print("1. 使用Spark内置Tokenizer（按空格分词），中文分词效果有限")
        print("2. 使用小规模示例数据")
        print("3. 避免了可能引发问题的Python UDF")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ====== 运行主程序 ======
if __name__ == "__main__":
    success = main()
    
    # 停止Spark会话
    if 'spark' in locals():
        spark.stop()
        print("\n🛑 Spark会话已停止。")
    
    if not success:
        print("\n⚠️  运行失败，请尝试以下方案:")
        print("1. 使用Pandas/Scikit-learn版本（推荐）")
        print("2. 在Linux/WSL环境中运行Spark")
        print("3. 使用云服务运行Spark")