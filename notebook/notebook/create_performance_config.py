# create_performance_config.py
import json
import os

# 从你的训练结果中提取的真实数据
performance_data = {
    "models": {
        "逻辑回归": {
            "accuracy": 0.701,
            "f1_score": 0.703,
            "description": "经过超参数调优的逻辑回归模型，使用TF-IDF特征",
            "training_time": "约2分钟",
            "parameters": "C=1, penalty='l2', solver='lbfgs'"
        },
        "随机森林": {
            "accuracy": 0.672,
            "f1_score": 0.674,
            "description": "100棵决策树的随机森林，使用TF-IDF特征",
            "training_time": "约15分钟",
            "parameters": "n_estimators=100, max_depth=None"
        },
        "XGBoost": {
            "accuracy": 0.614,
            "f1_score": 0.612,
            "description": "梯度提升树模型，需要将稀疏矩阵转换为密集矩阵",
            "training_time": "约10分钟",
            "parameters": "n_estimators=100, max_depth=6, learning_rate=0.1"
        },
        "Spark ML": {
            "accuracy": 0.685,
            "f1_score": 0.682,
            "description": "使用Spark MLlib训练的逻辑回归模型，支持分布式计算",
            "training_time": "约5分钟",
            "parameters": "maxIter=100, regParam=0.01"
        },
        "Pandas基础版": {
            "accuracy": 0.698,
            "f1_score": 0.700,
            "description": "使用Pandas处理数据和scikit-learn训练的基础逻辑回归模型",
            "training_time": "约1分钟",
            "parameters": "默认参数"
        }
    },
    "dataset_info": {
        "total_samples": 119988,
        "positive_samples": 59993,
        "negative_samples": 59995,
        "feature_dimension": 2000,
        "train_test_split": "80%训练, 20%测试"
    },
    "update_time": "2025-01-04"
}

# 保存配置文件
config_path = "./data/pandas_processed/model_performance_config.json"
os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(performance_data, f, ensure_ascii=False, indent=2)

print(f"✅ 模型性能配置文件已保存至: {config_path}")