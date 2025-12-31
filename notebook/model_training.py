# model_training.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import scipy.sparse
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("开始社交媒体情感分析模型训练与评估")
print("="*60)

# 1. 加载特征工程阶段保存的数据
data_dir = './data/pandas_processed'
X = scipy.sparse.load_npz(os.path.join(data_dir, 'X_tfidf_features.npz'))
y = pd.read_pickle(os.path.join(data_dir, 'y_labels.pkl'))

print(f"特征矩阵形状: {X.shape}")
print(f"标签分布:\n{y.value_counts()}")

# 2. 划分训练集和测试集 (80%训练，20%测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 3. 初始化要比较的模型
models = {
    '逻辑回归 (Logistic Regression)': LogisticRegression(random_state=42, max_iter=1000),
    '随机森林 (Random Forest)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    '支持向量机 (Linear SVM)': LinearSVC(random_state=42, max_iter=1000)
}

# 4. 训练、预测并评估每个模型
results = {}
predictions = {}
feature_importances = {}  # 用于存储模型的特征重要性（如果有）

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"训练模型: {name}")
    print(f"{'='*40}")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上预测
    y_pred = model.predict(X_test)
    predictions[name] = (y_test, y_pred)  # 保存真实值和预测值
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        '准确率 (Accuracy)': accuracy,
        '精确率 (Precision)': precision,
        '召回率 (Recall)': recall,
        'F1分数 (F1-Score)': f1
    }
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 尝试获取特征重要性（仅适用于有该属性的模型，如随机森林）
    if hasattr(model, 'coef_'):
        feature_importances[name] = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_

# 5. 对比所有模型结果
print("\n" + "="*60)
print("模型性能对比总结")
print("="*60)

results_df = pd.DataFrame(results).T.round(4)
print(results_df)

# 6. 可视化结果
# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_style("whitegrid")

# 图1: 模型性能对比条形图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

metrics_to_plot = ['准确率 (Accuracy)', '精确率 (Precision)', '召回率 (Recall)', 'F1分数 (F1-Score)']
for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    results_df[metric].plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(models)))
    ax.set_title(f'{metric} 对比', fontsize=14)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim([0.8, 1.0])  # 情感分析任务指标通常较高，聚焦于高分段
    ax.tick_params(axis='x', rotation=45)
    # 在柱子上显示数值
    for i, v in enumerate(results_df[metric]):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('./data/pandas_processed/model_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n模型对比图已保存至: ./data/pandas_processed/model_comparison.png")

# 图2: 最佳模型的混淆矩阵 (选择F1分数最高的模型)
best_model_name = results_df['F1分数 (F1-Score)'].idxmax()
y_test_best, y_pred_best = predictions[best_model_name]

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_best, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
plt.title(f'最佳模型混淆矩阵: {best_model_name}', fontsize=16)
plt.ylabel('真实标签', fontsize=14)
plt.xlabel('预测标签', fontsize=14)
plt.tight_layout()
plt.savefig('./data/pandas_processed/confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
print(f"最佳模型混淆矩阵已保存至: ./data/pandas_processed/confusion_matrix_best_model.png")

# 7. 保存最佳模型和所有结果
print("\n" + "="*60)
print("保存模型与结果")
print("="*60)

# 加载特征工程阶段保存的向量化器，与最佳模型一起保存
with open(os.path.join(data_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

best_model = models[best_model_name]
best_model.fit(X, y)  # 用全部数据重新训练一次，以获得最好的泛化能力

model_save_path = './data/pandas_processed/best_sentiment_model.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump({
        'model': best_model,
        'vectorizer': vectorizer,
        'model_name': best_model_name,
        'performance': results_df.loc[best_model_name].to_dict()
    }, f)

print(f"✅ 最佳模型 ({best_model_name}) 已保存至: {model_save_path}")
print(f"   包含: 训练好的模型 + TF-IDF向量化器 + 性能指标")

# 保存详细的评估报告
report_save_path = './data/pandas_processed/detailed_classification_report.txt'
with open(report_save_path, 'w', encoding='utf-8') as f:
    f.write("社交媒体情感分析项目 - 详细分类报告\n")
    f.write("="*50 + "\n\n")
    f.write(f"数据规模: {X.shape[0]} 条样本， {X.shape[1]} 个特征\n")
    f.write(f"训练/测试比例: 80%/20%\n")
    f.write(f"最佳模型: {best_model_name}\n\n")
    f.write("各模型性能对比:\n")
    f.write(results_df.to_string() + "\n\n")
    f.write(f"\n最佳模型 ({best_model_name}) 的详细分类报告:\n")
    f.write(classification_report(y_test_best, y_pred_best, target_names=['负面', '正面']))

print(f"📊 详细评估报告已保存至: {report_save_path}")

print("\n" + "="*60)
print("模型训练与评估阶段全部完成！")
print("="*60)
print("\n下一步建议:")
print("1. 查看生成的图表文件，了解模型性能。")
print("2. 可以创建一个简单的预测脚本，输入新文本进行情感预测。")
print("3. 考虑使用Streamlit构建一个交互式Web应用进行展示。")