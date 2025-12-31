# advanced_modeling.py - 完整版
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import scipy.sparse
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("社交媒体情感分析 - 进阶模型测试 (随机森林 & XGBoost)")
print("="*60)

# 1. 加载特征工程阶段保存的数据
data_dir = './data/pandas_processed'
X = scipy.sparse.load_npz(os.path.join(data_dir, 'X_tfidf_features.npz'))
y = pd.read_pickle(os.path.join(data_dir, 'y_labels.pkl'))

print(f"特征矩阵形状: {X.shape}")
print(f"标签分布:\n{y.value_counts()}")

# 2. 划分训练集和测试集 (与之前一致)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集: {X_train.shape[0]} 条样本")
print(f"测试集: {X_test.shape[0]} 条样本")

# 3. 将稀疏矩阵转换为密集矩阵 (XGBoost对稀疏矩阵支持有限，可能需要转换)
print("\n注意: XGBoost对稀疏矩阵支持有限，正在转换为密集矩阵...")
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
print(f"密集矩阵形状: {X_train_dense.shape}")

# 4. 测试随机森林模型 (可以使用稀疏矩阵)
print("\n" + "-"*50)
print("1. 随机森林 (Random Forest)")
print("-"*50)

rf_model = RandomForestClassifier(
    n_estimators=100,        # 树的数量，可调整
    max_depth=None,          # 树的最大深度
    min_samples_split=2,     # 内部节点再划分所需最小样本数
    min_samples_leaf=1,      # 叶子节点最少样本数
    random_state=42,
    n_jobs=-1                # 使用所有CPU核心
)

# 使用3折交叉验证评估
print("正在进行3折交叉验证...")
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, 
                               cv=3, scoring='f1_weighted', n_jobs=-1)
print(f"交叉验证F1分数: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# 在完整训练集上训练
print("在完整训练集上训练随机森林...")
rf_start = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - rf_start
print(f"训练完成，耗时: {rf_train_time:.2f}秒")

# 在测试集上评估
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

print(f"\n测试集性能:")
print(f"准确率: {rf_accuracy:.4f}")
print(f"F1分数: {rf_f1:.4f}")

# 5. 测试XGBoost模型 (需要使用密集矩阵)
print("\n" + "-"*50)
print("2. XGBoost")
print("-"*50)

xgb_model = XGBClassifier(
    n_estimators=100,           # 树的数量
    max_depth=6,                # 树的最大深度
    learning_rate=0.1,          # 学习率
    objective='binary:logistic', # 二分类问题
    use_label_encoder=False,    # 避免警告
    eval_metric='logloss',      # 评估指标
    random_state=42,
    n_jobs=-1                   # 使用所有CPU核心
)

# 使用3折交叉验证评估
print("正在进行3折交叉验证...")
xgb_cv_scores = cross_val_score(xgb_model, X_train_dense, y_train, 
                                cv=3, scoring='f1_weighted', n_jobs=-1)
print(f"交叉验证F1分数: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std():.4f})")

# 在完整训练集上训练
print("在完整训练集上训练XGBoost...")
xgb_start = time.time()
xgb_model.fit(X_train_dense, y_train)
xgb_train_time = time.time() - xgb_start
print(f"训练完成，耗时: {xgb_train_time:.2f}秒")

# 在测试集上评估
y_pred_xgb = xgb_model.predict(X_test_dense)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb, average='weighted')

print(f"\n测试集性能:")
print(f"准确率: {xgb_accuracy:.4f}")
print(f"F1分数: {xgb_f1:.4f}")

# 6. 与之前逻辑回归的基准对比
print("\n" + "="*50)
print("模型性能对比 (所有模型)")
print("="*50)

# 加载之前逻辑回归的基准结果
try:
    with open(os.path.join(data_dir, 'tuned_best_model.pkl'), 'rb') as f:
        lr_data = pickle.load(f)
    lr_f1 = lr_data.get('test_f1', 0.70)
except:
    lr_f1 = 0.7003  # 使用你之前运行得到的默认模型F1分数

comparison_data = {
    '逻辑回归 (基准)': lr_f1,
    '随机森林': rf_f1,
    'XGBoost': xgb_f1
}

comparison_df = pd.DataFrame.from_dict(comparison_data, orient='index', columns=['F1分数'])
comparison_df = comparison_df.sort_values('F1分数', ascending=False)
print("\nF1分数对比 (越高越好):")
print(comparison_df.to_string())

# 计算提升百分比
baseline_f1 = lr_f1
for model_name, f1_score_val in comparison_data.items():
    if model_name != '逻辑回归 (基准)':
        improvement = f1_score_val - baseline_f1
        percent_improvement = (improvement / baseline_f1) * 100
        print(f"\n{model_name} 对比基准:")
        print(f"  绝对提升: {improvement:.4f}")
        print(f"  相对提升: {percent_improvement:.2f}%")

# 7. 可视化对比结果
print("\n" + "="*50)
print("生成可视化图表")
print("="*50)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 模型性能对比条形图
plt.figure(figsize=(10, 6))
colors = ['#2E86AB', '#A23B72', '#F18F01']  # 为三个模型设置不同颜色
bars = plt.bar(comparison_df.index, comparison_df['F1分数'], color=colors, alpha=0.8)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.3f}', ha='center', va='bottom', fontsize=12)

plt.title('不同模型在微博情感分析上的F1分数对比', fontsize=16)
plt.ylabel('F1分数', fontsize=14)
plt.ylim([0.65, 0.75])  # 根据你的结果调整Y轴范围
plt.axhline(y=baseline_f1, color='r', linestyle='--', alpha=0.7, label=f'基准线 ({baseline_f1:.3f})')
plt.legend()
plt.tight_layout()

# 保存图表
chart_path = './data/pandas_processed/advanced_model_comparison.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"模型对比图已保存至: {chart_path}")

# 8. 保存最佳进阶模型
print("\n" + "="*50)
print("保存最佳进阶模型")
print("="*50)

# 确定最佳模型
best_model_name = comparison_df.index[0]
print(f"最佳模型: {best_model_name} (F1分数: {comparison_df.iloc[0]['F1分数']:.4f})")

if best_model_name == '随机森林':
    best_model = rf_model
    model_type = 'random_forest'
    # 使用全部数据重新训练最佳模型
    print("使用全部数据重新训练随机森林...")
    rf_model_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model_full.fit(X, y)
    
elif best_model_name == 'XGBoost':
    best_model = xgb_model
    model_type = 'xgboost'
    # 使用全部数据重新训练最佳模型
    print("使用全部数据重新训练XGBoost...")
    X_dense = X.toarray()
    xgb_model_full = XGBClassifier(n_estimators=100, use_label_encoder=False, 
                                   eval_metric='logloss', random_state=42)
    xgb_model_full.fit(X_dense, y)
else:
    print("逻辑回归仍是基准模型，请参考之前的调优结果。")
    best_model = None

# 如果找到了更好的模型，保存它
if best_model_name in ['随机森林', 'XGBoost']:
    # 加载特征工程阶段的向量化器
    with open(os.path.join(data_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    # 构建保存对象
    advanced_model_path = './data/pandas_processed/advanced_best_model.pkl'
    save_obj = {
        'model_type': model_type,
        'model': rf_model_full if best_model_name == '随机森林' else xgb_model_full,
        'vectorizer': vectorizer,
        'performance': {
            'f1_score': float(comparison_df.iloc[0]['F1分数']),
            'accuracy': float(rf_accuracy if best_model_name == '随机森林' else xgb_accuracy)
        },
        'requires_dense': best_model_name == 'XGBoost',  # XGBoost需要密集矩阵
        'feature_dimension': X.shape[1],
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(advanced_model_path, 'wb') as f:
        pickle.dump(save_obj, f)
    
    print(f"✅ 最佳进阶模型已保存至: {advanced_model_path}")

# 9. 生成详细报告
print("\n" + "="*50)
print("生成详细报告")
print("="*50)

report_path = './data/pandas_processed/advanced_modeling_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("社交媒体情感分析 - 进阶模型测试报告\n")
    f.write("="*60 + "\n\n")
    f.write(f"测试时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据规模: {X.shape[0]} 样本, {X.shape[1]} 特征\n")
    f.write(f"训练集/测试集: {X_train.shape[0]}/{X_test.shape[0]} 样本\n\n")
    
    f.write("各模型性能:\n")
    f.write("-"*40 + "\n")
    f.write(f"1. 逻辑回归 (基准): F1 = {lr_f1:.4f}\n")
    f.write(f"2. 随机森林: F1 = {rf_f1:.4f}\n")
    f.write(f"   训练时间: {rf_train_time:.2f}秒\n")
    f.write(f"3. XGBoost: F1 = {xgb_f1:.4f}\n")
    f.write(f"   训练时间: {xgb_train_time:.2f}秒\n\n")
    
    f.write("性能对比总结:\n")
    f.write("-"*40 + "\n")
    for idx, (model, score) in enumerate(comparison_data.items()):
        rank = idx + 1
        f.write(f"{rank}. {model}: {score:.4f}\n")
    
    f.write(f"\n最佳模型: {best_model_name}\n")
    f.write(f"最佳F1分数: {comparison_df.iloc[0]['F1分数']:.4f}\n")
    
    if best_model_name != '逻辑回归 (基准)':
        improvement = comparison_df.iloc[0]['F1分数'] - baseline_f1
        percent_improvement = (improvement / baseline_f1) * 100
        f.write(f"\n性能提升:\n")
        f.write(f"  绝对提升: {improvement:.4f}\n")
        f.write(f"  相对提升: {percent_improvement:.2f}%\n")

print(f"📊 详细报告已保存至: {report_path}")
print("\n" + "="*60)
print("进阶模型测试完成！")
print("="*60)