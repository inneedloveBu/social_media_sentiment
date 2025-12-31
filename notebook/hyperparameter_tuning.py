# hyperparameter_tuning.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import scipy.sparse
import pickle
import os
import time
import warnings
import json
warnings.filterwarnings('ignore')

print("="*60)
print("社交媒体情感分析模型 - 超参数调优")
print("="*60)

# 1. 加载数据
data_dir = './data/pandas_processed'
X = scipy.sparse.load_npz(os.path.join(data_dir, 'X_tfidf_features.npz'))
y = pd.read_pickle(os.path.join(data_dir, 'y_labels.pkl'))

# 划分训练集和测试集 (与之前一致)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]} 条样本")
print(f"测试集: {X_test.shape[0]} 条样本")
print(f"特征维度: {X_train.shape[1]}")

# 2. 选择并定义待调优的基模型
# 根据之前的结果，选择逻辑回归或LinearSVC。这里以逻辑回归为例，它通常更高效稳定。
print("\n" + "-"*40)
print("选择逻辑回归 (Logistic Regression) 进行超参数调优")
print("-"*40)

base_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)

# 3. 定义超参数网格
# C: 正则化强度，越小正则化越强。
# penalty: 正则化类型。注意：'l1'正则化需要solver支持（如'saga'或'liblinear')。
# solver: 优化算法。
param_grid = [
    {
        'C': [0.01, 0.1, 1, 10, 100],  # 覆盖从强正则化到弱正则化的范围
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']  # 'lbfgs'是默认，对l2高效；'saga'通用性强
    },
    {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1'],
        'solver': ['saga', 'liblinear'],  # 支持l1正则化的求解器
        'max_iter': [2000]  # l1可能需要更多迭代
    }
]

print(f"超参数组合总数: {sum(len(d) for d in param_grid)}")
print("开始网格搜索... (这可能需要几分钟，请耐心等待)")

# 4. 初始化GridSearchCV
# 使用5折交叉验证，以F1分数作为评估指标
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='f1_weighted',  # 使用加权F1分数，对不平衡数据更友好
    cv=5,                   # 5折交叉验证
    verbose=1,              # 输出详细进度
    n_jobs=-1               # 使用所有CPU核心并行计算
)

# 5. 执行网格搜索（这是最耗时的部分）
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"\n网格搜索完成！耗时: {end_time - start_time:.2f} 秒")

# 6. 输出最佳参数和交叉验证结果
print("\n" + "="*40)
print("最佳超参数组合")
print("="*40)
best_params = grid_search.best_params_
print(json.dumps(best_params, indent=4))
print(f"\n最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 查看所有参数组合的结果
cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_df = cv_results_df.sort_values('rank_test_score')
print(f"\n查看排名前5的参数组合:")
cols_to_display = ['rank_test_score', 'mean_test_score', 'std_test_score', 'param_C', 'param_penalty', 'param_solver']
print(cv_results_df[cols_to_display].head().to_string())

# 7. 用最佳模型在测试集上最终评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*40)
print("在独立测试集上的最终性能")
print("="*40)
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集F1分数: {test_f1:.4f}")
print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=['负面', '正面']))

# 8. 与默认参数模型对比（展示调优的增益）
print("\n" + "="*40)
print("性能提升对比")
print("="*40)
# 训练一个使用默认参数的模型
default_model = LogisticRegression(random_state=42, max_iter=1000)
default_model.fit(X_train, y_train)
y_pred_default = default_model.predict(X_test)
default_f1 = f1_score(y_test, y_pred_default, average='weighted')

improvement = test_f1 - default_f1
print(f"默认参数模型F1分数: {default_f1:.4f}")
print(f"调优后最佳模型F1分数: {test_f1:.4f}")
print(f"F1分数提升: {improvement:.4f} ({improvement/default_f1*100:.2f}%)")

# 9. 可视化结果
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 图1: 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['负面', '正面'], 
            yticklabels=['负面', '正面'])
plt.title(f'调优后最佳模型混淆矩阵\n(测试集 F1={test_f1:.3f})', fontsize=16)
plt.ylabel('真实标签', fontsize=14)
plt.xlabel('预测标签', fontsize=14)
plt.tight_layout()
conf_matrix_path = './data/pandas_processed/confusion_matrix_tuned.png'
plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
print(f"\n混淆矩阵已保存至: {conf_matrix_path}")

# 图2: 不同C值下的性能表现（可视化调优过程）
# 提取所有C值对应的结果
c_values = []
mean_scores = []
for i, params in enumerate(cv_results_df['params']):
    if 'C' in params:
        c_values.append(params['C'])
        mean_scores.append(cv_results_df.iloc[i]['mean_test_score'])

if c_values:  # 确保有数据可绘制
    plt.figure(figsize=(10, 6))
    # 为了清晰，按C值排序
    c_scores_df = pd.DataFrame({'C': c_values, 'F1_Score': mean_scores})
    c_scores_df = c_scores_df.sort_values('C')
    
    plt.plot(c_scores_df['C'], c_scores_df['F1_Score'], 'bo-', linewidth=2, markersize=8)
    plt.xscale('log')  # C值通常以对数尺度观察
    plt.xlabel('正则化强度 C (对数尺度)', fontsize=14)
    plt.ylabel('交叉验证 F1 分数', fontsize=14)
    plt.title('不同正则化强度 (C) 下的模型性能', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    c_curve_path = './data/pandas_processed/c_parameter_curve.png'
    plt.savefig(c_curve_path, dpi=300, bbox_inches='tight')
    print(f"C参数性能曲线已保存至: {c_curve_path}")

# 10. 保存最佳模型和相关元数据
print("\n" + "="*40)
print("保存调优后的最佳模型")
print("="*40)

# 加载特征工程阶段的向量化器
with open(os.path.join(data_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# 使用全部数据重新训练最佳模型（以获得最好的泛化性能）
print("使用全部数据重新训练最终模型...")
final_model = LogisticRegression(**best_params, random_state=42, max_iter=2000)
final_model.fit(X, y)  # 这次使用全部数据

# 构建保存对象
model_save_path = './data/pandas_processed/tuned_best_model.pkl'
save_obj = {
    'model': final_model,
    'vectorizer': vectorizer,
    'best_params': best_params,
    'test_accuracy': test_accuracy,
    'test_f1': test_f1,
    'feature_dimension': X.shape[1],
    'training_samples': X.shape[0]
}

with open(model_save_path, 'wb') as f:
    pickle.dump(save_obj, f)

print(f"✅ 调优后的最佳模型已保存至: {model_save_path}")
print(f"   包含: 最终模型 + 向量化器 + 最佳参数 + 性能指标")

# 保存调优报告
report_path = './data/pandas_processed/hyperparameter_tuning_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("社交媒体情感分析 - 超参数调优报告\n")
    f.write("="*50 + "\n\n")
    f.write(f"调优时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据规模: {X.shape[0]} 样本, {X.shape[1]} 特征\n")
    f.write(f"交叉验证折数: 5\n")
    f.write(f"评估指标: F1_weighted\n\n")
    f.write("最佳超参数:\n")
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"\n最佳交叉验证F1分数: {grid_search.best_score_:.4f}\n")
    f.write(f"独立测试集F1分数: {test_f1:.4f}\n")
    f.write(f"独立测试集准确率: {test_accuracy:.4f}\n\n")
    f.write("性能对比:\n")
    f.write(f"  默认参数模型F1分数: {default_f1:.4f}\n")
    f.write(f"  调优后模型F1分数: {test_f1:.4f}\n")
    f.write(f"  绝对提升: {improvement:.4f}\n")
    f.write(f"  相对提升: {improvement/default_f1*100:.2f}%\n")

print(f"📊 详细调优报告已保存至: {report_path}")
print("\n" + "="*60)
print("超参数调优完成！")
print("="*60)