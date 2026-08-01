"""
随机森林对照模型：作为传统树模型基线，与有序分类 XGBoost 进行公平对比。

核心思想：
1. 复用与原始 XGBoost 脚本一致的数据读取与特征预处理流程。
2. 将风险等级文本标签映射为整数类别，用于传统硬分类任务。
3. 使用 RandomForestClassifier 训练一个非线性树模型作为强基线。
4. 在训练集、验证集、测试集上分别计算准确率、分类报告和混淆矩阵。
5. 通过与提出方法的比较，验证“有序分类建模”的必要性与优势。
"""

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# ==================== 1. 读取数据 ====================
data_path = "data/Dataset.csv"
df = pd.read_csv(data_path)
print(f"原始数据集形状: {df.shape}")

X_raw = df.iloc[:, :23].copy()
y_raw = df.iloc[:, 23]


# ==================== 2. 标签编码 ====================
label_mapping = {'无风险': 0, '低风险': 1, '中风险': 2, '高风险': 3, '极高风险': 4}
if set(y_raw.unique()).issubset(set(label_mapping.keys())):
    y = y_raw.map(label_mapping).values.astype(int)
else:
    y = y_raw.values.astype(int)
print(f"标签分布: {np.bincount(y)}")


# ==================== 3. 特征预处理 ====================
# 二元特征：是/否 -> 1/0
binary_features = ['alcohol', 'authorized', 'ppe_fastened']
binary_mapping = {'是': 1, '否': 0}
for col in binary_features:
    if col in X_raw.columns:
        X_raw[col] = X_raw[col].map(binary_mapping).fillna(0).astype(int)

# 有序特征：按先后顺序编码
ordinal_features = {
    'skill': ['无证', '初级', '中级', '高级'],
    'accident_history': ['无', '1次', '≥2次'],
    'violation_history': ['无', '1次', '≥2次']
}
for col, categories in ordinal_features.items():
    if col in X_raw.columns:
        order_map = {cat: idx for idx, cat in enumerate(categories)}
        X_raw[col] = X_raw[col].map(order_map).fillna(0).astype(int)

# 无序类别：One-Hot 编码
nominal_features = ['hazard_type']
X_processed = pd.get_dummies(X_raw, columns=nominal_features, prefix=nominal_features)

X = X_processed.values.astype(float)
print(f"处理后特征矩阵形状: {X.shape}")


# ==================== 4. 划分数据集 ====================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")


# ==================== 5. 随机森林模型训练 ====================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)


# ==================== 6. 评估 ====================
train_pred = rf_model.predict(X_train)
val_pred = rf_model.predict(X_val)
test_pred = rf_model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"训练集准确率: {train_acc:.4f}")
print(f"验证集准确率: {val_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, test_pred, target_names=list(label_mapping.keys())))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, test_pred))


# ==================== 7. 保存模型 ====================
joblib.dump(rf_model, 'output/random_forest_risk_model.pkl')
print("\n随机森林模型已保存为 output/random_forest_risk_model.pkl")

log_filename = f"output/C_{time.strftime('%m-%d_%H%M')}_rf_baseline.txt"
with open(log_filename, 'w', encoding='utf-8') as f:
    f.write("模型类型: RandomForestClassifier\n")
    f.write(f"训练集准确率: {train_acc:.4f}\n")
    f.write(f"验证集准确率: {val_acc:.4f}\n")
    f.write(f"测试集准确率: {test_acc:.4f}\n")
    f.write("\n分类报告:\n")
    f.write(classification_report(y_test, test_pred, target_names=list(label_mapping.keys())))
    f.write("\n混淆矩阵:\n")
    f.write(str(confusion_matrix(y_test, test_pred)))

print(f"结果已保存到 {log_filename}")
