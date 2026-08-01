"""
风险等级有序分类改造版脚本：将标签视为有序数值，而不是纯离散类别。

核心思想：
1. 仍然沿用原有的数据读取与特征预处理逻辑：
   - 二值特征映射为 0/1；
   - 有序特征按显式顺序编码；
   - 无序类别使用 One-Hot 编码。
2. 不再直接把风险等级当成“无序多分类”处理，而是把风险等级映射为有序整数（0,1,2,3,4），
   然后使用 XGBoost 的回归模型进行训练，模型输出连续值。
3. 训练后通过最近整数映射（round / rint）将连续预测值转回风险等级类别，形成有序分类结果。
4. 在评估阶段加入两类指标：
   - 普通准确率（完全正确才算对）；
   - 距离加权正确率（相邻等级也视为部分正确），用于反映“错一档”比“错两档更可接受”的业务语义。
5. 这种思路更适合风险等级这种具有顺序关系、但边界不够清晰的场景。

说明：
- 这不是传统意义上的“多分类损失”，而是把风险等级任务转成“有序回归 + 近邻后处理”的方式。
- 若业务上认为“低风险和中风险”属于很接近的两类，则该方案通常比硬分类更合理。
"""

import time
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


# ==================== 1. 读取数据 ====================
data_path = "data/Dataset.csv"
df = pd.read_csv(data_path)
print(f"原始数据集形状: {df.shape}")

X_raw = df.iloc[:, :23].copy()
y_raw = df.iloc[:, 23]


# ==================== 2. 风险标签有序编码 ====================
# 这里保持和原脚本一致：把文本风险等级映射为 0,1,2,3,4
label_mapping = {'无风险': 0, '低风险': 1, '中风险': 2, '高风险': 3, '极高风险': 4}

if set(y_raw.unique()).issubset(set(label_mapping.keys())):
    y = y_raw.map(label_mapping).values.astype(int)
else:
    y = y_raw.values.astype(int)

print(f"标签分布: {np.bincount(y)}")


# ==================== 3. 特征预处理 ====================
# 01. 数值特征
numeric_features = [
    'health', 'fatigue', 'emotion', 'safety_knowledge', 'experience', 'distance', 'stay_time',
    'overlap', 'protection_eff', 'impact_range', 'stability_margin', 'device_abnormality',
    'sensor_coverage', 'alarm_effect', 'wind_speed', 'illumination'
]
numeric_features = [col for col in numeric_features if col in X_raw.columns]

# 02. 二元特征：是/否 -> 1/0
binary_features = ['alcohol', 'authorized', 'ppe_fastened']
binary_mapping = {'是': 1, '否': 0}
for col in binary_features:
    if col in X_raw.columns:
        X_raw[col] = X_raw[col].map(binary_mapping).fillna(0).astype(int)

# 03. 有序特征：按先后顺序编码
ordinal_features = {
    'skill': ['无证', '初级', '中级', '高级'],
    'accident_history': ['无', '1次', '≥2次'],
    'violation_history': ['无', '1次', '≥2次']
}
for col, categories in ordinal_features.items():
    if col in X_raw.columns:
        order_map = {cat: idx for idx, cat in enumerate(categories)}
        X_raw[col] = X_raw[col].map(order_map).fillna(0).astype(int)

# 04. 无序类别：One-Hot 编码
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


# ==================== 5. 有序分类训练：把标签看作有序数值 ====================
# 这里使用回归模型来拟合有序标签，输出连续值，再通过 round 映射回类别。
params = {
    'max_depth': 4,
    'learning_rate': 0.08,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

# 为了处理类别不平衡，使用样本权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = class_weights[y_train]

train_dmatrix = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
val_dmatrix = xgb.DMatrix(X_val, label=y_val)

early_stop = xgb.callback.EarlyStopping(
    rounds=20,
    metric_name='mae',
    data_name='validation_0',
    save_best=True
)

booster = xgb.train(
    params={
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'min_child_weight': params['min_child_weight'],
        'reg_alpha': params['reg_alpha'],
        'reg_lambda': params['reg_lambda'],
        'random_state': params['random_state'],
        'n_jobs': params['n_jobs']
    },
    dtrain=train_dmatrix,
    num_boost_round=params['n_estimators'],
    evals=[(val_dmatrix, 'validation_0')],
    callbacks=[early_stop],
    verbose_eval=False
)

model = booster


# ==================== 6. 预测结果转回类别 ====================
def ordinal_predict_to_label(y_pred_reg, min_label=0, max_label=4):
    """把连续回归值映射为最近的风险等级整数。"""
    y_pred = np.rint(y_pred_reg).astype(int)
    y_pred = np.clip(y_pred, min_label, max_label)
    return y_pred


# 训练、验证、测试集预测
train_dmatrix_pred = xgb.DMatrix(X_train)
val_dmatrix_pred = xgb.DMatrix(X_val)
test_dmatrix_pred = xgb.DMatrix(X_test)

train_pred_reg = model.predict(train_dmatrix_pred)
val_pred_reg = model.predict(val_dmatrix_pred)
test_pred_reg = model.predict(test_dmatrix_pred)

train_pred = ordinal_predict_to_label(train_pred_reg)
val_pred = ordinal_predict_to_label(val_pred_reg)
test_pred = ordinal_predict_to_label(test_pred_reg)


# ==================== 7. 评价指标：传统准确率 + 距离加权正确率 ====================
# 传统准确率
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

# 距离加权评价：相邻等级误差视为“部分正确”
def distance_weighted_score(y_true, y_pred):
    """
    设定规则：
    - 完全正确：1.0
    - 错 1 档：0.5
    - 错 >= 2 档：0.0
    """
    distances = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    scores = np.where(distances == 0, 1.0, np.where(distances == 1, 0.5, 0.0))
    return float(np.mean(scores))


train_weighted = distance_weighted_score(y_train, train_pred)
val_weighted = distance_weighted_score(y_val, val_pred)
test_weighted = distance_weighted_score(y_test, test_pred)

# 更细化的邻近误差统计：误差不超过1档的比例

def adjacent_accuracy(y_true, y_pred):
    distances = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.mean(distances <= 1))

train_adj = adjacent_accuracy(y_train, train_pred)
val_adj = adjacent_accuracy(y_val, val_pred)
test_adj = adjacent_accuracy(y_test, test_pred)


# ==================== 8. 输出结果 ====================
print(f"训练集准确率: {train_acc:.4f}")
print(f"验证集准确率: {val_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print(f"验证集相邻正确率(误差<=1): {val_adj:.4f}")
print(f"测试集相邻正确率(误差<=1): {test_adj:.4f}")
print(f"验证集距离加权正确率: {val_weighted:.4f}")
print(f"测试集距离加权正确率: {test_weighted:.4f}")
print("\n分类报告:")
print(classification_report(y_test, test_pred, target_names=list(label_mapping.keys())))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, test_pred))


# ==================== 9. 保存模型 ====================
joblib.dump(model, 'output/xgboost_ordinal_risk_model.pkl')
print("\n有序分类模型已保存为 output/xgboost_ordinal_risk_model.pkl")

# 日志文件
log_filename = f"output/B_{time.strftime('%m-%d_%H%M')}_ordinal_weighted.txt"
with open(log_filename, 'w', encoding='utf-8') as f:
    f.write("模型类型: XGBRegressor + 有序分类后处理\n")
    f.write("评估思路: 传统准确率 + 相邻误差加权准确率\n")
    f.write(f"训练集准确率: {train_acc:.4f}\n")
    f.write(f"验证集准确率: {val_acc:.4f}\n")
    f.write(f"测试集准确率: {test_acc:.4f}\n")
    f.write(f"验证集相邻正确率(误差<=1): {val_adj:.4f}\n")
    f.write(f"测试集相邻正确率(误差<=1): {test_adj:.4f}\n")
    f.write(f"验证集距离加权正确率: {val_weighted:.4f}\n")
    f.write(f"测试集距离加权正确率: {test_weighted:.4f}\n")
    f.write("\n分类报告:\n")
    f.write(classification_report(y_test, test_pred, target_names=list(label_mapping.keys())))
    f.write("\n混淆矩阵:\n")
    f.write(str(confusion_matrix(y_test, test_pred)))

print(f"结果已保存到 {log_filename}")
