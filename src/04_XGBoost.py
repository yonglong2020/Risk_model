import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from xgboost import callback
import warnings
import joblib

warnings.filterwarnings('ignore')

# ==================== 1. 读取数据（特征与标签在同一文件） ====================
data_path = "data/03_labeled/Dataset.csv"
df = pd.read_csv(data_path)
print(f"原始数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 假设前23列为特征，第24列为标签
# 根据变量顺序定义特征列名（按实际CSV列名调整）
# 如果CSV列名与下面列表不完全一致，请根据实际列名修改
expected_cols = [
    'health', 'fatigue', 'alcohol', 'emotion', 'safety_knowledge',
    'experience', 'skill', 'accident_history', 'violation_history',
    'distance', 'stay_time', 'overlap', 'authorized', 'ppe_fastened',
    'protection_eff', 'hazard_type', 'impact_range', 'stability_margin',
    'device_abnormality', 'sensor_coverage', 'alarm_effect', 'wind_speed', 'illumination'
]
# 检查CSV是否包含这些列
for col in expected_cols:
    if col not in df.columns:
        raise ValueError(f"CSV文件中缺少列: {col}")

# 分离特征和标签
X_raw = df[expected_cols].copy()
y_raw = df.iloc[:, 23]      # 第24列（索引23）为标签
print(f"标签唯一值: {y_raw.unique()}")

# ==================== 2. 标签编码 ====================
# 假设标签为文本风险等级，映射为整数（根据实际情况调整）
# 若标签已经是数字，可跳过此步
label_mapping = {'无风险': 0, '低风险': 1, '中风险': 2, '高风险': 3, '极高风险': 4}
# 检查标签值是否都在映射中
if set(y_raw.unique()).issubset(set(label_mapping.keys())):
    y = y_raw.map(label_mapping).values
else:
    # 如果标签已经是数值，直接使用
    y = y_raw.values.astype(int)
print(f"标签分布: {np.bincount(y)}")

# ==================== 3. 特征预处理 ====================
# 定义各类特征
numeric_features = [
    'health', 'fatigue', 'emotion', 'safety_knowledge', 'experience',
    'distance', 'stay_time', 'overlap', 'protection_eff',
    'impact_range', 'stability_margin', 'device_abnormality',
    'sensor_coverage', 'alarm_effect', 'wind_speed', 'illumination'
]

binary_features = ['alcohol', 'authorized', 'ppe_fastened']

ordinal_features = {
    'skill': ['无证', '初级', '中级', '高级'],
    'accident_history': ['无', '1次', '≥2次'],
    'violation_history': ['无', '1次', '≥2次']
}

nominal_features = ['hazard_type']

# 确保数值特征都存在（过滤掉可能缺失的）
numeric_features = [col for col in numeric_features if col in X_raw.columns]

# 处理二元特征：将 '是'/'否' 映射为 1/0，若已经是0/1则保持不变
binary_mapping = {'是': 1, '否': 0, True: 1, False: 0, '有': 1, '无': 0}
for col in binary_features:
    if col in X_raw.columns:
        X_raw[col] = X_raw[col].map(binary_mapping).fillna(0).astype(int)

# 处理有序特征：显式映射为0,1,2,...（保证顺序）
for col, categories in ordinal_features.items():
    if col in X_raw.columns:
        # 创建顺序映射字典
        order_map = {cat: idx for idx, cat in enumerate(categories)}
        X_raw[col] = X_raw[col].map(order_map).fillna(0).astype(int)

# 处理无序多类特征：one-hot编码
X_processed = pd.get_dummies(X_raw, columns=nominal_features, prefix=nominal_features)

# 最终特征矩阵
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


# ==================== 5. XGBoost 模型训练 ====================
early_stop = callback.EarlyStopping(
    rounds=10,
    metric_name='mlogloss',
    data_name='validation_0',
    save_best=True
)

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    callbacks=[early_stop]
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# ==================== 6. 评估 ====================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 特征重要性
importance = model.feature_importances_
feature_names = X_processed.columns.tolist()
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(10)
print("\nTop 10 重要特征:")
print(feat_imp)

# ==================== 7. 保存模型 ====================
joblib.dump(model, 'output/xgboost_risk_model.pkl')
print("\n模型已保存")