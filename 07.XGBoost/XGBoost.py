import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from xgboost import callback
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 读取数据 ====================
# 特征文件：第一行为列名，共26列
feature_path = "05.vine_copula/01.vine_copula_samples.csv"
label_path = "05.vine_copula/03.comprehensive_risk.csv"

# 加载特征
df_features = pd.read_csv(feature_path)
print(f"特征数据形状: {df_features.shape}")
print(f"特征列名: {list(df_features.columns)}")

# 加载标签：取第28列（索引27），假设文件可能无表头或第一行为数据
# 尝试先读取前几行判断，简单起见：用header=None，取第27列
df_labels_raw = pd.read_csv(label_path, header=0)
# 取第28列（索引27）
labels = df_labels_raw.iloc[:, 27].values
print(f"标签数据形状: {labels.shape}")
print(f"标签唯一值: {np.unique(labels)}")

# 检查样本数量是否一致
assert len(df_features) == len(labels), "特征与标签样本数不一致！"

# ==================== 2. 预处理 ====================
# 定义分类特征（需手动指定，根据数据样例）
# 数值特征（连续）
numeric_features = [
    'health', 'fatigue', 'emotion', 'safety_knowledge', 'experience',
    'distance', 'stay_time', 'overlap', 'protection_eff', 'visibility',
    'impact_range', 'stability_margin', 'device_abnormality', 'sensor_coverage',
    'alarm_effect', 'wind_speed', 'illumination'
]

# 二元名义特征（是/否 或 其他二元）
binary_features = [
    'alcohol', 'authorized', 'ppe_fastened', 'sign_visible', 'alert_perceptible'
]

# 有序分类特征（有明确顺序）
ordinal_features = {
    'skill': ['无证', '初级', '中级', '高级'],
    'accident_history': ['无', '1次', '≥2次'],
    'violation_history': ['无', '1次', '≥2次']
}

# 无序多类特征
nominal_features = ['hazard_type']

# 映射二元特征：'是'->1, '否'->0；若存在'有'/'无'等也统一处理
binary_mapping = {'是': 1, '否': 0, '有': 1, '无': 0}

# 开始预处理
df = df_features.copy()

# 处理二元特征
for col in binary_features:
    if col in df.columns:
        df[col] = df[col].map(binary_mapping).fillna(0).astype(int)

# 处理有序特征
for col, categories in ordinal_features.items():
    if col in df.columns:
        le = LabelEncoder()
        le.fit(categories)
        df[col] = le.transform(df[col])

# 处理无序多类特征（one-hot编码）
# 由于训练/验证/测试集划分后需保持一致，我们将在划分后使用ColumnTransformer处理所有特征
# 为简化流程，这里先对hazard_type进行one-hot，后续可统一使用Pipeline
# 更规范做法：对整体特征处理后再划分，避免数据泄露。但one-hot后维度固定，无影响。
df = pd.get_dummies(df, columns=nominal_features, prefix=nominal_features)

# 检查数值特征是否都存在
for col in numeric_features:
    if col not in df.columns:
        print(f"警告：数值特征 {col} 不在数据中，跳过")
        numeric_features.remove(col)

# 提取最终特征矩阵和标签
X = df.values.astype(float)  # 确保数值型
y = labels

# 将文本标签转换为数字编码
label_mapping = {'无风险': 0, '低风险': 1, '中风险': 2, '高风险': 3, '极高风险': 4}
y_encoded = np.array([label_mapping[lab] for lab in y])

print(f"特征矩阵形状: {X.shape}, 标签形状: {y_encoded.shape}")
print("标签分布:", np.bincount(y_encoded))

# ==================== 3. 划分数据集（训练/验证/测试） ====================
# 先分出训练集（60%）和临时集（40%）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
)
# 再从临时集中分出验证集（20%整体）和测试集（20%整体）
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"训练集样本数: {len(X_train)}")
print(f"验证集样本数: {len(X_val)}")
print(f"测试集样本数: {len(X_test)}")

# ==================== 4. 构建XGBoost模型并进行训练 ====================
# 定义模型参数（多分类）
early_stop = callback.EarlyStopping(
    rounds=10,                  # 连续几轮无提升则停止
    metric_name='mlogloss',     # 监控的评估指标
    data_name='validation_0',   # 监控数据集名称，对应 eval_set 中传入的验证集
    save_best=True              # 保存最佳模型而非最终迭代的模型
)

model = xgb.XGBClassifier(
    objective='multi:softmax',   # 多分类，输出类别索引
    num_class=5,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    callbacks=[early_stop]      # 传入回调列表
)

# 训练模型，使用验证集进行早停（可选）
# 注意：在 fit 方法中提供 eval_set
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# ==================== 5. 模型评估 ====================
# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("\n========== 模型评估结果 ==========")
print(f"测试集准确率: {accuracy:.4f}")

# 分类报告（类别名称使用原标签名）
target_names = list(label_mapping.keys())
print("\n分类报告:")
print(classification_report(y_test, y_pred, labels=[0,1,2,3,4], target_names=target_names))

# 混淆矩阵
print("混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 可选：特征重要性
importance = model.feature_importances_
feature_names = df.columns.tolist()
feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).head(10)
print("\nTop 10 重要特征:")
print(feat_imp_df)

# ==================== 6. 保存模型（可选） ====================
import joblib
joblib.dump(model, '06.XGBoost/xgboost_risk_model.pkl')
print("\n模型已保存为 xgboost_risk_model.pkl")