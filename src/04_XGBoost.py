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

# ==================== 1. 读取数据 ====================
data_path = "data/03_labeled/Dataset.csv"
df = pd.read_csv(data_path)
print(f"原始数据集形状: {df.shape}")
X_raw = df.iloc[:, :23].copy()  # 前23列为特征
y_raw = df.iloc[:, 23]          # 第24列为标签


# ==================== 2. 标签编码 ====================
# 假设标签为文本风险等级，映射为整数（根据实际情况调整）
label_mapping = {'无风险': 0, '低风险': 1, '中风险': 2, '高风险': 3, '极高风险': 4}
# 检查标签值是否都在映射中
if set(y_raw.unique()).issubset(set(label_mapping.keys())):
    y = y_raw.map(label_mapping).values
else:
    # 如果标签已经是数值，直接使用
    y = y_raw.values.astype(int)
print(f"标签分布: {np.bincount(y)}")

# ==================== 3. 特征预处理 ====================
# 01.定义数值特征
numeric_features = ['health', 'fatigue', 'emotion', 'safety_knowledge', 'experience','distance', 'stay_time', 'overlap', 'protection_eff',
    'impact_range', 'stability_margin', 'device_abnormality','sensor_coverage', 'alarm_effect', 'wind_speed', 'illumination']
numeric_features = [col for col in numeric_features if col in X_raw.columns]# 确保数值特征都存在（过滤掉可能缺失的）

# 02.定义二元特征
binary_features = ['alcohol', 'authorized', 'ppe_fastened']
binary_mapping = {'是': 1, '否': 0} # 处理二元特征：将 '是'/'否' 映射为 1/0，若已经是0/1则保持不变
for col in binary_features:
    if col in X_raw.columns:
        X_raw[col] = X_raw[col].map(binary_mapping).fillna(0).astype(int)

# 03.定义有序特征及其类别顺序
ordinal_features = {'skill': ['无证', '初级', '中级', '高级'],    'accident_history': ['无', '1次', '≥2次'],    'violation_history': ['无', '1次', '≥2次']}
for col, categories in ordinal_features.items():    # 处理有序特征：显式映射为0,1,2,...（保证顺序）
    if col in X_raw.columns:
        # 创建顺序映射字典
        order_map = {cat: idx for idx, cat in enumerate(categories)}
        X_raw[col] = X_raw[col].map(order_map).fillna(0).astype(int)

# 04.定义无序多类特征
nominal_features = ['hazard_type']
X_processed = pd.get_dummies(X_raw, columns=nominal_features, prefix=nominal_features)# 处理无序多类特征：one-hot编码

# 最终特征矩阵
X = X_processed.values.astype(float)
print(f"处理后特征矩阵形状: {X.shape}")


# ==================== 4. 划分数据集 ====================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)   #将完整数据集按 60%训练，40%临时拆分
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)   #将临时集进一步拆分为 75%验证集 和 25%测试集
print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")


# ==================== 5. XGBoost 模型训练 ====================
early_stop = callback.EarlyStopping(
    rounds=20,                  # 如果连续50轮验证集指标没有提升则停止训练
    metric_name='mlogloss',     # 监控多分类对数损失
    data_name='validation_0',   # 监控验证集
    save_best=True              # 训练过程中保存最佳模型
)

model = xgb.XGBClassifier(
    objective='multi:softmax',  # 多分类问题，使用 softmax 输出类别标签
    num_class=len(np.unique(y)),# 类别数量
    max_depth=6,                # 树的最大深度
    learning_rate=0.1,          # 学习率
    n_estimators=100,           # 树的数量
    subsample=0.8,              # 每棵树随机采样80%的数据
    colsample_bytree=0.8,       # 每棵树随机采样80%的特征
    random_state=42,            # 固定随机种子
    eval_metric='mlogloss',     # 评估指标为多分类对数损失
    callbacks=[early_stop]      # 使用早停回调函数
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)])  # 在训练过程中监控验证集性能，自动保存最佳模型并在性能不提升时提前停止训练

# ==================== 6. 评估 ====================
y_pred = model.predict(X_test)          # 预测测试集标签
acc = accuracy_score(y_test, y_pred)    # 计算测试集准确率
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