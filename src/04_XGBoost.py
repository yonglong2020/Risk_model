import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from xgboost import callback
import warnings
import joblib
import time

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

# ===================== 5.1. 记录超参数 =====================
params = {
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "rounds": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
}

hp_str = f"md{params['max_depth']}_lr{params['learning_rate']}_ne{params['n_estimators']}_r{params['rounds']}"
hp_str = hp_str.replace('.', '')
log_filename = f"output/结果_{hp_str}_{time.strftime('%m%d%H%M')}.txt"

# ===================== 5.2. 构建模型 =====================
early_stop = callback.EarlyStopping(
    rounds=params["rounds"],                  # 如果连续50轮验证集指标没有提升则停止训练
    metric_name='mlogloss',     # 监控多分类对数损失
    data_name='validation_0',   # 监控验证集
    save_best=True              # 训练过程中保存最佳模型
)

model = xgb.XGBClassifier(
    objective='multi:softmax',  # 多分类问题，使用 softmax 输出类别标签
    num_class=len(np.unique(y)),# 类别数量
    max_depth=params["max_depth"],                # 树的最大深度
    learning_rate=params["learning_rate"],          # 学习率
    n_estimators=params["n_estimators"],           # 树的数量
    subsample=params["subsample"],              # 每棵树随机采样80%的数据
    colsample_bytree=params["colsample_bytree"],       # 每棵树随机采样80%的特征
    #reg_lambda=1.2,              # L2正则化项
    #reg_alpha=0.1,               # L1正则化项
    random_state=42,            # 固定随机种子
    eval_metric='mlogloss',     # 评估指标为多分类对数损失
    callbacks=[early_stop]      # 使用早停回调函数
)



with open(log_filename, 'w', encoding='utf-8') as f:
    
    

    # ===================== 5.3.训练模型 =====================
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights[y_train]
    model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)])  # 在训练过程中监控验证集性能，自动保存最佳模型并在性能不提升时提前停止训练

     # 获取训练过程中的日志（关键！）
    eval_results = model.evals_result()
    val_log = eval_results['validation_0']['mlogloss']  # 损失函数名称

    # 把每一轮的损失写入文件
    batch = []
    for epoch, loss in enumerate(val_log, 1):
        batch.append(f"{loss:.5f}")  # 收集10个loss
        
        # 满10个 或 最后一轮 → 输出
        if epoch % 10 == 0 or epoch == len(val_log):
            start = epoch - len(batch) + 1
            end = epoch
            log_line = f"[{start}-{end}]\tvalidation_0-mlogloss: {', '.join(batch)}"
            f.write(log_line + "\n")
            batch = []  # 清空

    # ==================== 5.4. 评估模型 ====================
    # 1. 训练集评估
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # 2. 验证集评估
    y_pred_val = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)

    # 3. 测试集评估
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # 按顺序输出
    print(f"训练集准确率: {train_acc:.4f}")
    f.write(f"训练集准确率: {train_acc:.4f}\n")

    print(f"验证集准确率: {val_acc:.4f}")
    f.write(f"验证集准确率: {val_acc:.4f}\n")

    print(f"测试集准确率: {test_acc:.4f}\n")
    f.write(f"测试集准确率: {test_acc:.4f}\n")

    print("分类报告:")
    f.write("分类报告:\n")

    report = classification_report(y_test, y_pred, target_names=list(label_mapping.keys()))
    print(report)
    f.write(report + "\n")

    print("混淆矩阵:")
    f.write("混淆矩阵:\n")

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    f.write(str(cm) + "\n")


    # 特征重要性
    # importance = model.feature_importances_
    # feature_names = X_processed.columns.tolist()
    # feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    # feat_imp = feat_imp.sort_values('importance', ascending=False).head(10)
    # print("\nTop 10 重要特征:")
    # print(feat_imp)

# ==================== 6. 保存模型 ====================
joblib.dump(model, 'output/xgboost_risk_model.pkl')
print("\n模型已保存")