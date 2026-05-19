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
data_path = "data/Dataset.csv"
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

# ==================== 5. 超参数搜索（Optuna + 交叉验证） ====================
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# 合并训练集和验证集，用于最终训练
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])

class_weights_full = compute_class_weight('balanced', classes=np.unique(y_train_full), y=y_train_full)
sample_weights_full = class_weights_full[y_train_full]

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'n_estimators': 300,
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 15    # ✅ 关键：早停参数放在构造函数中
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_mlogloss = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val_cv = X_train[train_idx], X_train[val_idx]
        y_tr, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        class_w = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        sample_w = class_w[y_tr]
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_w,
            eval_set=[(X_val_cv, y_val_cv)],
            verbose=False
        )
        
        y_pred_prob = model.predict_proba(X_val_cv)
        loss = log_loss(y_val_cv, y_pred_prob)
        fold_mlogloss.append(loss)
    
    return np.mean(fold_mlogloss)

print("\n开始Optuna超参数搜索...")
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print("\n最佳超参数组合:", best_params)

best_params.update({
    'n_estimators': 300,
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train_full)),
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
})

# ==================== 6. 用全部训练数据重新训练最终模型 ====================
print("\n用全部训练数据重新训练最终模型...")
early_stop_final = xgb.callback.EarlyStopping(
    rounds=20,
    metric_name='mlogloss',
    data_name='validation_0',
    save_best=True
)

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(
    X_train_full, y_train_full,
    sample_weight=sample_weights_full,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stop_final],
    verbose=True
)

# ==================== 7. 评估 ====================
y_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n最终测试集准确率: {test_acc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# ==================== 8. 保存模型及结果 ====================
joblib.dump(final_model, 'output/xgboost_risk_model_best.pkl')
print("\n最优模型已保存为 output/xgboost_risk_model_best.pkl")

log_filename = f"output/optuna_best_{time.strftime('%m%d%H%M')}.txt"
with open(log_filename, 'w', encoding='utf-8') as f:
    f.write("最佳超参数组合:\n")
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\n测试集准确率: {test_acc:.4f}\n")
    f.write("\n分类报告:\n")
    f.write(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))
    f.write("\n混淆矩阵:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
print(f"搜索结果已保存至 {log_filename}")