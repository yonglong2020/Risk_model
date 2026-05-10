import pandas as pd
import numpy as np

# ------------------------------
# 1. 权重设定（基于专家打分 0-5）
# ------------------------------
weights_dict = {
    'health': 3.5,
    'fatigue': 5.0,
    'alcohol': 5.0,
    'emotion': 3.75,
    'safety_knowledge': 4.0,
    'experience': 3.5,
    'skill': 4.5,
    'accident_history': 3.25,
    'violation_history': 4.25,
    'distance': 4.75,
    'stay_time': 3.75,
    'overlap': 5.0,
    'authorized': 4.75,
    'ppe_fastened': 5.0,
    'protection_eff': 5.0,
    'visibility': 3.75,
    'sign_visible': 3.5,
    'alert_perceptible': 4.0,
    'hazard_type': 4.5,
    'impact_range': 4.25,
    'stability_margin': 5.0,
    'device_abnormality': 5.0,
    'sensor_coverage': 3.5,
    'alarm_effect': 4.5,
    'wind_speed': 4.25,
    'illumination': 3.5
}

# 归一化权重（使和为1）
total = sum(weights_dict.values())
weights_norm = {k: v/total for k, v in weights_dict.items()}

# 构建变量名到风险分列名的映射（原变量名 + '_risk'）
risk_cols = {var: f"{var}_risk" for var in weights_dict.keys()}

# ------------------------------
# 2. 读取风险分数据
# ------------------------------
file_path = 'data/02_sampling/vinecopula_mc_normalized.csv'   # 根据实际位置调整
df_risk = pd.read_csv(file_path)

# 检查必要的列是否存在
missing_cols = [col for col in risk_cols.values() if col not in df_risk.columns]
if missing_cols:
    raise KeyError(f"以下风险分列不存在: {missing_cols}\n实际列名: {df_risk.columns.tolist()}")

# ------------------------------
# 3. 计算综合得分
# ------------------------------
df_risk['comprehensive_score'] = 0.0
for var, weight in weights_norm.items():
    risk_col = risk_cols[var]
    df_risk['comprehensive_score'] += weight * df_risk[risk_col]

# 由于浮点误差，将得分截断在[0,1]区间
df_risk['comprehensive_score'] = df_risk['comprehensive_score'].clip(0, 1)

# ------------------------------
# 4. 划分风险等级（根据实际数据分布可调整阈值）
# ------------------------------
def classify_risk(score):
    if score < 0.25:
        return '无风险'
    if score < 0.35:
        return '低风险'
    elif score < 0.4:
        return '中风险'
    elif score < 0.45:
        return '高风险'
    else:
        return '极高风险'

df_risk['risk_level'] = df_risk['comprehensive_score'].apply(classify_risk)

# ------------------------------
# 5. 保存结果
# ------------------------------
output_path = 'data/comp_eval_labels.csv'
df_risk.to_csv(output_path, index=False)

print(f"综合风险计算完成，结果保存至：{output_path}")
print("\n前5行预览（综合得分 & 风险等级）：")
print(df_risk[['comprehensive_score', 'risk_level']].head())

# 可选：输出各等级统计
print("\n风险等级分布：")
print(df_risk['risk_level'].value_counts())