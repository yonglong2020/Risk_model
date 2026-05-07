import pandas as pd
import numpy as np

# 1. 读取原始数据
df = pd.read_csv('05.vine_copula/01.vine_copula_samples.csv')

# 2. 定义风险评分函数（每个指标一个函数，输入为Series或数值，输出风险分Series）
# 注意：部分函数依赖于其他列（如impact_range依赖hazard_type），单独处理

# 人员个体特质
def risk_health(s):
    return 1 - s ** 1.5

def risk_fatigue(s):
    return s ** 0.7

def risk_alcohol(s):
    # s为 "是" 或 "否"
    return s.apply(lambda x: 1.0 if x == '是' else 0.0)

def risk_emotion(s):
    return 1 - s

def risk_safety_knowledge(s):
    return 1 - s ** 0.8

def risk_experience(s):
    return np.where(s <= 1, 1.0,
                    np.where(s > 31, 0.0,
                             1 - ((s - 1) / 30) ** 1.2))

def risk_skill(s):
    mapping = {'无证': 1.0, '初级': 0.7, '中级': 0.4, '高级': 0.1}
    return s.map(mapping)

def risk_accident_history(s):
    mapping = {'无': 0.0, '1次': 0.5, '≥2次': 1.0}
    return s.map(mapping)

def risk_violation_history(s):
    mapping = {'无': 0.0, '1次': 0.4, '≥2次': 1.0}
    return s.map(mapping)

# 人与危险源时空关联
def risk_distance(s):
    return np.exp(-0.3 * s)

def risk_stay_time(s):
    return np.minimum(s / 60, 1.0)

def risk_overlap(s):
    return s ** 0.8

def risk_authorized(s):
    # "是" -> 0 (已准入，低风险), "否" -> 1
    return s.apply(lambda x: 0.0 if x == '是' else 1.0)

def risk_ppe_fastened(s):
    # "是" -> 0 (已系挂), "否" -> 1
    return s.apply(lambda x: 0.0 if x == '是' else 1.0)

def risk_protection_eff(s):
    return 1 - s ** 1.2

def risk_visibility(s):
    return 1 - s ** 0.8

def risk_sign_visible(s):
    return s.apply(lambda x: 0.0 if x == '是' else 1.0)

def risk_alert_perceptible(s):
    return s.apply(lambda x: 0.0 if x == '是' else 1.0)

# 危险源特性
def risk_hazard_type(s):
    mapping = {'高支模': 0.8, '深基坑': 0.7, '塔吊': 0.9, '车辆': 0.6, '配电箱': 0.5}
    return s.map(mapping)

def risk_impact_range(row):
    """依赖 hazard_type 和 impact_range"""
    L_max = {'塔吊': 80, '深基坑': 30, '高支模': 30, '车辆': 15, '配电箱': 5}
    ht = row['hazard_type']
    r = row['impact_range']
    if pd.isna(ht) or ht not in L_max:
        return np.nan
    return np.minimum(r / L_max[ht], 1.0)

def risk_stability_margin(s):
    return 1 - s ** 0.5

def risk_device_abnormality(s):
    return s ** 0.6

# 环境因素
def risk_sensor_coverage(s):
    return 1 - s ** 0.7

def risk_alarm_effect(s):
    return 1 - s ** 1.2

def risk_wind_speed(s):
    return np.where(s <= 2, 0.0, (s - 2) / 10)

def risk_illumination(s):
    return 1 - s ** 0.5

# 3. 计算各指标的风险分，创建新DataFrame
df_risk = pd.DataFrame(index=df.index)

# 应用单列函数
df_risk['health_risk'] = risk_health(df['health'])
df_risk['fatigue_risk'] = risk_fatigue(df['fatigue'])
df_risk['alcohol_risk'] = risk_alcohol(df['alcohol'])
df_risk['emotion_risk'] = risk_emotion(df['emotion'])
df_risk['safety_knowledge_risk'] = risk_safety_knowledge(df['safety_knowledge'])
df_risk['experience_risk'] = risk_experience(df['experience'])
df_risk['skill_risk'] = risk_skill(df['skill'])
df_risk['accident_history_risk'] = risk_accident_history(df['accident_history'])
df_risk['violation_history_risk'] = risk_violation_history(df['violation_history'])
df_risk['distance_risk'] = risk_distance(df['distance'])
df_risk['stay_time_risk'] = risk_stay_time(df['stay_time'])
df_risk['overlap_risk'] = risk_overlap(df['overlap'])
df_risk['authorized_risk'] = risk_authorized(df['authorized'])
df_risk['ppe_fastened_risk'] = risk_ppe_fastened(df['ppe_fastened'])
df_risk['protection_eff_risk'] = risk_protection_eff(df['protection_eff'])
df_risk['visibility_risk'] = risk_visibility(df['visibility'])
df_risk['sign_visible_risk'] = risk_sign_visible(df['sign_visible'])
df_risk['alert_perceptible_risk'] = risk_alert_perceptible(df['alert_perceptible'])
df_risk['hazard_type_risk'] = risk_hazard_type(df['hazard_type'])
# impact_range风险分依赖于行内hazard_type
df_risk['impact_range_risk'] = df.apply(risk_impact_range, axis=1)
df_risk['stability_margin_risk'] = risk_stability_margin(df['stability_margin'])
df_risk['device_abnormality_risk'] = risk_device_abnormality(df['device_abnormality'])
df_risk['sensor_coverage_risk'] = risk_sensor_coverage(df['sensor_coverage'])
df_risk['alarm_effect_risk'] = risk_alarm_effect(df['alarm_effect'])
df_risk['wind_speed_risk'] = risk_wind_speed(df['wind_speed'])
df_risk['illumination_risk'] = risk_illumination(df['illumination'])

# 4. 保存到新CSV文件
df_risk.to_csv('05.vine_copula/02.risk_scores.csv', index=False)

print("风险分计算完成，已保存至 05.vine_copula/02.risk_scores.csv")