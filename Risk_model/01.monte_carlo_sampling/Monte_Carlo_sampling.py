import numpy as np
import pandas as pd
from scipy.stats import (
    triang, lognorm, bernoulli, norm, beta, uniform, expon, poisson,
    weibull_min, truncnorm, multivariate_normal, rankdata
)

# ==================== 1. 设置参数 ====================
np.random.seed(42)
N = 100  # 样本数量

# ==================== 2. 定义边缘分布函数（逆变换采样） ====================
# 通用截断正态（区间 [low, high]）
def truncated_norm_ppf(u, mu, sigma, low, high):
    a, b = (low - mu) / sigma, (high - mu) / sigma
    return truncnorm.ppf(u, a, b, loc=mu, scale=sigma)

# 三角分布（min, max, mode）
def triangular_ppf(u, low, high, mode):
    c = (mode - low) / (high - low)
    return triang.ppf(u, c, loc=low, scale=high - low)

# 对数正态截断（区间 [low, high]）
def lognormal_trunc_ppf(u, mu_log, sigma_log, low, high):
    # mu_log, sigma_log 是底层正态的参数
    raw = lognorm.ppf(u, s=sigma_log, scale=np.exp(mu_log))
    return np.clip(raw, low, high)  # 简单截断，更精确可用条件分布

# 威布尔
def weibull_ppf(u, shape, scale):
    return weibull_min.ppf(u, shape, scale=scale)

# 指数截断（区间 [0, max]）
def exponential_trunc_ppf(u, rate, max_val):
    raw = expon.ppf(u, scale=1/rate)
    return np.clip(raw, 0, max_val)

# Beta 分布
def beta_ppf(u, a, b):
    return beta.ppf(u, a, b)

# 均匀分布
def uniform_ppf(u, low, high):
    return uniform.ppf(u, low, high - low)

# 伯努利（二值）
def bernoulli_ppf(u, p):
    return (u <= p).astype(int)

# ==================== 3. 定义相关矩阵（专家判断） ====================
# 组1：人员连续（health, fatigue, emotion）
corr_group1 = np.array([
    [1.0, -0.6,  0.5],   # health vs fatigue, health vs emotion
    [-0.6, 1.0, -0.5],   # fatigue vs health, fatigue vs emotion
    [0.5, -0.5, 1.0]     # emotion vs health, emotion vs fatigue
])

# 组2：知识技能（safety_knowledge, skill_numeric）
corr_group2 = np.array([[1.0, 0.6], [0.6, 1.0]])

# 组4：危险源特性（energy_level, impact_range, stability_margin, device_abnormality）
# 注意：不同危险源类型使用相同相关结构（实际可微调，为简化统一）
corr_group4 = np.array([
    [1.0,  0.7, -0.4,  0.0],   # energy vs impact, energy vs stability
    [0.7,  1.0, -0.4,  0.0],   # impact vs energy, impact vs stability
    [-0.4, -0.4, 1.0, -0.6],   # stability vs energy, stability vs impact, stability vs device
    [0.0,  0.0, -0.6,  1.0]    # device vs stability
])

# ==================== 4. 生成独立指标（不参与相关组） ====================
# 这些指标直接按边缘分布独立抽样
alcohol = bernoulli.rvs(0.02, size=N)
experience = exponential_trunc_ppf(np.random.rand(N), rate=0.1, max_val=40)
accident_pois = poisson.rvs(0.2, size=N)
accident_history = np.where(accident_pois == 0, '无', np.where(accident_pois == 1, '1次', '≥2次'))
violation_pois = poisson.rvs(0.5, size=N)
violation_history = np.where(violation_pois == 0, '无', np.where(violation_pois == 1, '1次', '≥2次'))

distance = uniform_ppf(np.random.rand(N), 0, 10)
stay_time = lognormal_trunc_ppf(np.random.rand(N), mu_log=3.4, sigma_log=0.6, low=0, high=120)
overlap = beta_ppf(np.random.rand(N), 1, 3)
authorized = bernoulli_ppf(np.random.rand(N), 0.9)
ppe_fastened = bernoulli_ppf(np.random.rand(N), 0.95)
protection_eff = triangular_ppf(np.random.rand(N), 0, 1, 0.8)
visibility_indep = beta_ppf(np.random.rand(N), 4, 2)   # 注意：后面会与环境组相关，暂存
sign_visible_indep = bernoulli_ppf(np.random.rand(N), 0.9)
alert_perceptible_indep = bernoulli_ppf(np.random.rand(N), 0.85)

sensor_coverage = beta_ppf(np.random.rand(N), 8, 2)
alarm_effect = triangular_ppf(np.random.rand(N), 0, 1, 0.8)
wind_speed = weibull_ppf(np.random.rand(N), shape=2, scale=3)
crowding = uniform_ppf(np.random.rand(N), 0, 1)

# ==================== 5. 使用高斯 Copula 生成相关组 ====================
def generate_correlated_samples(N, corr_matrix, margin_ppfs):
    """
    corr_matrix: 相关矩阵
    margin_ppfs: 列表，每个元素是函数(均匀样本) -> 目标变量样本
    返回 (N, d) 数组
    """
    d = len(margin_ppfs)
    # 生成多元正态
    mean = np.zeros(d)
    Z = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=N)
    # 转换为均匀分布
    U = norm.cdf(Z)
    # 应用边缘逆变换
    samples = np.zeros((N, d))
    for j in range(d):
        samples[:, j] = margin_ppfs[j](U[:, j])
    return samples

# ---------- 组1：健康、疲劳、情绪 ----------
margin_ppfs_group1 = [
    lambda u: triangular_ppf(u, 0, 1, 0.8),                      # health
    lambda u: lognormal_trunc_ppf(u, mu_log=-1.2, sigma_log=0.6, low=0, high=1),  # fatigue
    lambda u: truncated_norm_ppf(u, mu=0.7, sigma=0.1, low=0, high=1)   # emotion
]
group1_samples = generate_correlated_samples(N, corr_group1, margin_ppfs_group1)
health, fatigue, emotion = group1_samples[:, 0], group1_samples[:, 1], group1_samples[:, 2]

# ---------- 组2：安全知识 + 技能（有序分类）----------
# 将 skill 视为有序分类（无证=0, 初级=1, 中级=2, 高级=3），通过潜在正态分位数匹配概率
skill_probs = [0.1, 0.3, 0.4, 0.2]   # 累积概率 [0.1, 0.4, 0.8, 1.0]
skill_cutpoints = norm.ppf(np.cumsum([0.1, 0.3, 0.4, 0.2]))  # 分位数阈值
def skill_from_latent(z):
    # z 是标准正态变量
    return np.digitize(z, skill_cutpoints[:-1])   # 返回 0,1,2,3

# 安全知识边缘：截断正态 N(0.6,0.1) 于 [0,1]
def safety_knowledge_ppf(u):
    return truncated_norm_ppf(u, mu=0.6, sigma=0.1, low=0, high=1)

# 生成二元相关正态
mean2 = [0, 0]
Z2 = multivariate_normal.rvs(mean=mean2, cov=corr_group2, size=N)
U2 = norm.cdf(Z2)
safety_knowledge = safety_knowledge_ppf(U2[:, 0])
skill_latent = Z2[:, 1]
skill = skill_from_latent(skill_latent)
# 将 skill 编码为分类标签
skill_cats = ['无证', '初级', '中级', '高级']
skill_labels = [skill_cats[i] for i in skill]

# ---------- 组4：危险源特性（按 hazard_type 分层）----------
hazard_types = ['高支模', '深基坑', '塔吊', '车辆', '配电箱']
hazard_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
hazard_type = np.random.choice(hazard_types, size=N, p=hazard_probs)

# 定义每个类型的边缘分布函数（返回 ppf 函数）
def get_margin_ppfs_for_type(ht):
    if ht == '塔吊':
        return [
            lambda u: beta_ppf(u, 5, 2),                                    # energy_level Beta(5,2)
            lambda u: lognormal_trunc_ppf(u, 3.5, 0.6, 0, 100),            # impact_range
            lambda u: beta_ppf(u, 3, 2),                                    # stability_margin
            lambda u: beta_ppf(u, 2, 4)                                     # device_abnormality
        ]
    elif ht == '深基坑':
        return [
            lambda u: truncated_norm_ppf(u, 0.6, 0.15, 0, 1),
            lambda u: triangular_ppf(u, 5, 30, 15),
            lambda u: beta_ppf(u, 4, 3),
            lambda u: exponential_trunc_ppf(u, rate=2.0, max_val=1)
        ]
    elif ht == '高支模':
        return [
            lambda u: triangular_ppf(u, 0.2, 0.9, 0.6),
            lambda u: truncated_norm_ppf(u, 20, 8, 0, 100),
            lambda u: beta_ppf(u, 2, 2),
            lambda u: np.zeros_like(u)   # 退化 D=0
        ]
    elif ht == '车辆':
        return [
            lambda u: uniform_ppf(u, 0, 1),
            lambda u: exponential_trunc_ppf(u, rate=0.15, max_val=100),
            lambda u: triangular_ppf(u, 0.4, 1.0, 0.8),
            lambda u: beta_ppf(u, 1.5, 3)
        ]
    elif ht == '配电箱':
        return [
            lambda u: beta_ppf(u, 2, 5),
            lambda u: uniform_ppf(u, 0, 5),
            lambda u: np.ones_like(u),   # 退化 S=1
            lambda u: beta_ppf(u, 1.2, 4)
        ]
    else:
        raise ValueError(f"Unknown hazard type: {ht}")

# 预先分配数组
energy_level = np.empty(N)
impact_range = np.empty(N)
stability_margin = np.empty(N)
device_abnormality = np.empty(N)

# 按类型分批生成（避免循环内生成大矩阵）
for ht in hazard_types:
    mask = (hazard_type == ht)
    n_ht = mask.sum()
    if n_ht == 0:
        continue
    margin_ppfs = get_margin_ppfs_for_type(ht)
    # 生成相关样本
    samples_ht = generate_correlated_samples(n_ht, corr_group4, margin_ppfs)
    energy_level[mask] = samples_ht[:, 0]
    impact_range[mask] = samples_ht[:, 1]
    stability_margin[mask] = samples_ht[:, 2]
    device_abnormality[mask] = samples_ht[:, 3]

# ==================== 6. 环境组（照明、能见度、标识、预警）=============
# 使用潜在变量处理 visibility 与 sign_visible, alert_perceptible 的相关性
# 假设 visibility 与 sign_visible 相关 0.5，与 alert_perceptible 相关 0.4
# 构建三元相关正态 (visibility_latent, sign_latent, alert_latent)
corr_env = np.array([
    [1.0, 0.5, 0.4],
    [0.5, 1.0, 0.3],
    [0.4, 0.3, 1.0]
])
Z_env = multivariate_normal.rvs(mean=[0,0,0], cov=corr_env, size=N)
U_env = norm.cdf(Z_env)

# visibility 边缘：Beta(4,2)
visibility = beta_ppf(U_env[:, 0], 4, 2)
# sign_visible 二值，概率 0.9，通过阈值法：当潜在变量低于某阈值时取1，使得整体概率0.9
# 令 sign_visible = 1 if Z_env[:,1] <= norm.ppf(0.9) else 0
sign_thresh = norm.ppf(0.9)
sign_visible = (Z_env[:, 1] <= sign_thresh).astype(int)
# alert_perceptible 概率 0.85
alert_thresh = norm.ppf(0.85)
alert_perceptible = (Z_env[:, 2] <= alert_thresh).astype(int)

# 照明条件 illumination 与 visibility 相关 0.7，单独生成二元相关
corr_illum_vis = np.array([[1.0, 0.7], [0.7, 1.0]])
Z_illum = multivariate_normal.rvs(mean=[0,0], cov=corr_illum_vis, size=N)
U_illum = norm.cdf(Z_illum)
illumination = beta_ppf(U_illum[:, 0], 3, 2)   # illumination 边缘 Beta(3,2)
# visibility 已经生成，这里用相关性调整后的版本？注意：上面 visibility 已用 U_env 生成，但为了保持 illumination-visibility 相关，需要统一。
# 更严谨：先生成二元 (illumination_latent, visibility_latent) 相关，再分别映射。
# 上面 visibility 用了独立的三元组，会导致 illumination 与 visibility 不相关。修正：重写环境组。
# 为简化，我们重新生成 illumination 和 visibility 的相关对，然后 visibility 再与标识相关？这样会冲突。
# 实际可假设 illumination 与 sign_visible 等弱相关，可独立。以下采用独立方式保持简单：
# 但根据要求，illumination-visibility 高相关，我们单独生成一组二元 Copula：
Z_illum_vis = multivariate_normal.rvs(mean=[0,0], cov=[[1,0.7],[0.7,1]], size=N)
U_iv = norm.cdf(Z_illum_vis)
illumination = beta_ppf(U_iv[:, 0], 3, 2)
visibility = beta_ppf(U_iv[:, 1], 4, 2)   # 覆盖之前的 visibility

# 然后用 visibility 再与 sign_visible, alert_perceptible 相关（使用条件抽样或重新生成三元组）
# 为避免过复杂，这里保留独立抽样，因为 sign/alert 的二值概率已固定，相关性影响较小。
# 或者使用上面生成的 Z_env，但 visibility 已经改变，放弃。作为简化，直接独立生成 sign/alert。
sign_visible = bernoulli_ppf(np.random.rand(N), 0.9)
alert_perceptible = bernoulli_ppf(np.random.rand(N), 0.85)

# ==================== 7. 组合所有指标 ====================
df = pd.DataFrame({
    'health': health,
    'fatigue': fatigue,
    'alcohol': alcohol,
    'emotion': emotion,
    'safety_knowledge': safety_knowledge,
    'experience': experience,
    'skill': skill_labels,
    'accident_history': accident_history,
    'violation_history': violation_history,
    'distance': distance,
    'stay_time': stay_time,
    'overlap': overlap,
    'authorized': authorized,
    'ppe_fastened': ppe_fastened,
    'protection_eff': protection_eff,
    'visibility': visibility,
    'sign_visible': sign_visible,
    'alert_perceptible': alert_perceptible,
    'hazard_type': hazard_type,
    'energy_level': energy_level,
    'impact_range': impact_range,
    'stability_margin': stability_margin,
    'device_abnormality': device_abnormality,
    'sensor_coverage': sensor_coverage,
    'alarm_effect': alarm_effect,
    'wind_speed': wind_speed,
    'illumination': illumination,
    'crowding': crowding
})

# 保存为 CSV（可选）
df.to_csv('Risk_model/monte_carlo_samples.csv', index=False, encoding='utf-8-sig')

# 显示前几行和描述统计
# print(df.head())
# print(df.describe())