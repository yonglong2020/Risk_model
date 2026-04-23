"""
安全事故风险指标体系联合分布建模 (Vine Copula)
================================================
本代码实现以下功能：
1. 根据DEMATEL综合影响矩阵构造相关矩阵，生成伪观测数据。
2. 定义所有变量的边缘分布（包括连续、二值、有序分类以及条件混合分布）。
3. 使用pyvinecopulib自动选择R-Vine结构并拟合Copula。
4. 从拟合的Vine Copula中采样（N=10000），并转换为原始变量取值。
5. 输出采样结果CSV文件，并可视化部分变量对的散点图。

注意：需要预先安装 pyvinecopulib 和 scipy, pandas, numpy, matplotlib 等库。
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from pyvinecopulib import Vinecop, BicopFamily, FitControlsVinecop
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- 辅助函数 ---------------------------------
def near_psd(cov, epsilon=1e-8, max_iter=100):
    """
    将协方差矩阵修正为最接近的正定矩阵（基于特征值调整）。
    """
    cov = np.asarray(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, epsilon)
    corrected = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # 对称化处理
    corrected = (corrected + corrected.T) / 2
    return corrected

# ----------------------------- 边缘分布类 ---------------------------------
class TruncatedDistribution:
    """截断连续分布（基于scipy.stats的rv_continuous）"""
    def __init__(self, dist, a, b, **params):
        self.dist = dist
        self.a = a
        self.b = b
        self.params = params
        # 计算截断后的CDF缩放因子
        self.cdf_a = dist.cdf(a, **params)
        self.cdf_b = dist.cdf(b, **params)

    def cdf(self, x):
        """累积分布函数 P(X <= x)"""
        x_clip = np.clip(x, self.a, self.b)
        cdf_val = self.dist.cdf(x_clip, **self.params)
        return (cdf_val - self.cdf_a) / (self.cdf_b - self.cdf_a)

    def ppf(self, q):
        """分位数函数（逆CDF）"""
        q = np.clip(q, 1e-10, 1 - 1e-10)
        # 映射回原始分布的累积概率
        q_orig = self.cdf_a + q * (self.cdf_b - self.cdf_a)
        return self.dist.ppf(q_orig, **self.params)

class BernoulliDistribution:
    """伯努利分布（用于二值变量）"""
    def __init__(self, p):
        self.p = p

    def cdf(self, x):
        # x: 0 或 1 的取值（或其他值）
        x = np.asarray(x)
        res = np.zeros_like(x, dtype=float)
        res[x < 0] = 0.0
        res[(x >= 0) & (x < 1)] = 1 - self.p
        res[x >= 1] = 1.0
        return res

    def ppf(self, q):
        # 均匀分位数映射回0/1
        q = np.asarray(q)
        return (q > 1 - self.p).astype(float)

class CategoricalDistribution:
    """有序分类分布（假设类别为0,1,...,K-1，按给定概率）"""
    def __init__(self, probs):
        self.probs = np.asarray(probs)
        self.cumprobs = np.cumsum(self.probs)

    def cdf(self, x):
        # x 为类别整数
        x = np.asarray(x)
        res = np.zeros_like(x, dtype=float)
        for k, cp in enumerate(self.cumprobs):
            res[x >= k] = cp
        return res

    def ppf(self, q):
        # q ∈ [0,1] 映射到类别
        q = np.asarray(q)
        res = np.zeros_like(q, dtype=int)
        for k, cp in enumerate(self.cumprobs):
            res[q <= cp] = k
        return res

class MixtureDistribution:
    """
    混合分布：用于 impact_range, stability_margin, device_abnormality。
    根据 hazard_type 的先验概率（均匀）加权各组分分布。
    """
    def __init__(self, component_dists, weights=None):
        """
        component_dists: list of distribution objects, 每个对象必须有cdf和ppf方法
        weights: 各组分权重，默认等权
        """
        self.component_dists = component_dists
        self.n_comp = len(component_dists)
        if weights is None:
            self.weights = np.ones(self.n_comp) / self.n_comp
        else:
            self.weights = np.asarray(weights) / np.sum(weights)

    def cdf(self, x):
        # 加权求和
        x = np.asarray(x)
        cdf_val = np.zeros_like(x, dtype=float)
        for w, dist in zip(self.weights, self.component_dists):
            cdf_val += w * dist.cdf(x)
        return cdf_val

    def ppf(self, q):
        """
        通过数值求解逆CDF（因为混合分布没有解析逆）。
        采用二分法，对于标量q可快速求解。
        """
        q = np.asarray(q)
        # 处理标量
        if q.ndim == 0:
            if q <= 0:
                return self.component_dists[0].ppf(0.0)
            if q >= 1:
                return self.component_dists[-1].ppf(1.0)
            # 寻找根: cdf(x) - q = 0
            # 先确定搜索区间
            low = min(dist.ppf(0.0) for dist in self.component_dists)
            high = max(dist.ppf(1.0) for dist in self.component_dists)
            # 若区间内函数值同号，适当扩展
            while self.cdf(low) > q:
                low = low - (high - low) * 0.1
            while self.cdf(high) < q:
                high = high + (high - low) * 0.1
            return brentq(lambda x: self.cdf(x) - q, low, high)
        else:
            # 向量化处理（逐元素）
            res = np.zeros_like(q)
            for i, qi in enumerate(q):
                res[i] = self.ppf(qi)  # 递归调用标量版本
            return res

# ----------------------------- 构造相关矩阵 ---------------------------------
# 原始DEMATEL综合影响矩阵（26x26，文本格式）
# 变量顺序已给出
var_names = [
    'health', 'fatigue', 'alcohol', 'emotion', 'safety_knowledge',
    'experience', 'skill', 'accident_history', 'violation_history',
    'distance', 'stay_time', 'overlap', 'authorized', 'ppe_fastened',
    'protection_eff', 'visibility', 'sign_visible', 'alert_perceptible',
    'hazard_type', 'impact_range', 'stability_margin', 'device_abnormality',
    'sensor_coverage', 'alarm_effect', 'wind_speed', 'illumination'
]
n_vars = len(var_names)

# 矩阵数据（从用户提供的文本构造，注意仅给出非零值，零值已填充）
# 为简洁，直接使用numpy数组（手动录入，确保与表格一致）
# 这里使用硬编码，实际可从文件读取
dematel_matrix = np.zeros((n_vars, n_vars))
# 按行填充（变量索引按上述顺序 0~25）
# health(0) -> fatigue(1):0.184797703, emotion(3):0.161088678, authorized(12):0.137910566
dematel_matrix[0,1] = 0.184797703
dematel_matrix[0,3] = 0.161088678
dematel_matrix[0,12] = 0.137910566
# fatigue(1) -> health(0):0.15925203, emotion(3):0.219140477, ppe_fastened(13):0.133637465
dematel_matrix[1,0] = 0.15925203
dematel_matrix[1,3] = 0.219140477
dematel_matrix[1,13] = 0.133637465
# alcohol(2) -> health(0):0.175402635, fatigue(1):0.141112316, emotion(3):0.260013473, ppe_fastened(13):0.193897663
dematel_matrix[2,0] = 0.175402635
dematel_matrix[2,1] = 0.141112316
dematel_matrix[2,3] = 0.260013473
dematel_matrix[2,13] = 0.193897663
# emotion(3) -> health(0):0.135032985, ppe_fastened(13):0.141909135
dematel_matrix[3,0] = 0.135032985
dematel_matrix[3,13] = 0.141909135
# safety_knowledge(4) -> ppe_fastened(13):0.156747593
dematel_matrix[4,13] = 0.156747593
# experience(5) -> safety_knowledge(4):0.184546868, skill(6):0.163796114, ppe_fastened(13):0.142141318, visibility(15):0.143879579
dematel_matrix[5,4] = 0.184546868
dematel_matrix[5,6] = 0.163796114
dematel_matrix[5,13] = 0.142141318
dematel_matrix[5,15] = 0.143879579
# skill(6) -> safety_knowledge(4):0.127277685, ppe_fastened(13):0.132388009, protection_eff(14):0.125451211
dematel_matrix[6,4] = 0.127277685
dematel_matrix[6,13] = 0.132388009
dematel_matrix[6,14] = 0.125451211
# violation_history(8) -> authorized(12):0.114972617, ppe_fastened(13):0.162498195
dematel_matrix[8,12] = 0.114972617
dematel_matrix[8,13] = 0.162498195
# distance(9) -> protection_eff(14):0.127272727, alert_perceptible(17):0.127272727
dematel_matrix[9,14] = 0.127272727
dematel_matrix[9,17] = 0.127272727
# stay_time(10) -> fatigue(1):0.118713098
dematel_matrix[10,1] = 0.118713098
# hazard_type(18) -> distance(9):0.133188742, protection_eff(14):0.234677722, impact_range(19):0.2
dematel_matrix[18,9] = 0.133188742
dematel_matrix[18,14] = 0.234677722
dematel_matrix[18,19] = 0.2
# impact_range(19) -> distance(9):0.135437799
dematel_matrix[19,9] = 0.135437799
# stability_margin(20) -> protection_eff(14):0.145581044
dematel_matrix[20,14] = 0.145581044
# sensor_coverage(22) -> alarm_effect(24):0.2
dematel_matrix[22,24] = 0.2
# alarm_effect(23) -> alert_perceptible(17):0.163636364
dematel_matrix[23,17] = 0.163636364
# illumination(25) -> visibility(15):0.182535438, sign_visible(16):0.182306354
dematel_matrix[25,15] = 0.182535438
dematel_matrix[25,16] = 0.182306354

# 构造对称相关矩阵：取 (M + M^T)/2，对角线置1
corr_raw = (dematel_matrix + dematel_matrix.T) / 2
np.fill_diagonal(corr_raw, 1.0)
# 确保矩阵半正定（修正负特征值）
corr_matrix = near_psd(corr_raw)

# ----------------------------- 定义各变量的边缘分布 ---------------------------------
# 1. 连续变量
health_dist = TruncatedDistribution(stats.triang, 0, 1, c=0.8)   # Triangular(0,1,0.8) 参数c是mode
fatigue_dist = TruncatedDistribution(stats.lognorm, 0, 1, s=0.6, scale=np.exp(-1.2))  # LogNormal(μ=-1.2, σ=0.6)
alcohol_dist = BernoulliDistribution(p=0.02)   # 二值
emotion_dist = TruncatedDistribution(stats.norm, 0, 1, loc=0.7, scale=0.1)
safety_knowledge_dist = TruncatedDistribution(stats.norm, 0, 1, loc=0.6, scale=0.1)
experience_dist = TruncatedDistribution(stats.expon, 0, 40, scale=1/0.1)  # λ=0.1
skill_dist = CategoricalDistribution(probs=[0.1, 0.3, 0.4, 0.2])   # 类别:0=无证,1=初级,2=中级,3=高级
accident_history_dist = CategoricalDistribution(probs=[0.8187, 0.1637, 0.0175])  # 0=无,1=1次,2=≥2次
violation_history_dist = CategoricalDistribution(probs=[0.6065, 0.3033, 0.0902])
distance_dist = stats.uniform(loc=0, scale=10)  # Uniform(0,10)
stay_time_dist = TruncatedDistribution(stats.lognorm, 0, 120, s=0.6, scale=np.exp(3.4))
overlap_dist = stats.beta(a=1, b=3)   # Beta(1,3)
authorized_dist = BernoulliDistribution(p=0.9)
ppe_fastened_dist = BernoulliDistribution(p=0.95)
protection_eff_dist = TruncatedDistribution(stats.triang, 0, 1, c=0.8)  # Triangular(0,1,0.8)
visibility_dist = stats.beta(a=4, b=2)   # Beta(4,2)
sign_visible_dist = BernoulliDistribution(p=0.9)
alert_perceptible_dist = BernoulliDistribution(p=0.85)
# hazard_type: 五类等概率（视为有序分类，类别0~4）
hazard_type_dist = CategoricalDistribution(probs=[0.2, 0.2, 0.2, 0.2, 0.2])

# 2. 条件分布组件（用于混合分布）
# hazard_type类别顺序: 0=高支模,1=深基坑,2=塔吊,3=车辆,4=配电箱
# 定义各组分分布
# impact_range 组分
impact_comp = [
    TruncatedDistribution(stats.norm, 0, 100, loc=20, scale=8),                     # 高支模
    stats.triang(c=(15-5)/(30-5), loc=5, scale=25),                                 # 深基坑 (Triangular)
    TruncatedDistribution(stats.lognorm, 0, 100, s=0.6, scale=np.exp(3.5)),         # 塔吊
    TruncatedDistribution(stats.expon, 0, 100, scale=1/0.15),                       # 车辆
    stats.uniform(loc=0, scale=5)                                                   # 配电箱
]
# stability_margin 组分
stability_comp = [
    stats.beta(a=2, b=2),            # 高支模 Beta(2,2)
    stats.beta(a=4, b=3),            # 深基坑 Beta(4,3)
    stats.beta(a=3, b=2),            # 塔吊 Beta(3,2)
    stats.triang(c=(0.8-0.4)/(1.0-0.4), loc=0.4, scale=0.6), # 车辆 Triangular(0.4,1,0.8)
    stats.beta(a=1, b=1)             # 配电箱 常数1 => Beta(1,1) 退化
]
# device_abnormality 组分
device_comp = [
    stats.beta(a=0, b=0),            # 高支模 常数0 => 退化分布，使用Dirac近似
    TruncatedDistribution(stats.expon, 0, 1, scale=1/2.0),  # 深基坑 Exponential(λ=2)
    stats.beta(a=2, b=4),            # 塔吊 Beta(2,4)
    stats.beta(a=1.5, b=3),          # 车辆 Beta(1.5,3)
    stats.beta(a=1.2, b=4)           # 配电箱 Beta(1.2,4)
]
# 对于高支模的 device_abnormality = 0，用几乎退化的Beta分布替代（Beta(0.001,1000)）
device_comp[0] = stats.beta(a=0.001, b=1000)

# 构造混合分布（等权重，即 hazard_type 先验均匀）
impact_range_dist = MixtureDistribution(impact_comp, weights=[0.2]*5)
stability_margin_dist = MixtureDistribution(stability_comp, weights=[0.2]*5)
device_abnormality_dist = MixtureDistribution(device_comp, weights=[0.2]*5)

# 剩余连续变量
sensor_coverage_dist = stats.beta(a=8, b=2)
alarm_effect_dist = TruncatedDistribution(stats.triang, 0, 1, c=0.8)  # Triangular(0,1,0.8)
wind_speed_dist = stats.weibull_min(c=2, scale=3)  # Weibull(shape=2, scale=3)
illumination_dist = stats.beta(a=3, b=2)

# 将分布按变量顺序放入列表
dist_list = [
    health_dist, fatigue_dist, alcohol_dist, emotion_dist, safety_knowledge_dist,
    experience_dist, skill_dist, accident_history_dist, violation_history_dist,
    distance_dist, stay_time_dist, overlap_dist, authorized_dist, ppe_fastened_dist,
    protection_eff_dist, visibility_dist, sign_visible_dist, alert_perceptible_dist,
    hazard_type_dist, impact_range_dist, stability_margin_dist, device_abnormality_dist,
    sensor_coverage_dist, alarm_effect_dist, wind_speed_dist, illumination_dist
]

# ----------------------------- 生成伪观测数据 ---------------------------------
np.random.seed(42)
N_pseudo = 10000  # 伪观测样本量

# 1. 从多元正态采样（均值0，协方差=相关矩阵）
mean = np.zeros(n_vars)
samples_norm = np.random.multivariate_normal(mean, corr_matrix, N_pseudo)

# 2. 概率积分变换（PIT）得到均匀分布
U_pseudo = np.zeros_like(samples_norm)
for i in range(n_vars):
    U_pseudo[:, i] = stats.norm.cdf(samples_norm[:, i])

# 注意：对于离散变量，需要将均匀U映射为离散实际值，然后再通过jittering生成连续的均匀值（用于拟合copula）
# 更符合隐式连续化：先根据U_pseudo映射到离散值，然后添加均匀噪声得到新的U_fit
U_fit = np.zeros_like(U_pseudo)
X_pseudo = np.zeros_like(U_pseudo)  # 存储原始实际值（用于验证）
for i, dist in enumerate(dist_list):
    # 从U映射到实际变量值
    X_pseudo[:, i] = dist.ppf(U_pseudo[:, i])
    # 对于离散分布，添加jittering（均匀噪声）以获得连续的拟合用U
    if isinstance(dist, (BernoulliDistribution, CategoricalDistribution)):
        # 对每个样本，根据实际取值确定区间，在区间内均匀采样
        for j in range(N_pseudo):
            x_val = X_pseudo[j, i]
            # 计算cdf左端点
            if isinstance(dist, BernoulliDistribution):
                if x_val < 0.5:
                    # 类别0
                    U_fit[j, i] = np.random.uniform(0, 1 - dist.p)
                else:
                    U_fit[j, i] = np.random.uniform(1 - dist.p, 1)
            else:  # Categorical
                # 找到类别对应的区间 [cumprob[k-1], cumprob[k])
                cum = dist.cumprobs
                # 确保x_val是整数类别
                cat = int(x_val)
                if cat == 0:
                    low = 0.0
                else:
                    low = cum[cat-1]
                high = cum[cat]
                U_fit[j, i] = np.random.uniform(low, high)
    else:
        # 连续变量直接使用PIT后的均匀值
        U_fit[:, i] = U_pseudo[:, i]

# ----------------------------- 拟合R-Vine Copula ---------------------------------
# 使用pyvinecopulib自动选择树结构和双变量copula族
families = [BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel,
            BicopFamily.frank, BicopFamily.joe, BicopFamily.student]
# 注意：pyvinecopulib 要求输入为 numpy 数组，形状 (n_obs, n_vars)
# 1. 配置 Copula 族和截断等级
controls = FitControlsVinecop(
    family_set=[BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel,
                BicopFamily.frank, BicopFamily.joe, BicopFamily.student],
    select_trunc_lvl=True   # 自动选择最优截断等级
)
# 2. 拟合 R-Vine Copula
vc = Vinecop.from_data(U_fit, controls=controls)
print("Vine Copula 拟合完成。")
print(f"树结构: {vc.trunc_lvl} 层")

# ----------------------------- 蒙特卡洛采样 ---------------------------------
N_sample = 10000
U_sample = vc.simulate(N_sample)  # 得到均匀分布样本

# 转换为原始变量取值
samples_final = np.zeros((N_sample, n_vars))
for i, dist in enumerate(dist_list):
    # 直接使用ppf转换
    samples_final[:, i] = dist.ppf(U_sample[:, i])

# 创建DataFrame
df_samples = pd.DataFrame(samples_final, columns=var_names)

# 后处理：将分类变量转换为类别标签（方便阅读）
# skill: 0=无证,1=初级,2=中级,3=高级
skill_labels = ['无证', '初级', '中级', '高级']
df_samples['skill'] = df_samples['skill'].astype(int).map(lambda x: skill_labels[x])
# accident_history: 0=无,1=1次,2=≥2次
acc_labels = ['无', '1次', '≥2次']
df_samples['accident_history'] = df_samples['accident_history'].astype(int).map(lambda x: acc_labels[x])
# violation_history: 0=无,1=1次,2=≥2次
df_samples['violation_history'] = df_samples['violation_history'].astype(int).map(lambda x: acc_labels[x])
# hazard_type: 0=高支模,1=深基坑,2=塔吊,3=车辆,4=配电箱
hazard_labels = ['高支模', '深基坑', '塔吊', '车辆', '配电箱']
df_samples['hazard_type'] = df_samples['hazard_type'].astype(int).map(lambda x: hazard_labels[x])
# alcohol, authorized, ppe_fastened, sign_visible, alert_perceptible 二值映射
binary_vars = ['alcohol', 'authorized', 'ppe_fastened', 'sign_visible', 'alert_perceptible']
for var in binary_vars:
    df_samples[var] = df_samples[var].astype(int).map({0: '否', 1: '是'})

# 保存CSV
df_samples.to_csv('vine_copula_samples.csv', index=False)
print(f"采样完成，样本已保存至 vine_copula_samples.csv (共{N_sample}行)")

# ----------------------------- 可视化部分变量对 ---------------------------------
# 选取几对代表性变量绘制散点图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plot_pairs = [
    ('fatigue', 'health'),
    ('alcohol', 'ppe_fastened'),
    ('skill', 'experience'),
    ('hazard_type', 'impact_range'),
    ('visibility', 'illumination'),
    ('sensor_coverage', 'alarm_effect')
]
for ax, (xname, yname) in zip(axes.flat, plot_pairs):
    # 对于分类/二值变量，添加小抖动以便观察
    x_data = df_samples[xname]
    y_data = df_samples[yname]
    if x_data.dtype == 'object' and xname in binary_vars + ['skill', 'accident_history', 'violation_history', 'hazard_type']:
        # 转换为数值并添加抖动
        if xname in binary_vars:
            x_num = x_data.map({'否':0, '是':1}).astype(float)
        elif xname == 'skill':
            x_num = x_data.map({l:i for i,l in enumerate(skill_labels)}).astype(float)
        elif xname == 'hazard_type':
            x_num = x_data.map({l:i for i,l in enumerate(hazard_labels)}).astype(float)
        else:
            x_num = x_data.map({'无':0, '1次':1, '≥2次':2}).astype(float)
        x_jitter = x_num + np.random.normal(0, 0.05, size=len(x_num))
        ax.scatter(x_jitter, y_data, alpha=0.3, s=5)
    elif y_data.dtype == 'object':
        # 类似处理y
        if yname in binary_vars:
            y_num = y_data.map({'否':0, '是':1}).astype(float)
        elif yname == 'skill':
            y_num = y_data.map({l:i for i,l in enumerate(skill_labels)}).astype(float)
        elif yname == 'hazard_type':
            y_num = y_data.map({l:i for i,l in enumerate(hazard_labels)}).astype(float)
        else:
            y_num = y_data.map({'无':0, '1次':1, '≥2次':2}).astype(float)
        y_jitter = y_num + np.random.normal(0, 0.05, size=len(y_num))
        ax.scatter(x_data, y_jitter, alpha=0.3, s=5)
    else:
        ax.scatter(x_data, y_data, alpha=0.3, s=5)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title(f'{xname} vs {yname}')
plt.tight_layout()
plt.savefig('pair_plots.png', dpi=150)
plt.show()
print("散点图已保存至 pair_plots.png")