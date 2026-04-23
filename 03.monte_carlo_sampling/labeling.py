from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt

# ==================== 辅助函数：自动生成加权和节点的CPD ====================
def create_weighted_cpd(node_name, n_states, parents, parent_cards, weights, thresholds, noise=0.1):
    """
    生成基于加权和的条件概率表。
    node_name: 目标节点名
    n_states: 目标节点状态数（状态顺序: 0=低,1=中,2=高,3=极高）
    parents: 父节点列表
    parent_cards: 每个父节点的状态数（状态顺序必须为 0:低,1:中,2:高,3:极高）
    weights: 每个父节点的权重（总和不必为1，内部会归一化）
    thresholds: 等级划分阈值，长度为 n_states-1，例如 [0.2, 0.5, 0.8]
    noise: 不确定性，给目标等级的概率为 1-noise，其余均匀分配
    """
    # 归一化权重
    w = np.array(weights) / np.sum(weights)
    # 状态到分数的映射（低=0, 中=1/3, 高=2/3, 极高=1）
    state_score = {0: 0.0, 1: 1/3, 2: 2/3, 3: 1.0}
    
    total_comb = np.prod(parent_cards)
    cpd_values = np.zeros((n_states, total_comb))
    
    for idx, combo in enumerate(itertools.product(*[range(c) for c in parent_cards])):
        # 计算加权总分
        score = sum(w[i] * state_score[combo[i]] for i in range(len(parents)))
        # 确定等级
        level = 0
        for t in thresholds:
            if score >= t:
                level += 1
        # 填充概率
        for s in range(n_states):
            if s == level:
                cpd_values[s, idx] = 1 - noise
            else:
                cpd_values[s, idx] = noise / (n_states - 1)
    
    return TabularCPD(node_name, n_states, cpd_values,
                      evidence=parents, evidence_card=parent_cards)

# ==================== 定义网络结构 ====================
model = DiscreteBayesianNetwork([
    # 人员个体特质 → HumanFactor
    ('Health', 'HumanFactor'), ('Fatigue', 'HumanFactor'), ('Emotion', 'HumanFactor'),
    ('SafetyKnowledge', 'HumanFactor'), ('Experience', 'HumanFactor'), ('Skill', 'HumanFactor'),
    ('AccidentHistory', 'HumanFactor'), ('ViolationHistory', 'HumanFactor'),
    # 直接否决指标 → DirectTrigger
    ('Alcohol', 'DirectTrigger'), ('Authorized', 'DirectTrigger'), ('PpeFastened', 'DirectTrigger'),
    ('SignVisible', 'DirectTrigger'), ('AlertPerceptible', 'DirectTrigger'),
    # 暴露水平 → Exposure
    ('Distance', 'Exposure'), ('StayTime', 'Exposure'), ('Overlap', 'Exposure'), ('Crowding', 'Exposure'),
    # 物理隔离 → Isolation
    ('ProtectionEff', 'Isolation'),
    # 感知可达性 → Perception
    ('Visibility', 'Perception'), ('Illumination', 'Perception'),  # 照明影响感知
    # 危险源固有破坏力 → HazardIntensity
    ('EnergyLevel', 'HazardIntensity'), ('ImpactRange', 'HazardIntensity'),
    # 失控可能性 → HazardUncontrol
    ('StabilityMargin', 'HazardUncontrol'), ('DeviceAbnormality', 'HazardUncontrol'),
    # 管理环境 → Management
    ('SensorCoverage', 'Management'), ('AlarmEffect', 'Management'),
    # 以上中间节点 → RiskLevel
    ('HumanFactor', 'RiskLevel'), ('DirectTrigger', 'RiskLevel'), ('Exposure', 'RiskLevel'),
    ('Isolation', 'RiskLevel'), ('Perception', 'RiskLevel'), ('HazardIntensity', 'RiskLevel'),
    ('HazardUncontrol', 'RiskLevel'), ('Management', 'RiskLevel'),
    # RiskLevel → Accident
    ('RiskLevel', 'Accident')
])

# ==================== 根节点先验CPD ====================
# 根据您提供的分布参数设置离散状态概率（状态顺序：低,中,高,极高 或 0,1）
# 以下数值为示例，您可根据实际专家分布精确计算

# 人员个体特质（连续变量离散化后的概率）
cpd_health = TabularCPD('Health', 4, [[0.10], [0.30], [0.40], [0.20]])      # 健康程度
cpd_fatigue = TabularCPD('Fatigue', 4, [[0.45], [0.30], [0.15], [0.10]])     # 疲劳作业
cpd_emotion = TabularCPD('Emotion', 4, [[0.10], [0.30], [0.40], [0.20]])     # 情绪状态
cpd_safety_knowledge = TabularCPD('SafetyKnowledge', 4, [[0.20], [0.40], [0.30], [0.10]])  # 安全知识
# 经验（年）离散化：<5年(低), 5-15(中), >15(高) -> 3状态，这里映射为4状态(缺极高)
cpd_experience = TabularCPD('Experience', 4, [[0.30], [0.40], [0.30], [0.00]])   # 从业年限
# 技能（分类映射：无证->低,初级->中,中级->高,高级->极高）
cpd_skill = TabularCPD('Skill', 4, [[0.10], [0.30], [0.40], [0.20]])          # 资质水平
# 事故记录：无->低,1次->中,≥2次->高（无极高）
cpd_accident_history = TabularCPD('AccidentHistory', 4, [[0.82], [0.16], [0.02], [0.00]])
# 违规记录：无->低,1次->中,≥2次->高
cpd_violation_history = TabularCPD('ViolationHistory', 4, [[0.61], [0.30], [0.09], [0.00]])

# 直接否决指标（二值）
cpd_alcohol = TabularCPD('Alcohol', 2, [[0.98], [0.02]])          # 酒后作业
cpd_authorized = TabularCPD('Authorized', 2, [[0.90], [0.10]])    # 准入状态（1=已授权）
cpd_ppe_fastened = TabularCPD('PpeFastened', 2, [[0.95], [0.05]]) # 防护用品系挂（1=系挂）
cpd_sign_visible = TabularCPD('SignVisible', 2, [[0.90], [0.10]]) # 警示标识可见
cpd_alert_perceptible = TabularCPD('AlertPerceptible', 2, [[0.85], [0.15]]) # 预警可感知

# 暴露水平
cpd_distance = TabularCPD('Distance', 4, [[0.20], [0.30], [0.30], [0.20]])      # 距离（远,中,近,接触）
cpd_stay_time = TabularCPD('StayTime', 4, [[0.40], [0.35], [0.20], [0.05]])     # 逗留时间（短,中,长,超长）
cpd_overlap = TabularCPD('Overlap', 4, [[0.30], [0.40], [0.20], [0.10]])        # 重叠度
cpd_crowding = TabularCPD('Crowding', 4, [[0.25], [0.50], [0.20], [0.05]])      # 场地拥挤度

# 物理隔离
cpd_protection_eff = TabularCPD('ProtectionEff', 4, [[0.05], [0.20], [0.50], [0.25]])  # 防护有效性

# 感知可达性
cpd_visibility = TabularCPD('Visibility', 4, [[0.10], [0.30], [0.40], [0.20]])   # 环境辨识度
cpd_illumination = TabularCPD('Illumination', 4, [[0.15], [0.35], [0.40], [0.10]]) # 照明条件

# 危险源特性
cpd_energy_level = TabularCPD('EnergyLevel', 4, [[0.25], [0.40], [0.25], [0.10]])   # 能量等级
cpd_impact_range = TabularCPD('ImpactRange', 4, [[0.30], [0.30], [0.25], [0.15]])   # 波及范围
cpd_stability_margin = TabularCPD('StabilityMargin', 4, [[0.10], [0.30], [0.40], [0.20]])  # 稳定性裕度
cpd_device_abnormality = TabularCPD('DeviceAbnormality', 4, [[0.40], [0.35], [0.15], [0.10]]) # 设备异常度

# 环境因素
cpd_sensor_coverage = TabularCPD('SensorCoverage', 4, [[0.10], [0.20], [0.40], [0.30]])   # 监测覆盖率
cpd_alarm_effect = TabularCPD('AlarmEffect', 4, [[0.05], [0.15], [0.50], [0.30]])         # 报警系统有效性

# ==================== 中间节点CPD（自动生成） ====================
# HumanFactor: 8个父节点，状态数均为4，权重可根据专家经验调整
human_parents = ['Health','Fatigue','Emotion','SafetyKnowledge',
                 'Experience','Skill','AccidentHistory','ViolationHistory']
human_weights = [0.20, 0.25, 0.10, 0.15, 0.10, 0.05, 0.05, 0.10]  # 总和1.0
cpd_human = create_weighted_cpd('HumanFactor', 4, human_parents, [4]*8, human_weights, [0.2,0.5,0.8], noise=0.1)

# DirectTrigger: 确定性节点（任一否决指标触发则结果为1）
# 5个二值父节点，共32种组合，手动生成逻辑表
def create_direct_trigger_cpd():
    evidence = ['Alcohol','Authorized','PpeFastened','SignVisible','AlertPerceptible']
    evidence_card = [2,2,2,2,2]
    total = np.prod(evidence_card)
    values = np.zeros((2, total))
    for idx, combo in enumerate(itertools.product(*[range(c) for c in evidence_card])):
        # 危险条件：Alcohol==1 或 Authorized==0 或 PpeFastened==0 或 SignVisible==0 或 AlertPerceptible==0
        dangerous = (combo[0]==1 or combo[1]==0 or combo[2]==0 or combo[3]==0 or combo[4]==0)
        if dangerous:
            values[1, idx] = 1.0   # DirectTrigger=1
        else:
            values[0, idx] = 1.0   # DirectTrigger=0
    return TabularCPD('DirectTrigger', 2, values, evidence=evidence, evidence_card=evidence_card)
cpd_direct = create_direct_trigger_cpd()

# Exposure: 父节点 Distance(4), StayTime(4), Overlap(4), Crowding(4)
exp_parents = ['Distance','StayTime','Overlap','Crowding']
exp_weights = [0.4, 0.3, 0.2, 0.1]
cpd_exposure = create_weighted_cpd('Exposure', 4, exp_parents, [4,4,4,4], exp_weights, [0.2,0.5,0.8], noise=0.1)

# Isolation: 单父节点 ProtectionEff，直接映射（等级相同）
cpd_isolation = TabularCPD('Isolation', 4, np.eye(4), evidence=['ProtectionEff'], evidence_card=[4])

# Perception: 父节点 Visibility(4), Illumination(4)
perc_parents = ['Visibility','Illumination']
perc_weights = [0.6, 0.4]
cpd_perception = create_weighted_cpd('Perception', 4, perc_parents, [4,4], perc_weights, [0.2,0.5,0.8], noise=0.1)

# HazardIntensity: 父节点 EnergyLevel(4), ImpactRange(4)
intensity_parents = ['EnergyLevel','ImpactRange']
intensity_weights = [0.7, 0.3]
cpd_hazard_intensity = create_weighted_cpd('HazardIntensity', 4, intensity_parents, [4,4], intensity_weights, [0.2,0.5,0.8], noise=0.1)

# HazardUncontrol: 父节点 StabilityMargin(4), DeviceAbnormality(4)
uncontrol_parents = ['StabilityMargin','DeviceAbnormality']
uncontrol_weights = [0.5, 0.5]   # 注意 StabilityMargin 越高越好，需要反转分数，在 create_weighted_cpd 中 state_score 固定低=0好，需要特殊处理
# 因为 StabilityMargin 的值越高越稳定（风险越低），而我们的 state_score 是低=0（好），高=1（差），正好匹配：稳定性低（状态0）-> 风险高（分数1），稳定性高（状态3）-> 风险低（分数0）
# 但我们的加权和期望分数越高风险越大，所以直接使用 StabilityMargin 的状态值即可（状态0->低稳定性->高风险分数0？不对，状态0映射分数0，状态3映射分数1，则稳定性高反而分数高，不符合。需要反转）
# 修正：对 StabilityMargin 使用反向分数：1 - state_score[state]
def create_hazard_uncontrol_cpd():
    parents = ['StabilityMargin','DeviceAbnormality']
    parent_cards = [4,4]
    weights = [0.5, 0.5]
    w = np.array(weights) / np.sum(weights)
    state_score = {0:0.0, 1:1/3, 2:2/3, 3:1.0}
    total_comb = 16
    cpd_values = np.zeros((4, total_comb))
    for idx, (sm, da) in enumerate(itertools.product(range(4), range(4))):
        # StabilityMargin 状态越高越稳定，所以风险分数应为 1 - state_score[sm]
        risk_sm = 1 - state_score[sm]
        risk_da = state_score[da]
        score = w[0]*risk_sm + w[1]*risk_da
        level = 0
        for t in [0.2,0.5,0.8]:
            if score >= t:
                level += 1
        for s in range(4):
            if s == level:
                cpd_values[s, idx] = 0.9
            else:
                cpd_values[s, idx] = 0.1/3
    return TabularCPD('HazardUncontrol', 4, cpd_values, evidence=parents, evidence_card=parent_cards)
cpd_hazard_uncontrol = create_hazard_uncontrol_cpd()

# Management: 父节点 SensorCoverage(4), AlarmEffect(4)
mgmt_parents = ['SensorCoverage','AlarmEffect']
mgmt_weights = [0.4, 0.6]
cpd_management = create_weighted_cpd('Management', 4, mgmt_parents, [4,4], mgmt_weights, [0.2,0.5,0.8], noise=0.1)

# RiskLevel: 8个父节点，权重可沿用之前模糊综合评价的比例
risk_parents = ['HumanFactor','DirectTrigger','Exposure','Isolation',
                'Perception','HazardIntensity','HazardUncontrol','Management']
risk_parent_cards = [4,2,4,4,4,4,4,4]   # DirectTrigger 只有2状态
risk_weights = [0.25, 0.20, 0.15, 0.10, 0.05, 0.10, 0.10, 0.05]
# 注意：DirectTrigger 只有2状态（0,1），在状态分数映射中，状态0->0，状态1->1（因为一票否决直接满分）
# 我们需要自定义 RiskLevel 的生成，因为 DirectTrigger 特殊，且权重需按比例分配
def create_risklevel_cpd():
    n_states = 4
    parents = risk_parents
    parent_cards = risk_parent_cards
    weights = np.array(risk_weights) / np.sum(risk_weights)
    # 状态分数映射：普通节点 低=0,中=1/3,高=2/3,极高=1；DirectTrigger: 0=0, 1=1
    def get_score(node, state):
        if node == 'DirectTrigger':
            return float(state)   # 0->0, 1->1
        else:
            return {0:0.0, 1:1/3, 2:2/3, 3:1.0}[state]
    total_comb = np.prod(parent_cards)
    cpd_values = np.zeros((n_states, total_comb))
    for idx, combo in enumerate(itertools.product(*[range(c) for c in parent_cards])):
        score = sum(weights[i] * get_score(parents[i], combo[i]) for i in range(len(parents)))
        level = 0
        for t in [0.2,0.5,0.8]:
            if score >= t:
                level += 1
        for s in range(n_states):
            if s == level:
                cpd_values[s, idx] = 0.9
            else:
                cpd_values[s, idx] = 0.1 / (n_states-1)
    return TabularCPD('RiskLevel', n_states, cpd_values, evidence=parents, evidence_card=parent_cards)
cpd_risk = create_risklevel_cpd()

# Accident: 给定RiskLevel的条件概率（与之前相同）
cpd_accident = TabularCPD('Accident', 2,
                          [[0.999, 0.99, 0.90, 0.50],   # Accident=0
                           [0.001, 0.01, 0.10, 0.50]],  # Accident=1
                          evidence=['RiskLevel'], evidence_card=[4])

# ==================== 将所有CPD添加到模型 ====================
model.add_cpds(
    cpd_health, cpd_fatigue, cpd_emotion, cpd_safety_knowledge,
    cpd_experience, cpd_skill, cpd_accident_history, cpd_violation_history,
    cpd_alcohol, cpd_authorized, cpd_ppe_fastened, cpd_sign_visible, cpd_alert_perceptible,
    cpd_distance, cpd_stay_time, cpd_overlap, cpd_crowding,
    cpd_protection_eff, cpd_visibility, cpd_illumination,
    cpd_energy_level, cpd_impact_range, cpd_stability_margin, cpd_device_abnormality,
    cpd_sensor_coverage, cpd_alarm_effect,
    cpd_human, cpd_direct, cpd_exposure, cpd_isolation, cpd_perception,
    cpd_hazard_intensity, cpd_hazard_uncontrol, cpd_management,
    cpd_risk, cpd_accident
)

# 验证模型完整性
assert model.check_model()
print("模型构建成功，节点数:", len(model.nodes()))

# ==================== 推理示例 ====================
infer = VariableElimination(model)
# 示例证据：酒后作业（Alcohol=1），距离很近（Distance=3 表示接触）
query = infer.query(variables=['Accident'], evidence={'Alcohol': 1, 'Distance': 3})
print("P(Accident=1 | Alcohol=1, Distance=3):", query.values[1])

# 另一个示例：无否决项，距离远
query2 = infer.query(variables=['Accident'], evidence={'Alcohol': 0, 'Authorized': 1, 'PpeFastened': 1,
                                                       'SignVisible': 1, 'AlertPerceptible': 1, 'Distance': 0})
print("P(Accident=1 | 安全状态):", query2.values[1])




# 创建有向图
G = nx.DiGraph()
# 添加所有节点
G.add_nodes_from(model.nodes())
# 添加所有边
G.add_edges_from(model.edges())

# 绘图
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42, k=2)  # spring布局，可调节k值控制节点间距
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue',
        font_size=10, font_weight='bold', arrows=True, arrowstyle='->',
        arrowsize=20, edge_color='gray')
plt.title("贝叶斯网络结构图", fontsize=16)
plt.tight_layout()
plt.show()