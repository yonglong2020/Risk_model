import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题


# ==================== 定义节点及其所属类别 ====================
categories = {
    "人的状态": [
        "健康程度", "疲劳作业", "酒后作业", "情绪状态",
        "安全知识水平", "经验水平", "技能水平",
        "个人事故记录", "个人违规记录"
    ],
    "空间与时间暴露": [
        "空间关系", "时间尺度", "交叉作业垂直空间重叠度",
        "逗留时间", "场地拥挤度"
    ],
    "准入与防护": [
        "准入状态", "防护用品系挂状态"
    ],
    "危险源与设备": [
        "危险源类型", "能量等级", "波及范围",
        "承载/支撑结构稳定性裕度", "设备/机具运行状态异常度",
        "危险源防护措施有效性"
    ],
    "环境与感知": [
        "作业环境辨识度", "警示标识可见性", "预警信号可感知性",
        "实时气象条件", "照明条件"
    ],
    "监测与报警": [
        "实时监测设备覆盖率", "报警系统有效性"
    ],
    "事故风险": [
        "事故概率", "事故后果严重度"
    ]
}

node_to_cat = {node: cat for cat, nodes in categories.items() for node in nodes}

# 边关系（与之前相同）
edges = [
    # 人的状态 → 防护/违规
    ("健康程度", "防护用品系挂状态", "+"),
    ("疲劳作业", "防护用品系挂状态", "-"),
    ("酒后作业", "防护用品系挂状态", "-"),
    ("情绪状态", "防护用品系挂状态", "-"),
    ("安全知识水平", "防护用品系挂状态", "+"),
    ("经验水平", "防护用品系挂状态", "+"),
    ("技能水平", "防护用品系挂状态", "+"),
    ("健康程度", "个人违规记录", "-"),
    ("疲劳作业", "个人违规记录", "+"),
    ("酒后作业", "个人违规记录", "+"),
    ("情绪状态", "个人违规记录", "+"),
    ("安全知识水平", "个人违规记录", "-"),
    ("经验水平", "个人违规记录", "-"),
    ("技能水平", "个人违规记录", "-"),
    ("个人事故记录", "准入状态", "-"),
    ("个人违规记录", "准入状态", "-"),
    ("个人事故记录", "防护用品系挂状态", "-"),
    ("个人违规记录", "防护用品系挂状态", "-"),

    # 空间/时间 → 风险
    ("空间关系", "事故概率", "+"),
    ("时间尺度", "事故概率", "+"),
    ("交叉作业垂直空间重叠度", "空间关系", "+"),
    ("场地拥挤度", "空间关系", "+"),
    ("场地拥挤度", "交叉作业垂直空间重叠度", "+"),
    ("逗留时间", "时间尺度", "+"),
    ("疲劳作业", "逗留时间", "+"),
    ("作业环境辨识度", "逗留时间", "-"),

    # 危险源 → 后果与防护
    ("危险源类型", "能量等级", "+"),
    ("能量等级", "事故后果严重度", "+"),
    ("波及范围", "事故后果严重度", "+"),
    ("能量等级", "危险源防护措施有效性", "-"),
    ("承载/支撑结构稳定性裕度", "危险源防护措施有效性", "+"),
    ("设备/机具运行状态异常度", "危险源防护措施有效性", "-"),
    ("设备/机具运行状态异常度", "事故概率", "+"),

    # 环境 → 行为与暴露
    ("照明条件", "作业环境辨识度", "+"),
    ("警示标识可见性", "预警信号可感知性", "+"),
    ("预警信号可感知性", "逗留时间", "-"),
    ("实时气象条件", "承载/支撑结构稳定性裕度", "-"),
    ("实时气象条件", "疲劳作业", "+"),

    # 监测 → 预警与防护
    ("实时监测设备覆盖率", "报警系统有效性", "+"),
    ("报警系统有效性", "预警信号可感知性", "+"),
    ("报警系统有效性", "危险源防护措施有效性", "+"),

    # 最终风险
    ("防护用品系挂状态", "危险源防护措施有效性", "+"),
    ("准入状态", "防护用品系挂状态", "+"),
    ("危险源防护措施有效性", "事故概率", "-"),
    ("事故概率", "个人事故记录", "+"),
    ("事故概率", "事故后果严重度", "+"),
]

# 构建有向图
G = nx.DiGraph()
G.add_edges_from([(s, t) for s, t, _ in edges])

# ==================== 手动布局（按区域分列） ====================
cat_order = list(categories.keys())
cat_to_x = {cat: i * 2.5 for i, cat in enumerate(cat_order)}  # 列间距

pos = {}
for cat, nodes in categories.items():
    x = cat_to_x[cat]
    sorted_nodes = sorted(nodes)                # 统一排序，避免重叠
    n = len(sorted_nodes)
    y_vals = np.linspace(-1.5, 1.5, n)          # 垂直范围 -1.5 ~ 1.5
    for idx, node in enumerate(sorted_nodes):
        pos[node] = (x, y_vals[idx])

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(22, 14))

# 为每个区域添加背景矩形和标题
colors = ['#FFE4E1', '#E0FFFF', '#F0FFF0', '#FFFACD', '#E6E6FA', '#FFDAB9', '#F5DEB3']
for i, cat in enumerate(cat_order):
    nodes_in_cat = categories[cat]
    if not nodes_in_cat:
        continue
    xs = [pos[n][0] for n in nodes_in_cat]
    ys = [pos[n][1] for n in nodes_in_cat]
    x_min, x_max = min(xs) - 0.8, max(xs) + 0.8
    y_min, y_max = min(ys) - 0.6, max(ys) + 0.6
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     facecolor=colors[i % len(colors)], alpha=0.2,
                     edgecolor='gray', linestyle='--', linewidth=1)
    ax.add_patch(rect)
    # 区域标题
    ax.text((x_min + x_max)/2, y_max + 0.15, cat,
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=2400, node_color='lightblue',
                       edgecolors='black', linewidths=1.5, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

# 绘制边（带箭头和弯曲）
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=18,
                       arrowstyle='-|>', connectionstyle='arc3,rad=0.15', ax=ax, alpha=0.7)

# 添加边的极性标签
edge_labels = {(s, t): label for s, t, label in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                             label_pos=0.35, ax=ax, bbox=dict(facecolor='white', alpha=0.6))

plt.title("系统动力学因果回路图（按功能区域划分）", fontsize=18, pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig("causal_loop_by_region.png", dpi=300, bbox_inches='tight')
plt.show()