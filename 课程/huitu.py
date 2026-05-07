import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# ====================== 1. 配置参数 ======================
TERM_COUNT = 8
FIG_SIZE = (18, 12)
DPI = 120

COLOR_MAP = {
    "通识教育课": "#e0edf7",
    "学科基础课": "#d5e8d4",
    "专业核心课": "#fff2cc",
    "专业方向课": "#dae8fc",
    "自主特色课": "#c8e6c9",
    "独立实践环节": "#f8cbad",
    "独立实践环节（课外）": "#e0e0e0"
}

STYLE_MAP = {
    "必修": "solid",
    "选修": "dashed"
}

TYPE_Y_RANGES = {
    "通识教育课": {"y_min": 7.5, "y_max": 9.5},
    "学科基础课": {"y_min": 4.0, "y_max": 7.5},
    "专业核心课": {"y_min": 2.5, "y_max": 4.0},
    "专业方向课": {"y_min": 1.5, "y_max": 2.5},
    "自主特色课": {"y_min": 3.0, "y_max": 5.0},
    "独立实践环节": {"y_min": 0.5, "y_max": 1.5},
    "独立实践环节（课外）": {"y_min": 0.0, "y_max": 0.5}
}

BOX_HEIGHT = 0.35
MIN_BOX_WIDTH = 1.0
TEXT_FONT_SIZE = 8
TITLE_FONT_SIZE = 10

# ====================== 2. 读取数据（已修复） ======================
def read_course_data(file_path):
    df = pd.read_excel(file_path)
    df["前置课程"] = df["前置课程"].fillna("")
    df["课程名称"] = df["课程名称"].astype(str).str.strip()
    df["课程类型"] = df["课程类型"].astype(str).str.strip()
    df["必修选修"] = df["必修选修"].astype(str).str.strip()
    df["学期"] = df["学期"].astype(str).str.strip()
    
    def clean_term(t):
        if "-" in str(t):
            return int(str(t).split("-")[0])
        try:
            return int(t)
        except:
            return 1
    df["学期"] = df["学期"].apply(clean_term)
    return df

# ====================== 3. 计算位置 ======================
def calculate_positions(df):
    term_width = 10 / TERM_COUNT
    course_positions = {}
    type_term_counter = {}

    for _, row in df.iterrows():
        name = row["课程名称"]
        course_type = row["课程类型"]
        term = row["学期"]

        x = (term - 0.5) * term_width
        key = (course_type, term)
        if key not in type_term_counter:
            type_term_counter[key] = 0
        count = type_term_counter[key]
        type_term_counter[key] += 1

        y_range = TYPE_Y_RANGES[course_type]
        total_courses = len(df[(df["课程类型"] == course_type) & (df["学期"] == term)])
        y_step = (y_range["y_max"] - y_range["y_min"]) / (total_courses + 1)
        y = y_range["y_min"] + y_step * (count + 1)

        text_len = len(name)
        box_width = max(MIN_BOX_WIDTH, text_len * 0.12 + 0.2)
        course_positions[name] = {"x": x, "y": y, "box_width": box_width}
    return course_positions, term_width

# ====================== 4. 绘图 ======================
def draw_course_plan(df, course_positions, term_width):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 背景
    for course_type, color in COLOR_MAP.items():
        y_range = TYPE_Y_RANGES[course_type]
        rect = patches.Rectangle((0, y_range["y_min"]), 10, y_range["y_max"] - y_range["y_min"],
                                 facecolor=color, alpha=0.7, zorder=0)
        ax.add_patch(rect)
        ax.text(0.15, (y_range["y_min"] + y_range["y_max"])/2, course_type,
                va='center', fontsize=TITLE_FONT_SIZE, fontweight='bold', zorder=1)

    # 学期线
    for i in range(1, TERM_COUNT+1):
        x = i * term_width
        ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, zorder=1)
        ax.text(x-term_width/2, 9.7, f"第{i}学期", ha='center', fontsize=TITLE_FONT_SIZE, zorder=2)

    # 课程框
    for _, row in df.iterrows():
        name = row["课程名称"]
        pos = course_positions[name]
        x, y = pos["x"], pos["y"]
        box_width = pos["box_width"]
        line_style = STYLE_MAP[row["必修选修"]]

        rect = patches.Rectangle((x-box_width/2, y-BOX_HEIGHT/2), box_width, BOX_HEIGHT,
                                 facecolor='white', edgecolor='black', linestyle=line_style,
                                 linewidth=1, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=TEXT_FONT_SIZE, wrap=True, zorder=4)

    # 箭头
    for _, row in df.iterrows():
        name = row["课程名称"]
        prereq = str(row["前置课程"]).strip()
        if prereq and prereq in course_positions:
            pf = course_positions[prereq]
            pt = course_positions[name]
            sx = pf["x"] + pf["box_width"]/2
            sy = pf["y"]
            ex = pt["x"] - pt["box_width"]/2
            ey = pt["y"]
            arrow = FancyArrowPatch((sx,sy),(ex,ey), arrowstyle='->', color='black', linewidth=0.8, zorder=2)
            ax.add_patch(arrow)

    plt.savefig(r"course_plan_output.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 生成成功！图片已保存到当前文件夹")

# ====================== 主函数（已修复） ======================
def main():
    df = read_course_data(r"C:\Users\69462\OneDrive\W 文档\VScode\Risk_model\课程\courses.xlsx")
    cp, tw = calculate_positions(df)
    draw_course_plan(df, cp, tw)

if __name__ == "__main__":
    main()