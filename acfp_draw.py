import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义颜色方案
colors = {
    'spdconv': '#FFA07A',      # 橙色 - SPDConv
    'awm': '#6A5ACD',          # 深紫色 - AWM (核心创新)
    'cafe': '#FF6347',         # 深橙红色 - CAFE (核心创新)
    'repc3': '#9370DB',        # 蓝紫色 - RepC3
    'upsample': '#FFB84D',     # 浅橙色 - Upsample
    'conv': '#90EE90',         # 绿色 - Conv
    'feature': '#F0E68C'       # 浅黄色 - 特征层
}

# 定义模块绘制函数
def draw_module(ax, x, y, width, height, text, color, fontsize=11, fontweight='normal'):
    """绘制模块方框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            color='white' if color in [colors['awm'], colors['cafe']] else 'black')

def draw_arrow(ax, x1, y1, x2, y2, style='->'):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color='black',
                           linewidth=2,
                           mutation_scale=20)
    ax.add_patch(arrow)

def draw_feature_layer(ax, x, y, text, color='#F0E68C'):
    """绘制特征层（立体感）"""
    # 绘制多层效果
    for i in range(3):
        offset = i * 0.08
        box = FancyBboxPatch((x - offset, y + offset), 0.8, 0.6,
                           boxstyle="round,pad=0.02",
                           facecolor=color,
                           edgecolor='black',
                           linewidth=1.5,
                           alpha=0.7 - i*0.2)
        ax.add_patch(box)
    
    ax.text(x + 0.4, y + 0.3, text,
            ha='center', va='center',
            fontsize=12, fontweight='bold')

# ===== 绘制P3层（最上层，ACFP核心） =====
y_p3 = 8.5

# P2特征层
draw_feature_layer(ax, 0.5, y_p3, 'P2')

# SPDConv
draw_module(ax, 1.8, y_p3, 1.2, 0.6, 'SPDConv', colors['spdconv'], fontsize=10)

# P3特征层
draw_feature_layer(ax, 0.5, y_p3 - 1.2, 'P3')

# Y4上采样路径
draw_module(ax, 1.8, y_p3 - 2.4, 1.2, 0.6, 'Upsample', colors['upsample'], fontsize=10)
ax.text(1.0, y_p3 - 2.1, 'Y4', ha='center', va='center', fontsize=11, fontweight='bold')

# AWM模块（三路输入）
draw_module(ax, 3.8, y_p3 - 0.6, 1.4, 1.8, 'AWM\n自适应\n加权', colors['awm'], fontsize=11, fontweight='bold')

# CAFE模块
draw_module(ax, 5.8, y_p3, 1.4, 0.6, 'CAFE', colors['cafe'], fontsize=11, fontweight='bold')

# RepC3
draw_module(ax, 7.8, y_p3, 1.2, 0.6, 'RepC3', colors['repc3'], fontsize=10)

# 输出
ax.text(9.5, y_p3 + 0.3, '→ 输出P3', ha='left', va='center', fontsize=11, fontweight='bold')

# 绘制P3层的连接箭头
# P2 → SPDConv
draw_arrow(ax, 1.3, y_p3 + 0.3, 1.8, y_p3 + 0.3)
# SPDConv → AWM（上方输入）
draw_arrow(ax, 3.0, y_p3 + 0.3, 3.8, y_p3 + 0.6)
# P3 → AWM（中间输入）
draw_arrow(ax, 1.3, y_p3 - 0.9, 3.8, y_p3 - 0.0)
# Y4+Upsample → AWM（下方输入）
draw_arrow(ax, 3.0, y_p3 - 2.1, 3.8, y_p3 - 1.2)
# AWM → CAFE
draw_arrow(ax, 5.2, y_p3 + 0.3, 5.8, y_p3 + 0.3)
# CAFE → RepC3
draw_arrow(ax, 7.2, y_p3 + 0.3, 7.8, y_p3 + 0.3)
# RepC3 → 输出
draw_arrow(ax, 9.0, y_p3 + 0.3, 9.3, y_p3 + 0.3)

# ===== 绘制P4层 =====
y_p4 = 5.0

# P3输出向下
draw_arrow(ax, 8.4, y_p3, 8.4, y_p4 + 1.5, style='->')
draw_module(ax, 7.9, y_p4 + 1.5, 1.0, 0.5, 'Conv 3×3', colors['conv'], fontsize=9)
draw_arrow(ax, 8.4, y_p4 + 1.5, 8.4, y_p4 + 0.9)

# P4特征层
draw_feature_layer(ax, 0.5, y_p4, 'P4')

# P5→P4的上采样路径
draw_module(ax, 1.8, y_p4 - 1.0, 1.2, 0.5, 'Upsample', colors['upsample'], fontsize=9)

# AWM模块
draw_module(ax, 3.8, y_p4, 1.4, 0.8, 'AWM', colors['awm'], fontsize=11, fontweight='bold')

# RepC3
draw_module(ax, 5.8, y_p4, 1.2, 0.6, 'RepC3', colors['repc3'], fontsize=10)

# Conv 1×1（生成Y4）
draw_module(ax, 7.5, y_p4, 1.0, 0.5, 'Conv 1×1', colors['conv'], fontsize=9)

# 输出
ax.text(9.0, y_p4 + 0.3, '→ 输出P4', ha='left', va='center', fontsize=11, fontweight='bold')

# 绘制P4层的连接箭头
# P4特征 → AWM
draw_arrow(ax, 1.3, y_p4 + 0.3, 3.8, y_p4 + 0.4)
# P3输出 → AWM
draw_arrow(ax, 8.4, y_p4 + 0.9, 5.2, y_p4 + 0.6, style='->')
# Upsample → AWM
draw_arrow(ax, 3.0, y_p4 - 0.75, 3.8, y_p4 + 0.1)
# AWM → RepC3
draw_arrow(ax, 5.2, y_p4 + 0.4, 5.8, y_p4 + 0.3)
# RepC3 → Conv 1×1
draw_arrow(ax, 7.0, y_p4 + 0.3, 7.5, y_p4 + 0.25)
# Conv 1×1 → 输出
draw_arrow(ax, 8.5, y_p4 + 0.25, 8.8, y_p4 + 0.3)
# Conv 1×1 → Y4（向上到P3）
draw_arrow(ax, 8.0, y_p4 + 0.5, 2.4, y_p3 - 2.1, style='->')
ax.text(5.0, y_p4 + 1.8, 'Y4', ha='center', va='center', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# ===== 绘制P5层 =====
y_p5 = 1.5

# P4输出向下
draw_arrow(ax, 8.0, y_p4, 8.0, y_p5 + 1.2, style='->')
draw_module(ax, 7.5, y_p5 + 1.2, 1.0, 0.5, 'Conv 3×3', colors['conv'], fontsize=9)
draw_arrow(ax, 8.0, y_p5 + 1.2, 8.0, y_p5 + 0.8)

# P5特征层
draw_feature_layer(ax, 0.5, y_p5, 'P5')

# AWM模块
draw_module(ax, 3.8, y_p5, 1.4, 0.8, 'AWM', colors['awm'], fontsize=11, fontweight='bold')

# RepC3
draw_module(ax, 5.8, y_p5, 1.2, 0.6, 'RepC3', colors['repc3'], fontsize=10)

# 输出
ax.text(7.5, y_p5 + 0.3, '→ 输出P5', ha='left', va='center', fontsize=11, fontweight='bold')

# 绘制P5层的连接箭头
# P5特征 → AWM
draw_arrow(ax, 1.3, y_p5 + 0.3, 3.8, y_p5 + 0.4)
# P4输出 → AWM
draw_arrow(ax, 8.0, y_p5 + 0.8, 5.2, y_p5 + 0.6, style='->')
# AWM → RepC3
draw_arrow(ax, 5.2, y_p5 + 0.4, 5.8, y_p5 + 0.3)
# RepC3 → 输出
draw_arrow(ax, 7.0, y_p5 + 0.3, 7.3, y_p5 + 0.3)

# P5向上到P4的Upsample连接
draw_arrow(ax, 1.0, y_p5 + 0.7, 2.4, y_p4 - 0.75, style='->')

# ===== 添加虚线框和标题 =====
# 绘制大虚线框
rect = patches.Rectangle((0.3, 0.8), 10.0, 8.8,
                         linewidth=2.5,
                         edgecolor='gray',
                         facecolor='none',
                         linestyle='--')
ax.add_patch(rect)

# 添加标题
ax.text(7.0, 9.8, 'ACFP 自适应协同融合金字塔网络结构',
        ha='center', va='top',
        fontsize=16, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# 添加图例
legend_x = 11.0
legend_y = 7.5
ax.text(legend_x + 0.8, legend_y + 1.5, '图例', ha='center', va='center',
        fontsize=12, fontweight='bold')

legend_items = [
    ('SPDConv', colors['spdconv'], '无损下采样'),
    ('AWM', colors['awm'], '自适应加权'),
    ('CAFE', colors['cafe'], '特征增强'),
    ('RepC3', colors['repc3'], '特征整合'),
]

for i, (name, color, desc) in enumerate(legend_items):
    y = legend_y - i * 0.6
    # 绘制色块
    box = FancyBboxPatch((legend_x, y), 0.5, 0.4,
                         boxstyle="round,pad=0.02",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=1)
    ax.add_patch(box)
    # 添加文字
    ax.text(legend_x + 0.25, y + 0.2, name,
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            color='white' if color in [colors['awm'], colors['cafe']] else 'black')
    ax.text(legend_x + 0.7, y + 0.2, desc,
            ha='left', va='center', fontsize=9)

# 添加注释
note_y = 0.3
ax.text(7.0, note_y, '注：AWM和CAFE为本文提出的核心创新模块',
        ha='center', va='center',
        fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('./acfp/acfp_structure.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('./acfp/acfp_structure.pdf', bbox_inches='tight', facecolor='white')
print("ACFP网络结构图已生成！")
print("PNG: ./acfp/acfp_structure.png")
print("PDF: ./acfp/acfp_structure.pdf")