"""
RT-DETR 模型架构图自动绘制脚本
使用matplotlib绘制Baseline和改进版的架构对比图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def draw_box(ax, x, y, width, height, text, color='lightblue', is_modified=False):
    """绘制一个模块框"""
    if is_modified:
        color = 'lightcoral'  # 改进的模块用红色
        edgecolor = 'red'
        linewidth = 2.5
    else:
        edgecolor = 'black'
        linewidth = 1.5

    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        edgecolor=edgecolor,
        facecolor=color,
        linewidth=linewidth
    )
    ax.add_patch(box)

    # 添加文字
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=9, weight='bold')

def draw_arrow(ax, x1, y1, x2, y2, style='solid', color='black'):
    """绘制箭头"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->',
        mutation_scale=20,
        linewidth=2 if style=='solid' else 1.5,
        linestyle=style,
        color=color
    )
    ax.add_patch(arrow)

def draw_baseline_architecture():
    """绘制Baseline架构图"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # 标题
    ax.text(5, 19.5, 'Baseline RT-DETR-r18 架构', ha='center', fontsize=16, weight='bold')

    # 输入
    draw_box(ax, 3.5, 18.5, 3, 0.6, '输入图像\n[3, 640, 640]', 'lightgray')
    draw_arrow(ax, 5, 18.5, 5, 18.0)

    # Backbone标题
    ax.text(1, 17.5, 'Backbone', fontsize=12, weight='bold', color='darkblue')

    # Backbone层
    y_pos = 17.5
    backbone_layers = [
        ('ConvNormLayer', '[32, 320, 320]', 'P1/2'),
        ('MaxPool2d', '[64, 160, 160]', 'P2/4'),
        ('Blocks×2', '[64, 160, 160]', 'Stage2'),
        ('Blocks×2', '[128, 80, 80]', 'P3/8 →'),
        ('Blocks×2', '[256, 40, 40]', 'P4/16 →'),
        ('Blocks×2', '[512, 20, 20]', 'P5/32 →'),
    ]

    for i, (name, shape, note) in enumerate(backbone_layers):
        y = y_pos - i * 1.2
        draw_box(ax, 3, y, 4, 0.8, f'{name}\n{shape}\n{note}')
        if i < len(backbone_layers) - 1:
            draw_arrow(ax, 5, y, 5, y - 0.4)

    # Head标题
    y_head = y_pos - len(backbone_layers) * 1.2 - 0.5
    ax.text(1, y_head, 'Head (Neck + Decoder)', fontsize=12, weight='bold', color='darkgreen')

    # Head部分 - 简化表示
    y_pos = y_head - 0.5
    head_layers = [
        ('Conv 1×1', '[256, 20, 20]'),
        ('AIFI 自注意力', '[256, 20, 20]'),
        ('FPN 上采样融合', 'P5→P4→P3'),
        ('PAN 下采样融合', 'P3→P4→P5'),
        ('RTDETRDecoder', '检测结果'),
    ]

    for i, (name, info) in enumerate(head_layers):
        y = y_pos - i * 1.2
        draw_box(ax, 2.5, y, 5, 0.8, f'{name}\n{info}')
        if i < len(head_layers) - 1:
            draw_arrow(ax, 5, y, 5, y - 0.4)

    # 输出
    y_output = y_pos - len(head_layers) * 1.2
    draw_box(ax, 3.5, y_output, 3, 0.6, '检测结果输出', 'lightgreen')

    plt.tight_layout()
    return fig

def draw_improved_architecture():
    """绘制改进版架构图"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # 标题
    ax.text(5, 19.5, '改进版模型架构（标红为改动）', ha='center', fontsize=16, weight='bold')

    # 输入
    draw_box(ax, 3.5, 18.5, 3, 0.6, '输入图像\n[3, 640, 640]', 'lightgray')
    draw_arrow(ax, 5, 18.5, 5, 18.0)

    # Backbone标题
    ax.text(1, 17.5, 'Backbone', fontsize=12, weight='bold', color='darkblue')

    # Backbone层（标注改动）
    y_pos = 17.5
    backbone_layers = [
        ('🔴 SRFD', '[64, 160, 160]', 'P2/4', True),
        ('Blocks×2', '[64, 160, 160]', 'Stage2', False),
        ('🔴 C2f_DCMB', '[128, 80, 80]', 'P2 →', True),
        ('Conv 3×3 s=2', '[256, 40, 40]', 'P3/8 →', False),
        ('🔴 C2f_DCMB', '[256, 40, 40]', 'P3 →', True),
        ('Conv 3×3 s=2', '[384, 20, 20]', 'P4/16 →', False),
        ('🔴 C2f_DCMB', '[384, 20, 20]', 'P4 →', True),
        ('Conv 3×3 s=2', '[384, 10, 10]', 'P5/32 →', False),
        ('🔴 C2f_DCMB×3', '[384, 10, 10]', 'P5 →', True),
    ]

    for i, (name, shape, note, modified) in enumerate(backbone_layers):
        y = y_pos - i * 0.95
        draw_box(ax, 3, y, 4, 0.7, f'{name}\n{shape}\n{note}', is_modified=modified)
        if i < len(backbone_layers) - 1:
            draw_arrow(ax, 5, y, 5, y - 0.25)

    # Head标题
    y_head = y_pos - len(backbone_layers) * 0.95 - 0.5
    ax.text(1, y_head, 'Head (Neck + Decoder)', fontsize=12, weight='bold', color='darkgreen')

    # Head部分
    y_pos = y_head - 0.5
    head_layers = [
        ('Conv 1×1', '[256, 10, 10]', False),
        ('🔴 PolaAttention', '[256, 10, 10]', True),
        ('🔴 SPDConv+MFM', 'P2特征融合', True),
        ('🔴 CSPOmniKernel', '多尺度整合', True),
        ('🔴 MFM协同融合', '替换Concat', True),
        ('RTDETRDecoder', '检测结果', False),
    ]

    for i, (name, info, modified) in enumerate(head_layers):
        y = y_pos - i * 0.95
        draw_box(ax, 2.5, y, 5, 0.7, f'{name}\n{info}', is_modified=modified)
        if i < len(head_layers) - 1:
            draw_arrow(ax, 5, y, 5, y - 0.25)

    # 输出
    y_output = y_pos - len(head_layers) * 0.95
    draw_box(ax, 3.5, y_output, 3, 0.6, '检测结果输出', 'lightgreen')

    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='保持不变的模块'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='red', linewidth=2, label='改进/新增的模块')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig

def draw_comparison():
    """绘制对比图（左右对比）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 16))

    # 左边：Baseline
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 20)
    ax1.axis('off')
    ax1.text(5, 19.5, 'Baseline RT-DETR-r18', ha='center', fontsize=14, weight='bold')

    # Baseline简化架构
    y_pos = 18.5
    baseline_modules = [
        ('输入图像', '[3, 640, 640]', 'lightgray', False),
        ('Conv+MaxPool', 'P2/4下采样', 'lightblue', False),
        ('Blocks×4', 'P3/P4/P5', 'lightblue', False),
        ('AIFI', '自注意力', 'lightblue', False),
        ('Concat融合', 'FPN+PAN', 'lightblue', False),
        ('RTDETRDecoder', '检测', 'lightgreen', False),
    ]

    for i, (name, info, color, modified) in enumerate(baseline_modules):
        y = y_pos - i * 2.5
        draw_box(ax1, 2, y, 6, 1.5, f'{name}\n{info}', color, modified)
        if i < len(baseline_modules) - 1:
            draw_arrow(ax1, 5, y, 5, y - 1.0)

    # 右边：改进版
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 20)
    ax2.axis('off')
    ax2.text(5, 19.5, '改进版模型（5处改动）', ha='center', fontsize=14, weight='bold', color='red')

    # 改进版架构
    improved_modules = [
        ('输入图像', '[3, 640, 640]', 'lightgray', False),
        ('🔴 SRFD', '鲁棒下采样', 'lightcoral', True),
        ('🔴 C2f_DCMB', '动态卷积', 'lightcoral', True),
        ('🔴 PolaAttention', '极化注意力', 'lightcoral', True),
        ('🔴 SPDConv+MFM', '小目标增强', 'lightcoral', True),
        ('RTDETRDecoder', '检测', 'lightgreen', False),
    ]

    for i, (name, info, color, modified) in enumerate(improved_modules):
        y = y_pos - i * 2.5
        draw_box(ax2, 2, y, 6, 1.5, f'{name}\n{info}', color, modified)
        if i < len(improved_modules) - 1:
            draw_arrow(ax2, 5, y, 5, y - 1.0)

        # 添加改动标注
        if modified and i > 0:
            ax2.text(8.5, y + 0.75, f'改动{i}', fontsize=10, color='red', weight='bold')

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("正在生成架构图...")

    # 生成Baseline图
    print("1/3 生成Baseline架构图...")
    fig1 = draw_baseline_architecture()
    fig1.savefig('./架构图_Baseline.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: 架构图_Baseline.png")

    # 生成改进版图
    print("2/3 生成改进版架构图...")
    fig2 = draw_improved_architecture()
    fig2.savefig('./架构图_改进版.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: 架构图_改进版.png")

    # 生成对比图
    print("3/3 生成对比图...")
    fig3 = draw_comparison()
    fig3.savefig('./架构图_对比.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: 架构图_对比.png")

    print("\n✅ 所有架构图生成完成！")
    print("生成的文件：")
    print("  - 架构图_Baseline.png (详细版)")
    print("  - 架构图_改进版.png (详细版)")
    print("  - 架构图_对比.png (简化对比版，推荐用于论文)")

    plt.close('all')
