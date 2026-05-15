"""
PASE Module Structure Diagrams
- Figure 1: PASE Overall Architecture
- Figure 2: PolaAttention Detail
- Figure 3: ACGF Detail (with expanded spatial branch and single-direction gating)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def draw_box(ax, x, y, w, h, text, color='lightblue', fontsize=9, text_color='black'):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight='bold', wrap=True)
    return box


def draw_arrow(ax, start, end, color='black', style='->', connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                            color=color, linewidth=1.5,
                            connectionstyle=connectionstyle,
                            mutation_scale=15)
    ax.add_patch(arrow)
    return arrow


def draw_line(ax, start, end, color='black', linewidth=1.5, linestyle='-'):
    """Draw a simple line"""
    ax.plot([start[0], end[0]], [start[1], end[1]],
            color=color, linewidth=linewidth, linestyle=linestyle)


def draw_circle(ax, x, y, r, text, color='white'):
    """Draw a circle with text (for operations like +, x)"""
    circle = Circle((x, y), r, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=12, weight='bold')


# ============================================================
# Figure 1: PASE Overall Architecture
# ============================================================
def draw_pase_overall():
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 13.5, 'PASE: Polarity-Aware Spatial-Enhanced Encoder',
            ha='center', va='center', fontsize=14, weight='bold')

    # Input
    draw_box(ax, 5, 12.5, 2.5, 0.6, 'Input Feature\n[B, C, H, W]', color='#E8E8E8', fontsize=9)

    # Save for spatial branch
    ax.text(7.5, 11.8, 'save', ha='center', va='center', fontsize=8, style='italic', color='gray')
    draw_line(ax, (6.25, 12.2), (7.5, 12.2), color='gray', linestyle='--')
    draw_line(ax, (7.5, 12.2), (7.5, 5.5), color='gray', linestyle='--')
    draw_arrow(ax, (7.5, 5.5), (6.5, 5.5), color='gray', style='->')
    ax.text(7.7, 8.5, 'x_spatial', ha='left', va='center', fontsize=8, color='gray', rotation=90)

    # Arrow from input
    draw_arrow(ax, (5, 12.2), (5, 11.6), style='->')

    # PolaAttention Block
    pola_box = FancyBboxPatch((2, 9.2), 6, 2.2,
                               boxstyle="round,pad=0.02,rounding_size=0.2",
                               facecolor='#AED6F1', edgecolor='#2874A6', linewidth=2)
    ax.add_patch(pola_box)
    ax.text(5, 10.9, 'PolaAttention', ha='center', va='center', fontsize=11, weight='bold', color='#1A5276')
    ax.text(5, 10.3, 'Polarity-Aware Linear Attention', ha='center', va='center', fontsize=9, color='#2874A6')
    ax.text(5, 9.7, 'O(N) Complexity', ha='center', va='center', fontsize=8, color='#5D6D7E')

    # Arrow
    draw_arrow(ax, (5, 9.2), (5, 8.6), style='->')

    # Add + Residual
    draw_circle(ax, 5, 8.3, 0.25, '+', color='#FADBD8')
    draw_line(ax, (2, 12.2), (2, 8.3), color='#E74C3C', linestyle='--')
    draw_arrow(ax, (2, 8.3), (4.75, 8.3), color='#E74C3C', style='->')

    # LayerNorm
    draw_box(ax, 5, 7.6, 2, 0.5, 'LayerNorm', color='#F9E79F', fontsize=9)
    draw_arrow(ax, (5, 8.05), (5, 7.85), style='->')
    draw_arrow(ax, (5, 7.35), (5, 6.9), style='->')

    # ACGF Block
    acgf_box = FancyBboxPatch((2, 4.2), 6, 2.5,
                               boxstyle="round,pad=0.02,rounding_size=0.2",
                               facecolor='#ABEBC6', edgecolor='#1E8449', linewidth=2)
    ax.add_patch(acgf_box)
    ax.text(5, 6.2, 'ACGF', ha='center', va='center', fontsize=11, weight='bold', color='#145A32')
    ax.text(5, 5.6, 'Adaptive Channel-wise', ha='center', va='center', fontsize=9, color='#1E8449')
    ax.text(5, 5.1, 'Gated Feedforward', ha='center', va='center', fontsize=9, color='#1E8449')
    ax.text(5, 4.5, 'Dual-input Design', ha='center', va='center', fontsize=8, color='#5D6D7E')

    # Arrow
    draw_arrow(ax, (5, 4.2), (5, 3.6), style='->')

    # Add + Residual
    draw_circle(ax, 5, 3.3, 0.25, '+', color='#FADBD8')
    draw_line(ax, (1.5, 7.6), (1.5, 3.3), color='#E74C3C', linestyle='--')
    draw_arrow(ax, (1.5, 3.3), (4.75, 3.3), color='#E74C3C', style='->')

    # LayerNorm
    draw_box(ax, 5, 2.6, 2, 0.5, 'LayerNorm', color='#F9E79F', fontsize=9)
    draw_arrow(ax, (5, 3.05), (5, 2.85), style='->')
    draw_arrow(ax, (5, 2.35), (5, 1.9), style='->')

    # Output
    draw_box(ax, 5, 1.5, 2.5, 0.6, 'Output Feature\n[B, C, H, W]', color='#E8E8E8', fontsize=9)

    # Legend
    legend_y = 0.5
    draw_box(ax, 2, legend_y, 0.4, 0.3, '', color='#AED6F1')
    ax.text(2.4, legend_y, 'PolaAttention (Introduced)', ha='left', va='center', fontsize=8)
    draw_box(ax, 6, legend_y, 0.4, 0.3, '', color='#ABEBC6')
    ax.text(6.4, legend_y, 'ACGF (Proposed)', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('E:/Code/paper/rt-detr/figures/pase_overall.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('E:/Code/paper/rt-detr/figures/pase_overall.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 1: PASE Overall saved.")


# ============================================================
# Figure 2: PolaAttention Detail
# ============================================================
def draw_pola_attention():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'PolaAttention: Polarity-Aware Linear Attention',
            ha='center', va='center', fontsize=14, weight='bold')

    # Input
    draw_box(ax, 7, 8.7, 2.5, 0.5, 'Input x [B, N, C]', color='#E8E8E8', fontsize=9)

    # Linear projections
    draw_arrow(ax, (7, 8.45), (7, 8.1), style='->')

    # Split into Q,G and K,V
    draw_box(ax, 4.5, 7.6, 2, 0.5, 'Linear (Q, G)', color='#D5DBDB', fontsize=9)
    draw_box(ax, 9.5, 7.6, 2, 0.5, 'Linear (K, V)', color='#D5DBDB', fontsize=9)

    draw_line(ax, (7, 8.1), (7, 7.9))
    draw_line(ax, (7, 7.9), (4.5, 7.9))
    draw_line(ax, (7, 7.9), (9.5, 7.9))
    draw_arrow(ax, (4.5, 7.9), (4.5, 7.85), style='->')
    draw_arrow(ax, (9.5, 7.9), (9.5, 7.85), style='->')

    # Polarity Separation
    draw_box(ax, 4.5, 6.8, 2.5, 0.6, 'Polarity\nSeparation', color='#AED6F1', fontsize=9)
    draw_box(ax, 9.5, 6.8, 2.5, 0.6, 'Polarity\nSeparation', color='#AED6F1', fontsize=9)

    draw_arrow(ax, (4.5, 7.35), (4.5, 7.1), style='->')
    draw_arrow(ax, (9.5, 7.35), (9.5, 7.1), style='->')

    # Q+, Q- and K+, K-
    ax.text(3.2, 6.2, 'Q+ = ReLU(Q)^p', ha='center', va='center', fontsize=8, color='#1A5276')
    ax.text(5.8, 6.2, 'Q- = ReLU(-Q)^p', ha='center', va='center', fontsize=8, color='#1A5276')
    ax.text(8.2, 6.2, 'K+ = ReLU(K)^p', ha='center', va='center', fontsize=8, color='#1A5276')
    ax.text(10.8, 6.2, 'K- = ReLU(-K)^p', ha='center', va='center', fontsize=8, color='#1A5276')

    draw_line(ax, (4.5, 6.5), (4.5, 6.35))
    draw_line(ax, (4.5, 6.35), (3.2, 6.35))
    draw_line(ax, (4.5, 6.35), (5.8, 6.35))

    draw_line(ax, (9.5, 6.5), (9.5, 6.35))
    draw_line(ax, (9.5, 6.35), (8.2, 6.35))
    draw_line(ax, (9.5, 6.35), (10.8, 6.35))

    # Learnable power
    ax.text(7, 5.8, 'p = 1 + α·sigmoid(w)', ha='center', va='center', fontsize=9,
            style='italic', color='#7D3C98',
            bbox=dict(boxstyle='round', facecolor='#F5EEF8', edgecolor='#7D3C98'))

    # Two streams
    # Same-signed stream
    stream1_box = FancyBboxPatch((1.5, 3.5), 4.5, 1.8,
                                  boxstyle="round,pad=0.02,rounding_size=0.15",
                                  facecolor='#D4E6F1', edgecolor='#2E86C1', linewidth=1.5)
    ax.add_patch(stream1_box)
    ax.text(3.75, 5.0, 'Same-signed Stream', ha='center', va='center', fontsize=10, weight='bold', color='#1A5276')
    ax.text(3.75, 4.5, '[Q+; Q-] × [K+; K-]^T × V^s', ha='center', va='center', fontsize=9)
    ax.text(3.75, 4.0, '⊙ G^s', ha='center', va='center', fontsize=9, color='#2874A6')

    # Opposite-signed stream
    stream2_box = FancyBboxPatch((8, 3.5), 4.5, 1.8,
                                  boxstyle="round,pad=0.02,rounding_size=0.15",
                                  facecolor='#D5F5E3', edgecolor='#1E8449', linewidth=1.5)
    ax.add_patch(stream2_box)
    ax.text(10.25, 5.0, 'Opposite-signed Stream', ha='center', va='center', fontsize=10, weight='bold', color='#145A32')
    ax.text(10.25, 4.5, '[Q+; Q-] × [K-; K+]^T × V^o', ha='center', va='center', fontsize=9)
    ax.text(10.25, 4.0, '⊙ G^o', ha='center', va='center', fontsize=9, color='#1E8449')

    # Arrows to streams
    draw_line(ax, (3.2, 6.05), (3.2, 5.3))
    draw_line(ax, (5.8, 6.05), (5.8, 5.3))
    draw_line(ax, (3.2, 5.3), (3.75, 5.3))
    draw_line(ax, (5.8, 5.3), (3.75, 5.3))
    draw_arrow(ax, (3.75, 5.3), (3.75, 5.15), style='->')

    draw_line(ax, (8.2, 6.05), (8.2, 5.3))
    draw_line(ax, (10.8, 6.05), (10.8, 5.3))
    draw_line(ax, (8.2, 5.3), (10.25, 5.3))
    draw_line(ax, (10.8, 5.3), (10.25, 5.3))
    draw_arrow(ax, (10.25, 5.3), (10.25, 5.15), style='->')

    # Concat
    draw_box(ax, 7, 2.8, 2, 0.5, 'Concat', color='#FCF3CF', fontsize=9)
    draw_line(ax, (3.75, 3.5), (3.75, 2.8))
    draw_line(ax, (10.25, 3.5), (10.25, 2.8))
    draw_line(ax, (3.75, 2.8), (6, 2.8))
    draw_line(ax, (10.25, 2.8), (8, 2.8))

    # DWConv (local enhancement)
    draw_arrow(ax, (7, 2.55), (7, 2.2), style='->')
    draw_box(ax, 7, 1.9, 2.5, 0.5, 'DWConv (Local)', color='#FADBD8', fontsize=9)

    # Gate
    draw_arrow(ax, (7, 1.65), (7, 1.3), style='->')
    draw_box(ax, 7, 1.0, 1.5, 0.5, '× Gate g', color='#E8DAEF', fontsize=9)

    # G branch
    ax.text(11.5, 7.6, 'G', ha='center', va='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='#E8DAEF', edgecolor='black'))
    draw_line(ax, (5.5, 7.6), (5.5, 7.3), color='#7D3C98', linestyle='--')
    draw_line(ax, (5.5, 7.3), (11.5, 7.3), color='#7D3C98', linestyle='--')
    draw_line(ax, (11.5, 7.3), (11.5, 1.0), color='#7D3C98', linestyle='--')
    draw_arrow(ax, (11.5, 1.0), (7.75, 1.0), color='#7D3C98', style='->')

    # Output projection
    draw_arrow(ax, (7, 0.75), (7, 0.4), style='->')
    draw_box(ax, 7, 0.2, 2, 0.4, 'Output Proj', color='#D5DBDB', fontsize=9)

    plt.tight_layout()
    plt.savefig('E:/Code/paper/rt-detr/figures/pola_attention.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('E:/Code/paper/rt-detr/figures/pola_attention.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 2: PolaAttention saved.")


# ============================================================
# Figure 3: ACGF Detail (Revised - clearer data flow)
# ============================================================
def draw_acgf():
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.set_xlim(0, 16)
    ax.set_ylim(-1, 15)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(8, 14.5, 'ACGF: Adaptive Channel-wise Gated Feedforward',
            ha='center', va='center', fontsize=14, weight='bold')

    # ============ Two inputs ============
    draw_box(ax, 4, 13.7, 2.8, 0.5, 'x (current feature)', color='#AED6F1', fontsize=9)
    draw_box(ax, 12, 13.7, 3, 0.5, 'x_spatial (original)', color='#ABEBC6', fontsize=9)

    # ============ Main Branch ============
    # Project in
    draw_arrow(ax, (4, 13.45), (4, 13.1), style='->')
    draw_box(ax, 4, 12.8, 2.2, 0.5, 'Conv 1x1\n(C -> 2H)', color='#D5DBDB', fontsize=8)

    # DWConv
    draw_arrow(ax, (4, 12.55), (4, 12.2), style='->')
    draw_box(ax, 4, 11.9, 2, 0.5, 'DWConv 3x3', color='#D5DBDB', fontsize=8)

    # Split
    draw_arrow(ax, (4, 11.65), (4, 11.3), style='->')
    draw_box(ax, 4, 11.0, 1.5, 0.5, 'Split', color='#FCF3CF', fontsize=9)

    # x1 and x2 branches
    draw_line(ax, (4, 10.75), (4, 10.4))
    draw_line(ax, (4, 10.4), (2.5, 10.4))
    draw_line(ax, (4, 10.4), (5.5, 10.4))
    draw_arrow(ax, (2.5, 10.4), (2.5, 10.1), style='->')
    draw_arrow(ax, (5.5, 10.4), (5.5, 10.1), style='->')

    draw_box(ax, 2.5, 9.8, 1.2, 0.5, 'x1', color='#AED6F1', fontsize=10)
    draw_box(ax, 5.5, 9.8, 1.2, 0.5, 'x2', color='#D4AC0D', fontsize=10)  # 黄色突出x2

    # x2 的说明标注
    ax.text(6.3, 9.8, '(content feature)', ha='left', va='center', fontsize=7, style='italic', color='#7D6608')

    # ============ Spatial Branch (Expanded) ============
    spatial_box = FancyBboxPatch((9.5, 9.5), 5, 4.0,
                                  boxstyle="round,pad=0.02,rounding_size=0.2",
                                  facecolor='#D5F5E3', edgecolor='#1E8449', linewidth=2)
    ax.add_patch(spatial_box)
    ax.text(12, 13.2, 'Spatial Branch', ha='center', va='center', fontsize=10, weight='bold', color='#145A32')

    # Spatial branch internal components
    draw_arrow(ax, (12, 13.45), (12, 12.9), style='->')
    draw_box(ax, 12, 12.6, 2, 0.5, 'AvgPool 2x2', color='#ABEBC6', fontsize=8)

    draw_arrow(ax, (12, 12.35), (12, 12.0), style='->')
    draw_box(ax, 12, 11.7, 2.5, 0.5, 'Conv 3x3 + LN + ReLU', color='#ABEBC6', fontsize=8)

    draw_arrow(ax, (12, 11.45), (12, 11.1), style='->')
    draw_box(ax, 12, 10.8, 2.5, 0.5, 'Conv 3x3 + LN + ReLU', color='#ABEBC6', fontsize=8)

    draw_arrow(ax, (12, 10.55), (12, 10.2), style='->')
    draw_box(ax, 12, 9.9, 2, 0.5, 'Upsample x2', color='#ABEBC6', fontsize=8)

    # Spatial output label
    ax.text(12, 9.5, 's (spatial feature)', ha='center', va='center', fontsize=8, style='italic', color='#145A32')

    # ============ Fusion: x1 + spatial ============
    # x1 goes down
    draw_arrow(ax, (2.5, 9.55), (2.5, 9.0), style='->')

    # spatial goes left
    draw_line(ax, (12, 9.5), (12, 8.5))
    draw_line(ax, (12, 8.5), (6, 8.5))

    # Concat point
    draw_line(ax, (2.5, 9.0), (2.5, 8.5))
    draw_line(ax, (2.5, 8.5), (4, 8.5))
    draw_line(ax, (6, 8.5), (4, 8.5))

    draw_box(ax, 4, 8.0, 2.8, 0.5, 'Concat + Conv 1x1', color='#FCF3CF', fontsize=8)
    draw_arrow(ax, (4, 8.5), (4, 8.25), style='->')

    # DWConv after fusion
    draw_arrow(ax, (4, 7.75), (4, 7.4), style='->')
    draw_box(ax, 4, 7.1, 2, 0.5, 'DWConv 3x3', color='#D5DBDB', fontsize=8)

    # x1' label
    draw_arrow(ax, (4, 6.85), (4, 6.5), style='->')
    draw_box(ax, 4, 6.2, 1.5, 0.5, "x1'", color='#AED6F1', fontsize=10)
    ax.text(4.9, 6.2, '(spatial-enhanced)', ha='left', va='center', fontsize=7, style='italic', color='#2874A6')

    # ============ Single-direction Gating ============
    gating_box = FancyBboxPatch((1, 4.0), 10, 1.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.15",
                                 facecolor='#FEF9E7', edgecolor='#B7950B', linewidth=2)
    ax.add_patch(gating_box)
    ax.text(6, 5.55, 'Single-direction Gating', ha='center', va='center',
            fontsize=10, weight='bold', color='#7D6608')

    # GELU(x1')
    draw_box(ax, 3.5, 4.8, 2, 0.5, "GELU(x1')", color='#D4E6F1', fontsize=9)

    # Multiply symbol
    draw_circle(ax, 6, 4.8, 0.25, '×', color='#FADBD8')

    # x2 copy in gating
    draw_box(ax, 8.5, 4.8, 1.2, 0.5, 'x2', color='#D4AC0D', fontsize=9)

    # Arrows in gating
    draw_arrow(ax, (4.5, 4.8), (5.75, 4.8), style='->')
    draw_arrow(ax, (7.9, 4.8), (6.25, 4.8), style='->')

    # F_gate output
    draw_arrow(ax, (6, 4.55), (6, 4.2), style='->')
    draw_box(ax, 6, 4.0, 1.8, 0.35, 'F_gate', color='#FCF3CF', fontsize=9)

    # Connect x1' to gating
    draw_line(ax, (4, 5.95), (4, 5.3))
    draw_line(ax, (4, 5.3), (3.5, 5.3))
    draw_arrow(ax, (3.5, 5.3), (3.5, 5.05), style='->')

    # ============ x2 分叉点 - 关键！============
    # x2 从顶部分出两条路径
    # 路径1: 去 Single-direction Gating
    # 路径2: 去 Channel-wise Adaptive Gating 的 (1-gate)*x2

    # x2 的主干向下延伸
    draw_line(ax, (5.5, 9.55), (5.5, 6.8), color='#D4AC0D', linewidth=2)

    # 分叉点标记
    draw_circle(ax, 5.5, 6.8, 0.15, '', color='#D4AC0D')

    # 路径1: 去 Single-direction Gating (水平向右)
    draw_line(ax, (5.5, 6.8), (8.5, 6.8), color='#D4AC0D', linewidth=2)
    draw_line(ax, (8.5, 6.8), (8.5, 5.3), color='#D4AC0D', linewidth=2)
    draw_arrow(ax, (8.5, 5.3), (8.5, 5.05), color='#D4AC0D', style='->')

    # 路径2: 继续向下去 (1-gate)*x2
    draw_line(ax, (5.5, 6.8), (5.5, 2.0), color='#D4AC0D', linewidth=2)
    draw_line(ax, (5.5, 2.0), (8, 2.0), color='#D4AC0D', linewidth=2)
    draw_arrow(ax, (8, 2.0), (8, 1.75), color='#D4AC0D', style='->')

    # x2 路径标注
    ax.text(7.0, 6.5, 'x2 path 1', ha='center', va='center', fontsize=7, color='#7D6608')
    ax.text(5.8, 4.0, 'x2 path 2', ha='left', va='center', fontsize=7, color='#7D6608')

    # ============ Channel-wise Adaptive Gating ============
    channel_box = FancyBboxPatch((1, 0.3), 10, 3.2,
                                  boxstyle="round,pad=0.02,rounding_size=0.15",
                                  facecolor='#F5EEF8', edgecolor='#7D3C98', linewidth=2)
    ax.add_patch(channel_box)
    ax.text(6, 3.25, 'Channel-wise Adaptive Gating', ha='center', va='center',
            fontsize=10, weight='bold', color='#6C3483')

    # gate 参数说明
    gate_box = FancyBboxPatch((1.3, 2.2), 2.4, 0.8,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='white', edgecolor='#7D3C98', linewidth=1)
    ax.add_patch(gate_box)
    ax.text(2.5, 2.8, 'Learnable gate', ha='center', va='center', fontsize=8, weight='bold', color='#6C3483')
    ax.text(2.5, 2.4, 'gate = σ(w_g)', ha='center', va='center', fontsize=9, color='#7D3C98')

    # F_gate 输入
    draw_arrow(ax, (6, 4.0), (6, 3.5), style='->')
    draw_line(ax, (6, 3.5), (4.5, 3.5))

    # gate * F_gate
    draw_box(ax, 4.5, 2.5, 2, 0.5, 'gate × F_gate', color='#E8DAEF', fontsize=9)
    draw_arrow(ax, (4.5, 3.5), (4.5, 2.75), style='->')

    # (1-gate) * x2
    draw_box(ax, 8, 2.5, 2.2, 0.5, '(1-gate) × x2', color='#E8DAEF', fontsize=9)
    # x2 已经通过黄色线连接到这里

    # 解释文字
    ax.text(4.5, 1.9, 'spatial-modulated', ha='center', va='center', fontsize=7, style='italic', color='#6C3483')
    ax.text(8, 1.9, 'original content', ha='center', va='center', fontsize=7, style='italic', color='#6C3483')

    # Plus symbol
    draw_circle(ax, 6.25, 1.2, 0.25, '+', color='#FADBD8')

    # Arrows to plus
    draw_line(ax, (4.5, 2.25), (4.5, 1.2))
    draw_arrow(ax, (4.5, 1.2), (6.0, 1.2), style='->')
    draw_line(ax, (8, 2.25), (8, 1.2))
    draw_arrow(ax, (8, 1.2), (6.5, 1.2), style='->')

    # 输出公式
    ax.text(6.25, 0.65, 'gate×F_gate + (1-gate)×x2', ha='center', va='center', fontsize=9, weight='bold', color='#6C3483')

    # ============ Output ============
    draw_arrow(ax, (6.25, 0.3), (6.25, 0.0), style='->')
    draw_box(ax, 6.25, -0.3, 2.2, 0.4, 'Conv 1x1 (Proj out)', color='#D5DBDB', fontsize=8)

    # ============ Residual Connection ============
    draw_line(ax, (0.5, 13.7), (0.5, -0.3), color='#E74C3C', linestyle='--', linewidth=2)
    draw_line(ax, (0.5, -0.3), (5.1, -0.3), color='#E74C3C', linestyle='--', linewidth=2)

    ax.text(0.2, 6.5, 'Residual (+x)', ha='center', va='center', fontsize=8, color='#E74C3C',
            rotation=90, weight='bold')

    # Final output
    draw_arrow(ax, (6.25, -0.5), (6.25, -0.8), style='->')
    ax.text(6.25, -1.0, 'Output = x + Proj_out(...)', ha='center', va='center', fontsize=9, weight='bold')

    # ============ 右侧解释框 ============
    ann_box = FancyBboxPatch((12, 3.0), 3.8, 6.0,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor='#FAFAFA', edgecolor='#888888', linewidth=1)
    ax.add_patch(ann_box)

    ax.text(13.9, 8.7, 'Key Concepts:', ha='center', va='center', fontsize=10, weight='bold', color='#333333')

    # x2 explanation
    ax.text(12.2, 8.2, 'x2 (yellow line):', ha='left', va='center', fontsize=8, weight='bold', color='#D4AC0D')
    ax.text(12.2, 7.8, 'Content feature from Split', ha='left', va='center', fontsize=7, color='#666666')
    ax.text(12.2, 7.5, 'Used in TWO places:', ha='left', va='center', fontsize=7, color='#666666')
    ax.text(12.4, 7.2, '1. GELU(x1\') × x2 -> F_gate', ha='left', va='center', fontsize=7, color='#666666')
    ax.text(12.4, 6.9, '2. (1-gate) × x2', ha='left', va='center', fontsize=7, color='#666666')

    # gate explanation
    ax.text(12.2, 6.3, 'gate (per-channel):', ha='left', va='center', fontsize=8, weight='bold', color='#6C3483')
    ax.text(12.2, 5.9, 'Learnable, in [0,1]', ha='left', va='center', fontsize=7, color='#666666')
    ax.text(12.2, 5.6, 'gate→1: use F_gate', ha='left', va='center', fontsize=7, color='#666666')
    ax.text(12.2, 5.3, 'gate→0: use x2', ha='left', va='center', fontsize=7, color='#666666')

    # formula
    ax.text(12.2, 4.7, 'Formula:', ha='left', va='center', fontsize=8, weight='bold', color='#333333')
    ax.text(12.2, 4.3, 'out = gate × F_gate', ha='left', va='center', fontsize=7, color='#666666')
    ax.text(12.2, 4.0, '    + (1-gate) × x2', ha='left', va='center', fontsize=7, color='#666666')

    # meaning
    ax.text(12.2, 3.4, 'Meaning:', ha='left', va='center', fontsize=8, weight='bold', color='#333333')
    ax.text(12.2, 3.1, 'Weighted blend between', ha='left', va='center', fontsize=7, color='#666666')
    ax.text(12.2, 2.8, 'spatial-modulated & original', ha='left', va='center', fontsize=7, color='#666666')

    plt.tight_layout()
    plt.savefig('E:/Code/paper/rt-detr/figures/acgf_detail.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('E:/Code/paper/rt-detr/figures/acgf_detail.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Figure 3: ACGF saved.")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import os

    # Create figures directory
    os.makedirs('./figures', exist_ok=True)

    # Draw all figures
    draw_pase_overall()
    draw_pola_attention()
    draw_acgf()

    print("\nAll figures saved to E:/Code/paper/rt-detr/figures/")
    print("- pase_overall.png/pdf")
    print("- pola_attention.png/pdf")
    print("- acgf_detail.png/pdf")
