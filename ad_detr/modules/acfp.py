"""
ACFP: 自适应协同融合金字塔
=============================

【论文映射】对应论文第 3.3.2 节
全称: Adaptive Collaborative Fusion Pyramid (ACFP)

【问题背景】
RT-DETR 原始算法仅使用 P3、P4、P5 三层特征图。在密集小目标场景下，
8× 下采样的 P3 层上原本就少的像素进一步退化，目标区域不足 1 像素。

传统特征融合采用通道拼接（各层权重相等），隐含假设各层特征贡献相等。
但在自动驾驶中:
- 近处大目标 → 依赖 P5 高层语义
- 远处小目标 → 依赖 P2/P3 低层细节
固定权重无法适应尺度分布变化。

【直接引入 P2 检测头的困境】
- P2 特征图尺寸 = P3 的 4 倍 → neck 计算量平方级增长
- 传统步长卷积下采样会再次引入采样混叠 → 破坏 CAS-FD 建立的特征完整性
- 这是"引入高分辨率带来过多计算开销"与"保精度需要高分辨率"的矛盾

【ACFP 解决方案 — 三个子模块】
1. SPDConv (空间-深度卷积):
   P2 [B,C,160,160] → Space-to-Depth → [B,4C,80,80] → 3×3 Conv → [B,128,80,80]
   通过无损的空间→通道映射降低分辨率，不丢失任何像素信息。
   来自论文: "将空间维度的像素映射至通道维度，在降低空间分辨率的同时
   保留小目标的局部细节信息，确保细粒度信息在尺度转换中零损失传递。"

2. AWM (自适应加权模块, Adaptive Weighting Module):
   对应论文公式: 通过全局平均池化 + MLP + Softmax 生成自适应融合权重
   在代码中通过 MFM (Multi-Feature Merge) 模块实现。

3. CAFE (跨尺度自适应特征增强, Cross-scale Adaptive Feature Enhancement):
   对应 CSPOmniKernel — 内部 OmniKernel 提供全局+局部+大核三路多尺度增强。
   采用 CSP (Cross Stage Partial) 设计: 仅 25% 通道经过大核处理，
   借鉴部分卷积(Partial Convolution)思想降低计算量。

【论文公式对应】
- 自适应加权: W = Softmax(MLP(GAP(concat(F2, F3, Y4))))
- 最终融合: F_out = W2×F2 + W3×F3 + W4×Y4

【在 AD-DETR 中的位置】
neck (FPN/PAN) 中，处理 P2-P3 特征融合:
    backbone P2 → SPDConv(P2无损降维) → MFM融合(P2, P3, Y4) → CSPOmniKernel → RepC3 → PAN

【与 CAS-FD 的协同增益机理】
CAS-FD 提供的 P2 特征质量 → 直接决定 ACFP 的 AWM 表征上限
ACFP 通过 SPDConv 继续保持 CAS-FD 建立的特征完整性
两者构建了从 Stem → Backbone → Neck 的完整无损特征保护链

【文件内容】
- SPDConv:        空间-深度卷积 (P2 特征无损下采样)
- FGM:            频率门控模块 (OmniKernel 内部使用)
- OmniKernel:     全核多尺度特征增强 (CAFE 核心)
- CSPOmniKernel:  CSP 封装 OmniKernel (25%通道部分处理)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ._compat import Conv


# ============================================================================
# SPDConv: 空间-深度卷积
# ============================================================================

class SPDConv(nn.Module):
    """
    SPDConv: Space-to-Depth Convolution（空间-深度卷积）

    【论文映射】对应论文 3.3.2 节 ACFP 中的"空间-深度卷积模块"
    是 ACFP 无损引入 P2 特征的核心机制。

    原理: 将空间维度 (H,W) 的 2×2 块映射到通道维度 (C)，
    实现 2× 降采样但保留所有像素信息，零信息丢失。

    【SPDConv vs 步长卷积 vs MaxPool】
    | 操作          | 信息保留率 | 原理                                 |
    |--------------|-----------|--------------------------------------|
    | 步长卷积      | ~50%      | 步长=2 时仅对偶数位置采样               |
    | MaxPool 2×2  | ~25%      | 4 个像素中仅保留最大值                  |
    | SPDConv      | 100%      | 所有像素重排至通道维, 后续 Conv 加权融合  |

    【在 ACFP 中的完整流程】
    backbone P2 [B,C,160,160] → SPDConv → [B,128,80,80]
    → 与 P3 [B,128,80,80] 和 Y4 [B,128,80,80] 三路 MFM 融合
    → CSPOmniKernel 多尺度增强 → RepC3

    输入:  [B, inc, H, W]          输出: [B, ouc, H/2, W/2]
    降采样方式: Space-to-Depth (空间→通道，无信息丢失)
    """
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        # inc * 4 因为 2×2 空间块 → 4 个位置 → 4 倍通道
        self.conv = Conv(inc * 4, ouc, k=3)  # 3×3 Conv 在通道维融合 4 个空间位置

    def forward(self, x):
        """
        Space-to-Depth: 2×2块拆分 → 通道拼接 → 卷积融合

        x: [B, inc, H, W] → [B, ouc, H/2, W/2]

        以 inc=128, ouc=128, H=W=160 为例:
        [B,128,160,160] → cat 4×[B,128,80,80] → [B,512,80,80]
        → 3×3 Conv → [B,128,80,80]
        """
        x = torch.cat([
            x[..., ::2, ::2],    # 左上 (偶数行, 偶数列)
            x[..., 1::2, ::2],   # 右上 (奇数行, 偶数列)
            x[..., ::2, 1::2],   # 左下 (偶数行, 奇数列)
            x[..., 1::2, 1::2]   # 右下 (奇数行, 奇数列)
        ], dim=1)
        return self.conv(x)


# ============================================================================
# FGM: 频率门控模块 (OmniKernel 内部组件)
# ============================================================================

class FGM(nn.Module):
    """
    FGM: 频率门控模块 (Frequency Gating Module)

    OmniKernel 内部使用，通过频域操作增强特征判别能力。

    【频域门控原理】
    1. 输入 x [B,dim,H,W] → 两个 1×1 Conv → x1, x2 (两个视图)
    2. x2 通过 FFT 变换到频域 → x2_fft
       频域表示将图像分解为不同频率的正弦波分量
       - 低频: 平滑区域, 背景
       - 高频: 边缘, 纹理, 小目标
    3. x1 作为门控信号调制 x2_fft → out = x1 * x2_fft
       (在频域中"选择性增强或抑制某些频率成分")
    4. IFFT 回到空域 → 取幅值 → |IFFT(out)|
    5. 残差: alpha * 频域增强 + beta * 原始特征

    【可学习参数 alpha, beta 的作用】
    - alpha: 控制频域增强通道的强度 (初始化为 0, 逐步学习)
    - beta:  控制原始特征保留的强度 (初始化为 1, 完整保留)
    - 这种"从原始特征出发，逐步加强频域增强"的设计使训练稳定

    输入:  [B, dim, H, W]    输出: [B, dim, H, W]
    """
    def __init__(self, dim) -> None:
        super().__init__()
        # 深度卷积预处理: dim → 2*dim (每个通道独立处理)
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
        # 两个 1×1 Conv 生成门控/频域两路
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        # 可学习残差参数
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))  # 频域增强强度
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))    # 原始特征保留强度

    def forward(self, x):
        """
        x: [B, dim, H, W] → 频域门控 → [B, dim, H, W]
        """
        x1 = self.dwconv1(x)                     # 门控分支: [B, dim, H, W]
        x2 = self.dwconv2(x)                     # 频域分支: [B, dim, H, W]
        x2_fft = torch.fft.fft2(x2, norm='backward')  # FFT → 频域表示
        out = x1 * x2_fft                        # 门控调制: x1 ⊙ x2_fft
        out = torch.fft.ifft2(out, dim=(-2, -1))      # IFFT → 回到空域
        out = torch.abs(out)                     # 取模(去除复数相位)
        # 残差: alpha(频域增强) + beta(原始保留)
        return out * self.alpha + x * self.beta


# ============================================================================
# OmniKernel: 多尺度全核特征增强 (CAFE 核心)
# ============================================================================

class OmniKernel(nn.Module):
    """
    OmniKernel: 多尺度全核特征增强模块

    【论文映射】对应论文 3.3.2 节 ACFP 中 CAFE 模块的核心组件。
    集成全局、局部和大核三个并行分支，实现全尺度信息融合。

    【为什么需要 31×31 的大核？】
    自动驾驶场景的目标尺度跨度极大:
    - 近处车辆: 可能占据 200×200 像素 → 需要大感受野覆盖完整目标
    - 远处行人: 可能仅 3×3 像素 → 需要小感受野精确定位
    - 车道线/交叉口: 需要细长感受野 → 1×31/31×1 带状卷积

    31×31 的选择:
    - 7×7 太小: 无法完整覆盖近处大目标
    - 63×63 太大: 计算量巨大 + 在 P3 级 (80×80) 特征图上覆盖过半区域
    - 31×31 是折中: 在 20×20 P5 上覆盖 1/4 区域, 在 80×80 P3 上覆盖 1/8

    【三路分支详解】
    1. 全局分支 (FCA + SCA + FGM):
       通过 FCA (频率通道注意力) 捕获频域全局统计信息
       通过 SCA (空间通道注意力) 在空域进行通道级加权
       通过 FGM (频率门控) 融合频域和空域信息
       场景级语义 → 判断当前是停车场还是高速路

    2. 局部分支 (1×1 Conv):
       逐点卷积保留每个像素的独立响应
       维持小目标的边缘、纹理等高频信息 → 不被全局池化平滑掉

    3. 大核分支 (31×31 + 1×31 + 31×1):
       方形 31×31: 捕获大范围的空间上下文
       横向 1×31:  捕获水平方向长距离依赖 (道路走向)
       纵向 31×1:  捕获垂直方向空间关系 (与横向形成正交分解)
       三路均为 depthwise (groups=dim) → 参数量 = dim × 961 × 3

    【FCA 频率通道注意力的原理】
    通过 FFT 将特征变换到频域 → 在频域进行通道加权 → IFFT 回空域
    频域注意力可以捕捉到空域卷积难以感知的全局周期性模式。

    输入:  [B, dim, H, W]    输出: [B, dim, H, W]
    通道数不变, 空间尺寸不变。
    """
    def __init__(self, dim) -> None:
        super().__init__()
        ker = 31  # 大核尺寸
        pad = ker // 2  # 15

        # 输入/输出投影
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

        # ====== 多尺度深度卷积 (均为 depthwise, groups=dim) ======
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad),
                               stride=1, groups=dim)   # 横向带状: 捕获水平方向长程依赖
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0),
                               stride=1, groups=dim)   # 纵向带状: 与横向正交互补
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad,
                               stride=1, groups=dim)   # 方形大核: 方形大感受野
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0,
                               stride=1, groups=dim)   # 逐点: 局部细节保留

        self.act = nn.ReLU()

        # === SCA (空间通道注意力) ===
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # === FCA (频率通道注意力) ===
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)  # 频率门控融合 FCA 和 SCA

    def forward(self, x):
        """
        x: [B, dim, H, W] → 多尺度增强 → [B, dim, H, W]

        以 dim=64, H=W=80 为例:
        """
        out = self.in_conv(x)               # [B,64,80,80] 输入投影

        # === FCA: 频率通道注意力 ===
        # GAP → Conv → 频域注意力 [B,64,1,1]
        x_att = self.fac_conv(self.fac_pool(out))  # [B,64,1,1]
        x_fft = torch.fft.fft2(out, norm='backward')   # FFT 变换到频域 [B,64,80,80]
        x_fft = x_att * x_fft                         # 在频域施加通道注意力
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')  # IFFT 回空域
        x_fca = torch.abs(x_fca)                      # [B,64,80,80] 取幅值

        # === SCA: 空间通道注意力 ===
        x_att = self.conv(self.pool(x_fca))  # [B,64,1,1] 通道注意力权重
        x_sca = x_att * x_fca               # [B,64,80,80] 通道加权
        x_sca = self.fgm(x_sca)             # 频率门控精炼 [B,64,80,80]

        # === 多尺度融合 ===
        # x (残差) + 4个dwc分支 + FCA/SCA增强
        out = (x                               # 原始输入 (残差连接)
               + self.dw_13(out)               # 1×31 横向带状
               + self.dw_31(out)               # 31×1 纵向带状
               + self.dw_33(out)               # 31×31 方形大核
               + self.dw_11(out)               # 1×1 逐点局部
               + x_sca)                        # FCA+SCA+FGM 全局增强
        out = self.act(out)                    # ReLU 激活
        return self.out_conv(out)              # [B,64,80,80] 输出投影


# ============================================================================
# CSPOmniKernel: CSP 封装的 OmniKernel (CAFE 完整实现)
# ============================================================================

class CSPOmniKernel(nn.Module):
    """
    CSPOmniKernel: CSP 封装的 OmniKernel 模块

    【论文映射】对应论文 3.3.2 节 ACFP 中 CAFE 模块的完整实现。

    【CSP (Cross Stage Partial) 设计】
    输入 x [B, dim, H, W] → cv1(1×1) → split 为两路:
    - ok_branch [B, e*dim, H, W] → OmniKernel 深度多尺度增强
    - identity  [B, (1-e)*dim, H, W] → 恒等映射(不做任何处理)
    → cat(ok_branch, identity) → cv2(1×1) 跨组信息融合 → 输出

    【为什么 e=0.25？(而不是 0.5)】
    这是根据部分卷积(Partial Convolution)思想精心选择的:
    - e=0.5:   OmniKernel 的 31×31 大核计算量仍很大
    - e=0.125: 处理通道太少，增强效果不明显
    - e=0.25:  经验最佳平衡点:
      * OmniKernel 计算量 ↓75% (vs 全部通道)
      * 75% 恒等映射保留原始特征完整性
      * 1×1 Conv 提供充分的跨组信息交互

    【为什么需要恒等映射的 75%？】
    - 保留原始特征的"锚点" → 防止过度处理导致特征漂移
    - 恒等映射部分提供梯度直通路径 → 缓解 OmniKernel 深处理带来的梯度问题
    - 类似 ResNet 残差连接的思路: 学习残差比学习完整映射更容易

    【在 ACFP 中的完整流程】
    P2(SPDConv) + P3 + Y4 → MFM三路融合 → CSPOmniKernel → RepC3 → PAN后续

    输入:  [B, dim, H, W]    输出: [B, dim, H, W]
    参数:
        dim: 输入/输出通道数 (由 MFM 融合后的通道数决定)
        e:   OmniKernel 处理的通道比例 (默认 0.25)
    """
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        # cv1: 1×1 预处理 (dim → dim, 为 split 做准备)
        self.cv1 = Conv(dim, dim, 1)
        # cv2: 1×1 后融合 (dim → dim, 恢复通道)
        self.cv2 = Conv(dim, dim, 1)
        # 核心: OmniKernel 仅处理 e*dim 个通道
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        """
        x: [B, dim, H, W] → CSP + OmniKernel → [B, dim, H, W]

        以 dim=256, e=0.25 为例:
        x = [B, 256, H, W]
        → cv1 → [B, 256, H, W]
        → split: ok_branch [B, 64, H, W], identity [B, 192, H, W]
        → OmniKernel(ok_branch) → [B, 64, H, W]
        → cat → [B, 256, H, W]
        → cv2 → [B, 256, H, W]
        """
        x = self.cv1(x)                               # [B, dim, H, W] 预处理
        # 通道分割: ok_branch (经过OmniKernel) + identity (恒等)
        ok_branch, identity = torch.split(
            x,
            [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))],
            dim=1
        )
        # 深度处理(25%通道) + 恒等映射(75%通道) → 拼接 → 1×1 融合
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))
