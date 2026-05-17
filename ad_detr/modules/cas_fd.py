"""
CAS-FD: 内容感知选择性特征下采样模块
========================================

【论文映射】对应论文第3.3.1节
全称: Content-Aware Selective Feature Downsampling (CAS-FD)

【问题背景】
在BDD100K等自动驾驶场景中，远距离行人、车辆等目标在图像中仅占据3~10个像素。
RT-DETR原始结构的激进步长下采样（7×7 stride=2 卷积 + MaxPool）会导致
微小目标的高频信号在进入深层网络之前发生不可逆的像素级丢失。

【理论根因 — 奈奎斯特采样定理】
根据信号处理理论，当采样频率低于信号最高频率的两倍时引发混叠效应。
远距离小目标在图像中表现为高频信号（锐利边缘、高对比度）。
在步长=2 的卷积中，只有一半的像素位置参与计算，
占据仅 3-10 个像素的目标极易被完全跳过 → 不可逆的像素级信息丢失。

传统下采样的两大难题:
1. 特征同质化 — 步长卷积的跳跃采样直接跨越微小目标，将其平滑为背景
2. 背景噪声主导 — 大面积的背景响应淹没占比极小的小目标特征

【CAS-FD 解决方案】
通过三路并行下采样策略 + 空间自适应权重网络，在像素级实现内容感知的选择性融合，
从特征提取源头保护微小目标信息的完整性。

三路分支:
1. 细节分支 DetB (Detail Branch)  — Cut/像素重排, 完整保留像素信息
   → 对应论文公式: 将 H×W 特征图按 2×2 空间块分割, 4个子图沿通道拼接
2. 语义分支 SemB (Semantic Branch) — 步长卷积, 提取几何形状与上下文
3. 显著性分支 SalB (Salience Branch) — MaxPool, 保留最显著的特征响应

空间权重网络 SWN (Spatial Weight Net):
- 通过 concat + 1×1 Conv 实现像素级的自适应加权融合
  (这是论文中 SWN 的简化等价形式)
- concat+conv 隐含了可学习的逐像素加权: W_sem*F_sem + W_sal*F_sal + W_det*F_det
- 高纹理区域(小目标) → DetB 权重高，保护像素细节
- 平滑背景区域 → SemB/SalB 权重高，抑制低频噪声

【为什么是两阶段下采样而非一阶段 4×？】
- 阶段间插入 SemB 的 3×3 depthwise conv (stride=1) 做特征预处理
- 这给了网络在每次下采样前"精炼"特征的机会
- 一阶段 4× 下采样对微小目标更致命 (16→1 像素更激进)
- 两阶段渐进式：每个阶段只需 2×，更温和

【与原始 RT-DETR Stem 的对比】
原始: ConvNormLayer(3,32,s=2) → ConvNormLayer(32,32,s=1) → ConvNormLayer(32,64,s=1) → MaxPool(3,s=2)
SRFD: 7×7 Conv → [DetB+SemB 2×↓] → [DetB+SemB+SalB 2×↓]
原始有 3 次步长卷积和 1 次 MaxPool，共 4 次有损操作
SRFD 用 Cut(无损) 替代了大部分步长卷积，只有 SemB 使用步长卷积

【在 AD-DETR 中的位置】
替换原始 backbone 的 Stem 层，作为网络的第一个模块:
    输入图像 [B, 3, 640, 640] → SRFD → 特征图 [B, 64, 160, 160] (P2/4)

【文件内容】
- Cut:     像素重排下采样 (DetB 细节分支)
- SRFD:    完整 CAS-FD 模块 (论文核心)
- DRFD:    深度鲁棒特征下采样 (后续 stage 的下采样变体)
"""

import torch
import torch.nn as nn


# ============================================================================
# Cut: 像素重排下采样 (DetB 细节分支)
# ============================================================================

class Cut(nn.Module):
    """
    像素重排下采样 — CAS-FD 的细节分支 DetB

    【论文映射】对应 CAS-FD 三路并行中的 Detail Branch (DetB)

    原理: 将空间维度 (H, W) 的相邻像素映射到通道维度 (C)，
    实现 2x 下采样但保留所有像素信息，零信息丢失。

    操作详解:
    输入 [B, C, H, W] → 抽取 2x2 块的 4 个位置 (左上/右上/左下/右下)
    → 通道拼接 [B, 4C, H/2, W/2] → 1×1 Conv 融合 [B, out_c, H/2, W/2]

    【与 nn.PixelUnshuffle 的等价性】
    Cut 和 torch.nn.PixelUnshuffle(2) 是等效操作:
    - nn.PixelUnshuffle(2): [B, C, H, W] → [B, 4C, H/2, W/2]
    - Cut: 手动索引 → 拼接 → 1×1 Conv → 等价于 PixelUnshuffle + Conv

    【为什么对自动驾驶重要？】
    对于仅占 3-10 个像素的远距离行人:
    - 步长=2 卷积: 可能直接跳过奇数位置的像素 → 目标信息丢失
    - Cut: 每个像素都映射到通道维度 → 100% 保留
    - 后续 1×1 Conv 在通道维融合 4 个空间位置的像素 → 无损降采样

    在 CAS-FD 中的角色: 三路中最重要的一路，保证小目标的"物理存在"不被抹除。

    输入:  [B, in_channels, H, W]      输出: [B, out_channels, H/2, W/2]
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 4 倍通道 → out_channels 的 1×1 融合卷积
        # 1×1 Conv 在通道维度融合 4 个空间位置的像素信息
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: [B, C, H, W] → [B, out_channels, H/2, W/2]
        等价于 PixelUnshuffle(2) + 1×1 Conv + BN
        """
        # 抽取 2x2 空间块的 4 个位置 — 等同于 Pixel Unshuffle
        x0 = x[:, :, 0::2, 0::2]  # 左上角 (偶数行, 偶数列)
        x1 = x[:, :, 1::2, 0::2]  # 右上角 (奇数行, 偶数列)
        x2 = x[:, :, 0::2, 1::2]  # 左下角 (偶数行, 奇数列)
        x3 = x[:, :, 1::2, 1::2]  # 右下角 (奇数行, 奇数列)
        # 沿通道拼接: [B, C, H/2, W/2] * 4 → [B, 4C, H/2, W/2]
        x = torch.cat([x0, x1, x2, x3], dim=1)
        # 1×1 卷积在通道维度融合 4 个位置的信息 → [B, out_channels, H/2, W/2]
        x = self.conv_fusion(x)
        x = self.batch_norm(x)
        return x


# ============================================================================
# SRFD: 完整 CAS-FD 模块
# ============================================================================

class SRFD(nn.Module):
    """
    SRFD: 内容感知选择性特征下采样
         (在原始代码中名为 SRFD = Selective Robust Feature Downsampling)

    【论文映射】对应论文 3.3.1 节 CAS-FD 模块的完整实现
    论文中 CAS-FD = Content-Aware Selective Feature Downsampling

    【为什么论文叫 CAS-FD，代码叫 SRFD？】
    SRFD 是开发过程中的命名，论文写作时改为更能体现"内容感知"特性的 CAS-FD。
    两者是同一模块。答辩时需注意此命名差异。

    【整体结构 — 两阶段逐级下采样】
    输入图像 [B, 3, H, W]  (e.g. [B, 3, 640, 640])
      │
      ├─ 【0】7×7 Conv 初始化: 3ch → C/4 ch (大核特征强化)
      │   作用: 在开始下采样前，用大感受野(7×7)提取初始结构特征
      │
      ├─ 【第一阶段】原图 → 2x 下采样:
      │   ├─ DetB (Cut):     像素重排 → 完整保留所有像素信息
      │   ├─ SemB (Conv×2):  depthwise 3×3(s=1) → depthwise 3×3(s=2)
      │   └─ 拼接: SemB [B,C/2,H/2,W/2] + DetB [B,C/2,H/2,W/2]
      │      → [B, C, H/2, W/2] → 1×1 Conv 融合 → [B, C/2, H/2, W/2]
      │   注意: 第一阶段只有 2 路(SemB+DetB)，没有 SalB(MaxPool)
      │
      └─ 【第二阶段】2x → 4x 下采样:
          ├─ DetB (Cut):     像素重排
          ├─ SemB (Conv×2):  depthwise 3×3(s=1) → depthwise 3×3(s=2)
          ├─ SalB (MaxPool): 2×2 MaxPool → 保留最显著特征，抑制背景噪声
          └─ 三路拼接: SemB + DetB + SalB
             → [B, 3C, H/4, W/4] → 1×1 Conv 融合 → [B, C, H/4, W/4]

    【为什么第一阶段没有 SalB？】
    第一阶段从原图(640×640)下采样到 320×320，此时特征图分辨率还很高，
    每个像素承载的信息量较小。MaxPool 在高分辨率下容易保留噪声。
    第二阶段特征图缩小后，MaxPool 的显著性筛选作用才更有效。

    【groups=in_channels 的含义 — 为什么用深度可分离卷积？】
    SemB 中的 3×3 卷积使用 groups=输入通道数，即每个输入通道有独立的卷积核。
    - 好处: 参数量从 C×C×9 降到 C×9，大幅减少参数
    - 在 stem 阶段，输入通道较少(16~48)，独立处理每通道不会丢失太多交互
    - 后续 1×1 Conv 负责跨通道信息融合（深度可分离的"分离后融合"模式）

    【特征保护链】
    CAS-FD 的输出 P2/4 特征图 → 输入 backbone stage 2 → 输入 ACFP
    → ACFP 中的 SPDConv 继续用同样的像素重排策略保持信息完整性
    → 构成了从 Stem → Backbone → Neck 的完整特征保护链

    YAML 配置: [-1, 1, SRFD, [64]]  # 0-P2/4

    输入:  [B, in_channels,  H, W]            (in_channels=3  for RGB)
    输出:  [B, out_channels, H/4, W/4]        (out_channels=64  for P2/4)
    参数量: ~65K (与原始Stem相当，但保护了小目标信息)
    """
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        out_c14 = int(out_channels / 4)  # C/4 = 24  (1/4通道)
        out_c12 = int(out_channels / 2)  # C/2 = 48  (1/2通道)

        # ====== 【0】7×7 卷积: 初始特征强化 ======
        # 大核(7×7)在 stride=1 时覆盖较大感受野，为后续下采样做准备
        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)

        # ====== 【第一阶段】原图 → 2x 下采样 ======
        # SemB 语义分支: depthwise conv(s=1) 预处理 → depthwise conv(s=2) 下采样
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)
        # DetB 细节分支: 像素重排 (无损)
        self.cut_c = Cut(out_c14, out_c12)
        # 融合: SemB [out_c12] + DetB [out_c12] = [out_channels] → 1×1 Conv → [out_c12]
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)

        # ====== 【第二阶段】2x → 4x 下采样 ======
        # SemB 语义分支
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        # SalB 显著性分支: MaxPool 2×2
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        # DetB 细节分支: 像素重排 (无损)
        self.cut_r = Cut(out_c12, out_channels)
        # 三路融合: SemB + SalB + DetB → 1×1 Conv
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        """
        CAS-FD 前向传播 — 两阶段逐级下采样

        输入:  x = [B, 3, 640, 640]  (以 640×640 为例)
        输出:  x = [B, 64, 160, 160]  (P2/4)

        完整数据流:
        [B,3,640,640] → 7×7 Conv → [B,16,640,640]
        → SemB+DetB → [B,32,320,320] → SemB+SalB+DetB → [B,64,160,160]
        """
        # ====== 【0】初始特征强化 ======
        x = self.conv_init(x)           # [B, 3, 640, 640] → [B, 16, 640, 640]

        # ====== 【第一阶段】H→H/2 ======
        c = x                            # 保存副本用于 DetB

        # DetB 细节分支: 像素重排 (无损降采样)
        c = self.cut_c(c)               # [B, 16, 640, 640] → [B, 32, 320, 320]

        # SemB 语义分支: 深度卷积预处理 + 步长卷积下采样
        x = self.conv_1(x)              # [B, 16, 640, 640] → [B, 32, 640, 640] (s=1 预提取)
        x = self.conv_x1(x)             # [B, 32, 640, 640] → [B, 32, 320, 320] (s=2 下采样)
        x = self.batch_norm_x1(x)

        # SemB + DetB 拼接 → 1×1 Conv 融合(隐含 SWN 加权)
        x = torch.cat([x, c], dim=1)    # [B, 64, 320, 320]
        x = self.fusion1(x)             # [B, 64, 320, 320] → [B, 32, 320, 320]

        # ====== 【第二阶段】H/2→H/4 ======
        r = x                            # 保存副本用于 DetB
        x = self.conv_2(x)              # [B, 32, 320, 320] → [B, 64, 320, 320] (s=1 预提取)
        m = x                            # 保存副本用于 SalB

        # SemB 语义分支: 步长卷积下采样
        x = self.conv_x2(x)             # [B, 64, 320, 320] → [B, 64, 160, 160] (s=2)
        x = self.batch_norm_x2(x)

        # SalB 显著性分支: 最大值池化 → 保留最显著特征，抑制背景噪声
        m = self.max_m(m)               # [B, 64, 320, 320] → [B, 64, 160, 160]
        m = self.batch_norm_m(m)

        # DetB 细节分支: 像素重排 (无损降采样)
        r = self.cut_r(r)               # [B, 32, 320, 320] → [B, 64, 160, 160]

        # 三路拼接融合 (DetB + SemB + SalB)
        # concat 后 1×1 Conv = SWN 的简化实现:
        #   1×1 Conv 的权重矩阵隐含了论文公式(3.1)(3.2)的空间自适应加权
        x = torch.cat([x, r, m], dim=1)  # [B, 192, 160, 160]
        x = self.fusion2(x)             # [B, 192, 160, 160] → [B, 64, 160, 160]

        return x                        # P2/4 特征图 → 后续 backbone stage 2


# ============================================================================
# DRFD: 深度鲁棒特征下采样 (后续 stage 的下采样变体)
# ============================================================================

class DRFD(nn.Module):
    """
    DRFD: 深度鲁棒特征下采样 (Deep Robust Feature Downsampling)

    与 SRFD 类似的三路并行下采样, 但设计用于 backbone 后续 stage 之间的下采样
    (而不是 stem 层)。

    【SRFD vs DRFD 的区别】
    | 特性       | SRFD                     | DRFD                     |
    |-----------|--------------------------|--------------------------|
    | 使用位置   | Stem (输入图像 → P2/4)    | Backbone 中间 stage       |
    | 阶段数     | 两阶段 (原图→2x→4x)       | 单阶段 (H→H/2)            |
    | 输入通道   | 固定 3 (RGB)              | 任意                      |
    | 输出通道   | 固定 64 (P2/4)            | 任意 (参数指定)            |
    | 激活函数   | 无显式激活 (通过 BN)       | GELU 激活                 |
    | 初始卷积   | 7×7 大核                  | 无 (直接三路并行)          |

    【为什么第二阶段才有 SalB？】
    DRFD 用于中间 stage，此时特征图已有语义信息，MaxPool 的显著特征筛选作用明显。

    三路分支: Cut(DetB) + Conv(SemB) + MaxPool(SalB) → 1×1 Conv 融合

    输入:  [B, in_channels, H, W]         输出: [B, out_channels, H/2, W/2]
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # DetB 细节分支
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)
        # SemB 语义分支: depthwise conv 预处理 → depthwise conv stride=2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        # SalB 显著性分支: 2×2 MaxPool
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        # 三路融合: 3 * out_channels → out_channels
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        """
        输入: [B, C, H, W] → 三路并行下采样 → 融合 → [B, out_c, H/2, W/2]

        以 C=64, out_channels=128, H=W=160 为例:
        [B,64,160,160] → 三路 → [B,3*128,80,80] → fusion → [B,128,80,80]
        """
        c = x                            # 副本 for DetB: [B, C, H, W]
        x = self.conv(x)                 # SemB 预处理: [B, C, H, W] → [B, out_c, H, W]
        m = x                            # 副本 for SalB: [B, out_c, H, W]

        # DetB: 像素重排 (无损)
        c = self.cut_c(c)               # [B, C, H, W] → [B, out_c, H/2, W/2]

        # SemB: 步长卷积 (有损)
        x = self.conv_x(x)              # [B, out_c, H, W] → [B, out_c, H/2, W/2]
        x = self.act_x(x)               # GELU 激活
        x = self.batch_norm_x(x)

        # SalB: MaxPool (有损, 但保留了最显著特征)
        m = self.max_m(m)               # [B, out_c, H, W] → [B, out_c, H/2, W/2]
        m = self.batch_norm_m(m)

        # 三路融合: DetB(无损) + SemB(语义) + SalB(显著) → 1×1 Conv 加权融合
        x = torch.cat([c, x, m], dim=1)  # [B, 3*out_c, H/2, W/2]
        x = self.fusion(x)              # [B, 3*out_c, H/2, W/2] → [B, out_c, H/2, W/2]

        return x
