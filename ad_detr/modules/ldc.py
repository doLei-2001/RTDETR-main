"""
LDC: 动态Inception轻量化混合器
==================================

【论文映射】对应论文第 4 章
全称: Lightweight Dynamic Inception Convolution Mixer (LDC)

【问题背景 — ResNet-18 主干的参数冗余】
第三章引入 CAS-FD、ACFP、PASE 三个功能模块后:
- mAP50: 59.6% → 62.3% (精度大幅提升 ✓)
- 参数量: 19.9M → 22.8M (参数增加 ✗)
- FPS: 136 → 76 (推理速度下降 44.1% ✗)

瓶颈分析 (论文 §3.5.3):
1. ACFP 引入 P2 特征层 → 高分辨率特征图 → GPU 内存带宽饱和
2. CAS-FD 的空间权重生成 + PASE 的极性分离 → 大量逐点运算 → 访存密集
3. 多分支 + 张量解耦操作 → 显存访问碎片化 → 硬件利用率下降

【根本原因 — 标准卷积的参数冗余】
ResNet-18 的 BasicBlock 使用标准卷积:
- 参数量 = C_in × C_out × K²
- 例: 64→64, 3×3 → 64×64×9 = 36,864 参数 (一个 BasicBlock)
- 空间提取 + 通道融合耦合 → 大量参数用于通道间全连接
- 实际上并不需要每对通道都独立交互

【轻量化设计原则】
深度可分离卷积 = 空间卷积(DW) + 通道融合(PW)
参数量: C×K² + C²  vs  标准卷积: C²×K²
比值: 1/K² + 1/C → 当 C=64, K=3: (1/9+1/64) ≈ 12.7% (参数量减少 87.3%)

但仅用深度可分离卷积替换标准卷积是不够的:
- 单一卷积核(K×K)难以应对自动驾驶的多尺度目标
- 深度可分离卷积的表达能力弱于标准卷积 → 需要额外机制补偿

【LDC 解决方案 — 两个子模块】

1. MBDC (多分支深度卷积子模块):
   DynamicInceptionDWConv2d
   - 三路并行深度卷积: 方形 3×3 + 横向 1×11 + 纵向 11×1
   - DKW 动态权重: GAP → 1×1 Conv → Softmax → 自适应三路融合
     对应论文公式: w_i = Softmax(Conv(GAP(X)))_i
   - 多尺度: 方形捕获局部, 带状捕获方向性长程依赖

2. CSM (通道分割混合器):
   DynamicInceptionMixer
   - 输入 C 通道 → split 为两组 C/2
   - 每组独立 MBDC, 使用不同核尺寸 [3,5]
   - 拼接后 1×1 Conv 跨组融合
   - 计算量: 2 × C/2 × K² vs C × K² (相同), 但每组的 hidden dim 减半

【为什么 M 默认为 11？】
1×11 的带状卷积核覆盖范围:
- 在 P2/4 (160×160) 上: 11/160 ≈ 6.9% 的宽度
- 在 P5/32 (20×20) 上: 11/20  = 55% 的宽度 → 覆盖过半
折中选择 11: 不足以覆盖全图(避免退化), 但足够捕获车道线等中距结构。

【效果 (BDD100K 数据集) — 论文表 4.1】
| 指标      | 第三章方法  | AD-DETR(加LDC) | 变化       |
|----------|------------|----------------|------------|
| mAP50    | 62.3%      | 62.7%          | ↑0.4       |
| 参数量    | 22.8M      | 14.1M          | ↓38.2%     |
| GFLOPs   | 62.3       | 58.4           | ↓6.3%      |
| FPS      | 76         | 130            | ↑71.1%     |

关键分析: 精度不降反升(+0.4 mAP50)说明标准卷积存在过参数化,
LDC 以更少的参数实现了更有效的特征提取。

【在 AD-DETR 中的位置】
替换 ResNet-18 backbone 中的所有 BasicBlock:
    原: Blocks(BasicBlock×2) stage 2-5
    新: C2f_DCMB stage 2-5 (+ 独立 Conv stride=2 做下采样)

【文件内容】
- DynamicInceptionDWConv2d: MBDC + DKW (论文 4.2.2 节)
- DynamicInceptionMixer:    CSM (论文 4.2.3 节)
- DynamicIncMixerBlock:     LDC 基础块 (替代 BasicBlock)
- C2f_DCMB:                 C2f 封装 (YAML 兼容)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ._compat import Conv, C2f, DropPath
from .pase import ConvolutionalGLU


# ============================================================================
# DynamicInceptionDWConv2d: MBDC + DKW (论文 4.2.2 节)
# ============================================================================

class DynamicInceptionDWConv2d(nn.Module):
    """
    DynamicInceptionDWConv2d: 动态 Inception 深度卷积

    【论文映射】论文 4.2.2 节 MBDC (多分支深度卷积子模块)
    MBDC = Multi-Branch Depthwise Convolution

    【三路并行深度卷积分支 — 论文公式 4.2】
    1. 方形卷积分支 (3×3):
       - 提取局部方形邻域内的纹理与结构特征
       - 感知目标边缘与形状的主要分支
       - 参数量: C × 9

    2. 横向带状卷积分支 (1×11):
       - 非对称卷积 1×11, 捕获水平方向长程依赖
       - 适用于车载视角下的道路走向、车道线、水平排列目标
       - 参数量: C × 11

    3. 纵向带状卷积分支 (11×1):
       - 非对称卷积 11×1, 垂直方向空间关系
       - 与横向分支正交, 形成对 2D 空间的双轴分解
       - 参数量: C × 11

    三路合计参数量: C × (9+11+11) = C × 31
    对比标准 3×3 Conv: C² × 9 → 当 C=64: 31C vs 576C² → 31×64 vs 576×64²

    【DKW: 动态卷积核权重 — 论文公式 4.3】
    w_i = Softmax(Conv_{1×1}(AvgPool(X)))_i

    机制详解:
    1. GAP (全局平均池化): [B, C, H, W] → [B, C, 1, 1]
       → 提取场景级宏观统计信息
    2. 1×1 Conv: [B, C, 1, 1] → [B, 3C, 1, 1]
       → 为三路分支各生成 C 个权重
    3. Reshape + Softmax: [B, 3, C, 1, 1]
       → dim=0 上 Softmax 归一化, 三路权重之和=1

    DKW 使模型感知场景类型, 自适应调整各分支贡献:
    - 密集停车场(多目标, 高纹理) → 方形分支权重↑ (局部纹理重要)
    - 稀疏高速路(少目标, 远距离) → 带状分支权重↑ (远程上下文重要)

    【与普通 Inception 的区别】
    - 普通 Inception: 固定 1×1 Conv 融合权重 (静态)
    - 本模块: DKW 动态生成融合权重 (输入自适应)

    输入:  [B, in_channels, H, W]    输出: [B, in_channels, H, W]
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # 三路深度卷积 (depthwise = groups=in_channels)
        self.dwconv = nn.ModuleList([
            # 方形 K×K: 局部纹理与结构特征提取
            nn.Conv2d(in_channels, in_channels, square_kernel_size,
                      padding=square_kernel_size // 2, groups=in_channels),
            # 横向 1×M: 水平长程依赖, 道路走向/车道线感知
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size),
                      padding=(0, band_kernel_size // 2), groups=in_channels),
            # 纵向 M×1: 垂直空间关系, 与横向正交互补
            nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1),
                      padding=(band_kernel_size // 2, 0), groups=in_channels)
        ])

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

        # DKW: 动态卷积核权重生成器 (论文公式 4.3)
        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # GAP: [B,C,H,W] → [B,C,1,1]
            nn.Conv2d(in_channels, in_channels * 3, 1)  # 生成三路权重: [B,3C,1,1]
        )

    def forward(self, x):
        """
        x: [B, C, H, W] → DKW 生成权重 → 三路卷积加权融合 → [B, C, H, W]

        DKW 权重生成详解:
        x → GAP → [B, C, 1, 1] → 1×1 Conv → [B, 3C, 1, 1]
        → rearrange → [3, B, C, 1, 1] → Softmax(dim=0) → [3, B, C, 1, 1]
        → x_dkw[i] 广播到 [B, C, H, W] × dwconv_i(x)

        最终: output = BN(SiLU(Σ_i w_i · DWConv_i(x)))
        对应论文公式 4.2: Y = SiLU(BN(Σ w_i · DWConv_i(X)))
        """
        # DKW: [B, C, H, W] → [B, 3C, 1, 1] → rearrange → [3, B, C, 1, 1]
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)          # 三路权重归一化 (论文公式4.3)

        # 三路深度卷积 × DKW 权重 → 加权求和
        # 论文公式 4.2: Y = SiLU(BN(∑ w_i · DWConv_i(X)))
        x = torch.stack([
            self.dwconv[i](x) * x_dkw[i]
            for i in range(len(self.dwconv))
        ]).sum(0)

        return self.act(self.bn(x))


# ============================================================================
# DynamicInceptionMixer: CSM 通道分割混合器 (论文 4.2.3 节)
# ============================================================================

class DynamicInceptionMixer(nn.Module):
    """
    DynamicInceptionMixer: 动态Inception混合器

    【论文映射】对应论文 4.2.3 节 CSM (通道分割混合器)
    CSM = Channel Split Mixer

    【通道分割策略】
    输入 [B, C, H, W] → 沿 dim=1 均分为两组:
    - Group 1 [B, C/2, H, W] → MBDC(kernel=3) 处理
    - Group 2 [B, C/2, H, W] → MBDC(kernel=5) 处理
    - concat → [B, C, H, W] → 1×1 Conv 跨组融合

    【为什么两组的 kernel_size 不同 (3 vs 5)？】
    不同的方形核尺寸提供互补的感受野:
    - Group 1 (k=3): 小感受野 → 侧重局部细节、纹理
    - Group 2 (k=5): 大感受野 → 侧重中层结构、邻域上下文
    两组处理相同的输入特征, 但从不同空间尺度提取信息 → 多尺度互补

    【计算量分析】
    不分割: MBDC 处理 C 通道, 三路 3×3,1×11,11×1 各处理 C 个通道
    分割后: 每个 MBDC 处理 C/2 通道
    - 深度卷积计算量 = 2 × (C/2 × K²) = C × K² (相同)
    - 但每个 MBDC 内部的 hidden activation 尺寸减半
    - 实际 GPU 上的显存占用和带宽压力降低约 50%

    结构:
    x → [split C/2, C/2] → [MBDC(k1), MBDC(k2)] → concat → 1×1 Conv → y

    输入:  [B, channel, H, W]    输出: [B, channel, H, W]
    """
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 2  # 每组通道数

        # 两组 MBDC, 每组 C/2 通道, 不同方形核尺寸
        self.convs = nn.ModuleList([])
        for ks in kernels:
            # band_kernel_size = ks*3+2: 保证带状核远大于方形核
            # ks=3 → band=11, ks=5 → band=17
            self.convs.append(
                DynamicInceptionDWConv2d(min_ch, ks, ks * 3 + 2)
            )
        # 跨组信息融合: 1×1 Conv (逐点卷积)
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        """
        x: [B, C, H, W] → 通道分割 → 分组MBDC → concat → 1×1融合 → [B, C, H, W]
        """
        _, c, _, _ = x.size()
        # 通道均分为两组
        x_group = torch.split(x, [c // 2, c // 2], dim=1)
        # 每组独立 MBDC 处理 (不同核尺寸)
        x_group = torch.cat([
            self.convs[i](x_group[i])
            for i in range(len(self.convs))
        ], dim=1)
        # 跨组信息融合: 1×1 Conv 实现通道间交互
        x = self.conv_1x1(x_group)
        return x


# ============================================================================
# DynamicIncMixerBlock: LDC 基础块 (替代 BasicBlock)
# ============================================================================

class DynamicIncMixerBlock(nn.Module):
    """
    DynamicIncMixerBlock: LDC 基础块

    【论文映射】这是 LDC 模块的完整 block 实现。
    用于替换 ResNet-18 backbone 中的所有 BasicBlock。

    【MetaFormer 设计模式】
    MetaFormer 认为: Transformer/MLP-Mixer 等架构的成功不在于 Attention,
    而在于其通用的架构范式: TokenMixer + ChannelMLP + residual + norm

    本 Block 采用了 MetaFormer 风格:
        x = x + TokenMixer(Norm(x))    # MBDC+CSM 替代 Attention/Conv
        x = x + MLP(Norm(x))           # ConvolutionalGLU 替代 FFN

    其中:
    - TokenMixer = DynamicInceptionMixer (MBDC+DKW+CSM, 多尺度动态特征混合)
    - MLP         = ConvolutionalGLU (ACGF轻量版, 空间感知通道混合)

    【与 BasicBlock 的对比】
    | 组件        | BasicBlock       | LDC Block                        |
    |------------|------------------|----------------------------------|
    | 空间混合    | 3×3 Conv         | MBDC 三路 + DKW + CSM 通道分割     |
    | 通道混合    | 3×3 Conv(附带)    | ConvolutionalGLU (显式分离)       |
    | 参数量      | 2×C²×9           | C×31 + C×2×hidden                |
    | 下采样方式  | stride=2 in conv | 独立 Conv(stride=2)              |
    | 动态性      | 静态卷积核         | DKW 输入自适应权重                |

    【LayerScale 机制】
    LayerScale 初始化值 = 1e-2 → 训练初期"关闭"Block 的贡献
    → 模型退化为恒等映射 → 逐步学习 Block 的贡献
    → 相比直接学习完整 Block, LayerScale 使训练更稳定

    【为什么用 BatchNorm 而非 LayerNorm？】
    - MetaFormer 原版用 LayerNorm, 但 vision tasks 中 BN 效果通常更好
    - BN 在 batch 维度归一化, 保留通道间的差异 → 对卷积特征更友好
    - LN 在通道维度归一化, 抹平通道差异 → 对序列特征更友好
    LDC Block 在卷积 feature map 上操作, 故用 BN。

    输入:  [B, dim, H, W]    输出: [B, dim, H, W]
    """
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # Token Mixer: CSM (MBDC+DKW+通道分割)
        self.mixer = DynamicInceptionMixer(dim)
        # Channel MLP: ACGF 轻量版 (深度卷积门控)
        self.mlp = ConvolutionalGLU(dim)

        # DropPath (随机深度) — 训练时随机跳过此Block
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # LayerScale: 可学习的逐通道缩放因子
        # 初始化 1e-2 → 训练初期 Block 贡献几乎为0 → 从恒等映射出发逐步学习
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """
        x: [B, dim, H, W] → LDC Block → [B, dim, H, W]

        MetaFormer 风格:
        1. Norm → Mixer(CSM) → LayerScale → DropPath → residual
        2. Norm → MLP(CGLU) → LayerScale → DropPath → residual
        """
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.mixer(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x))
        )
        return x


# ============================================================================
# C2f_DCMB: C2f 封装的 LDC 块 (YAML 兼容)
# ============================================================================

class C2f_DCMB(C2f):
    """
    C2f_DCMB: C2f 封装的 LDC 模块

    【论文映射】这是 LDC 在 YAML 配置中使用的形式。
    通过 C2f 的 CSP 结构封装 DynamicIncMixerBlock。

    【双重通道压缩架构】
    CSP (Cross Stage Partial):
        x → cv1(1×1) → split C/2, C/2
        → 一半恒等 + 一半经 n×LDC Block → concat → cv2(1×1)

    LDC 内部的 CSM (Channel Split Mixer):
        经过 LDC 的 C/2 通道 → split C/4, C/4
        → MBDC(k=3) + MBDC(k=5) → concat → 1×1

    双重压缩效果:
    - CSP: C→C/2, 50%通道直连 → 50% 参数节省
    - CSM: C/2→C/4, 进一步分组处理 → 内部计算量再降
    - 总效果: 等效于每个 Block 仅处理 25% 的原始通道

    【为什么 C2f_DCMB 继承 C2f？】
    继承 C2f 的 CSP 框架 (cv1 → split → m → cv2):
    - 训练兼容: YAML 解析器识别 C2f 子类, 自动注入 n 参数
    - 代码复用: 无需重写 CSP 的 split/concat/conv 逻辑
    - 仅需替换 self.m: Bottleneck → DynamicIncMixerBlock

    在 AD-DETR YAML 中的使用:
        backbone:
          - [-1, 1, C2f_DCMB, [128]]   # n=1, C2f 封装 LDC
          - [-1, 1, C2f_DCMB, [256]]   # n=1, 不同通道

    输入:  [B, c1, H, W]    输出: [B, c2, H, W]
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 关键替换: C2f 内部默认用 Bottleneck, 这里替换为 LDC Block
        self.m = nn.ModuleList(
            DynamicIncMixerBlock(self.c) for _ in range(n)
        )
