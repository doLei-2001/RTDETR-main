"""
AD-DETR 论文核心模块

本包包含毕业论文《基于深度学习的自动驾驶目标检测算法研究》中
提出的 AD-DETR 算法的所有核心模块实现。

模块清单:
├── _compat.py  — ultralytics 依赖导入层 (Conv, C2f, LayerNorm, DropPath)
├── cas_fd.py   — CAS-FD 内容感知选择性特征下采样 (§3.3.1)
├── acfp.py     — ACFP 自适应协同融合金字塔 (§3.3.2)
├── pase.py     — PASE 极性感知空间增强编码器 (§3.4)
└── ldc.py      — LDC 动态Inception轻量化混合器 (§4.2)

原始代码位置:
- ultralytics/nn/extra_modules/block.py    — SRFD, OmniKernel, CSPOmniKernel, DynamicInception系列
- ultralytics/nn/extra_modules/transformer.py — PolaLinearAttention, ConvolutionalGLU, TransformerEncoderLayer_Pola系列
"""

# CAS-FD
from .cas_fd import Cut, SRFD, DRFD

# ACFP
from .acfp import SPDConv, OmniKernel, CSPOmniKernel, FGM

# PASE
from .pase import (
    PolaLinearAttention,
    ConvolutionalGLU,
    TransformerEncoderLayer_Pola,
    TransformerEncoderLayer_Pola_CGLU,
    TransformerEncoderLayer_Pola_FMFFN,
)

# LDC
from .ldc import (
    DynamicInceptionDWConv2d,
    DynamicInceptionMixer,
    DynamicIncMixerBlock,
    C2f_DCMB,
)

__all__ = [
    # CAS-FD
    'Cut', 'SRFD', 'DRFD',
    # ACFP
    'SPDConv', 'OmniKernel', 'CSPOmniKernel', 'FGM',
    # PASE
    'PolaLinearAttention', 'ConvolutionalGLU',
    'TransformerEncoderLayer_Pola', 'TransformerEncoderLayer_Pola_CGLU',
    'TransformerEncoderLayer_Pola_FMFFN',
    # LDC
    'DynamicInceptionDWConv2d', 'DynamicInceptionMixer',
    'DynamicIncMixerBlock', 'C2f_DCMB',
]
