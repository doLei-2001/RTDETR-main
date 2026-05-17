"""
ad_detr 依赖导入层
==================
本文件统一管理 paper 模块对 ultralytics 基础类的依赖。
所有导入均来自 ultralytics 原始实现，保证代码行为完全一致。
"""
# 基础卷积 — 用于 CSPOmniKernel, C2f_DCMB
from ultralytics.nn.modules.conv import Conv, autopad

# C2f 基类 — C2f_DCMB 继承自它
from ultralytics.nn.modules.block import C2f

# DropPath — 用于 DynamicIncMixerBlock
from timm.layers import DropPath

# LayerNorm — 与 transformer.py 中的 PASE 使用的完全一致
# 注意: 不是 block.py 中的 BiasFree LayerNorm!
# transformer.py line 33 的版本，支持 channels_first
from torch.nn import LayerNorm as _TorchLayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch


class LayerNorm(nn.Module):
    """与 ultralytics/nn/extra_modules/transformer.py:33 完全一致的 LayerNorm

    支持 channels_first (NCHW) 格式，PASE 编码器中 TransformerEncoderLayer_Pola 使用此实现。
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
