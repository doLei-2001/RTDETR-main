import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
try:
    from mamba_ssm import Mamba
except:
    pass
import numbers, math
from einops import rearrange

__all__ = ['SBSM', 'SEFN']

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# ======================================== PASE-ACGF: SEFN (Spatial-Enhanced FFN) ========================================
# 【论文映射】对应论文3.4.3节 ACGF (Adaptive Channel-Gated FFN)
# SEFN = Spatial-Enhanced Feedforward Network = ACGF
# 功能: 在PASE编码器中替换原始逐点FFN，通过双输入设计将空间位置信息注入语义特征
# ======================================== PASE-ACGF: SEFN ========================================

class SEFN(nn.Module):
    """SEFN: Spatial-Enhanced Feedforward Network（空间增强前馈网络）

    【论文映射】对应论文3.4.3节 PASE 中的 ACGF（自适应通道门控前馈网络）
    ACGF = Adaptive Channel-wise Gated Feedforward，用于替换Transformer原始逐点FFN。

    设计目标：原始FFN仅在通道维度逐点变换，忽略空间位置关系。
    SEFN通过双输入设计 + 门控机制，将原始特征保留的空间位置信息注入语义特征。

    结构解析：
    1.【Spatial Branch 空间分支】辅助输入spatial包含编码器原始空间判别信息
       → AvgPool(2x) → Conv3x3×2 → Upsample → 提取空间上下文特征y
       （对应论文"下采样-卷积处理-上采样"路径）

    2.【Content Branch 内容分支】主输入x为注意力聚合后的语义特征
       → 1x1升维 → DWConv → 分割为x1, x2

    3.【门控融合】x1与y拼接融合 → GELU → 与x2逐元素相乘
       正向门控 G_forward = GELU(x1_fused) ⊙ x2，空间引导内容选择
       对应论文公式(3.5)的双向交互门控简化版

    输入: x ∈ [B, C, H, W], spatial ∈ [B, C, H, W]
    输出: [B, C, H, W]
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SEFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)    


        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)
        

    def forward(self, x, spatial):
        
        x = self.project_in(x)
        
        #### Spatial branch
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)  
        ####
        
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y),dim=1))
        x1 = self.dwconv_afterfusion(x1)
        
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        
        self.mamba2 = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        
    
    @autocast(enabled=False)
    def forward(self, x, pe, mask):
        x_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            self.to(x.dtype)
        B, C = x.shape[:2]
        B, C, H, W = x.shape

        reversed_x1 = x.clone()

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        
        x2 = x.transpose(-1,-2)
  
        reversed_x2 = x2.clone()

        reversed_x1[:,:,1::2,:] = x[:,:,1::2,:].flip(-1)
        
        reversed_x2[:,:,1::2,:] = x2[:,:,1::2,:].flip(-1)

        #### add positional embedding
        x1_flat = reversed_x1.reshape(B, C, n_tokens).transpose(-1, -2)
        
        x1_flat = x1_flat + pe[:n_tokens, :]

        x2_flat = reversed_x2.reshape(B, C, n_tokens).transpose(-1, -2)
        
        x2_flat = x2_flat + pe[:n_tokens, :]
        ###### end adding positional embedding

        x1_norm = self.norm(x1_flat)
        x1_mamba = self.mamba(x1_norm)
        
        x2_norm = self.norm2(x2_flat)
        x2_mamba = self.mamba2(x2_norm)

        out1 = x1_mamba.transpose(-1, -2).reshape(B, C, H, W)
        out1_clone2 = out1
        out1_clone2[:,:,1::2,:] = out1[:,:,1::2,:].flip(-1)
        
        out2 = x2_mamba.transpose(-1, -2).reshape(B, C, W, H)
        out2_clone2 = out2
        out2_clone2[:,:,1::2,:] = out2[:,:,1::2,:].flip(-1)
        out2_clone2 = out2_clone2.transpose(-1,-2)
        
        out = out1_clone2 + out2_clone2
        
        return out

def PositionalEncoding(d_model, max_len=5000):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.cuda()

## Snake Bi-Directional Sequence Modelling (SBSM)
class SBSM(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(SBSM, self).__init__()
        self.dim = dim
        
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        ##### Try Mamba
        self.attn = MambaLayer(dim)
        #####

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = SEFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x_spatial = x
        B, C, H, W = x.size()
        pos = PositionalEncoding(self.dim, H*W).to(x.device)
        x = x + self.attn(self.norm1(x), pos, None).to(x_spatial.dtype)
        x = x + self.ffn(self.norm2(x), x_spatial)

        return x