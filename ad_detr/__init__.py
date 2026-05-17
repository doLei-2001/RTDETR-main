"""
AD-DETR: 面向自动驾驶场景的改进 RT-DETR 目标检测算法
=====================================================

论文: 基于深度学习的自动驾驶目标检测算法研究

四个核心创新模块:
  CAS-FD  — 内容感知选择性特征下采样 (§3.3.1)
  ACFP    — 自适应协同融合金字塔 (§3.3.2)
  PASE    — 极性感知空间增强编码器 (§3.4)
  LDC     — 动态Inception轻量化混合器 (§4.2)

目录结构:
  ad_detr/
  ├── __init__.py          # 本文件
  ├── model.py             # 模型创建 + YAML装配过程详解
  ├── train.py             # 训练入口
  ├── configs/
  │   └── ad_detr.yaml     # 完整 AD-DETR 模型配置
  └── modules/
      ├── __init__.py       # 模块导出
      ├── _compat.py        # ultralytics 依赖导入层
      ├── _registry.py      # YAML 模块注册器
      ├── cas_fd.py         # CAS-FD (SRFD)
      ├── acfp.py           # ACFP (SPDConv, OmniKernel, CSPOmniKernel)
      ├── pase.py           # PASE (PolaAttention, ConvolutionalGLU)
      └── ldc.py            # LDC (DynamicInception系列)

快速开始:
  # 训练完整 AD-DETR
  python -m ad_detr.train --data data_bdd100k.yaml

  # 查看架构
  python -m ad_detr.model

  # 代码审查 — 直接从 Python 导入
  from ad_detr.modules import SRFD, CSPOmniKernel, PolaLinearAttention

YAML 装配原理:
  YAML 的每一行 [from, repeats, ModuleName, args] 被 parse_model() 解析:
  1. ModuleName 字符串通过 globals()[name] 转为 Python 类
  2. args 计算通道数后传给 __init__
  3. from 指定输入来自哪一层 (-1=上一层, [a,b]=concat)
  4. 所有层按顺序存入 nn.Sequential

  模块注册通过 _registry.py 完成: 将 ad_detr 类注入 ultralytics.nn.tasks 命名空间。
"""
__version__ = '1.0.0'
