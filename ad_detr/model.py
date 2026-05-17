"""
AD-DETR 模型定义 + YAML 装配过程详解
=======================================

【论文映射】基于 RT-DETR 改进的自动驾驶目标检测算法 AD-DETR

本文件做三件事:
1. 详细解释 YAML 配置文件如何被解析为可执行的 PyTorch 模型
2. 逐模块说明 AD-DETR 的完整架构
3. 提供创建模型的 API

═══════════════════════════════════════════════════════════════════
YAML 装配原理
═══════════════════════════════════════════════════════════════════

ultralytics 使用 YAML 文件描述模型结构。每个 YAML 有 backbone 和 head 两个 section，
每一行是一个层，格式为 [from, repeats, module_name, args]。

【解析流程】parse_model() in ultralytics/nn/tasks.py
1. 读取 YAML → 得到 backbone 和 head 两个 list
2. 拼接: for layer in backbone + head:
3. 对每层:
   a. 解析 "from": -1 表示上一层, [5, 6] 表示从第5和第6层获取输入
   b. 通过 `globals()[module_name]` 把字符串转为 Python 类
      (这就是 _registry.py 做注册的原因)
   c. 计算通道数 c1, c2 (从上层输出和 args[0] 推断)
   d. 实例化: module = Class(*args)
   e. 存入 nn.Sequential
4. 返回完整的 nn.Sequential + 层的连接关系(save list)

【具体例子】看 ad_detr/configs/ad_detr.yaml 的第一行:
  - [-1, 1, SRFD, [64]]
    │   │  │     │
    │   │  │     └── args: [out_channels=64]
    │   │  └──────── module: 类名 'SRFD' → globals()['SRFD'] → SRFD(3, 64)
    │   └─────────── repeats: 1 → 实例化 1 次
    └─────────────── from: -1 → 输入来自上一层

再看 Concat 的例子:
  - [[-2, -1], 1, Concat, [1]]
     │          │   │       │
     │          │   │       └── args: [dim=1]
     │          │   └────────── module: Concat
     │          └────────────── repeats: 1
     └───────────────────────── from: [-2, -1] → 输入来自倒数第2层和倒数第1层
                                forward 时 torch.cat([layer[-2], layer[-1]], dim=1)

【backbone vs head】
- backbone 和 head 在 parse_model 中被拼接成一个大的 nn.Sequential
- backbone 中定义从属关系 save list (哪些层的输出需要保存供后续引用)
- head 的最后一层通常是 RTDETRDecoder (检测头)
- 模型整体是 backbone layers + head layers 的线性序列
- 分支和跳跃连接通过 "from" 索引和 Concat 层实现

【训练时如何运作】
1. RTDETRDetectionModel(yaml_path) → parse_model(yaml_dict) → nn.Sequential
2. RTDETRTrainer 持有 model, optimizer, dataloader
3. 每个 batch: image → model(image) → loss → backward → step
4. 验证: model(image) → postprocess → mAP 计算

═══════════════════════════════════════════════════════════════════
AD-DETR 架构总览
═══════════════════════════════════════════════════════════════════

backbone (主干网络):
  层0:  SRFD(CAS-FD)              [3,640,640]→[64,160,160] P2/4
  层1:  C2f_DCMB(LDC)×1          [64,160,160]→[128,160,160]
  层2:  Conv(s=2)                [128,160,160]→[256,80,80]  P3/8
  层3:  C2f_DCMB(LDC)×1          [256,80,80]→[256,80,80]
  层4:  Conv(s=2)                [256,80,80]→[384,40,40]   P4/16
  层5:  C2f_DCMB(LDC)×1          [384,40,40]→[384,40,40]
  层6:  Conv(s=2)                [384,40,40]→[384,20,20]   P5/32
  层7:  C2f_DCMB(LDC)×3          [384,20,20]→[384,20,20]

head (混合编码器+解码器):
  P5编码 (PASE):
    层8-10: Conv→PASE(PolaAttention+ACGF)→Conv  [256,20,20]
  FPN自顶向下:
    层11-15: Upsample→Conv→MFM→RepC3  P5→P4
    层16-20: Upsample→SPDConv(P2无损)→MFM→CSPOmniKernel→RepC3  (ACFP)
  PAN自底向上:
    层21-23: Conv(s=2)→MFM→RepC3  P3→P4
    层24-26: Conv(s=2)→MFM→RepC3  P4→P5
  Decoder:
    层27: RTDETRDecoder(nc,256,300,4,8,3) → 300检测框
"""

import os


def create_model(yaml_path=None, nc=80):
    """
    创建 AD-DETR 模型

    参数:
        yaml_path: YAML 配置文件路径 (None=使用默认 ad_detr.yaml)
        nc:        类别数 (BDD100K=10, KITTI=3, COCO=80)

    返回:
        RTDETRDetectionModel 实例

    用法:
        from ad_detr.model import create_model
        model = create_model(nc=10)
        model.train(data='data_bdd100k.yaml', epochs=300, imgsz=640)
    """
    from .modules._registry import register
    register()  # 将模块类注入 ultralytics 命名空间

    from ultralytics.nn.tasks import RTDETRDetectionModel

    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), 'configs', 'ad_detr.yaml')

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f'找不到 YAML 配置文件: {yaml_path}\n'
            f'请确认 ad_detr/configs/ad_detr.yaml 存在。'
        )

    return RTDETRDetectionModel(yaml_path, nc=nc)


def print_architecture():
    """打印 AD-DETR 完整架构, 用于论文答辩展示"""
    from .modules._registry import register
    register()
    from ultralytics.nn.tasks import RTDETRDetectionModel
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), 'configs', 'ad_detr.yaml')
    model = RTDETRDetectionModel(yaml_path, nc=80, verbose=True)

    print("""
╔═══════════════════════════════════════════════════════╗
║  YAML 装配说明                                        ║
╠═══════════════════════════════════════════════════════╣
║  每行格式: [from, repeats, ModuleClass, [args...]]   ║
║  from:  -1=上一层  [a,b]=concat第a,b层                ║
║  repeats: 模块重复次数 (乘depth_multiple)              ║
║  ModuleClass: 类名字符串, 由globals()[name]解析        ║
║  args: 传给 __init__ 的参数                            ║
╚═══════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    print_architecture()
