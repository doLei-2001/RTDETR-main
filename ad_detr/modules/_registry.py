"""
YAML 模块注册器
================
ultralytics 的 parse_model() 通过 `globals()[module_name]` 按字符串查找类。
本模块将 ad_detr 的类注入到 ultralytics.nn.tasks 模块的全局命名空间，
使 YAML 配置文件中的模块名字符串能被正确解析为 Python 类。

【为什么需要注册？】
parse_model() 定义在 ultralytics/nn/tasks.py 中，关键代码 (line 744):
    m = globals()[m]      # m 是类名字符串, 如 'SRFD', 'C2f_DCMB'
    m_ = m(*args)         # 实例化

这里的 globals() 是 tasks.py 模块的全局命名空间，它通过以下导入链获得模块类:
    tasks.py → from ultralytics.nn.extra_modules import *
            → extra_modules/__init__.py → from .block import *
                                        → from .transformer import *

ad_detr/modules/ 中的实现与原始代码完全一致，
注册后 YAML 配置可以直接使用类名来装配模型。

【使用方式】
    from ad_detr.modules._registry import register
    register()  # 必须在创建 RTDETRDetectionModel 之前调用
    model = RTDETRDetectionModel('ad_detr/configs/ad_detr.yaml')
"""
import ultralytics.nn.tasks as _tasks


def register():
    """将 ad_detr 模块类注入到 ultralytics.nn.tasks 的全局命名空间

    必须在 parse_model() 被调用之前执行(即 RTDETRDetectionModel 构造之前)。
    重复调用是安全的——已存在的类不会被覆盖。
    """
    from . import __all__ as _module_names
    import ad_detr.modules as _mod

    count = 0
    for name in _module_names:
        cls = getattr(_mod, name, None)
        if cls is not None and not hasattr(_tasks, name):
            setattr(_tasks, name, cls)
            count += 1

    # 验证关键类已注册
    _check = ['SRFD', 'CSPOmniKernel', 'PolaLinearAttention', 'C2f_DCMB',
              'TransformerEncoderLayer_Pola', 'TransformerEncoderLayer_Pola_CGLU']
    missing = [n for n in _check if not hasattr(_tasks, n)]
    if missing:
        raise RuntimeError(f'模块注册失败: {missing} 未能在 tasks 命名空间中找到')
    print(f'[ad_detr] 已注册 {count} 个模块类到 ultralytics.nn.tasks 命名空间')
