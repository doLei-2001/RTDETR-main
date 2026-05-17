"""
AD-DETR 训练入口
================

用法:
    python -m ad_detr.train --data data_bdd100k.yaml --epochs 300

论文消融实验 (BDD100K):
    # Baseline
    python -m ad_detr.train --config rtdetr-r18 --data data_bdd100k.yaml

    # + CAS-FD
    python -m ad_detr.train --config rtdetr-SRFD --data data_bdd100k.yaml

    # + ACFP
    python -m ad_detr.train --config rtdetr-SRFD-SOEP --data data_bdd100k.yaml

    # + PASE
    python -m ad_detr.train --config rtdetr-SRFD-SOEP-MFM-Pola --data data_bdd100k.yaml

    # 完整 AD-DETR
    python -m ad_detr.train --config ad_detr --data data_bdd100k.yaml

跨数据集验证 (KITTI):
    python -m ad_detr.train --config ad_detr --data data_kitti.yaml --epochs 300
"""
import os, sys, warnings, argparse
warnings.filterwarnings('ignore')


def _find_yaml(config_name):
    """查找 YAML 配置文件, 返回绝对路径"""
    # 1. ad_detr/configs/ 下的文件
    local = os.path.join(os.path.dirname(__file__), 'configs', f'{config_name}.yaml')
    if os.path.exists(local):
        return local

    # 2. 项目根目录下的文件 (如 rtdetr-r18.yaml, rtdetr-SRFD-SOEP.yaml)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_yaml = os.path.join(root, f'{config_name}.yaml')
    if os.path.exists(root_yaml):
        return root_yaml

    # 3. ultralytics 内置的 rt-detr 配置
    builtin = os.path.join(root, 'ultralytics', 'cfg', 'models', 'rt-detr', f'{config_name}.yaml')
    if os.path.exists(builtin):
        return builtin

    raise FileNotFoundError(
        f'找不到配置文件: {config_name}.yaml\n'
        f'搜索路径:\n  {local}\n  {root_yaml}\n  {builtin}'
    )


def train(config='ad_detr', data='data_bdd100k.yaml', imgsz=640,
          epochs=300, batch=4, workers=4, device='0',
          project='runs/train', name='exp', weights=None,
          resume=None, cache=False):
    """
    AD-DETR 训练函数

    参数:
        config:  模型配置名 (ad_detr, rtdetr-r18, rtdetr-SRFD-SOEP 等)
        data:    数据集 yaml 路径
        imgsz:   输入尺寸 (默认 640)
        epochs:  训练轮数 (论文=300, 无早停)
        batch:   批次大小
        workers: 数据加载线程数
        device:  GPU 编号
        project: 输出根目录
        name:    实验名
        weights: 预训练权重路径
        resume:  续训 checkpoint 路径
        cache:   是否缓存图片
    """
    # ---- 注册 ad_detr 模块到 ultralytics 命名空间 ----
    from ad_detr.modules._registry import register
    register()

    from ultralytics import RTDETR

    yaml_path = _find_yaml(config)

    print(f'配置: {yaml_path}')
    print(f'数据: {data}')
    print(f'轮数: {epochs}  batch: {batch}  imgsz: {imgsz}')

    model = RTDETR(weights) if weights else RTDETR(yaml_path)

    model.train(
        data=data, cache=cache, imgsz=imgsz, epochs=epochs,
        batch=batch, workers=workers, device=device,
        resume=resume, project=project, name=name,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AD-DETR 训练')
    parser.add_argument('--config', default='ad_detr', help='模型配置名')
    parser.add_argument('--data', default='data_bdd100k.yaml', help='数据集配置')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs/train')
    parser.add_argument('--name', default='exp-ad-detr')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()

    train(
        config=args.config, data=args.data, imgsz=args.imgsz,
        epochs=args.epochs, batch=args.batch, workers=args.workers,
        device=args.device, project=args.project, name=args.name,
        weights=args.weights, resume=args.resume, cache=args.cache,
    )
