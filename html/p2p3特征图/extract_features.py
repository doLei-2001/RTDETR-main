"""
P2/P3 特征图可视化 — 对比传统下采样 vs CAS-FD
================================================

【用途】
论文 PPT 素材: 展示原始 RT-DETR 的下采样机制如何"忽略"小目标，
以及 CAS-FD (SRFD) 如何从源头保护小目标信息。

【原理】
- 传统 Stem: ConvNormLayer(3,32,s=2) → ConvNormLayer(32,32) → ConvNormLayer(32,64) → MaxPool(3,s=2)
  问题: stride=2 跳过一半像素, MaxPool 只保留最大值 → 小目标信息多次丢失
- CAS-FD (SRFD): 7×7Conv → DetB(Cut无损)+SemB(步长Conv) → DetB+SemB+SalB(MaxPool)
  优势: Cut 保留所有像素, 三路融合选择性保护小目标

【输出图片】
- 01_原始图像.png
- 02_传统Stem_P2特征热力图.png
- 03_CAS-FD_P2特征热力图.png
- 04_传统主干_P3特征热力图.png
- 05_CAS-FD_P3特征热力图.png
- 06_对比总览.png (6 合 1 大图)

【用法】
python extract_features.py

图片命名为 img.png 放在本目录下即可。
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
# 注意: DCNv4 等依赖在 import 时就需要 GPU，不能设 CUDA_VISIBLE_DEVICES=-1
# 如果有 GPU 就用 GPU，比 CPU 快很多

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cv2

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===============================================================
# 0. 路径设置
# ===============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
IMG_PATH = os.path.join(SCRIPT_DIR, 'img.png')
OUTPUT_DIR = SCRIPT_DIR

sys.path.insert(0, PROJECT_ROOT)

# ===============================================================
# 1. 特征提取器 — 通过 hook 截取中间层输出
# ===============================================================
class FeatureExtractor:
    """注册 hook 到目标层，捕获前向传播的中间特征

    用法:
        extractor = FeatureExtractor(model)
        extractor.register_hook(target_layer_index)
        model(image_tensor)
        features = extractor.get_features()
    """
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.handles = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach().cpu()
        return hook

    def register_layer(self, layer_idx, name):
        """对 model.model[layer_idx] 注册 hook"""
        if layer_idx < len(self.model.model):
            handle = self.model.model[layer_idx].register_forward_hook(
                self._hook_fn(name)
            )
            self.handles.append(handle)
            print(f"  ✓ Hook 注册: layer[{layer_idx}] → {name}")
        else:
            print(f"  ✗ 层 {layer_idx} 不存在 (模型共 {len(self.model.model)} 层)")

    def clear(self):
        self.features = {}
        for h in self.handles:
            h.remove()
        self.handles = []


# ===============================================================
# 2. 加载模型
# ===============================================================
def load_models():
    """加载传统 RT-DETR r18 和 SRFD 变体两个模型"""
    from ultralytics import RTDETR
    from ultralytics.nn.tasks import RTDETRDetectionModel

    print("=" * 60)
    print("  加载模型")
    print("=" * 60)

    # --- 模型 A: 传统 RT-DETR r18 ---
    print("\n[模型A] 传统 RT-DETR r18 — 步长卷积 + MaxPool Stem")
    yaml_r18 = os.path.join(PROJECT_ROOT, 'ultralytics', 'cfg', 'models', 'rt-detr', 'rtdetr-r18.yaml')
    model_traditional = RTDETRDetectionModel(yaml_r18, nc=80, verbose=False)
    # 使用随机权重即可 — 我们只关注特征响应的空间分布模式, 不需要训练好的权重
    # 随机权重下, 高频/低频响应的对比仍然能体现下采样机制的区别
    model_traditional.eval()

    # --- 模型 B: CAS-FD (SRFD) ---
    print("\n[模型B] AD-DETR — CAS-FD(SRFD) + LDC 主干")
    yaml_srfd = os.path.join(PROJECT_ROOT, 'rtdetr-SRFD-SOEP.yaml')
    if not os.path.exists(yaml_srfd):
        print("  ⚠ rtdetr-SRFD-SOEP.yaml 不存在, 尝试其他路径...")
        # 备用: 使用 rtdetr-SRFD-SOEP-MFM-Pola-DIMB.yaml
        yaml_srfd = os.path.join(PROJECT_ROOT, 'rtdetr-SRFD-SOEP-MFM-Pola-DIMB.yaml')
    model_srfd = RTDETRDetectionModel(yaml_srfd, nc=80, verbose=False)
    model_srfd.eval()

    print("\n  模型A 层数:", len(model_traditional.model))
    print("  模型B 层数:", len(model_srfd.model))
    return model_traditional, model_srfd


# ===============================================================
# 3. 提取特征
# ===============================================================
def extract_features(model_traditional, model_srfd, img_tensor):
    """对两个模型分别提取 P2 和 P3 级别的特征

    传统 r18 结构 (backbone 前 6 层):
      [0] ConvNormLayer(3,32,s=2) → P1/2
      [1] ConvNormLayer(32,32,s=1)
      [2] ConvNormLayer(32,64,s=1)
      [3] MaxPool2d(3,s=2)        → P2/4  ← 提取点
      [4] Blocks(64,BasicBlock×2) → P2/4 refined
      [5] Blocks(128,BasicBlock×2)→ P3/8  ← 提取点

    SRFD 结构 (backbone 前几层):
      [0] SRFD(3→64)             → P2/4  ← 提取点
      [1] C2f_DCMB(64→128)       → P2/4 refined
      [2] Conv(128→256,s=2)      → P3/8
      [3] C2f_DCMB(256→256)      → P3/8  ← 提取点
    """

    print("\n" + "=" * 60)
    print("  提取中间层特征")
    print("=" * 60)

    # === 模型 A: 传统 r18 ===
    print("\n[模型A] 注册 hook...")
    ext_a = FeatureExtractor(model_traditional)
    ext_a.register_layer(3, 'trad_p2')   # MaxPool 输出 → P2/4 (stem 最终输出)
    ext_a.register_layer(5, 'trad_p3')   # Blocks stage 3 → P3/8

    # === 模型 B: SRFD (CAS-FD) ===
    print("\n[模型B] 注册 hook...")
    ext_b = FeatureExtractor(model_srfd)
    ext_b.register_layer(0, 'srfd_p2')   # SRFD 输出 → P2/4 (CAS-FD stem 输出)
    # 找到 SRFD 模型 backbone 中 P3 对应的层
    # SRFD yaml backbone: [0]SRFD→P2/4, [1]C2f_DCMB→P2/4, [2]Conv→P3/8, [3]C2f_DCMB→P3/8
    ext_b.register_layer(3, 'srfd_p3')   # C2f_DCMB 输出 → P3/8

    # === 前向传播 ===
    print("\n  执行前向传播...")
    with torch.no_grad():
        _ = model_traditional(img_tensor)
        _ = model_srfd(img_tensor)

    # 合并两个 dict
    all_features = {}
    all_features.update(ext_a.features)
    all_features.update(ext_b.features)

    print(f"\n  提取到的特征:")
    for name, feat in all_features.items():
        print(f"    {name}: shape = {list(feat.shape)}")

    ext_a.clear()
    ext_b.clear()
    return all_features


# ===============================================================
# 4. 热力图生成
# ===============================================================
def feature_to_heatmap(feature):
    """
    将特征图转换为 2D 热力图

    方法: 取通道维度的最大值 (max across channels),
    这反映了"每个空间位置的最大特征响应强度"。
    小目标区域如果有信息保留 → 会有可观测的热力响应。

    输入: feature [1, C, H, W] 或 [B, C, H, W]
    输出: heatmap [H, W] — numpy array, 已归一化到 [0, 1]
    """
    if feature.dim() == 4:
        feature = feature[0]  # [C, H, W]
    # 通道维度取最大值 → 每个位置的最强响应
    heatmap = feature.abs().max(dim=0)[0].numpy()
    # 归一化到 [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


def create_heatmap_overlay(image_np, heatmap, alpha=0.6):
    """
    将热力图叠加到原始图像上

    参数:
        image_np: 原始图像 [H, W, 3] numpy (RGB, 0-1)
        heatmap:  热力图 [h, w] numpy (0-1)
        alpha:    热力图透明度
    返回:
        overlay:  [H, W, 3] numpy
    """
    # 上采样热力图到图像尺寸
    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # 使用 JET 色彩映射
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = heatmap_colored[..., ::-1] / 255.0  # BGR→RGB, 0-1

    # Alpha 混合
    overlay = image_np * (1 - alpha) + heatmap_colored * alpha
    overlay = np.clip(overlay, 0, 1)
    return overlay


# ===============================================================
# 5. 可视化主函数
# ===============================================================
def visualize(image_path, features, output_dir):
    """
    生成 6 张可视化图片:
    1. 原始图像
    2. 传统 Stem P2 热力图 + 叠加
    3. CAS-FD P2 热力图 + 叠加
    4. 传统主干 P3 热力图 + 叠加
    5. CAS-FD P3 热力图 + 叠加
    6. 对比总览大图 (6 合 1)
    """
    print("\n" + "=" * 60)
    print("  生成可视化")
    print("=" * 60)

    # 读取并预处理原始图像
    original = Image.open(image_path).convert('RGB')
    original_np = np.array(original).astype(np.float32) / 255.0

    # 创建专用热力图色彩映射 (白→黄→橙→红→深红)
    colors = ['white', 'yellow', 'orange', 'red', 'darkred']
    cmap_heat = LinearSegmentedColormap.from_list('heat_custom', colors, N=256)

    # 提取各热力图
    hm_trad_p2 = feature_to_heatmap(features['trad_p2'])
    hm_srfd_p2 = feature_to_heatmap(features['srfd_p2'])
    hm_trad_p3 = feature_to_heatmap(features['trad_p3'])
    hm_srfd_p3 = feature_to_heatmap(features['srfd_p3'])

    # 创建叠加图
    overlay_trad_p2 = create_heatmap_overlay(original_np, hm_trad_p2)
    overlay_srfd_p2 = create_heatmap_overlay(original_np, hm_srfd_p2)
    overlay_trad_p3 = create_heatmap_overlay(original_np, hm_trad_p3)
    overlay_srfd_p3 = create_heatmap_overlay(original_np, hm_srfd_p3)

    # ======== 图1: 原始图像 ========
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.imshow(original_np)
    ax1.set_title('原始图像 (Raw Image)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    fig1.tight_layout()
    path1 = os.path.join(output_dir, '01_原始图像.png')
    fig1.savefig(path1, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"  ✓ {path1}")

    # ======== 图2: 传统 Stem P2 ========
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    # 左: 纯热力图
    im2a = axes2[0].imshow(hm_trad_p2, cmap=cmap_heat, interpolation='bilinear')
    axes2[0].set_title('传统 Stem P2 特征热力图\n(ConvNormLayer×3 + MaxPool)', fontsize=11)
    axes2[0].axis('off')
    plt.colorbar(im2a, ax=axes2[0], fraction=0.046, pad=0.04)
    # 右: 叠加到原图
    axes2[1].imshow(overlay_trad_p2)
    axes2[1].set_title('热力图叠加原图\n红色=高响应 蓝色=低响应', fontsize=11)
    axes2[1].axis('off')
    fig2.suptitle('传统 Stem → P2/4 特征响应', fontsize=14, fontweight='bold',
                  color='#d93025')
    fig2.tight_layout()
    path2 = os.path.join(output_dir, '02_传统Stem_P2特征热力图.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  ✓ {path2}")

    # ======== 图3: CAS-FD P2 ========
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    im3a = axes3[0].imshow(hm_srfd_p2, cmap=cmap_heat, interpolation='bilinear')
    axes3[0].set_title('CAS-FD (SRFD) P2 特征热力图\n(DetB+SemB+SalB 三路并行)', fontsize=11)
    axes3[0].axis('off')
    plt.colorbar(im3a, ax=axes3[0], fraction=0.046, pad=0.04)
    axes3[1].imshow(overlay_srfd_p2)
    axes3[1].set_title('热力图叠加原图\n红色=高响应 蓝色=低响应', fontsize=11)
    axes3[1].axis('off')
    fig3.suptitle('CAS-FD (SRFD) → P2/4 特征响应', fontsize=14, fontweight='bold',
                  color='#0f9d58')
    fig3.tight_layout()
    path3 = os.path.join(output_dir, '03_CAS-FD_P2特征热力图.png')
    fig3.savefig(path3, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print(f"  ✓ {path3}")

    # ======== 图4: 传统 P3 ========
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    im4a = axes4[0].imshow(hm_trad_p3, cmap=cmap_heat, interpolation='bilinear')
    axes4[0].set_title('传统主干 P3 特征热力图\n(Blocks BasicBlock×2, P3/8)', fontsize=11)
    axes4[0].axis('off')
    plt.colorbar(im4a, ax=axes4[0], fraction=0.046, pad=0.04)
    axes4[1].imshow(overlay_trad_p3)
    axes4[1].set_title('热力图叠加原图', fontsize=11)
    axes4[1].axis('off')
    fig4.suptitle('传统 ResNet-18 主干 → P3/8 特征响应', fontsize=14, fontweight='bold',
                  color='#d93025')
    fig4.tight_layout()
    path4 = os.path.join(output_dir, '04_传统主干_P3特征热力图.png')
    fig4.savefig(path4, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    print(f"  ✓ {path4}")

    # ======== 图5: CAS-FD P3 ========
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
    im5a = axes5[0].imshow(hm_srfd_p3, cmap=cmap_heat, interpolation='bilinear')
    axes5[0].set_title('CAS-FD + LDC P3 特征热力图\n(C2f_DCMB, P3/8)', fontsize=11)
    axes5[0].axis('off')
    plt.colorbar(im5a, ax=axes5[0], fraction=0.046, pad=0.04)
    axes5[1].imshow(overlay_srfd_p3)
    axes5[1].set_title('热力图叠加原图', fontsize=11)
    axes5[1].axis('off')
    fig5.suptitle('CAS-FD + LDC 主干 → P3/8 特征响应', fontsize=14, fontweight='bold',
                  color='#0f9d58')
    fig5.tight_layout()
    path5 = os.path.join(output_dir, '05_CAS-FD_P3特征热力图.png')
    fig5.savefig(path5, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig5)
    print(f"  ✓ {path5}")

    # ======== 图6: 对比总览 (6 合 1) ========
    fig6, axes6 = plt.subplots(2, 3, figsize=(18, 11))

    # 第一行: P2 级别的对比
    axes6[0, 0].imshow(original_np)
    axes6[0, 0].set_title('原始图像', fontsize=13, fontweight='bold')
    axes6[0, 0].axis('off')

    im6_1 = axes6[0, 1].imshow(hm_trad_p2, cmap=cmap_heat, interpolation='bilinear')
    axes6[0, 1].set_title('传统Stem P2/4\n(步长卷积+MaxPool)', fontsize=13, color='#d93025')
    axes6[0, 1].axis('off')
    plt.colorbar(im6_1, ax=axes6[0, 1], fraction=0.046, pad=0.04)

    im6_2 = axes6[0, 2].imshow(hm_srfd_p2, cmap=cmap_heat, interpolation='bilinear')
    axes6[0, 2].set_title('CAS-FD P2/4\n(三路并行+无损Cut)', fontsize=13, color='#0f9d58')
    axes6[0, 2].axis('off')
    plt.colorbar(im6_2, ax=axes6[0, 2], fraction=0.046, pad=0.04)

    # 第二行: P3 级别的对比
    axes6[1, 0].imshow(original_np)
    axes6[1, 0].set_title('原始图像', fontsize=13, fontweight='bold')
    axes6[1, 0].axis('off')

    im6_3 = axes6[1, 1].imshow(hm_trad_p3, cmap=cmap_heat, interpolation='bilinear')
    axes6[1, 1].set_title('传统主干 P3/8\n(ResNet BasicBlock)', fontsize=13, color='#d93025')
    axes6[1, 1].axis('off')
    plt.colorbar(im6_3, ax=axes6[1, 1], fraction=0.046, pad=0.04)

    im6_4 = axes6[1, 2].imshow(hm_srfd_p3, cmap=cmap_heat, interpolation='bilinear')
    axes6[1, 2].set_title('CAS-FD+LDC P3/8\n(C2f_DCMB 轻量化)', fontsize=13, color='#0f9d58')
    axes6[1, 2].axis('off')
    plt.colorbar(im6_4, ax=axes6[1, 2], fraction=0.046, pad=0.04)

    fig6.suptitle('P2/P3 特征热力图对比: 传统下采样 vs CAS-FD',
                  fontsize=16, fontweight='bold', y=1.02)
    fig6.tight_layout()
    path6 = os.path.join(output_dir, '06_对比总览.png')
    fig6.savefig(path6, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig6)
    print(f"  ✓ {path6}")

    return path6


# ===============================================================
# 6. 主程序
# ===============================================================
def main():
    # 检查输入图片
    if not os.path.exists(IMG_PATH):
        print("=" * 60)
        print("  错误: 找不到输入图片!")
        print("=" * 60)
        print(f"\n  请将自动驾驶场景图片放到:")
        print(f"  {IMG_PATH}")
        print(f"\n  命名为: img.png")
        print(f"\n  支持: jpg, png, bmp 等常见格式")
        sys.exit(1)

    print("=" * 60)
    print("  AD-DETR P2/P3 特征图可视化")
    print("  传统下采样 vs CAS-FD 对比")
    print("=" * 60)
    print(f"\n  输入图片: {IMG_PATH}")
    print(f"  输出目录: {OUTPUT_DIR}")

    # ---- 加载图片 ----
    img = Image.open(IMG_PATH).convert('RGB')
    w, h = img.size
    print(f"  图片尺寸: {w}×{h}")

    # Resize 到 640×640 (模型输入尺寸)
    img_640 = img.resize((640, 640), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img_640)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    print(f"  预处理后: {list(img_tensor.shape)}")

    # ---- 加载模型 ----
    model_trad, model_srfd = load_models()

    # ---- 提取特征 ----
    features = extract_features(model_trad, model_srfd, img_tensor)

    # ---- 可视化 ----
    result_path = visualize(IMG_PATH, features, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)
    print(f"\n  输出文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png') and f != os.path.basename(IMG_PATH):
            print(f"    {os.path.join(OUTPUT_DIR, f)}")
    print(f"\n  核心输出: {result_path}")
    print(f"  建议用于 PPT: 06_对比总览.png (6合1 大图)")


if __name__ == '__main__':
    main()
