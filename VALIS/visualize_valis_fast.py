import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
from skimage.transform import resize  # <--- 换回 skimage，但这次是在小图上跑，很快！

warnings.filterwarnings("ignore")

# ================= 配置区 =================
src_dir = "/data/guoxs/Syx/ROSIE/VALIS/Test_QPTIFF/HE-37/slides"
raw_he_name = "HE-37.ome.tiff"
raw_dapi_name = "CODEX-37_DAPI.ome.tiff"

reg_dir = "/data/guoxs/Syx/ROSIE/VALIS/Test_QPTIFF/HE-37/registered_ometiff"
reg_he_name = "HE-37.ome.tiff" 
reg_dapi_name = "CODEX-37_DAPI.ome.tiff"

out_dir = "/data/guoxs/Syx/ROSIE/VALIS/Test_QPTIFF/HE-37/visualize"
# ================= 核心逻辑 =================

def read_smart_thumb(base_path, filename, target_width=1024):
    """
    智能读取：只负责通过切片把超级大图变小，不负责缩放。
    """
    full_path = os.path.join(base_path, filename)
    if not os.path.exists(full_path):
        print(f"❌ 文件缺失: {filename}")
        return None
    
    try:
        with tifffile.TiffFile(full_path) as tif:
            series = tif.series[0]
            shape = series.shape
            
            # 1. 维度判断
            if len(shape) == 3 and shape[0] < 100: # (C, Y, X)
                h, w = shape[1], shape[2]
                is_channel_first = True
            elif len(shape) == 3: # (Y, X, C)
                h, w = shape[0], shape[1]
                is_channel_first = False
            else: # (Y, X)
                h, w = shape[0], shape[1]
                is_channel_first = False

            # 2. 计算切片步长 (让读进来的图大约是 target_width 的 2 倍大)
            step = 1
            if w > target_width * 2:
                step = int(w / (target_width * 2))
            
            print(f"   ℹ️ 读取 {filename}: 原图 {w}x{h} -> 切片 step={step}")

            # 3. 读取并切片
            raw_data = series.asarray()
            
            if step > 1:
                if is_channel_first:
                    img_view = raw_data[:, ::step, ::step] 
                else:
                    img_view = raw_data[::step, ::step]
            else:
                img_view = raw_data

            # 4. 转为标准 Numpy (float32)
            img = np.array(img_view, copy=True).astype(np.float32)

            # 5. 维度调整 -> (H, W, C)
            if is_channel_first: 
                img = img.transpose(1, 2, 0)
            
            # 6. 去除单通道维度 -> (H, W)
            if img.ndim == 3 and img.shape[2] == 1: 
                img = img[:, :, 0]

            return img

    except Exception as e:
        print(f"❌ 读取错误 {filename}: {e}")
        return None

def normalize_dapi(img):
    if img.max() == 0: return img
    p_low, p_high = np.percentile(img, (5, 99.8))
    img = np.clip(img, p_low, p_high)
    denom = p_high - p_low
    if denom > 0:
        img = (img - p_low) / denom
    return np.clip(img, 0, 1)

def normalize_he(img):
    if img.max() > 255: img /= 65535.0
    elif img.max() > 1.0: img /= 255.0
    return np.clip(img, 0, 1)

def draw_overlay_robust(ax, he_img, dapi_img, title):
    TARGET_WIDTH = 1024 

    h_he, w_he = he_img.shape[:2]
    if w_he == 0: return

    aspect_ratio = h_he / w_he
    target_height = int(TARGET_WIDTH * aspect_ratio)

    print(f"   🎨 正在绘制: {title} (目标尺寸: {TARGET_WIDTH}x{target_height})")

    try:
        # === 关键修改：使用 skimage.transform.resize ===
        # skimage 的 resize 接收 (Height, Width)，这点和 cv2 相反，要注意
        # preserve_range=True 保证数值不被乱改，方便我们后面自己归一化
        
        he_small = resize(he_img, (target_height, TARGET_WIDTH), preserve_range=True, anti_aliasing=True)
        dapi_small = resize(dapi_img, (target_height, TARGET_WIDTH), preserve_range=True, anti_aliasing=True)
        
    except Exception as e:
        print(f"⚠️ Resize Error (skimage): {e}")
        return

    # 归一化
    if he_small.ndim == 2: he_small = np.stack([he_small]*3, axis=-1)
    he_norm = normalize_he(he_small)
    dapi_norm = normalize_dapi(dapi_small)

    # 叠加
    green_overlay = np.zeros((target_height, TARGET_WIDTH, 4), dtype=np.float32)
    green_overlay[..., 1] = 1.0 
    green_overlay[..., 3] = dapi_norm * 0.7 

    ax.imshow(he_norm)
    ax.imshow(green_overlay, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

# ================= 主程序 =================

Path(out_dir).mkdir(parents=True, exist_ok=True)
print("🚀 [Skimage 稳健版] 开始处理...")

# 1. 读取
he_raw = read_smart_thumb(src_dir, raw_he_name)
dapi_raw = read_smart_thumb(src_dir, raw_dapi_name)

try:
    he_reg = read_smart_thumb(reg_dir, reg_he_name) 
    dapi_reg = read_smart_thumb(reg_dir, reg_dapi_name)
except:
    he_reg, dapi_reg = None, None
    print("⚠️ 未找到配准后图像")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# 2. 绘制左图
if he_raw is not None and dapi_raw is not None:
    draw_overlay_robust(axes[0], he_raw, dapi_raw, "Before Registration\n(DAPI)")
    print("✅ 左图绘制完成")
else:
    print("❌ 左图数据缺失")

# 3. 绘制右图
if he_reg is not None and dapi_reg is not None:
    draw_overlay_robust(axes[1], he_reg, dapi_reg, "After Registration\n(DAPI)")
    print("✅ 右图绘制完成")
else:
    axes[1].text(0.5, 0.5, "Result Not Available", ha='center')

out_path = os.path.join(out_dir, "HE-37_Skimage_Check.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n🎉 结果已保存: {out_path}")