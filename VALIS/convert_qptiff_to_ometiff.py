import tifffile
import numpy as np
import os

# ==========================================
#              核心转换函数
# ==========================================

def read_qptiff_and_convert(qptiff_path, out_path, pixel_size, is_he=False):
    """
    读取 QPTIFF 并转换为 OME-TIFF
    :param qptiff_path: 输入路径
    :param out_path: 输出路径
    :param pixel_size: 物理分辨率 (单位: um/pixel)。20x填0.5, 10x填1.0
    :param is_he: True=处理HE(RGB), False=处理CODEX(提取DAPI)
    """
    print(f"🔄 正在转换: {os.path.basename(qptiff_path)}")
    print(f"   📏 设定分辨率: {pixel_size} µm/px ({'H&E' if is_he else 'CODEX'})")
    
    try:
        with tifffile.TiffFile(qptiff_path) as tif:
            # 读取 Series 0 (通常是最高分辨率层级)
            raw_data = tif.series[0].asarray()
            print(f"   📖 原始数据形状: {raw_data.shape}, 类型: {raw_data.dtype}")

            # === H&E 处理逻辑 (RGB) ===
            if is_he:
                image_data = raw_data
                # 如果是 (3, H, W) -> 转为 (H, W, 3)
                if image_data.ndim == 3 and image_data.shape[0] == 3:
                    image_data = image_data.transpose(1, 2, 0)
                # 归一化为 uint8
                if image_data.dtype != np.uint8:
                    print("   ⚠️ 将 H&E 转换为 uint8...")
                    image_data = (image_data / 256).astype(np.uint8) if image_data.max() > 255 else image_data.astype(np.uint8)
                
                photometric_mode = 'rgb'
                # 元数据中写入正确的分辨率
                metadata = {
                    'PhysicalSizeX': pixel_size, 'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': pixel_size, 'PhysicalSizeYUnit': 'µm'
                }

            # === CODEX 处理逻辑 (提取 DAPI) ===
            else:
                dapi_idx = 0  # 假设 DAPI 是第 0 个通道
                
                # 判断形状是 (C, H, W) 还是 (H, W, C)
                if raw_data.ndim == 3:
                    if raw_data.shape[0] < 100: # (C, H, W)
                        image_data = raw_data[dapi_idx, :, :]
                    else: # (H, W, C)
                        image_data = raw_data[:, :, dapi_idx]
                else:
                    image_data = raw_data # 单通道

                print(f"   🧪 已提取单通道 (Index {dapi_idx}), 用于 DAPI 配准")
                photometric_mode = 'minisblack'
                
                # 元数据中写入正确的分辨率
                metadata = {
                    'PhysicalSizeX': pixel_size, 'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': pixel_size, 'PhysicalSizeYUnit': 'µm',
                    'Channel': {'Name': ['DAPI']} 
                }

            # === 写入 OME-TIFF ===
            print(f"   💾 正在写入 OME-TIFF: {image_data.shape} ...")
            with tifffile.TiffWriter(out_path, bigtiff=True) as out_tif:
                out_tif.write(
                    image_data,
                    photometric=photometric_mode,
                    tile=(512, 512),
                    compression='lzw',
                    metadata=metadata
                )
                
        print(f"✅ 转换成功: {out_path}\n")

    except Exception as e:
        print(f"❌ 转换失败 {qptiff_path}: {e}")
        import traceback
        traceback.print_exc()

# ================= 用户配置与执行 =================

# 1. 输入文件路径
he_qptiff_path = r"/data/guoxs/Syx/ROSIE/train_data_1/HE/HE-37/Scan2/HE-37_Scan2.qptiff"
codex_qptiff_path = r"/data/guoxs/Syx/ROSIE/train_data_1/广医肺癌多色/37-25.11.24/Scan1/37-25.11.24_Scan1.qptiff"

# 2. 输出目录
output_dir = "/data/guoxs/Syx/ROSIE/VALIS/Test_QPTIFF/HE-37/slides"
os.makedirs(output_dir, exist_ok=True)

# 3. 输出文件名
he_out_path = os.path.join(output_dir, "HE-37.ome.tiff")
dapi_out_path = os.path.join(output_dir, "CODEX-37_DAPI.ome.tiff")

if __name__ == "__main__":
    
    # --- 关键修改：分别为两个文件指定不同的分辨率 ---
    
    # 1. 转换 HE (20x -> 0.5 um/px)
    if os.path.exists(he_qptiff_path):
        read_qptiff_and_convert(
            he_qptiff_path, 
            he_out_path, 
            pixel_size=0.2485,  # <--- HE 是 20x
            is_he=True
        )
    else:
        print(f"⚠️ 找不到 HE 文件")

    # 2. 转换 CODEX (10x -> 1.0 um/px)
    if os.path.exists(codex_qptiff_path):
        read_qptiff_and_convert(
            codex_qptiff_path, 
            dapi_out_path, 
            pixel_size=0.5005,  # <--- CODEX 是 10x (分辨率更低，像素更大)
            is_he=False
        )
    else:
        print(f"⚠️ 找不到 CODEX 文件")