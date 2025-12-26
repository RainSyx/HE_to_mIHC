import os
import numpy as np
import cv2
import pandas as pd
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from preprocess import register_he_codex_arrays, create_patch_label_table_from_arrays,visualize_registration

# ================= 配置路径 =================
ROOT_DIR = "/home/guoxs/Syx/ROISE/ROSIE_datatry"  # 请修改为你的服务器真实路径
HE_IMAGES_DIR = os.path.join(ROOT_DIR, "images")
CODEX_IMAGES_DIR = os.path.join(ROOT_DIR, "codex_images") # 新建的文件夹
OUTPUT_DIR = os.path.join(ROOT_DIR, "preprocessed_data")

# ================= 数据加载函数 =================

def get_he_array(sample_id):
    """读取 H&E Zarr (保持不变)"""
    zarr_path = os.path.join(HE_IMAGES_DIR, sample_id, "image.ome.zarr")
    if not os.path.exists(zarr_path):
        return None
    try:
        channels = []
        for i in range(3):
            url = os.path.join(zarr_path, str(i))
            reader = Reader(parse_url(url))
            nodes = list(reader())
            channels.append(nodes[0].data[0].compute())
        return np.stack(channels, axis=-1).astype(np.uint8)
    except Exception as e:
        print(f"  ❌ Error reading H&E: {e}")
        return None

def get_codex_array(sample_id):
    """
    【修改版】读取 PNG 格式的 CODEX 数据
    逻辑：
    1. 去 codex_images/{sample_id}/ 目录下找所有 png
    2. 找到 DAPI.png (或者包含 DAPI 字样的文件) 放在第一个通道用于配准
    3. 其他 png 依次堆叠
    """
    sample_dir = os.path.join(CODEX_IMAGES_DIR, sample_id)
    if not os.path.exists(sample_dir):
        print(f"  ❌ CODEX dir not found: {sample_dir}")
        return None, None, None

    # 1. 找到所有 PNG 文件
    files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.png')]
    if not files:
        print(f"  ❌ No PNGs found in {sample_dir}")
        return None, None, None

    # 2. 区分 DAPI 和其他 Marker
    # 假设文件名包含 "DAPI" (大小写不敏感)
    dapi_file = None
    marker_files = []
    
    for f in files:
        if "dapi" in f.lower():
            dapi_file = f
        else:
            marker_files.append(f)
            
    if dapi_file is None:
        print("  ⚠️ Warning: No 'DAPI' file found. Using the first file for registration.")
        dapi_file = files[0]
        marker_files = [f for f in files if f != dapi_file]

    # 排序其他 marker，保证顺序固定
    marker_files.sort()
    
    # 最终的通道列表：[DAPI, Marker1, Marker2, ...]
    all_files = [dapi_file] + marker_files
    channel_names = ["DAPI"] + [os.path.splitext(f)[0] for f in marker_files]
    
    print(f"  Found {len(all_files)} channels: {channel_names}")

    # 3. 读取图片并堆叠
    img_list = []
    for f in all_files:
        path = os.path.join(sample_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 强制读为灰度
        if img is None:
            print(f"  ❌ Failed to read {f}")
            return None, None, None
        img_list.append(img)
        
    # 堆叠成 (C, H, W)
    codex_array = np.stack(img_list, axis=0)
    
    # 返回数组、DAPI索引(永远是0)、所有Marker名称(存到csv里有用)
    return codex_array, 0, channel_names

# ================= 主流程 =================

# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # 遍历 HE 文件夹下的 ID
#     sample_ids = [d for d in os.listdir(HE_IMAGES_DIR) if os.path.isdir(os.path.join(HE_IMAGES_DIR, d))]
    
#     for sample_id in sample_ids:
#         print(f"\nProcessing {sample_id}...")
        
#         # 在 run_preprocess.py 的 main() 循环中找到这里：

#         # 1. 加载 H&E
#         # he_array_original = get_he_array(sample_id) # 将变量名改为 original
#         # if he_array_original is None: continue

#         he_array = get_he_array(sample_id) # 将变量名改为 original
#         if he_array is None: continue
        
#         # # ============================================================
#         # # 【新增模块：人为制造干扰 (Artificial Disturbance)】
#         # # 目的是为了检验算法能否把严重错位的图像修正回来
#         # # ============================================================
#         # print("  [TEST MODE] Applying artificial rotation and shift to H&E...")
#         # h_he, w_he = he_array_original.shape[:2]
#         # center = (w_he // 2, h_he // 2)
        
#         # # 定义干扰参数：旋转角度和平移距离
#         # angle = 20.0       # 旋转 20 度 (足够明显)
#         # shift_x = 30       # 向右平移 30 像素
#         # shift_y = -20      # 向上平移 20 像素
        
#         # # 1. 计算旋转矩阵
#         # M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
#         # # 应用旋转 (背景填充白色，避免黑边干扰视觉)
#         # he_disturbed = cv2.warpAffine(he_array_original, M_rot, (w_he, h_he), 
#         #                               borderMode=cv2.BORDER_CONSTANT, 
#         #                               borderValue=(255, 255, 255))
        
#         # # 2. 计算平移矩阵
#         # M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
#         # # 应用平移
#         # he_disturbed = cv2.warpAffine(he_disturbed, M_trans, (w_he, h_he),
#         #                               borderMode=cv2.BORDER_CONSTANT,
#         #                               borderValue=(255, 255, 255))
        
#         # # 【关键】：把干扰后的图像赋值给 he_array，让后面的流程处理它
#         # he_array = he_disturbed 
#         # # ============================================================


#         # 2. 加载 CODEX PNGs (这段不用变)
#         codex_array, dapi_idx, marker_names = get_codex_array(sample_id)
#         if codex_array is None: continue
        
#         print(f"  Arrays loaded. HE (Disturbed): {he_array.shape}, CODEX: {codex_array.shape}")
        
#         # 3. 配准 (这段不用变)
#         print("  Running registration...")
#         try:
#             dapi_img = codex_array[dapi_idx]
#             # 确保对于小图这里是 downsample_factor=1
#             M_full = register_he_codex_arrays(he_array, dapi_img, calc_w=4096)
#             print("  Registration successful.")
            
#             # 【修改可视化调用路径】
#             # 保存到 OUTPUT_DIR 下的一个专门文件夹，方便查看
#             viz_dir = os.path.join(OUTPUT_DIR, "registration_checks")
#             os.makedirs(viz_dir, exist_ok=True)
#             viz_base_path = os.path.join(viz_dir, f"{sample_id}_test")

#             from preprocess import visualize_registration
#             # 调用新的白色背景可视化函数
#             visualize_registration(he_array, codex_array, M_full, viz_base_path, he_alpha=0.7)
            
#         except Exception as e:
#             print(f"  ⚠️ Registration failed: {e}")
#             # 如果配准失败，保存一下干扰后的图看看原因
#             cv2.imwrite("debug_disturbed_he_failed.jpg", cv2.cvtColor(he_array, cv2.COLOR_RGB2BGR))
#             continue
            
#         # ... (后面的生成 CSV 代码不用变) ...
            
#         # 4. 生成 CSV
#         out_csv_path = os.path.join(OUTPUT_DIR, f"{sample_id}_patches.csv")
#         try:
#             # 注意：preprocess.py 原版代码生成的 CSV 列名是 marker_0, marker_1...
#             # 这里我们先生成，如果需要具体名字可以后续修改 preprocess.py
#             create_patch_label_table_from_arrays(
#                 he_img=he_array,
#                 codex_img=codex_array,
#                 M_full=M_full,
#                 slide_id=sample_id,
#                 out_csv_path=out_csv_path,
#                 patch_size=128,
#                 label_window=8,
#                 stride=128
#             )
#             print(f"  ✅ Saved CSV to {out_csv_path}")
            
#             # 【可选优化】把 CSV 的列名改成真实的 Marker 名字
#             df = pd.read_csv(out_csv_path)
#             # 建立映射: marker_0 -> DAPI, marker_1 -> CD4 ...
#             rename_dict = {f"marker_{i}": name for i, name in enumerate(marker_names)}
#             df.rename(columns=rename_dict, inplace=True)
#             df.to_csv(out_csv_path, index=False)
#             print(f"  ✅ Renamed columns to {marker_names}")
            
#         except Exception as e:
#             print(f"  ❌ CSV generation failed: {e}")

# if __name__ == "__main__":
#     main()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    sample_ids = [d for d in os.listdir(HE_IMAGES_DIR) if os.path.isdir(os.path.join(HE_IMAGES_DIR, d))]
    
    for sample_id in sample_ids:
        print(f"\nProcessing {sample_id}...")
        
        # 1. 加载 H&E
        he_array = get_he_array(sample_id) 
        if he_array is None: continue
        
        # 2. 加载 CODEX PNGs
        codex_array, dapi_idx, marker_names = get_codex_array(sample_id)
        if codex_array is None: continue
        
        print(f"  Arrays loaded. HE: {he_array.shape}, CODEX: {codex_array.shape}")
        
        # 3. 配准
        print("  Running registration...")
        try:
            dapi_img = codex_array[dapi_idx]
            
            # 【重要修改】: 传入 sample_id，用于 debug 图片命名
            M_full = register_he_codex_arrays(
                he_array, 
                dapi_img, 
                clip_limit=2.0,
                calc_w=None, 
                debug_dir=os.path.join(OUTPUT_DIR, "debug_sift"),
                sample_id=sample_id,  # <--- 这里传入了 ID
                ransac_thresh=40.0
            )
            print("  Registration successful.")
            
            # 可视化融合结果 (白底图)
            viz_dir = os.path.join(OUTPUT_DIR, "registration_checks")
            os.makedirs(viz_dir, exist_ok=True)
            viz_base_path = os.path.join(viz_dir, f"{sample_id}_check")

            visualize_registration(he_array, codex_array, M_full, viz_base_path, he_alpha=0.7)
            
        except Exception as e:
            print(f"  ⚠️ Registration failed: {e}")
            continue
            
        # 4. 生成 CSV
        out_csv_path = os.path.join(OUTPUT_DIR, f"{sample_id}_patches.csv")
        try:
            create_patch_label_table_from_arrays(
                he_img=he_array,
                codex_img=codex_array,
                M_full=M_full,
                slide_id=sample_id,
                out_csv_path=out_csv_path,
                patch_size=128,
                label_window=8,
                stride=128
            )
            
            # 把 CSV 的列名改成真实的 Marker 名字
            df = pd.read_csv(out_csv_path)
            rename_dict = {f"marker_{i}": name for i, name in enumerate(marker_names)}
            df.rename(columns=rename_dict, inplace=True)
            df.to_csv(out_csv_path, index=False)
            print(f"  ✅ Saved CSV to {out_csv_path} (Cols: {marker_names})")
            
        except Exception as e:
            print(f"  ❌ CSV generation failed: {e}")

if __name__ == "__main__":
    main()