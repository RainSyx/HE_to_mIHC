from valis import registration, preprocessing, feature_detectors, feature_matcher
from valis import warp_tools
from pathlib import Path
import shutil
import os
import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

# ==========================================
#           ULTIMATE PATCH AREA (核心补丁区)
# ==========================================

# --- 补丁 1: RANSAC 避坑 (Skimage 替代版) ---
# 彻底解决 OpenCV findHomography 对输入数据挑剔导致的崩溃
def safe_filter_matches_ransac(kp1_xy, kp2_xy, method=None, ransac_val=3.0, **kwargs):
    if kp1_xy is None or kp2_xy is None:
        return np.empty((0, 2)), np.empty((0, 2)), np.array([])
    try:
        kp1 = np.array(list(kp1_xy))
        kp2 = np.array(list(kp2_xy))
        if len(kp1) < 4 or len(kp2) < 4:
            return np.empty((0, 2)), np.empty((0, 2)), np.array([])
        
        # 使用 skimage 计算变换矩阵
        model, inliers = ransac((kp1, kp2), ProjectiveTransform, min_samples=4, 
                                residual_threshold=ransac_val, max_trials=2000)
        
        if inliers is None or not np.any(inliers):
            return np.empty((0, 2)), np.empty((0, 2)), np.array([])
            
        good_matches_idx = np.where(inliers)[0]
        return kp1[good_matches_idx], kp2[good_matches_idx], good_matches_idx
    except Exception:
        return np.empty((0, 2)), np.empty((0, 2)), np.array([])

feature_matcher.filter_matches_ransac = safe_filter_matches_ransac
print("✅ [补丁1] RANSAC 修复完成")


# --- 补丁 2: LightGlue 类型强转 ---
# 解决 PyTorch 报错 double != float
_original_match_images = feature_matcher.LightGlueMatcher.match_images

def safe_match_images(self, *args, **kwargs):
    # 动态检查参数，强制转为 float32
    for key in ['desc1', 'kp1_xy', 'desc2', 'kp2_xy']:
        if key in kwargs and kwargs[key] is not None:
            if not isinstance(kwargs[key], np.ndarray):
                 kwargs[key] = np.array(kwargs[key])
            kwargs[key] = kwargs[key].astype(np.float32)
    return _original_match_images(self, *args, **kwargs)

feature_matcher.LightGlueMatcher.match_images = safe_match_images
print("✅ [补丁2] LightGlue 修复完成")


# --- 补丁 3: 特征检测器输入清洗 ---
# 解决 OpenCV detect 报错
class SafeDiskFD(feature_detectors.DiskFD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.light_glue_feature_name = "disk" 

    def detect_and_compute(self, image, mask=None):
        if not isinstance(image, np.ndarray):
            try:
                image = warp_tools.vips2numpy(image)
            except:
                if hasattr(image, 'write_to_memory'):
                    mem = image.write_to_memory()
                    image = np.ndarray(buffer=mem, dtype=np.uint8, shape=[image.height, image.width])
        
        if isinstance(image, np.ndarray) and not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        return super().detect_and_compute(image, mask)


# --- [更新] 补丁 4: Mask2Contours 完全重写 ---
# 既然原函数内部有问题，我们就自己实现一遍逻辑，确保数据绝对安全
def safe_mask2contours(mask, kernel_size=11, **kwargs):
    # 1. 强制转 Numpy
    if not isinstance(mask, np.ndarray):
        try:
            mask = warp_tools.vips2numpy(mask)
        except:
            if hasattr(mask, 'write_to_memory'):
                mem = mask.write_to_memory()
                mask = np.ndarray(buffer=mem, dtype=np.uint8, shape=[mask.height, mask.width])
    
    # 2. 强制转 uint8 且连续 (这是 OpenCV 崩溃的根源)
    if isinstance(mask, np.ndarray):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if not mask.flags['C_CONTIGUOUS']:
            mask = np.ascontiguousarray(mask)

    # 3. 手动实现核心逻辑 (不再调用库函数)
    try:
        # 创建核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 膨胀 (之前就是这里崩的，现在数据清洗过，应该稳了)
        mask_dilated = cv2.dilate(mask, kernel)
        
        # 找轮廓
        contours, hierarchy = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 填充轮廓
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, 255, -1)
        
        return filled_mask
    except Exception as e:
        print(f"⚠️ 掩膜生成警告: {e}")
        # 如果出错，为了不崩程序，直接返回原掩膜
        return mask

preprocessing.mask2contours = safe_mask2contours
print("✅ [补丁4] Mask2Contours 重写完成")


# ==========================================
#              MAIN LOGIC (主程序)
# ==========================================

# ================= 空间大挪移配置 (/data 版) =================

# 1. [最关键] 将临时缓存搬家到 data 盘
# 防止 VIPS 处理大图时把 /tmp 撑爆
data_tmp = "/data/guoxs/Syx/ROSIE/tmp_cache"
os.makedirs(data_tmp, exist_ok=True)
os.environ['TMPDIR'] = data_tmp
# 降低内存阈值，让大图数据更早写入硬盘缓存，防止内存溢出
os.environ['VIPS_DISC_THRESHOLD'] = '500m'


# 1. 路径
slide_src_dir = "/data/guoxs/Syx/ROSIE/VALIS/Test_QPTIFF/HE-37/slides"
results_dst_dir = "/data/guoxs/Syx/ROSIE/VALIS/Test_QPTIFF/HE-37/valis_out"
registered_slide_dst_dir = "/data/guoxs/Syx/ROSIE/VALIS/Test_QPTIFF/HE-37/registered_ometiff"
he_filename = "HE-37.ome.tiff"
dapi_filename = "CODEX-37_DAPI.ome.tiff"
#cd57_filename = "B2155897_CD57.ome.tiff"
#hla-dr_filename = "B2155897_HLA-DR.ome.tiff"
full_he_path = os.path.join(slide_src_dir, he_filename)

# 2. 清理
if os.path.exists(results_dst_dir):
    shutil.rmtree(results_dst_dir)
Path(registered_slide_dst_dir).mkdir(parents=True, exist_ok=True)

# 3. 构建匹配器 (必须使用 SafeFD)
safe_fd = SafeDiskFD()
safe_matcher = feature_matcher.LightGlueMatcher(feature_detector=safe_fd, match_filter_method="ransac")
safe_matcher_for_sorting = feature_matcher.LightGlueMatcher(feature_detector=safe_fd, match_filter_method="ransac")

# 4. 初始化 VALIS
registrar = registration.Valis(
    src_dir=slide_src_dir,
    dst_dir=results_dst_dir,
    imgs_ordered=True,
    reference_img_f=full_he_path,
    align_to_reference=True,
    max_processed_image_dim_px=4096,
    max_non_rigid_registration_dim_px=4096,
    check_for_reflections=True,
    create_masks=False,       
    crop_for_rigid_reg=False, 
    matcher=safe_matcher,
    matcher_for_sorting=safe_matcher_for_sorting
)

# 5. 配置处理器
processor_dict = {
    he_filename: [preprocessing.OD, {"adaptive_eq": False}],
    dapi_filename: [preprocessing.ChannelGetter, {"channel": "dapi", "adaptive_eq": True}],
    #cd57_filename: [preprocessing.ChannelGetter, {"channel": "cd57", "adaptive_eq": True}]
}

# 6. 运行
print("\n--- Starting Registration ---")
try:
    rigid_registrar, non_rigid_registrar, error_df = registrar.register(
        processor_dict=processor_dict
    )
    
    if registrar:
        print("\n✅ Registration Finished. Now Saving...")
        registrar.warp_and_save_slides(
            registered_slide_dst_dir,
            crop="reference", 
            compression="lzw"
        )
        print("\n🎉 All Done! 结果已保存至:", registered_slide_dst_dir)
        print(error_df)
    else:
        print("❌ Registrar is None")

except Exception:
    import traceback
    traceback.print_exc()
finally:
    registration.kill_jvm()