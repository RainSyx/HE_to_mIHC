import os
import numpy as np
import pandas as pd
import cv2

"""
预处理脚本：
- H&E vs CODEX(DAPI) 的仿射配准（SIFT + RANSAC）
- 基于配准结果从 H&E 采样 patch，并在 CODEX 上取 marker 平均作为 label
- 输出一个 CSV：每行一个 patch，包含中心坐标和 marker_0..marker_(C-1)
"""


# ======================
# 基础工具函数
# ======================

# ==============================================================================
# 在 preprocess.py 文件中找到原来的 visualize_registration 函数并替换为以下内容
# ==============================================================================
# ==============================================================================
# 在 preprocess.py 文件中找到并替换 visualize_registration 函数
# ==============================================================================


def visualize_registration(he_img, codex_img, M, out_path_base, he_alpha=1.0):
    """
    生成配准前后的对比图（以 H&E 尺寸为基准）。
    
    逻辑修改：
    1. 目标尺寸：完全锁定为 H&E 图片的 (Height, Width)。
    2. 配准前：强制将 CODEX(mIHC) 拉伸/缩小到 H&E 的尺寸。
    3. 配准后：使用逆矩阵 (Inverse Matrix) 将 CODEX 变换到 H&E 的坐标系上。
    
    配色：Green (DAPI) + Magenta (HE)
    """
    # 1. 确定基准尺寸 (以 H&E 为主)
    h_he, w_he = he_img.shape[:2]
    
    # --- 2. 准备 H&E (Source) ---
    # H&E 不需要动，它是我们的基准
    he_gray = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY)
    # 反色 (为了显示为品红色)
    he_inv = 255 - he_gray
    
    def normalize(img):
        img = img.astype(np.float32)
        return ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        
    he_norm = normalize(he_inv)

    # --- 3. 准备 CODEX (Target) ---
    if codex_img.ndim == 3:
        dapi = codex_img[0]
    else:
        dapi = codex_img
    dapi_norm_orig = normalize(dapi) # 原始尺寸的 DAPI

    # === A. 配准前 (Before) ===
    # 强制把 DAPI resize 成 H&E 的大小
    # 注意 cv2.resize 是 (width, height)
    dapi_before = cv2.resize(dapi_norm_orig, (w_he, h_he), interpolation=cv2.INTER_LINEAR)

    # === B. 配准后 (After) ===
    # 我们现在的矩阵 M 是: HE -> CODEX
    # 我们想要显示的底图是 HE，所以我们需要把 CODEX -> HE
    # 这需要计算 "逆矩阵" (Inverse Affine Transform)
    try:
        M_inv = cv2.invertAffineTransform(M)
        # 使用逆矩阵把 DAPI 变换到 HE 的坐标系
        dapi_after = cv2.warpAffine(dapi_norm_orig, M_inv, (w_he, h_he))
    except Exception as e:
        print(f"  ⚠️ Warning: Could not invert affine matrix: {e}")
        dapi_after = dapi_before # 降级处理

    # --- 4. 生成融合图 (Green + Magenta) ---
    def create_gm_fusion(he_layer, dapi_layer, alpha):
        # 初始化画布 (大小等于 HE)
        fusion = np.zeros((h_he, w_he, 3), dtype=np.uint8)
        
        # H&E 强度控制
        he_val = (he_layer.astype(np.float32) * alpha).astype(np.uint8)
        
        # 通道分配:
        # DAPI (Target/Warped Target) -> Green (通道 1)
        fusion[..., 1] = dapi_layer
        
        # H&E (Source/Base) -> Magenta (Blue 0 + Red 2)
        fusion[..., 0] = he_val # Blue
        fusion[..., 2] = he_val # Red
        
        return fusion

    # 生成图片
    fusion_before = create_gm_fusion(he_norm, dapi_before, he_alpha)
    fusion_after = create_gm_fusion(he_norm, dapi_after, he_alpha)
    
    # --- 5. 保存 ---
    path_before = out_path_base + "_before_GM.jpg"
    path_after = out_path_base + "_after_GM.jpg"
    
    cv2.imwrite(path_before, fusion_before, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    cv2.imwrite(path_after, fusion_after, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    
    print(f"  ✅ Saved check (Before): {path_before}")
    print(f"  ✅ Saved check (After) : {path_after}")

# ==============================================================================

def to_gray_he(he_img):
    """
    将 H&E 图转成灰度图。
    输入可以是 (H, W, 3) 或 (H, W)，输出 uint8 (H, W)。
    """
    if he_img.ndim == 3:
        he_rgb = he_img.astype(np.uint8)
        gray = cv2.cvtColor(he_rgb, cv2.COLOR_RGB2GRAY)
    elif he_img.ndim == 2:
        gray = he_img.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected H&E image shape: {he_img.shape}")
    return gray


def clahe_normalize(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对 2D 图像做 CLAHE，输入 img 为 uint8 (H, W)，返回 uint8。
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


# ======================
# SIFT + RANSAC 配准
# ======================

# def estimate_affine_sift_ransac(he_gray, dapi_gray,
#                                 max_features=5000,
#                                 keep_ratio=0.15,
#                                 ransac_thresh=3.0):
#     """
#     使用 SIFT 特征 + RANSAC 来估计 H&E -> CODEX(DAPI) 的仿射变换矩阵 (2x3)

#     he_gray: uint8 (H, W)，H&E 灰度（已做过 CLAHE）
#     dapi_gray: uint8 (H, W)，DAPI 灰度（已做过 CLAHE）
#     """

#     sift = cv2.SIFT_create(nfeatures=max_features)

#     keypts1, desc1 = sift.detectAndCompute(he_gray, None)
#     keypts2, desc2 = sift.detectAndCompute(dapi_gray, None)

#     if desc1 is None or desc2 is None:
#         raise RuntimeError("No SIFT features detected in one of the images.")

#     # FLANN 匹配
#     index_params = dict(algorithm=1, trees=5)  # KDTree
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(desc1, desc2, k=2)

#     # Lowe ratio test
#     good = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good.append(m)

#     if len(good) < 10:
#         raise RuntimeError(f"Not enough good matches: {len(good)}")

#     # 按距离排序，保留一部分最好的
#     good = sorted(good, key=lambda x: x.distance)
#     keep_n = max(int(len(good) * keep_ratio), 10)
#     good = good[:keep_n]

#     pts_he = np.float32([keypts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     pts_dapi = np.float32([keypts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

#     # 估计仿射（部分仿射，相似变换）
#     M, inliers = cv2.estimateAffinePartial2D(
#         pts_he, pts_dapi,
#         method=cv2.RANSAC,
#         ransacReprojThreshold=ransac_thresh
#     )
#     if M is None:
#         raise RuntimeError("Affine transform estimation failed.")

#     # M: (2, 3)
#     return M, inliers


# def register_he_codex_arrays(he_img, dapi_img, downsample_factor=None, clip_limit=2.0):
#     """
#     输入：
#     - he_img: H&E 图，(H, W, 3) 或 (H, W)
#     - dapi_img: DAPI 图，(H, W)
#     输出：
#     - M_full: H&E -> CODEX 的仿射变换矩阵 (2, 3)

#     【核心修复】：
#     自动处理 H&E 和 CODEX 分辨率不一致的问题（例如 H&E 是 10k，CODEX 是 20k）。
#     不再依赖单一的 downsample_factor，而是把两张图都缩放到固定的计算尺寸（例如宽 1024）进行配准，
#     然后通过矩阵运算还原回原始分辨率的对应关系。
#     """
#     # 1. 预处理 H&E
#     he_gray = to_gray_he(he_img)
#     he_gray = 255 - he_gray # 反色
    
#     # 2. 获取原始尺寸
#     h_he, w_he = he_gray.shape[:2]
#     h_da, w_da = dapi_img.shape[:2]
    
#     # 3. 定义配准时的计算尺寸 (Calculation Size)
#     # 将两张图都强制缩放到宽度为 1024 (或者 2048)，高度按比例或强制一致
#     # 这里为了稳健，强制让 DAPI 的计算尺寸 = H&E 的计算尺寸
#     calc_w = 1024
    
#     # 计算 H&E 的缩放比例 (Original -> Small)
#     scale_he_x = calc_w / w_he
#     scale_he_y = scale_he_x # 保持宽高比
#     calc_h = int(h_he * scale_he_y)
    
#     # 计算 CODEX 的缩放比例 (Original -> Small)
#     # 注意：我们要把 CODEX 也缩放到 (calc_w, calc_h)
#     # 这样在小图上，它们的大小是完全一样的，SIFT 才能工作
#     scale_da_x = calc_w / w_da
#     scale_da_y = calc_h / h_da 
    
#     # 4. 生成小图
#     he_small = cv2.resize(he_gray, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
#     dapi_small = cv2.resize(dapi_img, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
    
#     # CLAHE 增强对比度
#     he_clahe = clahe_normalize(he_small, clip_limit=clip_limit)
#     dapi_clahe = clahe_normalize(dapi_small, clip_limit=clip_limit)
    
#     # 5. 计算小图之间的变换矩阵 M_small
#     # M_small maps (he_small_x, he_small_y) -> (dapi_small_x, dapi_small_y)
#     try:
#         M_small, inliers = estimate_affine_sift_ransac(he_clahe, dapi_clahe)
#     except Exception as e:
#         print(f"  ⚠️ SIFT failed: {e}. Fallback to simple resize alignment.")
#         # 如果 SIFT 失败，因为 "Before" 看着挺对齐，
#         # 我们假设它们是对齐的，只存在分辨率差异。
#         # M_full 应该只包含缩放 (w_da / w_he)
#         sx = w_da / w_he
#         sy = h_da / h_he
#         return np.float32([[sx, 0, 0], [0, sy, 0]])

#     # 6. 【数学推导】将 M_small 还原为 M_full
#     # 关系链：
#     # P_he_orig --(S_he)--> P_he_small --(M_small)--> P_dapi_small --(S_dapi_inv)--> P_dapi_orig
#     #
#     # S_he: 缩放矩阵 (H&E Original -> Small)
#     S_he = np.array([[scale_he_x, 0, 0],
#                      [0, scale_he_y, 0],
#                      [0, 0, 1]])
                     
#     # S_dapi: 缩放矩阵 (CODEX Original -> Small)
#     S_dapi = np.array([[scale_da_x, 0, 0],
#                        [0, scale_da_y, 0],
#                        [0, 0, 1]])
    
#     # 我们需要 M_full 使得: P_dapi_orig = M_full @ P_he_orig
#     # 路径是: P_dapi_orig = S_dapi_inv @ (M_small @ (S_he @ P_he_orig))
#     # 所以: M_full = S_dapi_inv @ M_small @ S_he
    
#     S_dapi_inv = np.linalg.inv(S_dapi)
    
#     # 扩展 M_small 为 3x3 方便矩阵乘法
#     M_small_3x3 = np.vstack([M_small, [0, 0, 1]])
    
#     M_full_3x3 = S_dapi_inv @ M_small_3x3 @ S_he
    
#     # 取前两行作为最终仿射矩阵
#     M_full = M_full_3x3[:2, :]
    
#     return M_full



# ======================
# SIFT + RANSAC 配准 (含可视化与特征筛选)
# ======================



def estimate_affine_sift_ransac(img1_gray, img2_gray,
                                out_vis_path=None,
                                keep_ratio=0.20,
                                ransac_thresh=15.0, # <--- 修改点：默认阈值 15.0
                                min_kp_size=0.0):
    """
    img1_gray: H&E (Source)
    img2_gray: DAPI (Target)
    """

    # 1. SIFT 检测
    sift = cv2.SIFT_create(nfeatures=200000,          
    nOctaveLayers=8,     
    contrastThreshold=0.005,  
    edgeThreshold=80,     
    sigma=1.0            
    )
    keypts1, desc1 = sift.detectAndCompute(img1_gray, None)
    keypts2, desc2 = sift.detectAndCompute(img2_gray, None)

    if desc1 is None or desc2 is None:
        raise RuntimeError("No SIFT features detected.")

    # =======================================================
    # 【修改后的代码】: 针对大图优化，手动绘制超大特征点
    # =======================================================
    if out_vis_path is not None:
        base_path = out_vis_path.replace("_match_lines_vis.jpg", "")
        path_kp1 = f"{base_path}_keypoints_he.jpg"
        path_kp2 = f"{base_path}_keypoints_dapi.jpg"

        print(f"   Saving LARGE keypoints vis to: {path_kp1} & {path_kp2}")

        # --- 动态计算绘制尺寸 (关键步骤) ---
        # 无论图片多大，圆圈半径总是图片宽度的 0.5%
        # 例如：1000px 图 -> 半径 5px
        #       20000px 图 -> 半径 100px (这就看得很清了)
        h_ref, w_ref = img1_gray.shape
        draw_radius = max(5, int(w_ref * 0.005)) 
        draw_thickness = max(2, int(draw_radius * 0.3)) # 线宽也随比例变粗

        def draw_large_circles(img_gray, keypoints, color):
            # 转成彩色图以便画彩色圆圈
            vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                # 画圆圈
                cv2.circle(vis, (x, y), draw_radius, color, draw_thickness)
                # (可选) 画圆心点，确保精确定位
                cv2.circle(vis, (x, y), draw_thickness * 2, color, -1) 
            return vis

        # 绘制 H&E (红色)
        vis_kp1 = draw_large_circles(img1_gray, keypts1, (0, 0, 255))
        cv2.imwrite(path_kp1, vis_kp1)
        
        # 绘制 DAPI (绿色)
        vis_kp2 = draw_large_circles(img2_gray, keypts2, (0, 255, 0))
        cv2.imwrite(path_kp2, vis_kp2)
    # =======================================================

    # 2. FLANN 匹配
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc1, desc2, k=2)

    # 3. Ratio Test 
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance: 
            good_matches.append(m)

    if len(good_matches) < 10:
        raise RuntimeError(f"Not enough matches (Ratio Test): {len(good_matches)}")

    # 按距离排序
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    keep_n = max(int(len(good_matches) * keep_ratio), 15)
    good_matches = good_matches[:keep_n]

    # 提取坐标
    pts1 = np.float32([keypts1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypts2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 4. RANSAC 估计
    M, inliers_mask = cv2.estimateAffinePartial2D(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh
    )

    if M is None:
        raise RuntimeError("RANSAC affine estimation failed.")

    # ==========================================
    # 5. 可视化：手动画粗线 (解决红线太淡的问题)
    # ==========================================
    if out_vis_path is not None:
        print(f"   Writing Outlier/Inlier visualization to: {out_vis_path}")
        
        # 准备画布 (左右拼接)
        h1, w1 = img1_gray.shape
        h2, w2 = img2_gray.shape
        vis_h = max(h1, h2)
        vis_w = w1 + w2
        vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        
        # 左边放 Source (HE)，右边放 Target (DAPI)
        vis_img[:h1, :w1] = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
        vis_img[:h2, w1:w1+w2] = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)

        mask_flat = inliers_mask.ravel().tolist()

        # 辅助绘制函数 (thickness=2 让线变粗)
        def draw_line_thick(img, pt_src, pt_dst, color, thickness=2):
            # pt_dst 需要加上左图的宽度 w1
            p1 = (int(pt_src[0]), int(pt_src[1]))
            p2 = (int(pt_dst[0] + w1), int(pt_dst[1]))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.circle(img, p1, 4, color, -1) # 画个圆点更明显
            cv2.circle(img, p2, 4, color, -1)

        # 先画红线 (Outliers)
        for i, is_inlier in enumerate(mask_flat):
            if not is_inlier:
                m = good_matches[i]
                pt_s = keypts1[m.queryIdx].pt
                pt_d = keypts2[m.trainIdx].pt
                draw_line_thick(vis_img, pt_s, pt_d, (0, 0, 255), thickness=3)

        # 后画绿线 (Inliers) - 保证绿线压在红线上
        for i, is_inlier in enumerate(mask_flat):
            if is_inlier:
                m = good_matches[i]
                pt_s = keypts1[m.queryIdx].pt
                pt_d = keypts2[m.trainIdx].pt
                draw_line_thick(vis_img, pt_s, pt_d, (0, 255, 0), thickness=2)

        cv2.imwrite(out_vis_path, vis_img)

    return M, inliers_mask

# ======================
# 修改 register_he_codex_arrays 以调用可视化
# ======================

def register_he_codex_arrays(he_img, dapi_img, 
                             calc_w,          # <--- 修改点：默认 4096
                             clip_limit, 
                             debug_dir,
                             sample_id,
                             ransac_thresh): # <--- 修改点：新增 sample_id 参数
    """
    calc_w: 计算配准时使用的宽度。推荐 2048 或 4096。
    """
    # 1. 预处理 H&E
    he_gray = to_gray_he(he_img)
    he_gray = 255 - he_gray # 反色
    
    # 获取原始尺寸
    h_he, w_he = he_gray.shape[:2]
    h_da, w_da = dapi_img.shape[:2]
    
    # 2. 决定缩放比例
    if calc_w is None or calc_w >= w_he:
        print(f"   [Info] Running registration at FULL resolution ({w_he}x{h_he})...")
        scale_he_x = 1.0
        scale_he_y = 1.0
        he_small = he_gray
        dapi_small = dapi_img 
        if (h_da, w_da) != (h_he, w_he):
             dapi_small = cv2.resize(dapi_img, (w_he, h_he), interpolation=cv2.INTER_LINEAR)
        scale_da_x = w_he / w_da 
        scale_da_y = h_he / h_da
        
    else:
        print(f"   [Info] Downsampling for calculation: {w_he} -> {calc_w} width.")
        scale_he_x = calc_w / w_he
        scale_he_y = scale_he_x 
        calc_h = int(h_he * scale_he_y)
        he_small = cv2.resize(he_gray, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
        
        scale_da_x = calc_w / w_da
        scale_da_y = calc_h / h_da
        dapi_small = cv2.resize(dapi_img, (calc_w, calc_h), interpolation=cv2.INTER_AREA)

    # 3. CLAHE 增强
    he_clahe = clahe_normalize(he_small, clip_limit=clip_limit)
    dapi_clahe = clahe_normalize(dapi_small, clip_limit=clip_limit)
    
    # 4. 计算变换矩阵
    try:
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            # <--- 修改点：文件名加入 sample_id，防止覆盖
            vis_path = os.path.join(debug_dir, f"{sample_id}_match_lines_vis.jpg")
        else:
            vis_path = None

        # 动态调整阈值
        #base_thresh = ransac_thresh # 基础阈值
        if calc_w is not None:
             current_thresh = ransac_thresh * (calc_w / 1024.0)
        else:
             current_thresh =80.0 

        # 限制一下最小阈值
        current_thresh = max(10.0, current_thresh)
        
        print(f"   [Info] Using RANSAC threshold: {current_thresh:.1f}")

        M_small, inliers = estimate_affine_sift_ransac(
            he_clahe, dapi_clahe, 
            out_vis_path=vis_path,
            min_kp_size=0.0,
            ransac_thresh=ransac_thresh 
        )
        
    except Exception as e:
        print(f"  ⚠️ SIFT failed: {e}. Fallback to simple resize alignment.")
        sx = w_da / w_he
        sy = h_da / h_he
        return np.float32([[sx, 0, 0], [0, sy, 0]])

    # 5. 还原矩阵到原始尺寸
    S_he = np.array([[scale_he_x, 0, 0], [0, scale_he_y, 0], [0, 0, 1]])
    S_dapi = np.array([[scale_da_x, 0, 0], [0, scale_da_y, 0], [0, 0, 1]])
    
    S_dapi_inv = np.linalg.inv(S_dapi)
    M_small_3x3 = np.vstack([M_small, [0, 0, 1]])
    M_full_3x3 = S_dapi_inv @ M_small_3x3 @ S_he
    M_full = M_full_3x3[:2, :]
    
    return M_full

# 简单的“是否有组织”判断函数，例如看灰度直方图里是不是绝大多数像素都非常白

def is_valid_patch_from_he(he_img, x, y, patch_size=128,
                           white_thr=240, max_white_frac=0.9):
    """
    简单规则：如果 patch 里 > max_white_frac 的像素都接近白色，就认为是无效（背景）。

    he_img: (H, W, 3) 或 (H, W)
    (x, y): patch 中心坐标（H&E 坐标系）
    """
    if he_img.ndim == 3:
        Hh, Wh, _ = he_img.shape
        patch_rgb = he_img[
            y - patch_size // 2 : y + patch_size // 2,
            x - patch_size // 2 : x + patch_size // 2,
            :
        ]
        patch_gray = cv2.cvtColor(patch_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        Hh, Wh = he_img.shape
        patch_gray = he_img[
            y - patch_size // 2 : y + patch_size // 2,
            x - patch_size // 2 : x + patch_size // 2
        ].astype(np.uint8)

    if patch_gray.size == 0:
        return 0  # 越界就当无效

    white_frac = np.mean(patch_gray >= white_thr)
    return int(white_frac < max_white_frac)  # 1=有效，0=无效


# ======================
# 采 patch + 生成 CSV
# ======================

def create_patch_label_table_from_arrays(
    he_img,
    codex_img,
    M_full,
    slide_id,
    out_csv_path,
    patch_size=128,
    label_window=8,
    stride=128,
    valid_margin=64
):
    """
    he_img: H&E 图，(H, W, 3) 或 (H, W)
    codex_img: CODEX 多通道图，(C, Hc, Wc)
    M_full: H&E -> CODEX 仿射变换矩阵 (2, 3)
    slide_id: 字符串
    out_csv_path: 输出 CSV 路径

    在 H&E 上以 stride 采样 patch 中心，映射到 CODEX 坐标，
    用 label_window x label_window 计算每个 marker 的平均强度。
    """
    if he_img.ndim == 3:
        Hh, Wh = he_img.shape[:2]
    else:
        Hh, Wh = he_img.shape

    if codex_img.ndim != 3:
        raise ValueError(f"codex_img expected shape (C, H, W), got {codex_img.shape}")

    C, Hc, Wc = codex_img.shape

    half_patch = patch_size // 2
    half_label = label_window // 2

    records = []

    # 避免 patch 超出 H&E 边界
    y_start = valid_margin
    y_end = Hh - valid_margin
    x_start = valid_margin
    x_end = Wh - valid_margin

    for y in range(y_start, y_end, stride):
        for x in range(x_start, x_end, stride):
            # H&E 坐标 (x, y) 映射到 CODEX
            src = np.array([x, y, 1.0], dtype=np.float32)
            dst = M_full @ src
            xc, yc = float(dst[0]), float(dst[1])

            # CODEX 上窗口
            x0 = int(round(xc - half_label))
            y0 = int(round(yc - half_label))
            x1 = x0 + label_window
            y1 = y0 + label_window

            if x0 < 0 or y0 < 0 or x1 > Wc or y1 > Hc:
                continue

            label_vec = []
            for c in range(C):
                patch_c = codex_img[c, y0:y1, x0:x1]
                label_vec.append(float(patch_c.mean()))

            # 有效 = biomarker 有效 ∧ patch 有效

            patch_valid = is_valid_patch_from_he(
                he_img,
                x,
                y,
                patch_size=patch_size,
                white_thr=240,
                max_white_frac=0.9,
            )
            # 如果你想直接丢掉无效 patch，可以在这里 continue：
            # if patch_valid == 0:
            #     continue

            record = {
                "slide_id": slide_id,
                "center_x": x,
                "center_y": y,
                "patch_valid": patch_valid,
            }
            for i, v in enumerate(label_vec):
                record[f"marker_{i}"] = v

            records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    print(f"Saved {len(df)} patches to {out_csv_path}")
    return df


# ======================
# Toy 测试
# ======================

# def toy_test(output_dir="toy_output"):
#     """
#     使用随机合成的 512x512 图像做一个简单的 toy 测试：
#     - 构造 base 图像
#     - H&E: 用 base 生成 RGB
#     - CODEX: 用 base 生成 DAPI + 若干 marker 通道
#     - 自己配自己（he_gray 和 dapi 一样），M ~ identity
#     - 生成 patch-level CSV
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     H = W = 512
#     # base 图像：0~255 随机，但为了特征明显一点可以做个简单的 pattern
#     y = np.linspace(0, 1, H)
#     x = np.linspace(0, 1, W)
#     xx, yy = np.meshgrid(x, y)
#     base = ((np.sin(10 * xx) + np.cos(10 * yy)) * 0.5 + 0.5) * 255
#     base = base.astype(np.uint8)

#     # H&E: 简单地把 base 复制到 RGB 三通道
#     he_img = np.stack([base, base, base], axis=-1)  # (H, W, 3)

#     # CODEX: C=4，channel 0 作为 DAPI，别的 channel 做一些简单变换
#     C = 4
#     codex_channels = []
#     for c in range(C):
#         # 不同的线性变换 + 噪声，保持结构相似
#         ch = np.clip(base * (0.5 + 0.2 * c) +
#                      np.random.randn(H, W) * 5.0, 0, 255).astype(np.uint8)
#         codex_channels.append(ch)
#     codex_img = np.stack(codex_channels, axis=0)  # (C, H, W)
#     dapi_img = codex_img[0]

#     # 配准：因为 he_gray 和 dapi_img 实际上来自同一个 base，
#     # 理论上 M 应该接近 identity。
#     M_full = register_he_codex_arrays(
#         he_img,
#         dapi_img,
#         calc_w=4096,
#         clip_limit=2.0
#     )
#     print("Estimated affine matrix M_full:")
#     print(M_full)

#     # 用配准结果生成 patch-label CSV
#     out_csv_path = os.path.join(output_dir, "toy_patches.csv")
#     df = create_patch_label_table_from_arrays(
#         he_img,
#         codex_img,
#         M_full,
#         slide_id="toy_slide",
#         out_csv_path=out_csv_path,
#         patch_size=128,
#         label_window=8,
#         stride=64,
#         valid_margin=64
#     )

#     print(df.head())

# def run_on_real_pair(he_array, codex_array, slide_id, out_csv_path):
#     dapi_img = codex_array[dapi_channel_index]   # 比如 0
#     M_full = register_he_codex_arrays(he_array, dapi_img, downsample_factor=1)
#     create_patch_label_table_from_arrays(
#         he_array,
#         codex_array,
#         M_full,
#         slide_id=slide_id,
#         out_csv_path=out_csv_path,
#         patch_size=128,
#         label_window=8,
#         stride=64,
#         valid_margin=64,
#     )



# if __name__ == "__main__":
#     # 直接运行 python preprocess.py 就会执行 toy 测试
#     toy_test()
    #run_on_real_pair(he_array, codex_array, slide_id, out_csv_path)
