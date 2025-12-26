"""
Training script for H&E to multiplex protein prediction model.

This script trains a deep learning model to predict protein expression levels
from H&E images. It supports both training and evaluation modes.

Required directory structure:
ROOT_DIR/
    ├── data/                     # Contains training data
    │   └── cell_measurements.pqt # Parquet file with cell measurements
    ├── images/                   # H&E image data
    │   └── {uuid}/image.ome.zarr # Zarr formatted image files  
    ├── metadata/                 # Metadata files
    │   └── metadata_dict.pkl     # Dictionary with experiment metadata
    └── runs/                     # Training run outputs
"""

import os
import shutil

# 在导入torch之前设置CUDA环境
# 确保conda环境的库路径优先
conda_env = os.environ.get('CONDA_PREFIX', '/NAS/guoxs/miniconda3/envs/SyxROISE')
if 'LD_LIBRARY_PATH' not in os.environ or conda_env not in os.environ.get('LD_LIBRARY_PATH', ''):
    conda_lib = f"{conda_env}/lib"
    cuda_cupti = "/usr/local/cuda-12.0/extras/CUPTI/lib64"
    cuda_lib = "/usr/local/cuda-12.0/lib64"
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{conda_lib}:{cuda_cupti}:{cuda_lib}:{current_ld}"

# 设置CUDA_VISIBLE_DEVICES（如果未设置，默认使用0,2,3，排除有问题的GPU 1和4）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
    print(f"自动设置 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (排除有问题的GPU)")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import wandb
from typing import Tuple, List, Dict, Optional
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# Configure torch multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Configuration constants
ROOT_DIR = "/home/guoxs/Syx/ROISE/ROSIE_datatry"  # Base directory for project
DATA_FILE = os.path.join(ROOT_DIR, "data/cell_measurements.pqt")
METADATA_FILE = os.path.join(ROOT_DIR, "metadata/metadata_dict.pkl")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Model training constants
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EVAL_INTERVAL = 50
PATIENCE = 500
# GPU与驱动已恢复，开启多进程数据加载
NUM_WORKERS = 0
PATCH_SIZE = 128

# Dataset splits for train/val/test
DATASET_SPLITS = {
    'train': [ "B2121081","B2155897"
        # TODO: Add training dataset splits
    ],
    'val': [ "B2155897"
        # TODO: Add validation dataset splits
    ],
    'test': [ "B2155897"
        # TODO: Add test dataset splits
    ]
}

def pad_patch(patch: np.ndarray, 
             original_size: Tuple[int, int], 
             x_center: int, 
             y_center: int, 
             patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Pads the given patch if its size is less than patch_size x patch_size pixels.

    Args:
        patch: NumPy array representing the patch image
        original_size: Tuple of (width, height) of the original image
        x_center: X coordinate of the center of the patch in the original image
        y_center: Y coordinate of the center of the patch in the original image
        patch_size: The target size of the patch

    Returns:
        Padded patch as a NumPy array
    """
    original_height, original_width = original_size
    current_height, current_width = patch.shape[:2]
    
    if current_height == patch_size and current_width == patch_size:
        return patch
        
    # Calculate padding needed
    pad_left = max(patch_size // 2 - x_center, 0)
    pad_right = max(x_center + patch_size // 2 - original_width, 0)
    pad_top = max(patch_size // 2 - y_center, 0)
    pad_bottom = max(y_center + patch_size // 2 - original_height, 0)

    # Apply padding
    pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if patch.ndim == 3 else ((pad_top, pad_bottom), (pad_left, pad_right))
    padded_patch = np.pad(patch, pad_shape, mode='constant', constant_values=0)

    # Ensure the patch is exactly patch_size x patch_size
    padded_patch = padded_patch[:patch_size, :patch_size]

    return padded_patch

def masked_mse_loss(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss with a mask.

    Args:
        pred: Predicted tensor
        target: Target tensor
        mask: Mask tensor with 1s for elements to include and 0s to exclude

    Returns:
        Loss value
    """
    mask = mask.bool()
    masked_pred = torch.masked_select(pred, mask)
    masked_target = torch.masked_select(target, mask)
    return F.mse_loss(masked_pred, masked_target, reduction='mean')

def get_model(num_outputs: Optional[int] = None, 
             use_context: bool = False, 
             use_mask: bool = False) -> nn.Module:
    """
    Creates and returns the model architecture.

    Args:
        num_outputs: Number of output features to predict
        use_context: Whether to use contextual features
        use_mask: Whether to use masking in the model

    Returns:
        PyTorch model instance
    """
    model = models.convnext_small(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model

class ImageDataset(Dataset):
    """
    Dataset class for loading H&E image patches and their corresponding protein expression values.
    
    Args:
        data_df: DataFrame containing cell measurements
        root_dir: Root directory containing image data
        is_test: Whether this is a test dataset
        use_mask: Whether to use cell segmentation masks
        transform: Transforms to apply to images
        metadata_dict: Dictionary containing experiment metadata
        test_acq_ids: List of acquisition IDs to use for testing
        subset: Subset of coverslip IDs to use
        pred_only: Whether to only generate predictions (no ground truth)
    """
    def __init__(self,
                data_df: pd.DataFrame,
                root_dir: str,
                is_test: bool = False,
                use_mask: bool = False,
                transform: Optional[Dict] = None,
                metadata_dict: Optional[Dict] = None,
                test_acq_ids: Optional[List[str]] = None,
                subset: Optional[List[str]] = None,
                pred_only: bool = False):
        
        self.df = data_df
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = PATCH_SIZE
        self.ps = self.patch_size//2
        self.metadata_dict = metadata_dict
        self.use_mask = use_mask
        self.invalid_acq_ids = set()
        self.zarr_cache = {}
        self.is_test = is_test
        self.pred_only = pred_only

        self.all_biomarkers = self.metadata_dict['all_biomarkers']
        assert len(self.all_biomarkers) != 0, "No biomarker labels found"
        
        if subset is not None:
            self.df = self.df[self.df['HE_COVERSLIP_ID'].isin(subset)]
        if test_acq_ids is not None:
            self.df = self.df[self.df['CODEX_ACQUISITION_ID'].isin(test_acq_ids)]
            
        self.df.reset_index(inplace=True)
        self.acq_map = {i: x for i, x in enumerate(self.df['CODEX_ACQUISITION_ID'].unique())}
        self.acq_map.update({x: i for i, x in enumerate(self.df['CODEX_ACQUISITION_ID'].unique())})

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[Tuple]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Tuple containing (image_patch, expression_values, mask, metadata)
            Returns None if the item is invalid
        """

        row = self.df.iloc[idx]
        seg_acq_id = row['CODEX_ACQUISITION_ID']
        he_region_uuid = self.metadata_dict[seg_acq_id]['HE_REGION_UUID']
        he_region_path = os.path.join(IMAGE_DIR, he_region_uuid, 'image.ome.zarr')
        
        if he_region_path in self.invalid_acq_ids or not os.path.exists(he_region_path):
            self.invalid_acq_ids.add(he_region_uuid)
            return None

        if self.use_mask and seg_acq_id in self.invalid_acq_ids:
            return None

        # Handle expression values
        if self.pred_only:
            exp_row = np.zeros(len(self.all_biomarkers))
            nan_mask = np.zeros(len(self.all_biomarkers))
            valid_mask = np.ones(len(self.all_biomarkers))
            exp_vec = exp_row.astype(np.float32)
        else:
            exp_row = row[self.all_biomarkers]
            nan_mask = exp_row.isnull().values
            exp_row[nan_mask] = 0
            valid_mask = ~nan_mask
            exp_vec = exp_row.values.astype(np.float32)

        # Get image patch coordinates
        X, Y = row['X'], row['Y']
        
        # Handle segmentation masks if needed
        if self.use_mask:
            seg_path = os.path.join(self.root_dir, 'codex_segs', f'{seg_acq_id}.ome.zarr')
            if not os.path.exists(seg_path):
                self.invalid_acq_ids.add(seg_acq_id)
                return None

        # Load H&E image
        if he_region_path in self.zarr_cache:
            he_zarr = self.zarr_cache[he_region_path]
        else:
            # import pdb
            # pdb.set_trace()
            
            he_zarr = [list(Reader(parse_url(he_region_path+f'/{i}', mode="r"))())[0].data[0].compute() 
                      for i in range(3)]
            self.zarr_cache[he_region_path] = he_zarr

        # Extract patch
        b = np.clip(Y-self.ps, 0, he_zarr[0].shape[0])
        t = np.clip(Y+self.ps, 0, he_zarr[0].shape[0])
        l = np.clip(X-self.ps, 0, he_zarr[0].shape[1])
        r = np.clip(X+self.ps, 0, he_zarr[0].shape[1])
        
        he_patch = np.array([channel[b:t, l:r] for channel in he_zarr]).transpose(1, 2, 0)
        he_patch = pad_patch(he_patch, he_zarr[0].shape, X, Y)
        assert he_patch.shape == (128, 128, 3), f'H&E patch shape is {he_patch.shape}'

        # Apply transforms
        seed = np.random.randint(2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if isinstance(self.transform, dict):
            he_patch_pt = self.transform['all_channels'](he_patch)
            patch = self.transform['image_only'](he_patch_pt)
        else:
            patch = self.transform(he_patch)
        
        # import pdb 
        # pdb.set_trace()
        
        assert patch.shape == (3, 224, 224), f'Patch shape is {patch.shape}'

        metadata = {
            'CODEX_ACQUISITION_ID': seg_acq_id,
            'X': X,
            'Y': Y,
            'HE_REGION_UUID': he_region_uuid
        }

        return patch, exp_vec, torch.from_numpy(valid_mask.astype(np.float32)),metadata


# def evaluate(model: nn.Module,
#             data_loader: DataLoader,
#             device: torch.device,
#             run_dir: str,
#             step: int,
#             bm_labels: List[str],
#             acq_dict: Optional[Dict] = None,
#             save_predictions: bool = False,
#             pred_biomarkers: Optional[List[str]] = None) -> Optional[Tuple[float, float]]:
#     """
#     Evaluate the model on a dataset.
    
#     Args:
#         model: The model to evaluate
#         data_loader: DataLoader for the evaluation dataset
#         device: Device to run evaluation on
#         run_dir: Directory to save results
#         step: Current training step
#         bm_labels: List of biomarker labels
#         acq_dict: Dictionary mapping acquisition IDs to metadata
#         save_predictions: Whether to save predictions to disk
#         pred_biomarkers: List of biomarkers to predict (if different from bm_labels)
        
#     Returns:
#         None
#     """
#     model.eval()
#     if pred_biomarkers is None:
#         pred_biomarkers = bm_labels
        
#     try:
#         acq_map = data_loader.dataset.acq_map
#     except:
#         acq_map = data_loader.dataset.dataset.acq_map
    
#     eval_dataset = []
#     save_interval = 2000
    
#     with torch.no_grad():
#         #for idx, (inputs, exp_vec, nan_mask, X, Y, indices) in tqdm(enumerate(data_loader)):
#         for idx, (inputs, exp_vec, nan_mask, metadata) in tqdm(enumerate(data_loader)):
#             if idx==63:
#                 import pdb
#                 pdb.set_trace()
#             X = np.array([m for m in metadata['X']])
#             Y = np.array([m for m in metadata['Y']])
#             acq_ids = [m for m in metadata['CODEX_ACQUISITION_ID']]
#             # import pdb
#             # pdb.set_trace()
#             inputs = inputs.to(device)
#             outputs = model(inputs).detach().cpu().numpy()
#             #indices = indices.numpy()
#             exp_vec = exp_vec.numpy()
            
#             #acq_ids = [acq_map[x] for x in indices]
#             rows = [[a,b,c]+list(d)+list(e) for a,b,c,d,e in zip(X, Y, acq_ids, outputs, exp_vec)]
#             eval_dataset.extend(rows)
            
#             if save_predictions and idx % save_interval == 0:
#                 temp_df = pd.DataFrame(eval_dataset, 
#                                      columns=['X', 'Y', 'CODEX_ACQUISITION_ID'] + 
#                                              [f'pred_{x}' for x in pred_biomarkers] + 
#                                              [f'gt_{x}' for x in bm_labels])
                
#                 if os.path.exists(f'{run_dir}/predictions_{step}_{idx}.pqt'):
#                     temp_df.to_parquet(f'{run_dir}/predictions_{step}_{idx}.pqt', 
#                                      engine='fastparquet', append=True)
#                 else:
#                     temp_df.to_parquet(f'{run_dir}/predictions_{step}_{idx}.pqt')
#                 eval_dataset = []

#     eval_dataset = pd.DataFrame(eval_dataset, 
#                               columns=['X', 'Y', 'CODEX_ACQUISITION_ID'] + 
#                                       [f'pred_{x}' for x in pred_biomarkers] + 
#                                       [f'gt_{x}' for x in bm_labels])

#     if save_predictions:
#         eval_dataset.to_parquet(f'{run_dir}/predictions_{step}.pqt')
    
#     return None

# --- 请确保添加这个辅助函数 (放在 evaluate 函数之前) ---
def get_gaussian_kernel(kernel_size=16, sigma=4.0):
    """生成用于平滑拼接的高斯核"""
    x, y = np.mgrid[-kernel_size//2:kernel_size//2, -kernel_size//2:kernel_size//2]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g

def evaluate(model: nn.Module,
            data_loader: DataLoader,
            device: torch.device,
            run_dir: str,
            step: int,
            bm_labels: List[str],
            acq_dict: Optional[Dict] = None,
            save_predictions: bool = False,
            pred_biomarkers: Optional[List[str]] = None,
            save_visualizations: bool = True,
            num_biomarkers_to_viz: int = 6) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.

    返回:
        val_pr   - patch 级别的平均 Pearson R（跨 biomarker 平均）
        val_ssim - 在重建 CODEX-like 图像上的平均 SSIM
    """
    model.eval()
    if pred_biomarkers is None:
        pred_biomarkers = bm_labels

    # ---- 用于计算指标的累积容器 ----
    all_preds = []       # [N, B]
    all_targets = []     # [N, B]
    all_valid = []       # [N, B] bool
    all_x = []           # [N]
    all_y = []           # [N]
    all_acq_ids = []     # [N]，CODEX_ACQUISITION_ID

    # ---- 用于可选保存 parquet 的缓存 ----
    eval_rows = []
    save_interval = 2000

    with torch.no_grad():
        for idx, (inputs, exp_vec, valid_mask, metadata) in enumerate(tqdm(data_loader)):
            # DataLoader 默认会把 dict collate 成 {key: [batch 值]}
            # if idx==63:
            #     import pdb
            #     pdb.set_trace()
            X = np.array(metadata['X'])
            Y = np.array(metadata['Y'])
            acq_ids = np.array(metadata['CODEX_ACQUISITION_ID'])

            inputs = inputs.to(device)
            outputs = model(inputs).detach().cpu().numpy()   # [bs, B]
            targets = exp_vec.numpy()                        # [bs, B]
            valid_mask_np = valid_mask.numpy().astype(bool)  # [bs, B]

            # ---- 累积到大数组里，之后统一算指标 ----
            all_preds.append(outputs)
            all_targets.append(targets)
            all_valid.append(valid_mask_np)
            all_x.append(X)
            all_y.append(Y)
            all_acq_ids.append(acq_ids)

            # ---- 构造 DataFrame 行（如果要保存预测）----
            rows = [[x, y, a] + list(pred_row) + list(gt_row)
                    for x, y, a, pred_row, gt_row
                    in zip(X, Y, acq_ids, outputs, targets)]
            eval_rows.extend(rows)

            # 分段写 parquet 防止内存爆
            if save_predictions and (idx + 1) % save_interval == 0:
                temp_df = pd.DataFrame(
                    eval_rows,
                    columns=['X', 'Y', 'CODEX_ACQUISITION_ID'] +
                            [f'pred_{x}' for x in pred_biomarkers] +
                            [f'gt_{x}' for x in bm_labels]
                )
                os.makedirs(run_dir, exist_ok=True)
                out_path = os.path.join(run_dir, f'predictions_{step}_{idx}.pqt')
                if os.path.exists(out_path):
                    temp_df.to_parquet(out_path, engine='fastparquet', append=True)
                else:
                    temp_df.to_parquet(out_path)
                eval_rows = []

    # 如果 loader 里什么都没有，直接返回 NaN
    if not all_preds:
        return float('nan'), float('nan')

    # import pdb
    # pdb.set_trace()

    #all_preds: a list of numpy (64,50), 64:batch_size, 50:biomarker_num

    # ---- 把所有 batch 拼起来 ----
    all_preds = np.concatenate(all_preds, axis=0)       # [N, B]
    all_targets = np.concatenate(all_targets, axis=0)   # [N, B]
    all_valid = np.concatenate(all_valid, axis=0)       # [N, B]
    all_x = np.concatenate(all_x, axis=0)               # [N]
    all_y = np.concatenate(all_y, axis=0)               # [N]
    all_acq_ids = np.concatenate(all_acq_ids, axis=0)   # [N],the name of CODEX_ACQUISITION_ID

    # import pdb
    # pdb.set_trace()

    # ---- 最后一块 parquet（如果需要保存）----
    if save_predictions:
        eval_dataset = pd.DataFrame(
            eval_rows,
            columns=['X', 'Y', 'CODEX_ACQUISITION_ID'] +
                    [f'pred_{x}' for x in pred_biomarkers] +
                    [f'gt_{x}' for x in bm_labels]
        )
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, f'predictions_{step}.pqt')
        eval_dataset.to_parquet(out_path)

    # ===================== 1. 计算 Patch-level Pearson R =====================
    # import pdb
    # pdb.set_trace()
    def _compute_patch_pearson(preds, targets, valid_mask):
        """
        preds, targets, valid_mask: [N, B]
        对每个 biomarker j 计算 Pearson R，在 valid_mask[:, j] 为 True 的样本上。
        最后对所有 biomarker 的 R 取平均。
        """
        N, B = targets.shape
        rs = []
        # import pdb
        # pdb.set_trace()
        for j in range(B):
            mask_j = valid_mask[:, j]
            if mask_j.sum() < 2:
                continue
            gt_j = targets[mask_j, j]
            pred_j = preds[mask_j, j]
            # 全常数就没法算相关系数，跳过
            if gt_j.std() == 0 or pred_j.std() == 0:
                continue
            r = np.corrcoef(gt_j, pred_j)[0, 1]
            if np.isfinite(r):
                rs.append(r)
        if not rs:
            return float('nan')
        return float(np.mean(rs))
    # import pdb
    # pdb.set_trace()
    val_pr = _compute_patch_pearson(all_preds, all_targets, all_valid)

    # # ===================== 2. 计算重建图像上的 SSIM =====================
    # # import pdb
    # # pdb.set_trace()
    # def _compute_image_ssim(preds, targets, valid_mask, xs, ys, acq_ids, downscale: int = 1):
    #     """
    #     使用 X/Y 坐标和 CODEX_ACQUISITION_ID 把 patch 表达值撒回 2D 网格，
    #     对每个 acquisition + biomarker 计算一对 (gt_img, pred_img) 的 SSIM。

    #     downscale > 1 时会对坐标做整除下采样，用于控制图像尺寸（如果内存吃紧可以改成 2、4 之类）。
        
    #     返回:
    #         ssim_vals: SSIM值列表
    #         images_for_viz: 用于可视化的图像字典 {acq_id: {biomarker_idx: (img_pred, img_gt)}}
    #     """
    #     xs = xs.astype(np.int64)
    #     ys = ys.astype(np.int64)
    #     acq_ids = np.asarray(acq_ids)
    #     N, B = targets.shape
    #     ssim_vals = []
    #     images_for_viz = {}  # {acq_id: {biomarker_idx: (img_pred, img_gt, ssim_val)}}
        
    #     # import pdb
    #     # pdb.set_trace()
    #     for acq in np.unique(acq_ids):
    #         idx_acq = np.where(acq_ids == acq)[0]
    #         if idx_acq.size == 0:
    #             continue

    #         xs_acq = xs[idx_acq]
    #         ys_acq = ys[idx_acq]
    #         preds_acq = preds[idx_acq]
    #         targets_acq = targets[idx_acq]
    #         valid_acq = valid_mask[idx_acq]

    #         # 平移到从 0 开始，并可选下采样
    #         xs0 = xs_acq - xs_acq.min()
    #         ys0 = ys_acq - ys_acq.min()
    #         if downscale > 1:
    #             xs0 = xs0 // downscale
    #             ys0 = ys0 // downscale

    #         H = int(ys0.max() + 1)
    #         W = int(xs0.max() + 1)
    #         if H <= 1 or W <= 1:
    #             continue
            
    #         images_for_viz[acq] = {}
            
    #         # import pdb
    #         # pdb.set_trace()
    #         for j in range(B):
    #             vmask = valid_acq[:, j]
    #             if vmask.sum() < 2:
    #                 continue

    #             img_pred = np.zeros((H, W), dtype=np.float32)
    #             img_gt = np.zeros((H, W), dtype=np.float32)

    #             yj = ys0[vmask]
    #             xj = xs0[vmask]
    #             v_pred = preds_acq[vmask, j]
    #             v_gt = targets_acq[vmask, j]

    #             img_pred[yj, xj] = v_pred
    #             img_gt[yj, xj] = v_gt

    #             data_range = float(img_gt.max() - img_gt.min())
    #             if data_range <= 0:
    #                 continue

    #             s = ssim(img_gt, img_pred, data_range=data_range)
    #             if np.isfinite(s):
    #                 ssim_vals.append(s)
    #                 images_for_viz[acq][j] = (img_pred.copy(), img_gt.copy(), s)

    #     if not ssim_vals:
    #         return float('nan'), {}
    #     return float(np.mean(ssim_vals)), images_for_viz
    
    # # import pdb
    # # pdb.set_trace()
    # # 如果觉得图像太大，可以把 downscale 改成 2 / 4
    # val_ssim, images_for_viz = _compute_image_ssim(
    #     all_preds, all_targets, all_valid, all_x, all_y, all_acq_ids, downscale=1
    # )
    
    # # ===================== 3. 保存可视化图像 =====================
    # if save_visualizations and images_for_viz:
    #     viz_dir = os.path.join(run_dir, 'visualizations_all_biomarkers', f'step_{step}')
    #     os.makedirs(viz_dir, exist_ok=True)
        
    #     for acq_id, acq_images in images_for_viz.items():
    #         # 为每个acquisition创建可视化
    #         if len(acq_images) == 0:
    #             continue
            
    #         # 如果num_biomarkers_to_viz为-1，可视化所有biomarker
    #         if num_biomarkers_to_viz == -1:
    #             biomarkers_to_viz = sorted(acq_images.keys())
    #         else:
    #             biomarkers_to_viz = list(range(min(num_biomarkers_to_viz, len(bm_labels))))
            
    #         # 选择有效的biomarker
    #         valid_biomarker_indices = [j for j in biomarkers_to_viz if j in acq_images]
    #         if not valid_biomarker_indices:
    #             continue
            
    #         # 计算全局归一化范围（用于统一显示所有biomarker）
    #         # 收集所有pred和GT的值范围
    #         all_pred_values = []
    #         all_gt_values = []
    #         for j in valid_biomarker_indices:
    #             img_pred, img_gt, _ = acq_images[j]
    #             all_pred_values.extend(img_pred[img_pred > 0].flatten())  # 只统计非零值
    #             all_gt_values.extend(img_gt[img_gt > 0].flatten())
            
    #         # 使用百分位数进行归一化，避免异常值影响
    #         pred_min = np.percentile(all_pred_values, 1) if all_pred_values else 0
    #         pred_max = np.percentile(all_pred_values, 99) if all_pred_values else 1
    #         gt_min = np.percentile(all_gt_values, 1) if all_gt_values else 0
    #         gt_max = np.percentile(all_gt_values, 99) if all_gt_values else 1
            
    #         # 创建figure，每个biomarker一行，包含pred和GT并排显示
    #         # 如果biomarker太多，分成多个文件
    #         max_biomarkers_per_file = 12  # 每个文件最多显示12个biomarker
    #         num_files = (len(valid_biomarker_indices) + max_biomarkers_per_file - 1) // max_biomarkers_per_file
            
    #         for file_idx in range(num_files):
    #             start_idx = file_idx * max_biomarkers_per_file
    #             end_idx = min(start_idx + max_biomarkers_per_file, len(valid_biomarker_indices))
    #             current_biomarkers = valid_biomarker_indices[start_idx:end_idx]
                
    #             # 每个biomarker一行，两列（Pred和GT）
    #             fig, axes = plt.subplots(len(current_biomarkers), 2, 
    #                                     figsize=(12, 4 * len(current_biomarkers)))
    #             if len(current_biomarkers) == 1:
    #                 axes = axes.reshape(1, -1)
                
    #             for row_idx, j in enumerate(current_biomarkers):
    #                 img_pred, img_gt, ssim_val = acq_images[j]
    #                 biomarker_name = bm_labels[j] if j < len(bm_labels) else f'Biomarker_{j}'
                    
    #                 # 归一化：使用全局范围进行归一化，然后clip到[0,1]
    #                 def normalize_with_range(img, vmin, vmax):
    #                     """使用给定的范围归一化图像"""
    #                     if vmax > vmin:
    #                         img_norm = (img - vmin) / (vmax - vmin)
    #                         img_norm = np.clip(img_norm, 0, 1)
    #                     else:
    #                         # 如果范围无效，使用局部归一化
    #                         if img.max() > img.min():
    #                             img_norm = (img - img.min()) / (img.max() - img.min())
    #                         else:
    #                             img_norm = np.zeros_like(img)
    #                     return img_norm
                    
    #                 # 对每个图像，优先使用全局范围，如果该图像范围超出全局范围则使用局部归一化
    #                 pred_vmin = min(pred_min, img_pred[img_pred > 0].min() if np.any(img_pred > 0) else pred_min)
    #                 pred_vmax = max(pred_max, img_pred.max() if np.any(img_pred > 0) else pred_max)
    #                 gt_vmin = min(gt_min, img_gt[img_gt > 0].min() if np.any(img_gt > 0) else gt_min)
    #                 gt_vmax = max(gt_max, img_gt.max() if np.any(img_gt > 0) else gt_max)
                    
    #                 img_pred_norm = normalize_with_range(img_pred, pred_vmin, pred_vmax)
    #                 img_gt_norm = normalize_with_range(img_gt, gt_vmin, gt_vmax)
                    
    #                 # 预测图（灰度图）
    #                 im1 = axes[row_idx, 0].imshow(img_pred_norm, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    #                 axes[row_idx, 0].set_title(f'Prediction: {biomarker_name}\nSSIM: {ssim_val:.4f}', 
    #                                            fontsize=10, fontweight='bold')
    #                 axes[row_idx, 0].axis('off')
                    
    #                 # GT图（灰度图）
    #                 im2 = axes[row_idx, 1].imshow(img_gt_norm, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    #                 axes[row_idx, 1].set_title(f'Ground Truth: {biomarker_name}', 
    #                                            fontsize=10, fontweight='bold')
    #                 axes[row_idx, 1].axis('off')
                
    #             # 添加说明
    #             file_suffix = f"_part{file_idx+1}" if num_files > 1 else ""
    #             title = f'Protein Expression Visualization - {acq_id} (Step {step})\n'
    #             title += f'Biomarkers {start_idx+1}-{end_idx} of {len(valid_biomarker_indices)} | '
    #             title += f'Each point represents average of 8×8 patch | Global avg SSIM: {val_ssim:.4f}'
                
    #             plt.suptitle(title, fontsize=11, y=0.995)
    #             plt.tight_layout(rect=[0, 0, 1, 0.98])
                
    #             # 保存图像
    #             save_path = os.path.join(viz_dir, f'{acq_id}_grayscale{file_suffix}.png')
    #             plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    #             plt.close()
                
    #             print(f"  已保存灰度可视化: {save_path}")
            
    #         # 额外保存一个所有biomarker的平均SSIM信息文件
    #         all_ssims = [acq_images[j][2] for j in valid_biomarker_indices]
    #         avg_ssim = np.mean(all_ssims) if all_ssims else 0.0
    #         info_path = os.path.join(viz_dir, f'{acq_id}_ssim_info.txt')
    #         with open(info_path, 'w') as f:
    #             f.write(f"Acquisition: {acq_id}\n")
    #             f.write(f"Step: {step}\n")
    #             f.write(f"Number of biomarkers visualized: {len(valid_biomarker_indices)}\n")
    #             f.write(f"Average SSIM (this acquisition, visualized biomarkers): {avg_ssim:.4f}\n")
    #             f.write(f"Global average SSIM (all acquisitions, all biomarkers): {val_ssim:.4f}\n")
    #             f.write("\nPer-biomarker SSIM:\n")
    #             for j in valid_biomarker_indices:
    #                 biomarker_name = bm_labels[j] if j < len(bm_labels) else f'Biomarker_{j}'
    #                 ssim_val = acq_images[j][2]
    #                 f.write(f"  {biomarker_name}: {ssim_val:.4f}\n")

    
    # ===================== 2. 计算重建图像上的 SSIM (使用高斯平滑重建) =====================
    def _compute_image_ssim(preds, targets, valid_mask, xs, ys, acq_ids, 
                          stride: int = 8, kernel_size: int = 16):
        """
        使用高斯加权融合将稀疏的预测点重建为全尺寸平滑灰度图像，并计算 SSIM。
        原理类似 evaluate.py 的 process_image，但针对的是点状数据。
        """
        # 确保坐标是整数
        xs = xs.astype(np.int64)
        ys = ys.astype(np.int64)
        acq_ids = np.asarray(acq_ids)
        
        N, B = targets.shape
        ssim_vals = []
        images_for_viz = {} 
        
        # 预计算高斯核 (kernel_size 建议设为 2 * stride 以保证覆盖)
        # sigma 设为 kernel_size / 4 可以保证边缘平滑衰减
        weight_kernel = get_gaussian_kernel(kernel_size=kernel_size, sigma=kernel_size/4)
        half_k = kernel_size // 2

        for acq in np.unique(acq_ids):
            idx_acq = np.where(acq_ids == acq)[0]
            if idx_acq.size == 0:
                continue

            xs_acq = xs[idx_acq]
            ys_acq = ys[idx_acq]
            preds_acq = preds[idx_acq]
            targets_acq = targets[idx_acq]
            valid_acq = valid_mask[idx_acq]

            # 计算画布大小 (归一化坐标到从0开始)
            x_min, y_min = xs_acq.min(), ys_acq.min()
            xs0 = xs_acq - x_min
            ys0 = ys_acq - y_min
            
            # 画布尺寸需要留出边缘 padding，防止 kernel 越界
            # 加大一点尺寸确保容纳所有 kernel
            H = int(ys0.max() + kernel_size + 1)
            W = int(xs0.max() + kernel_size + 1)
            
            if H <= 1 or W <= 1:
                continue
            
            images_for_viz[acq] = {}
            
            # 遍历每个 biomarker
            for j in range(B):
                vmask = valid_acq[:, j]
                # 如果有效点太少，跳过
                if vmask.sum() < 2:
                    continue

                # 初始化累加器 (分子) 和 权重图 (分母)
                pred_num = np.zeros((H, W), dtype=np.float32)
                pred_den = np.zeros((H, W), dtype=np.float32)
                
                gt_num = np.zeros((H, W), dtype=np.float32)
                gt_den = np.zeros((H, W), dtype=np.float32)

                # 筛选出当前 acquisition + 当前 biomarker 的有效点
                valid_indices = np.where(vmask)[0]
                
                # --- 核心：高斯加权累加循环 ---
                for k in valid_indices:
                    cx, cy = xs0[k], ys0[k]
                    val_p = preds_acq[k, j]
                    val_g = targets_acq[k, j]
                    
                    # 确定 ROI 坐标 (以点为中心放置 kernel)
                    t = int(max(0, cy - half_k))
                    b = int(min(H, cy + half_k))
                    l = int(max(0, cx - half_k))
                    r = int(min(W, cx + half_k))
                    
                    # 确定 kernel 的切片 (处理边缘裁剪情况)
                    kt = int(t - (cy - half_k))
                    kb = int(b - (cy - half_k))
                    kl = int(l - (cx - half_k))
                    kr = int(r - (cx - half_k))
                    
                    # 截取对应的 kernel
                    curr_kernel = weight_kernel[kt:kb, kl:kr]
                    
                    # 累加 Pred (加权)
                    pred_num[t:b, l:r] += val_p * curr_kernel
                    pred_den[t:b, l:r] += curr_kernel
                    
                    # 累加 GT (加权)
                    gt_num[t:b, l:r] += val_g * curr_kernel
                    gt_den[t:b, l:r] += curr_kernel

                # --- 归一化 (除以权重和) ---
                # 避免除以 0
                pred_den = np.maximum(pred_den, 1e-8)
                gt_den = np.maximum(gt_den, 1e-8)
                
                img_pred = pred_num / pred_den
                img_gt = gt_num / gt_den

                # --- 计算 SSIM ---
                data_range = float(img_gt.max() - img_gt.min())
                # 如果 GT 是纯平的(例如全0)，SSIM 无意义，设为 0 或 1
                if data_range <= 1e-6:
                    s = 0.0 
                else:
                    s = ssim(img_gt, img_pred, data_range=data_range)
                
                if np.isfinite(s):
                    ssim_vals.append(s)
                    # 保存重建好的图用于可视化
                    images_for_viz[acq][j] = (img_pred.copy(), img_gt.copy(), s)

        if not ssim_vals:
            return float('nan'), {}
            
        return float(np.mean(ssim_vals)), images_for_viz
    
    # 调用计算函数
    # 假设你的数据生成脚本里 STRIDE = 8，这里 kernel_size 设为 16 效果最好
    val_ssim, images_for_viz = _compute_image_ssim(
        all_preds, all_targets, all_valid, all_x, all_y, all_acq_ids, 
        stride=8, kernel_size=16
    )
    
    # ===================== 3. 保存可视化图像 (适配平滑灰度图) =====================
    if save_visualizations and images_for_viz:
        viz_dir = os.path.join(run_dir, 'visualizations_all_biomarkers', f'step_{step}')
        os.makedirs(viz_dir, exist_ok=True)
        
        for acq_id, acq_images in images_for_viz.items():
            if len(acq_images) == 0:
                continue
            
            # 确定要可视化的 biomarker 列表
            if num_biomarkers_to_viz == -1:
                biomarkers_to_viz = sorted(acq_images.keys())
            else:
                biomarkers_to_viz = list(range(min(num_biomarkers_to_viz, len(bm_labels))))
            
            valid_biomarker_indices = [j for j in biomarkers_to_viz if j in acq_images]
            if not valid_biomarker_indices:
                continue
            
            # --- 归一化辅助函数 ---
            def normalize_img(img, p_min=1, p_max=99):
                """基于百分位数的鲁棒归一化"""
                if img.size == 0 or img.max() == img.min():
                    return np.zeros_like(img)
                # 排除背景 0 值的影响，只统计有信号的区域
                valid_pixels = img[img > 1e-6]
                if valid_pixels.size == 0:
                    vmin, vmax = img.min(), img.max()
                else:
                    vmin = np.percentile(valid_pixels, p_min)
                    vmax = np.percentile(valid_pixels, p_max)
                
                if vmax > vmin:
                    return np.clip((img - vmin) / (vmax - vmin), 0, 1)
                else:
                    return np.zeros_like(img)

            # 分页保存，防止图片太长
            max_biomarkers_per_file = 10
            num_files = (len(valid_biomarker_indices) + max_biomarkers_per_file - 1) // max_biomarkers_per_file
            
            for file_idx in range(num_files):
                start_idx = file_idx * max_biomarkers_per_file
                end_idx = min(start_idx + max_biomarkers_per_file, len(valid_biomarker_indices))
                current_biomarkers = valid_biomarker_indices[start_idx:end_idx]
                
                
                # 绘图：每行一个 biomarker，左边 Pred，右边 GT
                fig, axes = plt.subplots(len(current_biomarkers), 2, 
                                       figsize=(10, 4 * len(current_biomarkers)))
                if len(current_biomarkers) == 1:
                    axes = axes.reshape(1, -1)
                
                for row_idx, j in enumerate(current_biomarkers):
                    img_pred, img_gt, ssim_val = acq_images[j]
                    biomarker_name = bm_labels[j] if j < len(bm_labels) else f'Biomarker_{j}'
                    
                    # 归一化用于显示
                    disp_pred = normalize_img(img_pred)
                    disp_gt = normalize_img(img_gt)
                    
                    # Pred
                    ax_p = axes[row_idx, 0]
                    im1 = ax_p.imshow(disp_pred, cmap='gray', interpolation='nearest')
                    ax_p.set_title(f'Pred: {biomarker_name}\nSSIM: {ssim_val:.3f}')
                    ax_p.axis('off')
                    plt.colorbar(im1, ax=ax_p, fraction=0.046, pad=0.04)
                    
                    # GT
                    ax_g = axes[row_idx, 1]
                    im2 = ax_g.imshow(disp_gt, cmap='gray', interpolation='nearest')
                    ax_g.set_title(f'GT: {biomarker_name}')
                    ax_g.axis('off')
                    plt.colorbar(im2, ax=ax_g, fraction=0.046, pad=0.04)
                
                file_suffix = f"_part{file_idx+1}" if num_files > 1 else ""
                plt.suptitle(f'Reconstruction: {acq_id} (Step {step})', y=0.995)
                plt.tight_layout()
                
                save_path = os.path.join(viz_dir, f'{acq_id}_smooth{file_suffix}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  已保存平滑重建图: {save_path}")

            # 保存统计信息 txt
            all_ssims = [acq_images[j][2] for j in valid_biomarker_indices]
            avg_ssim = np.mean(all_ssims) if all_ssims else 0.0
            with open(os.path.join(viz_dir, f'{acq_id}_stats.txt'), 'w') as f:
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
                for j in valid_biomarker_indices:
                    name = bm_labels[j] if j < len(bm_labels) else str(j)
                    f.write(f"{name}: {acq_images[j][2]:.4f}\n")

    return val_pr, val_ssim

    


def get_available_gpus(preferred_gpus: Optional[List[int]] = None) -> List[int]:
    """
    检测并返回可用的GPU设备列表。
    会测试每个GPU是否可以正常使用，过滤掉有问题的GPU。
    
    Args:
        preferred_gpus: 优先使用的GPU ID列表。如果指定，只测试这些GPU。
    
    Returns:
        可用GPU设备ID的列表（相对于CUDA_VISIBLE_DEVICES的本地ID）
    """
    available_gpus = []
    
    # 检查CUDA_VISIBLE_DEVICES环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible:
        print(f"检测到CUDA_VISIBLE_DEVICES={cuda_visible}")
    
    if not torch.cuda.is_available():
        print("警告: torch.cuda.is_available()返回False")
        # 即使is_available返回False，也尝试检测GPU（可能是驱动问题但GPU实际可用）
        print("尝试强制检测GPU设备...")
    
    try:
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU设备")
        
        if num_gpus == 0:
            print("没有检测到GPU设备")
            return available_gpus
        
        # 如果指定了优先使用的GPU列表，只测试这些GPU
        # 注意：如果设置了CUDA_VISIBLE_DEVICES，这里的ID应该是相对于可见设备的
        if preferred_gpus is not None:
            gpu_ids_to_test = [gpu_id for gpu_id in preferred_gpus if gpu_id < num_gpus]
            if not gpu_ids_to_test:
                # 如果所有preferred_gpus都超出范围，测试所有可用GPU
                gpu_ids_to_test = list(range(num_gpus))
                print(f"警告: 指定的GPU {preferred_gpus}超出范围，将测试所有可用GPU")
        else:
            gpu_ids_to_test = list(range(num_gpus))
        
        for gpu_id in gpu_ids_to_test:
            try:
                # 尝试在每个GPU上创建一个tensor来测试
                torch.cuda.set_device(gpu_id)
                test_tensor = torch.zeros(1).cuda(gpu_id)
                # 尝试一个简单的计算来确保GPU真的可用
                result = test_tensor + 1
                del test_tensor, result
                torch.cuda.empty_cache()
                
                # 获取GPU名称
                try:
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                except:
                    gpu_name = "Unknown"
                print(f"  GPU {gpu_id}: {gpu_name} - 可用 ✓")
                available_gpus.append(gpu_id)
                
            except Exception as e:
                print(f"  GPU {gpu_id}: 不可用 ✗ (错误: {str(e)})")
                continue
                
    except Exception as e:
        print(f"检测GPU时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    if available_gpus:
        print(f"将使用以下GPU: {available_gpus}")
    else:
        print("没有可用的GPU，将使用CPU")
    
    return available_gpus


def main():
    """Main training and evaluation function."""
    
    # Initialize wandb for experiment tracking
    # 增大初始化超时时间，避免代理或网络慢导致超时
    wandb.init(
        project='hande_to_codex',
        name='model_training',
        settings=wandb.Settings(init_timeout=180)
    )

    # Set up data transforms
    transform_train = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize(224, antialias=True),
            transforms.RandomRotation(degrees=(-10, 10)),
        ]),
        'image_only': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    transform_eval = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
        ]),
        'image_only': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    # Load metadata and data
    metadata_dict = pd.read_pickle(METADATA_FILE)
    data_df = pd.read_parquet(DATA_FILE)
    # import pdb
    # pdb.set_trace()

    # # --- Debug prints to inspect data and splits ---
    # print("ROOT_DIR:", ROOT_DIR)
    # print("IMAGE_DIR:", IMAGE_DIR)
    # print("METADATA_FILE:", METADATA_FILE)
    # print("DATA_FILE:", DATA_FILE)
    # print("DATASET_SPLITS:", DATASET_SPLITS)
    # print()

    # # show keys in metadata_dict
    # print("metadata_dict keys (sample 10):", list(metadata_dict.keys())[:10])
    # print("all_biomarkers:", metadata_dict.get('all_biomarkers', [])[:10])
    # print()

    # # show data_df basic info
    # print("data_df shape:", data_df.shape)
    # print("data_df columns:", data_df.columns.tolist())
    # if 'HE_COVERSLIP_ID' in data_df.columns:
    #     print("Unique HE_COVERSLIP_ID (sample 20):", data_df['HE_COVERSLIP_ID'].unique()[:20])
    # if 'CODEX_ACQUISITION_ID' in data_df.columns:
    #     print("Unique CODEX_ACQUISITION_ID (sample 20):", data_df['CODEX_ACQUISITION_ID'].unique()[:20])
    # print()
    # # check intersection between your subset strings and data_df[HE_COVERSLIP_ID]
    # for split_name, ids in DATASET_SPLITS.items():
    #     if len(ids) == 0:
    #         print(f"{split_name} : EMPTY list in DATASET_SPLITS")
    #         continue
    #     # intersection with HE_COVERSLIP_ID
    #     if 'HE_COVERSLIP_ID' in data_df.columns:
    #         inter = set(ids).intersection(set(data_df['HE_COVERSLIP_ID'].unique()))
    #         print(f"{split_name} - subset count in HE_COVERSLIP_ID: {len(inter)} / {len(ids)} ; matches: {list(inter)[:10]}")
    #     # intersection with metadata_dict values (HE_REGION_UUID)
    #     # build set of HE_REGION_UUIDs from metadata_dict values if available
    #     he_region_uuids = set()
    #     for k,v in metadata_dict.items():
    #         try:
    #             he_region_uuids.add(v.get('HE_REGION_UUID'))
    #         except:
    #             pass
    #     inter2 = set(ids).intersection(he_region_uuids)
    #     print(f"{split_name} - subset count in metadata HE_REGION_UUID: {len(inter2)} / {len(ids)} ; matches: {list(inter2)[:10]}")
    # print("---------")


    
    # Create datasets
    # import pdb 
    # pdb.set_trace()
    train_dataset = ImageDataset(
        data_df=data_df,
        root_dir=ROOT_DIR,
        transform=transform_train,
        metadata_dict=metadata_dict,
        subset=DATASET_SPLITS['train']
    )

    val_dataset = ImageDataset(
        data_df=data_df,
        root_dir=ROOT_DIR,
        transform=transform_eval,
        metadata_dict=metadata_dict,
        subset=DATASET_SPLITS['val'],
        is_test=True
    )

    # Create data loaders
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        # 如果batch为空（所有样本都被过滤掉），返回None，让训练循环跳过
        if len(batch) == 0:
            return None
        return torch.utils.data.default_collate(batch)

    # 打印数据集信息
    print(f"\n数据集信息:")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Num workers: {NUM_WORKERS}")
    
    # 测试数据集样本有效性
    print(f"\n检查训练数据集样本有效性:")
    valid_samples = 0
    invalid_samples = 0
    for i in range(min(100, len(train_dataset))):  # 检查前100个样本
        try:
            sample = train_dataset[i]
            if sample is None:
                invalid_samples += 1
            else:
                valid_samples += 1
        except Exception as e:
            invalid_samples += 1
            if i == 0:
                print(f"  样本{i}加载错误: {e}")
    
    print(f"  前{min(100, len(train_dataset))}个样本中: 有效={valid_samples}, 无效={invalid_samples}")
    
    if valid_samples == 0:
        print(f"  严重警告: 训练集中没有有效样本！")
        print(f"  请检查数据文件路径和数据集配置")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # 测试DataLoader
    print(f"\nDataLoader测试:")
    print(f"  预计batch数量: {len(train_loader)}")
    try:
        # 测试获取一个batch
        test_batch = next(iter(train_loader))
        print(f"  成功获取batch: inputs shape={test_batch[0].shape}, labels shape={test_batch[1].shape}")
    except StopIteration:
        print("  错误: DataLoader为空，无法获取任何batch")
    except RuntimeError as e:
        if "Empty batch" in str(e):
            print(f"  警告: 第一个batch为空（所有样本被过滤）")
        else:
            print(f"  错误: {e}")
    except Exception as e:
        print(f"  错误: 无法获取batch: {e}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 4,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Set up model and training
    # 如果设置了CUDA_VISIBLE_DEVICES，GPU ID会重新映射，所以使用None让函数自动检测所有可见GPU
    # CUDA_VISIBLE_DEVICES已经在代码开头设置了，所以这里不需要指定preferred_gpus
    available_gpus = get_available_gpus(preferred_gpus=None)
    
    if available_gpus:
        # 使用第一个可用GPU作为主设备
        device = torch.device(f'cuda:{available_gpus[0]}')
        print(f"✓ 使用设备: {device}")
        
        # 如果有多个可用GPU，使用DataParallel
        if len(available_gpus) > 1:
            model = get_model(num_outputs=len(metadata_dict['all_biomarkers']))
            model = nn.DataParallel(model, device_ids=available_gpus)
            print(f"✓ 使用DataParallel，GPU设备: {available_gpus}")
        else:
            model = get_model(num_outputs=len(metadata_dict['all_biomarkers']))
            # 确保模型在正确的GPU上
            torch.cuda.set_device(available_gpus[0])
            print(f"✓ 使用单GPU: {available_gpus[0]}")
    else:
        # 没有可用GPU，使用CPU
        device = torch.device('cpu')
        print("⚠️  警告: 没有可用的GPU，将在CPU上运行（训练速度会很慢）")
        print("   建议检查CUDA驱动和PyTorch CUDA版本是否匹配")
        print(f"   PyTorch版本: {torch.__version__}")
        model = get_model(num_outputs=len(metadata_dict['all_biomarkers']))
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = masked_mse_loss

    # Training loop
    step = 0
    best_val_score = 0
    steps_since_best_val_score = 0

    while True:
        model.train()
        for inputs, labels, mask, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            # import pdb
            # pdb.set_trace()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, mask)

            if step % 10 == 0:  # 每10步打印一次
                print(f"[Step {step}] Loss = {loss.item():.6f}")
            
            if torch.isnan(loss):
                print(f"[Step {step}] Warning: Loss is NaN, skipping batch")
                continue
                
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                wandb.log({'train_loss': loss.item()})

            # Validation
            if step % EVAL_INTERVAL == 0:
                # import pdb
                # pdb.set_trace()
                # val_r2, val_ssim = evaluate(
                #     model, val_loader, device, 
                #     os.path.join(ROOT_DIR, 'runs'), 
                #     step, metadata_dict['all_biomarkers']
                # )
                
                # val_score = val_r2 + val_ssim
                # wandb.log({
                #     'val_r2': val_r2,
                #     'val_ssim': val_ssim,
                #     'val_score': val_score
                # })

                val_r2, val_ssim = evaluate(
                    model, val_loader, device, 
                    os.path.join(ROOT_DIR, 'runs'), 
                    step, metadata_dict['all_biomarkers'],
                    save_visualizations=True,
                    num_biomarkers_to_viz=-1  # -1表示可视化所有biomarker，或指定数量如6
                )

                # 防止 NaN 影响 early stopping 和 scheduler
                if np.isnan(val_r2):
                    val_r2 = 0.0
                if np.isnan(val_ssim):
                    val_ssim = 0.0

                val_score = val_r2 + val_ssim

                print(f"[Step {step}] val_r2 (Pearson R) = {val_r2:.4f}, val_ssim = {val_ssim:.4f}, val_score = {val_score:.4f}")

                wandb.log({
                    'val_r2': val_r2,
                    'val_ssim': val_ssim,
                    'val_score': val_score
                })


                # Save best model
                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), os.path.join(ROOT_DIR, 'runs', 'best_model.pth'))
                    steps_since_best_val_score = 0
                else:
                    steps_since_best_val_score += EVAL_INTERVAL

                scheduler.step(val_score)

                # Early stopping check
                if steps_since_best_val_score >= PATIENCE:
                    print(f'Early stopping after {step} steps')
                    return

                model.train()

            step += 1

if __name__ == '__main__':
    main()