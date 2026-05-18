import os
import random
import math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, get_worker_info
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import scipy.signal
from typing import NamedTuple, Dict, List, Tuple


H, W = 3024, 4032 ## Hardcoded for MATADOR dataset

## UTILS FUNCTIONS

def matador_paths(texture_folder: Path, label_folder: Path, context_folder: Path = None, random_seed: int = 42, split_ratios: tuple = (0.7, 0.1, 0.2)):
    # Load textures paths
    paths = list(Path(texture_folder).iterdir())
    if context_folder is not None:
        paths +=  list(Path(context_folder).iterdir())
    print(f"Found {len(paths)} images ...")

    # Filter out textures with associated labels defined in MATADOR_IGNORE
    if label_folder is not None:
        MATADOR_IGNORE = {"thermoplastic", "thermoset", "elastomer",
                "paint", "glass", "porcelain",
                "aluminum", "steel", "brass", 
                "iron", "bronze", "copper",
                "silk", "polyester", "fiber",
            }
        ignored_basenames = set()
        for label_file in Path(label_folder).glob('*.txt'):
            label = label_file.read_text().splitlines()[0].strip()
            if label in MATADOR_IGNORE:
                ignored_basenames.add(label_file.stem)
        paths = [path_obj for path_obj in paths if path_obj.stem not in ignored_basenames]
        print(f"Filtered out {MATADOR_IGNORE} textures")
        print(f"Remaining: {len(paths)}")

    rng = random.Random(random_seed)
    rng.shuffle(paths)
    # Split paths into train, val, test sets
    num_total = len(paths)
    num_train = int(num_total * split_ratios[0])
    num_val = int(num_total * split_ratios[1])
    
    train_paths = paths[:num_train]
    print(f"Initialized training set with {len(train_paths)} textures.")
   
    val_paths = paths[num_train : num_train + num_val]
    print(f"Initialized validation set with {len(val_paths)} textures.")
    
    test_paths = paths[num_train + num_val:]
    print(f"Initialized test set with {len(test_paths)} textures.")

    return train_paths, val_paths, test_paths

def list_image_paths(folder: Path):
    paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not paths:
        raise RuntimeError(f"No images found in {folder}")
    return paths

def inspect_and_load(p: Path):
    with Image.open(p) as im:
        im = im.convert('L')
        if im.size != (W, H):
            return p, None  # Return None if shape is incorrect
        arr = np.array(im, dtype=np.uint8)
    return p, arr

# --------------------------------#
# Shared memory loading of images #
# --------------------------------#
def preload_to_tensor(path_list, n_threads=8):
    """
    Preload all grayscale images into a shared torch.uint8 tensor of shape (N, H, W).
    Uses a ThreadPoolExecutor to parallelize image decoding (IO-bound).
    Returns the shared tensor and the path list used.
    """
    if not path_list:
        print("Warning: No images found in path_list.")
        return torch.empty(0), []

    print(f"Target shape is {H}x{W}. Loading and filtering images concurrently...")

    # Use ThreadPoolExecutor for IO-bound loads
    n_threads = min(n_threads, os.cpu_count() or 4)
    print(f"Using {n_threads} threads ...")
    
    loaded_results = {}
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(inspect_and_load, p): p for p in path_list}
        
        for fut in as_completed(futures):
            p, arr = fut.result()
            if arr is not None:
                loaded_results[p] = arr

    paths_to_load = [p for p in path_list if p in loaded_results]
    loaded_arrays = [loaded_results[p] for p in paths_to_load]
    
    original_count = len(path_list)
    skipped_count = original_count - len(paths_to_load)
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images with incorrect dimensions.")

    if not paths_to_load:
        print("Warning: No images with the correct shape found.")
        return torch.empty(0), []

    N = len(paths_to_load)
    print(f"Loading {N} images of shape {H}x{W}")

    # quick memory estimate
    per_img_mb = (H * W) / (1024**2)
    total_gb = (N * per_img_mb) / 1024
    print(f"Estimated memory (uint8): {per_img_mb:.3f} MB per image, {total_gb:.2f} GB total")

    # sanity check: compare to system memory
    vm = psutil.virtual_memory()
    print(f"System RAM total: {vm.total/1024**3:.2f} GB, available: {vm.available/1024**3:.2f} GB")

    if total_gb > vm.available * 0.9 / 1024:
        print("WARNING: estimated memory for all images may exceed available memory!")

    # Allocate shared tensor (uint8)
    shared = torch.empty((N, H, W), dtype=torch.uint8)
    # Reserve shared memory before loading
    shared.share_memory_()
    print("Allocated shared tensor and called share_memory_()")

    # Now, copy the already-loaded numpy arrays into the shared tensor
    for i, arr in enumerate(loaded_arrays):
        shared[i].copy_(torch.from_numpy(arr))

    print("All images loaded into shared tensor.")
    return shared, paths_to_load

def dynamic_pano_collate_fn(batch):
    """
    Custom collate function to handle panoramas of different sizes.
    It finds the max dimensions in the batch and pads all panoramas to that size.
    """
    # Separate the components of the batch
    panos, all_offsets, thetas, targets = zip(*batch)

    # --- Find max dimensions for panoramas ---
    max_h = max([p.shape[0] for p in panos])
    max_w = max([p.shape[1] for p in panos])

    # --- Pad panoramas and stack them ---
    # Create a tensor to hold the padded panoramas
    padded_panos = torch.zeros(len(panos), max_h, max_w)
    for i, p in enumerate(panos):
        h, w = p.shape
        padded_panos[i, :h, :w] = p

    # --- Stack other tensors normally ---
    stacked_offsets = torch.stack(all_offsets, 0)
    stacked_thetas = torch.stack(thetas, 0)
    stacked_targets = torch.stack(targets, 0)

    return padded_panos, stacked_offsets, stacked_thetas, stacked_targets


def build_tiled_pano(shared_images: torch.Tensor, target_h: int, target_w: int,
                     randomize_tiles: bool = True) -> torch.Tensor:
    """
    Stitch full-resolution tiles from shared_images until covering target size,
    then crop to exactly [target_h, target_w].

    Args:
      shared_images: uint8 tensor [N, H_tex, W_tex], all same H_tex, W_tex.
      target_h, target_w: required pano size in pixels.
      randomize_tiles: if True, choose a random image for each tile position.

    Returns:
      pano: float32 [target_h, target_w] in [0, 1].
    """
    assert shared_images.dtype == torch.uint8 and shared_images.dim() == 3, "shared_images must be uint8 [N,H,W]"
    N, H_tex, W_tex = shared_images.shape

    # Number of tiles needed to cover target size
    ny = math.ceil(target_h / H_tex)
    nx = math.ceil(target_w / W_tex)

    canvas_h = ny * H_tex
    canvas_w = nx * W_tex
    canvas = torch.empty((canvas_h, canvas_w), dtype=torch.uint8)

    if randomize_tiles:
        idxs = torch.randint(low=0, high=N, size=(ny, nx))
    else:
        # Deterministic tiling pattern
        idxs = torch.arange(ny * nx).reshape(ny, nx) % N

    y = 0
    for iy in range(ny):
        x = 0
        for ix in range(nx):
            img = shared_images[int(idxs[iy, ix])]
            canvas[y:y+H_tex, x:x+W_tex] = img
            x += W_tex
        y += H_tex

    # Crop to exact target size
    pano = canvas[:target_h, :target_w].to(torch.float32).div_(255.0)
    return pano




class Trajectory(NamedTuple):
    pos:  torch.Tensor   # [T, >=2], smoothed
    ori:  torch.Tensor   # [T, 3],   smoothed (roll, pitch, yaw)
    vel:  torch.Tensor   # [T, 3],   body-frame velocity (vx, vy, vz)
    gyro: torch.Tensor   # [T, 3],   body-frame gyro (wx, wy, wz)
    Hz:   int

def _ma1d_np(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.astype(np.float64, copy=True)
    left = (w - 1) // 2
    right = w - 1 - left
    xpad = np.pad(x.astype(np.float64), (left, right), mode='reflect')
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(xpad, kernel, mode='valid')

def _smooth_pos_ori_np(pos_global: np.ndarray,
                              ori_global: np.ndarray,
                              w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Common smoothing: identical for both datasets."""
    pos_sm = np.array(pos_global, copy=True, dtype=np.float64)
    x_s = _ma1d_np(pos_global[:, 0], w)
    y_s = _ma1d_np(pos_global[:, 1], w)
    pos_sm[:, 0] = x_s
    pos_sm[:, 1] = y_s

    ori_sm = np.array(ori_global, copy=True, dtype=np.float64)
    # smooth roll, pitch
    if ori_sm.shape[1] >= 2:
        ori_sm[:, 0] = _ma1d_np(ori_sm[:, 0], w)
        ori_sm[:, 1] = _ma1d_np(ori_sm[:, 1], w)

    # yaw: unwrap -> MA -> rewrap [-pi, pi]
    yaw_un = np.unwrap(ori_global[:, 2].astype(np.float64))
    yaw_un_s = _ma1d_np(yaw_un, w)
    yaw_s = (yaw_un_s + np.pi) % (2.0 * np.pi) - np.pi
    ori_sm[:, 2] = yaw_s
    return pos_sm, ori_sm

def _derive_vel_gyro_np(pos_sm: np.ndarray,
                               ori_sm: np.ndarray,
                               Hz: int) -> Tuple[np.ndarray, np.ndarray]:
    """Common derivation: identical for both datasets."""
    dt = 1.0 / float(Hz)
    x_s = pos_sm[:, 0].astype(np.float64)
    y_s = pos_sm[:, 1].astype(np.float64)
    yaw_s = ori_sm[:, 2].astype(np.float64)

    dxdt = np.gradient(x_s, dt)
    dydt = np.gradient(y_s, dt)

    yaw_un_s = np.unwrap(yaw_s.astype(np.float64))
    wz_s = np.gradient(yaw_un_s, dt)

    c = np.cos(yaw_s)
    s = np.sin(yaw_s)
    vx_w = dxdt
    vy_w = dydt
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w

    N = pos_sm.shape[0]
    vel_body_sm = np.zeros((N, 3), dtype=np.float64)
    vel_body_sm[:, 0] = vx_b
    vel_body_sm[:, 1] = vy_b

    gyro_sm = np.zeros((N, 3), dtype=np.float64)
    gyro_sm[:, 2] = wz_s
    return vel_body_sm, gyro_sm

def load_and_process_trajectory(path: Path,
                                Hz: int,
                                ma_window: int) -> Trajectory:
    """
    Shared loader for one Tartan trajectory P*:
    - loads pos_global/ori_global
    - smooths with MA
    - derives body frame vel/gyro
    """
    pos_global_path = path / 'imu' / 'pos_global.txt'
    ori_global_path = path / 'imu' / 'ori_global.txt'
    if not (pos_global_path.exists() and ori_global_path.exists()):
        raise FileNotFoundError(f"Missing pos/ori in {path}")

    pos_global = np.loadtxt(pos_global_path)
    ori_global = np.loadtxt(ori_global_path)

    alpha_v = 0.4/5.25 #scaling factor if you want smaller speeds
    pos_global = pos_global # * alpha_v 

    pos_sm_np, ori_sm_np = _smooth_pos_ori_np(pos_global, ori_global, ma_window)
    vel_np, gyro_np = _derive_vel_gyro_np(pos_sm_np, ori_sm_np, Hz)

    return Trajectory(
        pos=torch.from_numpy(pos_sm_np).float(),
        ori=torch.from_numpy(ori_sm_np).float(),
        vel=torch.from_numpy(vel_np).float(),
        gyro=torch.from_numpy(gyro_np).float(),
        Hz=Hz,
    )

def resample_trajectory(traj: Trajectory,
                        Hz_out: int) -> Trajectory:
    """Shared resampling: pos, ori (yaw unwrapped), vel, gyro – same code for both datasets."""
    Hz_in = traj.Hz
    if Hz_in == Hz_out:
        return traj

    pos_m, ori_rad, vel_ms, gyro_rads = traj.pos, traj.ori, traj.vel, traj.gyro

    T_in = pos_m.shape[0]
    t_in = torch.linspace(0, (T_in - 1) / Hz_in, T_in, dtype=torch.float64)
    T_out = int(math.ceil((T_in - 1) * Hz_out / Hz_in)) + 1
    t_out = torch.linspace(0, (T_in - 1) / Hz_in, T_out, dtype=torch.float64)

    def interp_1d(x_in, y_in, x_out):
        y = np.interp(x_out.cpu().numpy(), x_in.cpu().numpy(), y_in.cpu().numpy())
        return torch.from_numpy(y).to(dtype=torch.float64)

    pos_out = torch.zeros(T_out, pos_m.shape[1], dtype=torch.float64)
    pos_out[:, 0] = interp_1d(t_in, pos_m[:, 0].double(), t_out)
    pos_out[:, 1] = interp_1d(t_in, pos_m[:, 1].double(), t_out)
    for k in range(2, pos_m.shape[1]):
        pos_out[:, k] = interp_1d(t_in, pos_m[:, k].double(), t_out)

    yaw_src = ori_rad[:, 2].double().cpu().numpy()
    yaw_unwrapped = np.unwrap(yaw_src)
    yaw_out = np.interp(t_out.cpu().numpy(), t_in.cpu().numpy(), yaw_unwrapped)
    ori_out = torch.zeros(T_out, ori_rad.shape[1], dtype=torch.float64)
    if ori_rad.shape[1] >= 2:
        ori_out[:, 0] = interp_1d(t_in, ori_rad[:, 0].double(), t_out)
        ori_out[:, 1] = interp_1d(t_in, ori_rad[:, 1].double(), t_out)
    ori_out[:, 2] = torch.from_numpy(yaw_out)

    vel_out = torch.zeros(T_out, vel_ms.shape[1], dtype=torch.float64)
    gyro_out = torch.zeros(T_out, gyro_rads.shape[1], dtype=torch.float64)
    for k in range(vel_ms.shape[1]):
        vel_out[:, k] = interp_1d(t_in, vel_ms[:, k].double(), t_out)
    for k in range(gyro_rads.shape[1]):
        gyro_out[:, k] = interp_1d(t_in, gyro_rads[:, k].double(), t_out)

    return Trajectory(
        pos=pos_out.float(),
        ori=ori_out.float(),
        vel=vel_out.float(),
        gyro=gyro_out.float(),
        Hz=Hz_out,
    )

class TartanTrajectoryDataset_1000Hz_V2(Dataset):
    """
    Dataset that uses real-world trajectory data from a dataset like TartanAir
    to drive the visual odometry simulator. It loads long trajectories, chunks
    them into smaller sub-trajectories, and generates the necessary inputs
    for the MetaGridPreprocessor by simulating the motion relative to the
    start of each chunk.

    Canonical pipeline:
      - load & smooth at 100 Hz using load_and_process_trajectory
      - resample whole trajectories to 1000 Hz using resample_trajectory
      - chunks defined in 1000 Hz index space
      - targets computed from smoothed 100 Hz trajectory over last 0.1 s of the chunk
    """
    def __init__(self,
                 shared_images: torch.Tensor,
                 texture_paths,
                 trajectory_paths,
                 trajectory_len: int = 100,   # chunk length at 1000 Hz (0.1 s)
                 sample_rate: int = 100,      # base Hz, canonical 100 Hz
                 camera_fov: float = 90.0,
                 window_width: int = 128,
                 height_range: tuple = (0.15, 0.15),
                 grid_displacement: float = 0.48,
                 camera_displacement: float = 0.016,
                 one_grid_only: bool = False,
                 debug_mode: bool = False,
                 ma_window: int = 101,
                 gt_type: str = 'mean', # 'mid' or 'last'
                 legacy_mode : bool = False,
                 max_vx_mps: float = 5.0,
                 flip_augment : bool = True,
                 # legacy args kept for backwards compatibility but unused:
                 subsample_len: int = 10,
                 output_seq_len: int = 100,
                 target_window_seq: float = 0.1,
                 ):
        super().__init__()

        # --- Store parameters ---
        self.shared_images = shared_images
        self.texture_paths = list(texture_paths)
        self.trajectory_paths = trajectory_paths
        self.trajectory_len_1000 = trajectory_len
        self.win = window_width
        self.h = window_width
        self.height_range = height_range
        self.camera_fov_rad = math.radians(camera_fov)
        self.sample_rate = sample_rate           # 100 Hz canonical
        self.debug_mode = debug_mode
        self.max_vx_mps = max_vx_mps
        self.max_wz_radps = 1.0
        self.ma_window = max(1, int(ma_window))
        self.dt_100 = 1.0 / float(self.sample_rate)
        self.gt_type = gt_type
        self.legacy_mode = legacy_mode
        self.camera_displacement = camera_displacement
        
        if self.legacy_mode:
            self.stride_1000 = 500 #0.5 s stride
        else:
            self.stride_1000 = 50 #0.05 s stride

        # --- Load 100 Hz trajectories and build 1000 Hz versions ---
        self.loaded_trajectories_100: Dict[int, Trajectory] = {}
        self.loaded_trajectories_1000: Dict[int, Trajectory] = {}
        self.trajectory_pointers: List[Tuple[int, int]] = []  # (traj_id, start_idx_1000)


        print(f"Found {len(trajectory_paths)} trajectory folders. Loading, smoothing, resampling to 1000 Hz ...")

        for i, path in enumerate(trajectory_paths):
            if not path.is_dir():
                continue
            try:
                traj_100 = load_and_process_trajectory(path, Hz=self.sample_rate, ma_window=self.ma_window)
            except Exception as e:
                print(f"Warning: Skipping {path.name} due to a loading error: {e}")
                continue

            # Filter by max |vx| at 100 Hz
            if torch.abs(traj_100.vel[:, 0]).max() >= self.max_vx_mps and not self.legacy_mode:
                print(f"Skipping {path.name}, max |vx| exceeds limit.")
                continue

            self.loaded_trajectories_100[i] = traj_100
            traj_1000 = resample_trajectory(traj_100, Hz_out=1000)
            self.loaded_trajectories_1000[i] = traj_1000

            T_1000 = traj_1000.pos.shape[0]
            for start_idx_1000 in range(0, T_1000 - self.trajectory_len_1000 + 1, self.stride_1000):
                end_idx_1000 = start_idx_1000 + self.trajectory_len_1000
                if self.legacy_mode and torch.abs(traj_1000.vel[start_idx_1000:end_idx_1000, 0]).max() <= self.max_vx_mps:
                    self.trajectory_pointers.append((i, start_idx_1000))
                elif not self.legacy_mode:
                    self.trajectory_pointers.append((i, start_idx_1000))

        print(f"Found {len(self.loaded_trajectories_100)} trajectories, created {len(self.trajectory_pointers)} sub-trajectory samples.")
        total_trajectories = len(self.trajectory_pointers)
        train_split = int(0.7 * total_trajectories)
        val_split = int(0.8 * total_trajectories)

        print(f"Using {len(self.trajectory_pointers)} 1s trajectories.")

        # --- Pano size bound based on max speed and 0.1 s at 100 Hz ---
        min_height_m, _ = self.height_range
        max_ppm = self.win / (2 * min_height_m * math.tan(self.camera_fov_rad / 2))
        max_vx_pxps = self.max_vx_mps * max_ppm

        if self.trajectory_len_1000 == 100:
            t_total = 0.1
        elif self.trajectory_len_1000 == 1000:
            t_total = 1  # 0.1 s window
        max_travel_dist_px = max_vx_pxps * t_total
        self.pano_width = math.ceil(2 * (max_travel_dist_px + self.win) + 100)
        self.pano_height = math.ceil(2 * (max_travel_dist_px + self.h) + 100)
        print(f"Dynamic Pano size set to: {self.pano_width} x {self.pano_height}")

        # Geometry
        if not one_grid_only:
            meta_coords = torch.tensor([[-0.05, -0.5], [0.05, -0.5], [-0.05, 0.5], [0.05, 0.5]])  # BL, BR, TL, TR
            self.num_cams = 16
            self.num_grids = 4
        else:
            meta_coords = torch.tensor([[0.0, 0.0]])
            self.num_cams = 4
            self.num_grids = 1

        self.meta_grid_offsets_m = meta_coords * grid_displacement
        intra_coords = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
        self.intra_grid_offsets_m = intra_coords * self.camera_displacement

        self.flip_augment = flip_augment

    def __len__(self):
        return len(self.trajectory_pointers)

    def __getitem__(self, idx):
        traj_id, start_idx_1000 = self.trajectory_pointers[idx]
        traj_100 = self.loaded_trajectories_100[traj_id]
        traj_1000 = self.loaded_trajectories_1000[traj_id]

        end_idx_1000 = start_idx_1000 + self.trajectory_len_1000

        # --- 1000 Hz slice for simulation (pano + offsets) ---
        pos_chunk_m_1000 = traj_1000.pos[start_idx_1000:end_idx_1000, :2].to(torch.float64)
        ori_chunk_rad_1000 = traj_1000.ori[start_idx_1000:end_idx_1000].to(torch.float64)
        
        flip_dir = random.random() < 0.5
        if flip_dir and self.flip_augment:
            pos_chunk_m_1000 = pos_chunk_m_1000.flip(0)
            ori_chunk_rad_1000 = ori_chunk_rad_1000.flip(0)
            
            # Normalize angles to [-π, π]
            ori_chunk_rad_1000[:, 2] = torch.atan2(
                torch.sin(ori_chunk_rad_1000[:, 2]),
                torch.cos(ori_chunk_rad_1000[:, 2])
            )

        thetas_abs = ori_chunk_rad_1000[:, 2]
        cos_t, sin_t = torch.cos(thetas_abs), torch.sin(thetas_abs)

        current_height = random.uniform(*self.height_range)
        texture_ppm = self.win / (2 * current_height * math.tan(self.camera_fov_rad / 2))

        robot_path_m = pos_chunk_m_1000

        static_cam_offsets_m = torch.zeros(self.num_cams, 2, dtype=torch.float64)
        meta_offsets_m_64 = self.meta_grid_offsets_m.to(torch.float64)
        intra_offsets_m_64 = self.intra_grid_offsets_m.to(torch.float64)
        for grid_idx in range(self.num_grids):
            for cam_in_grid_idx in range(4):
                cam_global_idx = grid_idx * 4 + cam_in_grid_idx
                static_cam_offsets_m[cam_global_idx] = meta_offsets_m_64[grid_idx] + intra_offsets_m_64[cam_in_grid_idx]

        all_offsets_m = torch.zeros(self.num_cams, self.trajectory_len_1000, 2, dtype=torch.float64)
        for i_cam in range(self.num_cams):
            r0_x, r0_y = static_cam_offsets_m[i_cam, 0], static_cam_offsets_m[i_cam, 1]
            rotated_offset_x_m = r0_x * cos_t - r0_y * sin_t
            rotated_offset_y_m = r0_x * sin_t + r0_y * cos_t
            all_offsets_m[i_cam, :, 0] = robot_path_m[:, 0] + rotated_offset_x_m
            all_offsets_m[i_cam, :, 1] = robot_path_m[:, 1] + rotated_offset_y_m

        if self.camera_displacement > 0:
            for g in range(self.num_grids):
                blk = all_offsets_m[g*4:(g+1)*4, 0]
                assert torch.unique(blk, dim=0).size(0) == 4, f"path collapse at grid {g} (post-rotation)"

        min_x_m, max_x_m = all_offsets_m[..., 0].min(), all_offsets_m[..., 0].max()
        min_y_m, max_y_m = all_offsets_m[..., 1].min(), all_offsets_m[..., 1].max()

        path_width_px = (max_x_m - min_x_m) * texture_ppm
        path_height_px = (max_y_m - min_y_m) * texture_ppm

        pano_width = math.ceil(path_width_px + 2 * self.win + 50)
        pano_height = math.ceil(path_height_px + 2 * self.h + 50)

        img_idx = random.randint(0, len(self.texture_paths) - 1)
        tex = self.shared_images[img_idx].to(dtype=torch.float32).div_(255.0)
        H_tex, W_tex = tex.shape
        x0 = random.randint(0, max(0, W_tex - pano_width))
        y0 = random.randint(0, max(0, H_tex - pano_height))
        pano = tex[y0:y0+pano_height, x0:x0+pano_width]

        # world -> pano pixels
        all_offsets_relative_m = all_offsets_m - torch.tensor([min_x_m, min_y_m], dtype=torch.float64)
        all_offsets_pixels = all_offsets_relative_m * texture_ppm
        all_offsets_pixels[..., 1] = (max_y_m - min_y_m) * texture_ppm - all_offsets_pixels[..., 1]
        margin_offset = torch.tensor([self.win, self.h], dtype=torch.float64)
        all_offsets_final = (all_offsets_pixels + margin_offset).to(torch.float32)
        thetas = thetas_abs.to(torch.float32)

        # --- Canonical target from 100 Hz trajectory over last 0.1 s of this chunk ---
        vx_100 = traj_100.vel[:, 0]
        wz_100 = traj_100.gyro[:, 2]

        # Time of last frame in chunk at 1000 Hz
        t_end = (end_idx_1000 - 1) / 1000.0
        t_start = max(0.0, t_end - 0.1)

        # Map time window [t_start, t_end] to 100 Hz indices
        idx_start_100 = int(round(t_start * self.sample_rate))
        idx_end_100 = int(round(t_end * self.sample_rate)) + 1  # +1 for slicing

        idx_start_100 = max(0, idx_start_100)
        idx_end_100 = min(vx_100.shape[0], idx_end_100)

        vx_slice = vx_100[idx_start_100:idx_end_100]
        wz_slice = wz_100[idx_start_100:idx_end_100]
        
        if flip_dir and self.flip_augment:
            # Reverse velocity sequence and flip sign
            vx_slice = -vx_slice.flip(0)
            wz_slice = wz_slice.flip(0)

        if self.gt_type == 'mean':
            vx_target = vx_slice.mean()
            wz_target = wz_slice.mean()
        elif self.gt_type == 'last':
            vx_target = vx_slice[-1]
            wz_target = wz_slice[-1]
        elif self.gt_type == 'mid':
            mid_idx = vx_slice.shape[0] // 2
            vx_target = vx_slice[mid_idx]
            wz_target = wz_slice[mid_idx]

        target = torch.tensor([vx_target, wz_target, current_height], dtype=torch.float32)

        if self.debug_mode:
            target_unnorm = torch.tensor([vx_target, wz_target, current_height], dtype=torch.float32)
            target_seq = torch.stack([vx_slice, wz_slice], dim=1)
            return pano, all_offsets_final, -thetas, target, target_seq, target_unnorm
        else:
            return pano, all_offsets_final, -thetas, target


class MetaGridPreprocessor(nn.Module):
    def __init__(self, seq_len, h, win, num_cams=16):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.win = win
        self.num_cams = num_cams

    @staticmethod
    def make_grid(center_offsets: torch.Tensor, H: int, W: int, H_p: int, W_p: int, 
                 thetas: torch.Tensor = None) -> torch.Tensor:
        """Creates a sampling grid for F.grid_sample with translation and rotation."""
        N = center_offsets.shape[0]
        device = center_offsets.device

        # Create base meshgrid for the camera window, centered at (0,0)
        ys_win, xs_win = torch.meshgrid(
            torch.linspace(- (H - 1) / 2.0, (H - 1) / 2.0, H, device=device, dtype=torch.float32),
            torch.linspace(- (W - 1) / 2.0, (W - 1) / 2.0, W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        if thetas is not None:
            # Recreate the same rotation matrix
            cos_t = torch.cos(thetas).view(N, 1, 1)
            sin_t = torch.sin(thetas).view(N, 1, 1)
            
            # Rotate the meshgrid points
            xs_rot = xs_win.unsqueeze(0) * cos_t - ys_win.unsqueeze(0) * sin_t
            ys_rot = xs_win.unsqueeze(0) * sin_t + ys_win.unsqueeze(0) * cos_t
        else:
            xs_rot = xs_win.unsqueeze(0)
            ys_rot = ys_win.unsqueeze(0)
            
        # Add center offsets to translate the grid
        grid_x_abs = xs_rot + center_offsets.view(N, 1, 1, 2)[..., 0]
        grid_y_abs = ys_rot + center_offsets.view(N, 1, 1, 2)[..., 1]

        # Normalize to [-1, 1] for grid_sample
        grid_x_norm = 2 * grid_x_abs / (W_p - 1) - 1
        grid_y_norm = 2 * grid_y_abs / (H_p - 1) - 1
        
        grid = torch.stack((grid_x_norm, grid_y_norm), dim=3)
        return grid
    
    @torch.no_grad()
    def forward(self, panos: torch.Tensor, all_offsets: torch.Tensor, thetas=None) -> torch.Tensor:
        B, H_p, W_p = panos.shape
        
        # Offsets are the center points for each camera at each time step
        # Thetas are the rotations angles at each time step - they are the same for all the cameras
        if thetas is not None:
            thetas_expanded = thetas.unsqueeze(1).expand(-1, self.num_cams, -1)
            thetas_flat = thetas_expanded.reshape(-1)
        else:
            thetas_flat = None

        # Flatten offsets for batch processing
        center_offsets_flat = all_offsets.view(-1, 2)
        N_total = center_offsets_flat.shape[0]
        
        # Prepare panos
        p = panos.view(B, 1, 1, H_p, W_p).expand(-1, self.num_cams, self.seq_len, -1, -1)
        p = p.reshape(N_total, 1, H_p, W_p)

        # Create grid and warp with rotation
        grid = self.make_grid(
            center_offsets_flat, 
            H=self.h, 
            W=self.win, 
            H_p=H_p, 
            W_p=W_p, 
            thetas=thetas_flat
        )
        warped = F.grid_sample(p, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Reshape to final output format
        final_shape = (B, self.num_cams, self.seq_len, 1, self.h, self.win)
        return warped.view(final_shape)

###############################
## FOR TESTING ON FULL PATHS ## 
###############################

class TartanFullTrajectoryDataset(Dataset):

    def __init__(self,
                 shared_images: torch.Tensor,
                 texture_paths,
                 trajectory_root_dir: str,
                 sample_rate_in_hz: int = 100,    # original Hz in Tartan Dataset
                 resample_to_hz: int = 1000,       
                 camera_fov: float = 70.0,
                 window_width: int = 128,
                 height_range: tuple = (0.15, 0.65),
                 grid_displacement: float = 0.048,
                 camera_displacement: float = 0.019,
                 mode: str = 'train',
                 one_grid_only: bool = True,
                 debug_mode: bool = False,
                 ma_window: int = 101,
                 max_vx_mps: float = 5.5):              
        super().__init__()
        self.shared_images = shared_images
        self.texture_paths = list(texture_paths)
        self.camera_fov_rad = math.radians(camera_fov)
        self.win = window_width
        self.h = window_width
        self.height_range = height_range
        self.sample_rate_in_hz = sample_rate_in_hz
        self.resample_to_hz = resample_to_hz
        self.debug_mode = debug_mode
        self.safety_margin_px = 64
        self.max_vx_mps = max_vx_mps
        self.max_wz_radps = 1.0
        self.ma_window = max(1, int(ma_window))
        self.dt_in = 1.0 / float(self.sample_rate_in_hz)

        # Geometry
        if not one_grid_only:
            meta_coords = torch.tensor([[-0.05, -0.5], [0.05, -0.5], [-0.05, 0.5], [0.05, 0.5]])  # BL, BR, TL, TR
            self.num_cams = 16
            self.num_grids = 4
        else:
            meta_coords = torch.tensor([[+0.08, 0.0]])
            self.num_cams = 4
            self.num_grids = 1
        self.meta_grid_offsets_m = meta_coords * grid_displacement
        intra_coords = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
        self.intra_grid_offsets_m = intra_coords * camera_displacement

        # Load all trajectories (smooth + re-derive) -> because there is noise in the paths 
        self.trajectory_pointers: List[int] = []
        self.loaded_trajectories: Dict[int, Trajectory] = {}

        traj_paths = sorted(list(Path(trajectory_root_dir).glob('*/*/P*')))
        if not traj_paths:
            print(f"Warning: No trajectory folders found in {trajectory_root_dir} with pattern '*/*/P*'")
            return
        print(f"Found {len(traj_paths)} trajectory folders. Loading full trajectories ...")

        for i, path in enumerate(traj_paths):
            if not path.is_dir():
                continue
            try:
                traj = load_and_process_trajectory(path, Hz=self.sample_rate_in_hz, ma_window=self.ma_window)
            except Exception as e:
                print(f"Warning: Skipping {path.name} due to load error: {e}")
                continue

            # filter by max |vx|
            vx_abs_max = float(torch.abs(traj.vel[:, 0]).max())
            if vx_abs_max >= self.max_vx_mps:
                print(f"Skipping {path.name}, max |vx| {vx_abs_max:.2f} m/s exceeds limit.")
                continue

            self.loaded_trajectories[i] = traj
            self.trajectory_pointers.append(i)

        total = len(self.trajectory_pointers)
        train_split = int(0.7 * total)
        val_split = int(0.8 * total)
        if mode == 'train':
            self.trajectory_pointers = self.trajectory_pointers[:train_split]
        elif mode == 'val':
            self.trajectory_pointers = self.trajectory_pointers[train_split:val_split]
        elif mode == 'test':
            self.trajectory_pointers = self.trajectory_pointers[val_split:]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        print(f"Mode: {mode}. Using {len(self.trajectory_pointers)} full trajectories out of {total}.")
    

    def _compute_camera_paths_and_pano(self, pos_m, ori_rad):
        """
        Compute per-camera absolute paths in meters, then determine needed pano size and stitch.
        Returns:
          - pano [H_p, W_p] float32
          - all_offsets_final [num_cams, T, 2] float32 pixel coords within pano
          - thetas [T] float32
          - pixels-per-meter scalar used
        """
        T = pos_m.shape[0]
        pos_m_64 = pos_m[:, :2].to(torch.float64)
        thetas_abs = ori_rad[:, 2].to(torch.float64)
        cos_t, sin_t = torch.cos(thetas_abs), torch.sin(thetas_abs)

        # Sensor geometry
        static_cam_offsets_m = torch.zeros(self.num_cams, 2, dtype=torch.float64)
        meta_offsets_m_64 = self.meta_grid_offsets_m.to(torch.float64)
        intra_offsets_m_64 = self.intra_grid_offsets_m.to(torch.float64)
        for grid_idx in range(self.num_grids):
            for cam_in_grid_idx in range(4):
                cam_global_idx = grid_idx * 4 + cam_in_grid_idx
                static_cam_offsets_m[cam_global_idx] = meta_offsets_m_64[grid_idx] + intra_offsets_m_64[cam_in_grid_idx]

        # Full sequence offsets in meters: [num_cams, T, 2]
        all_offsets_m = torch.zeros(self.num_cams, T, 2, dtype=torch.float64)
        for i in range(self.num_cams):
            r0_x, r0_y = static_cam_offsets_m[i, 0], static_cam_offsets_m[i, 1]
            rx = r0_x * cos_t - r0_y * sin_t
            ry = r0_x * sin_t + r0_y * cos_t
            all_offsets_m[i, :, 0] = pos_m_64[:, 0] + rx
            all_offsets_m[i, :, 1] = pos_m_64[:, 1] + ry

        # Bounds in meters over all cameras and time
        min_x_m, max_x_m = all_offsets_m[..., 0].min(), all_offsets_m[..., 0].max()
        min_y_m, max_y_m = all_offsets_m[..., 1].min(), all_offsets_m[..., 1].max()

        # Single height (ppm) for the whole trajectory
        current_height = random.uniform(*self.height_range)
        ppm = self.win / (2 * current_height * math.tan(self.camera_fov_rad / 2))

        # Pano size with half-window margin + safety margin on both sides
        safety = self.safety_margin_px
        path_w_px = float((max_x_m - min_x_m) * ppm)
        path_h_px = float((max_y_m - min_y_m) * ppm)
        pano_w = math.ceil(path_w_px + 2 * (self.win + safety))
        pano_h = math.ceil(path_h_px + 2 * (self.h + safety))

        pano = build_tiled_pano(self.shared_images, pano_h, pano_w, randomize_tiles=True)

        # Transform world meters -> pano pixels
        all_offsets_rel_m = all_offsets_m - torch.tensor([min_x_m, min_y_m], dtype=torch.float64)
        all_offsets_px = all_offsets_rel_m * ppm
        # Flip y to image space
        all_offsets_px[..., 1] = (max_y_m - min_y_m) * ppm - all_offsets_px[..., 1]
        margin = torch.tensor([self.win + safety, self.h + safety], dtype=torch.float64)
        all_offsets_final = (all_offsets_px + margin).to(torch.float32)

        thetas = thetas_abs.to(torch.float32)
        return pano, all_offsets_final, thetas, float(ppm)

    def __len__(self):
        return len(self.trajectory_pointers)
    
    def __getitem__(self, idx):
        traj_id = self.trajectory_pointers[idx]
        traj = self.loaded_trajectories[traj_id]

        if self.resample_to_hz is not None and self.resample_to_hz != traj.Hz:
            traj_used = resample_trajectory(traj, Hz_out=self.resample_to_hz)
        else:
            traj_used = traj

        pos = traj_used.pos
        ori = traj_used.ori
        vel = traj_used.vel
        gyro = traj_used.gyro

        pano, all_offsets_final, thetas_abs, ppm = self._compute_camera_paths_and_pano(pos, ori)

        target = torch.stack([vel[:, 0], gyro[:, 2]], dim=1).to(torch.float32)

        return {
            'pano': pano,
            'offsets': all_offsets_final,
            'thetas': -thetas_abs,
            'target': target,
            'ppm': ppm,
            'path': str(self.loaded_trajectories[traj_id].pos),
            'pos': pos,
            'ori': ori,
            'vel': vel,
            'gyro': gyro
        }


class MetaGridStreamer(nn.Module):
    """
    Streaming variant of MetaGridPreprocessor. Warps in chunks to bound memory.
    """
    def __init__(self, h, win, num_cams=16):
        super().__init__()

        self.h = h
        self.win = win
        self.num_cams = num_cams

    @torch.no_grad()
    def stream(self, pano: torch.Tensor, all_offsets: torch.Tensor, thetas: torch.Tensor,
               chunk_len: int = 100, device: str = 'cpu'):
        """
        Yields warped chunks:
          - pano: [H_p, W_p] float32 in [0,1]
          - all_offsets: [num_cams, T, 2] float32 pixel coords
          - thetas: [T] float32 (rotation per-frame, radians)
        Yields tensors of shape [num_cams, chunk, 1, h, win].
        """
        pano = pano.to(device, non_blocking=True)
        thetas = thetas.to(device, non_blocking=True)
        all_offsets = all_offsets.to(device, non_blocking=True)

        B = 1  # single trajectory
        H_p, W_p = pano.shape
        total_T = all_offsets.shape[1]
        assert thetas.shape[0] == total_T, "thetas and offsets length mismatch"

        # Pre-expand pano for grid_sample shape expectations per chunk
        def warp_chunk(t0, t1):
            # centers: [num_cams, chunk, 2] -> [N_total, 2]
            centers = all_offsets[:, t0:t1, :].contiguous()
            chunk = centers.shape[1]
            centers_flat = centers.view(-1, 2)

            # thetas: [chunk] -> expanded to [num_cams * chunk]
            th = thetas[t0:t1]
            th_exp = th.view(1, chunk).expand(self.num_cams, chunk).reshape(-1)

            # Prepare pano per-grid_sample
            p = pano.view(B, 1, 1, H_p, W_p).expand(B, self.num_cams, chunk, -1, -1)
            p = p.reshape(-1, 1, H_p, W_p)  # [N_total, 1, H_p, W_p]

            # Reuse existing static grid builder for consistency
            grid = MetaGridPreprocessor.make_grid(
                center_offsets=centers_flat,
                H=self.h, W=self.win,
                H_p=H_p, W_p=W_p,
                thetas=th_exp
            )
            warped = F.grid_sample(p, grid, mode='bilinear', padding_mode='border', align_corners=True)
            return warped.view(self.num_cams, chunk, 1, self.h, self.win)

        for t0 in range(0, total_T, chunk_len):
            t1 = min(t0 + chunk_len, total_T)
            yield warp_chunk(t0, t1)