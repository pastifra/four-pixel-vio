import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import sklearn.metrics
import logging
module_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
from pathlib import Path
import pprint
from typing import Dict
from utils.constants import LOG_PATH, MODEL_PATH
import utils.utils as utils
import yaml
from itertools import product
from simulator.cam_models import GaborCam, FreeformCam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator.views_simulator import MetaGridPreprocessor, matador_paths, preload_to_tensor, TartanTrajectoryDataset_1000Hz_V2, dynamic_pano_collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Set random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_extended_metrics(writer, actuals, preds, stds, prefix, epoch, validation=False):
    """Calculates and logs a comprehensive set of metrics for a given prediction task."""
    
    metrics_dict = {}
    
    # 1. Standard Metrics
    metrics_dict[f'{prefix}/rmse'] = np.sqrt(sklearn.metrics.mean_squared_error(actuals, preds))
    if validation==False:
        metrics_dict[f'{prefix}/mae'] = sklearn.metrics.mean_absolute_error(actuals, preds)
        metrics_dict[f'{prefix}/r2'] = sklearn.metrics.r2_score(actuals, preds)

    # 2. 95% Confidence Interval Coverage
    z_score = 1.96  # for 95% confidence
    lower_bounds = preds - z_score * stds
    upper_bounds = preds + z_score * stds
    is_within_interval = (actuals >= lower_bounds) & (actuals <= upper_bounds)
    coverage = np.mean(is_within_interval) * 100
    if validation==False:
        metrics_dict[f'{prefix}/coverage_95'] = coverage

    # 3. Percentile-Based Filtering and Plotting
    confidence_percentile = 95
    std_threshold = np.percentile(stds, confidence_percentile)
    mask = stds < std_threshold
    if np.any(mask):
        metrics_dict[f'{prefix}/std_threshold_95'] = std_threshold
        metrics_dict[f'{prefix}/rmse_top{confidence_percentile}'] = np.sqrt(sklearn.metrics.mean_squared_error(actuals[mask], preds[mask]))
        
        # 4. Scatter plot for the filtered (top 95%) data ---
        fig_scatter_filtered = plt.figure(figsize=(10,8))
        sc_filtered = plt.scatter(actuals[mask], preds[mask], c=stds[mask], cmap='viridis', alpha=0.7)
        plt.colorbar(sc_filtered, label='Predicted Std Dev')
        # Use the full range for the ideal line for better comparison
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', label='Ideal')
        plt.plot([], [], ' ', label=f'Std Threshold < {std_threshold:.4f}')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title(f'{prefix.upper()} Predictions (Top {confidence_percentile}% Confidence)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        writer.add_figure(f'{prefix}/confidence_scatter_top{confidence_percentile}', fig_scatter_filtered, epoch)
        plt.close(fig_scatter_filtered)

    # 5. Confidence-Colored Scatter Plot (for ALL data)
    fig_scatter = plt.figure(figsize=(10, 8))
    sc = plt.scatter(actuals, preds, c=stds, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Predicted Std Dev')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', label='Ideal')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(f'{prefix.upper()} Predictions vs. Ground Truth (All Data)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    writer.add_figure(f'{prefix}/confidence_scatter', fig_scatter, epoch)
    plt.close(fig_scatter)

    return metrics_dict

def log_and_save_masks(writer, model, epoch, exp_model_dir):
    """
    Pulls the current masks from the model, formats them into a seamless 2x2 grid,
    saves the image to disk, and pushes it to TensorBoard.
    """
    # 1. Get the current masks
    masks = model.visualize_overlap_masks()
    
    # 2. Set up the seamless 2x2 figure
    layout = [2, 3, 0, 1]
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), gridspec_kw={'wspace': 0, 'hspace': 0}) 
    axs = axs.flatten()
    
    for plot_idx, view_idx in enumerate(layout):
        mask = masks[view_idx].permute(1, 2, 0).cpu().numpy()
        axs[plot_idx].imshow(mask)
        axs[plot_idx].axis('off')
        
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # 3. Save to disk (e.g., inside your experiment's model directory)
    # This creates files like "masks_epoch_10.png"
    save_path = os.path.join(exp_model_dir, f"masks_epoch_{epoch}.png")
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    # 4. Push to TensorBoard
    writer.add_figure('visuals/masks', fig, epoch)
    plt.close(fig)

def run_experiment(exp_config: Dict, exp_name: str):
    exp_log_dir = Path(LOG_PATH) / exp_name
    exp_model_dir = Path(MODEL_PATH) / exp_name
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    exp_model_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting experiment: {exp_name}")
    logging.info(f"Configuration:\n{pprint.pformat(exp_config)}")
    ################# LOAD MODEL and DATASET ##########################
    img_size = tuple(exp_config.get("img_size", (128, 128)))
    displacement = exp_config.get("displacement", 0.016)
    fov = exp_config.get("fov", 90.0)
    height = tuple(exp_config.get("height", (0.15, 0.15)))
    epochs = exp_config.get("epochs", 100)
    base_lr = exp_config.get("base_lr", 1e-4)
    batch_size = exp_config.get("batch_size", 8)
    switch_epoch = exp_config.get("switch_epoch", 10) # Epoch to switch from MSE to NLL loss
    gpu_n = exp_config.get("gpu_n", None)
    texture_path = exp_config.get("texture_path", "/mnt/minVO/mincam/data/sensor_gray")
    label_path = exp_config.get("label_path", "/mnt/minVO/mincam/data/label")
    context_path = exp_config.get("context_path", None)
    seed = exp_config.get("seed", 42)
    sample_rate = exp_config.get("sample_rate", 100)  # Hz
    tartan_dir = exp_config.get("tartan_dir", "/mnt/minVO/Tartan")
    trajectory_len = exp_config.get("trajectory_len", 1000)
    grid_displacement = exp_config.get("grid_displacement", 0.048)
    target_batch_size = 32
    sensor_hertz = exp_config.get("sensor_hertz", 1000)
    gt_type = exp_config.get("gt_type", 'mean')  # 'mid' or 'last'
    legacy_mode = exp_config.get("legacy_mode", False)
    max_vx_mps = exp_config.get("max_vx_mps", 5.0)
    freeform = exp_config.get("freeform", False)
    if batch_size < target_batch_size:
        accumulation_steps = target_batch_size // batch_size
        logging.info(f"Using gradient accumulation with {accumulation_steps} accumulation steps")
    else:
        accumulation_steps = 1

    if not freeform:
        #Quadrature pair is computed w.r.t to center of overlap region
        #By default, non overlapping regions are blackened out
        #Phase is not a learnable parameter
        logging.info("Using Gabors")
        model = GaborCam(
            img_size=img_size,
            seq_len=trajectory_len,
            realistic_sensor=exp_config.get("realistic_sensor", True),
            simulate_pd_area_blur=exp_config.get("simulate_pd_area_blur", True),
            simulate_solid_angle=exp_config.get("simulate_solid_angle", False),
            mask_min_value=exp_config.get("mask_min_value", 0.0),
            mask_max_value=exp_config.get("mask_max_value", 1.0),
            simulate_directivity=exp_config.get("simulate_directivity", True),
            mask_init_method=exp_config.get("mask_init_method", "gabors"),
            sensor_gain=exp_config.get("mincam_sensor_gain"),
            sensor_n_bits=exp_config.get("mincam_sensor_n_bits"),
            sensor_saturation_val=exp_config.get("mincam_sensor_saturation_val"),
            read_noise_std=exp_config.get("mincam_read_noise_std"),
            model_vert_fov=fov,
            model_horiz_fov=fov,
            n_gabors=1,
            init_freqs=exp_config.get("init_freqs", [6.0]),
            batch_size=batch_size,
            height=(height[0]+height[1])/2,
            camera_displacement=displacement,
            differential_overlap=exp_config.get("differential_overlap", True), #BLACKEN OUT ONLY Positive/negative pair non overlapping
            full_masks=exp_config.get("full_masks", False) ##NO BLACKENING OUT
            )
    else:    
        logging.info("Using Freeform Pixels")
        model = FreeformCam(
            img_size=img_size,
            seq_len=trajectory_len,
            realistic_sensor=exp_config.get("realistic_sensor", True),
            simulate_pd_area_blur=exp_config.get("simulate_pd_area_blur", True),
            simulate_solid_angle=exp_config.get("simulate_solid_angle", True),
            mask_min_value=exp_config.get("mask_min_value", 0.0),
            mask_max_value=exp_config.get("mask_max_value", 1.0),
            simulate_directivity=exp_config.get("simulate_directivity", True),
            mask_init_method=exp_config.get("mask_init_method", "flat"),
            sensor_gain=exp_config.get("mincam_sensor_gain"),
            sensor_n_bits=exp_config.get("mincam_sensor_n_bits"),
            sensor_saturation_val=exp_config.get("mincam_sensor_saturation_val"),
            read_noise_std=exp_config.get("mincam_read_noise_std"),
            model_vert_fov=fov,
            model_horiz_fov=fov
        )

    #device = torch.device(("cuda:%d" % gpu_n) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Seed for reproducibility
    set_seed(seed)

    # MATADOR IMAGES
    logging.info(f"Model: {model}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Using device: {device}")
    train_paths, val_paths, test_paths = matador_paths(Path(texture_path), Path(label_path), context_folder=Path(context_path) if context_path is not None else None, random_seed=seed, split_ratios=(0.7, 0.1, 0.2))
    logging.info('Loading train dataset...')
    train_img_tensor, train_paths_loaded = preload_to_tensor(train_paths, n_threads=8)
    logging.info('Loading val dataset...')
    val_img_tensor, val_paths_loaded = preload_to_tensor(val_paths, n_threads=8)

    # TARTAN TRAJECTORIES
    trajectory_paths = sorted(list(Path(tartan_dir).glob('*/*/P*')))
    random.shuffle(trajectory_paths)
    total_tartan_paths = len(trajectory_paths)
    train_end = int(total_tartan_paths * 0.7)
    val_end = int(total_tartan_paths * 0.8)
    train_tartan_paths = trajectory_paths[:train_end]
    val_tartan_paths = trajectory_paths[train_end:val_end]
    test_tartan_paths = trajectory_paths[val_end:]
    print(f"Tartan Paths: {total_tartan_paths} | Train: {len(train_tartan_paths)} | Val: {len(val_tartan_paths)} | Test: {len(test_tartan_paths)}")


    train_dataset = TartanTrajectoryDataset_1000Hz_V2(
        shared_images=train_img_tensor,
        texture_paths=train_paths_loaded,
        trajectory_paths=train_tartan_paths,
        trajectory_len=trajectory_len,
        window_width=img_size[1],
        height_range=height,
        camera_displacement=displacement,
        grid_displacement=grid_displacement,
        one_grid_only=  True,
        gt_type=gt_type,
        legacy_mode=legacy_mode,
        max_vx_mps=max_vx_mps,
        camera_fov=fov,
        flip_augment=exp_config.get('flip_augment', False)
        )
    val_dataset = TartanTrajectoryDataset_1000Hz_V2(
        shared_images=val_img_tensor,
        texture_paths=val_paths_loaded,
        trajectory_paths=val_tartan_paths,
        trajectory_len=trajectory_len,
        window_width=img_size[1],
        height_range=height,
        camera_displacement=displacement,
        grid_displacement=grid_displacement,
        one_grid_only=  True,
        gt_type=gt_type,
        legacy_mode=legacy_mode,
        max_vx_mps=max_vx_mps,
        camera_fov=fov,
        flip_augment=exp_config.get('flip_augment', False)
        )

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              pin_memory=True, 
                              num_workers=8, 
                              persistent_workers=True,
                              collate_fn=dynamic_pano_collate_fn)

    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size,
                            shuffle= False,
                            pin_memory=True,
                            num_workers=8,
                            persistent_workers=True,
                            collate_fn=dynamic_pano_collate_fn)
    
    preproc = MetaGridPreprocessor(
                seq_len= trajectory_len,
                h=train_dataset.h, 
                win=train_dataset.win,
                num_cams=4).to(device)
    
    logging.info(f"Dataset loaded. Train size: {len(train_loader.dataset)}, "
                f"Validation size: {len(val_loader.dataset)}")

    writer = SummaryWriter(log_dir=exp_log_dir)

    if not freeform:
        logging.info('Initializing optimizer with fixed Learning Rate')
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay = exp_config.get("weight_decay", 0.0))
    else:
        logging.info('Initializing optimizer with higher masks learning rate')
        optimizer = optim.Adam([
            {'params': model.chirp_network.parameters(), 'lr': 1e-4},
            {'params': [model.masks], 'lr': 1e-4}
        ])


    nll_criterion = nn.GaussianNLLLoss(reduction='none')
    init_criterion = nn.MSELoss()

    if not freeform:
        print('Initial GABOR parameters')
        print(f'Frequency {torch.sigmoid(model.gabor_raw_frequencies)*model.nyquist_limit}')
        print(f'Standard deviation {F.softplus(model.gabor_raw_gaussian_stds)+1e-6}')
        print(f'Amplitude {torch.sigmoid(model.gabor_raw_amplitudes)}')
        print(f'Phase {model.gabor_phases}')

        if exp_config.get("freeze_masks", False):
            mask_params = [
                'gabor_raw_amplitudes',
                'gabor_raw_frequencies',
                'gabor_phases',
                'gabor_raw_gaussian_stds'
            ]

            for name, param in model.named_parameters():
                if name in mask_params:
                    param.requires_grad = False
                    print(f"Frozen: {name}")

    logging.info(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    ################ TRAINING LOOP ##########################
    global_step = 0
    for epoch in range(epochs):     
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (panos, offsets, thetas, targets)  in enumerate(train_loader):
            panos = panos.to(device)
            offsets = offsets.to(device)
            thetas = thetas.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                frames = preproc(panos, offsets, thetas)  # no grad for preproc
            frames = frames.detach()

            pred_speed, pred_log_var  = model(frames) # (batch_size, seq_len, 2)
            pred_speed = pred_speed.squeeze(-1)
            pred_log_var = pred_log_var.squeeze(-1)

            if epoch > switch_epoch:
                speed_loss = nll_criterion(pred_speed, targets[:, 0], torch.exp(pred_log_var)).mean()
                tot_loss = speed_loss

            else:
                speed_loss = init_criterion(pred_speed, targets[:, 0])
                tot_loss = speed_loss

            unscaled_loss = tot_loss.item()

            if accumulation_steps > 1:
                tot_loss = tot_loss / accumulation_steps

            tot_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += unscaled_loss
            writer.add_scalar('train/tot_loss', unscaled_loss, global_step)
            writer.add_scalar('train/speed_loss',speed_loss.item(), global_step)
            global_step += 1
        
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()   

        if not freeform:
            freqs = torch.sigmoid(model.gabor_raw_frequencies) * model.nyquist_limit
            stds = F.softplus(model.gabor_raw_gaussian_stds) + 1e-6
            amps = torch.sigmoid(model.gabor_raw_amplitudes)
            phases = model.gabor_phases
            for i in range(model.n_gabors):
                writer.add_scalar(f'gabor/frequency_{i}', freqs[i].item(), epoch)
                writer.add_scalar(f'gabor/std_dev_{i}', stds[i].item(), epoch)
                writer.add_scalar(f'gabor/amplitude_{i}', amps[i].item(), epoch)
                writer.add_scalar(f'gabor/phase_{i}', phases[i].item(), epoch)

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('train/loss', avg_loss, epoch)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        log_and_save_masks(writer, model, epoch, exp_model_dir)
        
        if epoch % 10 == 0:
            print('Running validation...')
            val_running_loss = 0.0
            n_val_samples   = 0
            predicted_vels = []
            actual_vels = []
            speed_log_var = []

            model.eval()
            with torch.no_grad():
                for panos, offsets, thetas, targets in val_loader:
                    panos = panos.to(device)
                    offsets = offsets.to(device)
                    thetas = thetas.to(device)
                    targets = targets.to(device)
                    with torch.no_grad():
                        frames = preproc(panos, offsets, thetas)  # no grad for preproc
                    frames = frames.detach()

                    pred_speed, pred_log_var  = model(frames) # (batch_size, seq_len, 2)
                    pred_speed = pred_speed.squeeze(-1)
                    pred_log_var = pred_log_var.squeeze(-1)

                    if epoch > switch_epoch:
                        batch_loss = nll_criterion(pred_speed, targets[:, 0], torch.exp(pred_log_var)).mean()
                    else:
                        batch_loss = init_criterion(pred_speed, targets[:, 0])
                    
                    
                    # Accumulate weighted by batch size
                    bsz = panos.size(0)
                    val_running_loss += batch_loss.item() * bsz
                    n_val_samples   += bsz

                    # De‑normalize and collect predictions + ground truth
                    # Vel:
                    vel_pred = (pred_speed).cpu().tolist()
                    vel_true = (targets[:,0]).cpu().tolist()


                    predicted_vels.extend(vel_pred)
                    actual_vels.extend(vel_true)
                    speed_log_var.extend(pred_log_var.cpu().tolist())

                #Metrics
                avg_val_loss = val_running_loss / n_val_samples
                writer.add_scalar('val/loss', avg_val_loss, epoch)

                actual_vels_np = np.array(actual_vels)
                predicted_vels_np = np.array(predicted_vels)
                predicted_stds_vel_np = np.exp(0.5 * np.array(speed_log_var))

                
                vel_metrics = log_extended_metrics(writer, actual_vels_np, predicted_vels_np, predicted_stds_vel_np, 'val/speed', epoch, validation=True)
                for metric_name, metric_value in vel_metrics.items():
                    writer.add_scalar(metric_name, metric_value, epoch)

                print("Val metrics calculated and logged to TensorBoard.")

                checkpoint_path = os.path.join(MODEL_PATH, exp_name, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, checkpoint_path)
                logging.info(f"Checkpoint saved at {checkpoint_path}")
    
    ################ TEST ##########################
    # Release shared tensors and resources for training and validation datasets
    del train_img_tensor
    del val_img_tensor
    del train_dataset
    del val_dataset
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    logging.info("Released shared tensors and resources for training and validation datasets.")
    logging.info('Loading test dataset...')

    test_img_tensor, test_paths_loaded = preload_to_tensor(test_paths, n_threads=8)

    test_dataset = TartanTrajectoryDataset_1000Hz_V2(
        shared_images=test_img_tensor,
        texture_paths=test_paths_loaded,
        trajectory_paths=test_tartan_paths,
        trajectory_len=trajectory_len,
        window_width=img_size[1],
        height_range=height,
        camera_displacement=displacement,
        grid_displacement=grid_displacement,
        one_grid_only= True,
        gt_type=gt_type,
        legacy_mode=legacy_mode,
        max_vx_mps=max_vx_mps,
        camera_fov=fov,
        flip_augment=exp_config.get('flip_augment', False))
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle= False,
                             pin_memory=True, 
                             num_workers=8,
                             persistent_workers=True,
                             collate_fn=dynamic_pano_collate_fn)

    predicted_vels = []
    actual_vels = []
    speed_log_var = []

    model.eval()
    with torch.no_grad():
        for panos, offsets, thetas, targets in test_loader:
            # Move the whole batch at once
            panos   = panos.to(device,   non_blocking=True)
            offsets = offsets.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            thetas  = thetas.to(device,  non_blocking=True)
            with torch.no_grad():
                frames = preproc(panos, offsets, thetas)  # → [B, seq_len, 1, h, w]
            frames = frames.detach()

            pred_speed, pred_log_var  = model(frames) # (batch_size, seq_len, 2)
            pred_speed = pred_speed.squeeze(-1)
            pred_log_var = pred_log_var.squeeze(-1)

            
            # De‑normalize and collect predictions + ground truth
            vel_pred = (pred_speed).cpu().tolist()
            vel_true = (targets[:,0]).cpu().tolist()
            predicted_vels.extend(vel_pred)
            actual_vels.extend(vel_true)
            speed_log_var.extend(pred_log_var.cpu().tolist())


        actual_vels_np = np.array(actual_vels)
        predicted_vels_np = np.array(predicted_vels)
        predicted_stds_vel_np = np.exp(0.5 * np.array(speed_log_var))
        
        all_metrics = {}
        vel_metrics = log_extended_metrics(writer, actual_vels_np, predicted_vels_np, predicted_stds_vel_np, 'test/speed', epoch)
        all_metrics.update(vel_metrics)
        
        hparams_to_log = {k: v for k, v in exp_config.items() if isinstance(v, (int, float, str, bool))}
        writer.add_hparams(hparams_to_log, all_metrics)

        print("Test metrics calculated and logged to TensorBoard.")


    checkpoint_path = os.path.join(MODEL_PATH, exp_name, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")
    del test_img_tensor
    del test_dataset
    del test_loader
    del train_paths, val_paths, test_paths
    del train_paths_loaded, val_paths_loaded, test_paths_loaded
    del predicted_vels, actual_vels, speed_log_var
    del model
    del optimizer
    torch.cuda.empty_cache()
    logging.info("Experiment completed.")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN model for velocity estimation.")
    parser.add_argument('--config', type=str,
                        default='data/configs/demo_cfg.yml',
                        help='Path to the experiment configuration file.')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        D = yaml.safe_load(f)

    global_options = {}
    if "global_options" in D:
        global_options = utils.list_of_dicts_to_dict(D["global_options"])

    for exp_name, exp_config_list in D.items():
        if exp_name == "global_options":
            continue

        exp_config = {**global_options, **utils.list_of_dicts_to_dict(exp_config_list)}
        
        run_experiment(exp_config, exp_name)
