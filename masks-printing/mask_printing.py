import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 

from simulator.cam_models import GaborCam 
import utils.utils as utils
import argparse
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import yaml
import torch

PRINTER = "printer"
PHOTOMASK = "photomask"

# --- 4-PIXEL CONFIGURATION ---
PROTOTYPE_NUM_PIXELS = 4
HANDLE_PARALLAX = False #DO NOT CHANGE
FOCAL_LENGTH = 11.4252e-3 # m, FOV-dependent
SCENE_DEPTH = 7.816 # m

# Physical layout parameters (in mm)
MASK_WIDTH_MM = 16.0
INTER_MASK_DIST_MM = 3.0
IMG_BORDER_WIDTH_MM = 7.0
PIXEL_SPACING_MM = 19.0 # Distance between adjacent photodiode centers

def load_config_yml(config_path: Path):
    with open(config_path, "r") as f:
        D = yaml.safe_load(f)

    global_options = {}
    if "global_options" in D:
        global_options = utils.list_of_dicts_to_dict(D["global_options"])
    
    found_one_exp = False
    
    for exp_name, exp_config_list in D.items():
        if exp_name == "global_options":
            continue
        
        if found_one_exp:
            print("Multiple experiments in yml config file")
            sys.exit(1)

        exp_config = {**global_options, **utils.list_of_dicts_to_dict(exp_config_list)}
        found_one_exp = True
    
    return exp_config

def load_model(args, config_path: Path, checkpoint_path: Path):
    exp_config = load_config_yml(config_path)

    if args.disable_printer_trans_fn:
        mask_min_value = 0
        mask_max_value = 1
    else:
        mask_min_value = exp_config.get("mask_min_value")
        mask_max_value = exp_config.get("mask_max_value")

    model = GaborCam(
        img_size=(755,755), #increasing resolution for the printer
        seq_len=exp_config.get("trajectory_len", 1000),
        realistic_sensor=exp_config.get("realistic_sensor", True),
        simulate_pd_area_blur=False,
        simulate_solid_angle=False,
        mask_min_value=mask_min_value,
        mask_max_value=mask_max_value,
        simulate_directivity=False,
        mask_init_method=exp_config.get("mask_init_method", "gabors"),
        sensor_gain=exp_config.get("mincam_sensor_gain"),
        sensor_n_bits=exp_config.get("mincam_sensor_n_bits"),
        sensor_saturation_val=exp_config.get("mincam_sensor_saturation_val"),
        read_noise_std=exp_config.get("mincam_read_noise_std"),
        model_vert_fov=exp_config.get("fov", 70),
        model_horiz_fov=exp_config.get("fov", 70),
        n_gabors=1, # 1 Gabor generates 4 masks (pos/neg cos/sin)
        init_freqs=exp_config.get("init_freqs", (8.0)),
        batch_size=exp_config.get("batch_size", 1),
        height=exp_config.get("height", 0.06)[0],
        camera_displacement=exp_config.get("displacement", 19e-3),
        differential_overlap=exp_config.get("differential_overlap", False),
        full_masks=exp_config.get("full_masks", False)
    )

    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    exclude_keys = ["X_shifted", "Y_shifted", "X", "Y", "roi_gates", "_view_shifts", "_roi_hw", "mask_blur_kernel", "solid_angle", "diode_directivity"]
    filtered_state_dict = {k: v for k, v in state_dict.items() if k not in exclude_keys}

    model.load_state_dict(filtered_state_dict, strict=False)

    epoch = state_dict.get('epoch', -1)
    print(f"Loaded model from epoch {epoch}")
    model.eval()

    return model

def shift_img(img, shift_r, shift_c, pad_value=0):
    img_shift = np.roll(img, (shift_r, shift_c), axis=(0, 1))

    if shift_r > 0:
        img_shift[:shift_r,:] = pad_value
    elif shift_r < 0:
        img_shift[shift_r:,:] = pad_value

    if shift_c > 0:
        img_shift[:,:shift_c] = pad_value
    elif shift_c < 0:
        img_shift[:,shift_c:] = pad_value
    
    return img_shift

def per_pixel_parallax(scene_depth: float, f: float):
    """
    Calculates parallax offsets for a 2x2 grid.
    """
    pixel_spacing = PIXEL_SPACING_MM * 1e-3 # convert mm to m
    
    # 2x2 grid centered relative to the origin
    displacement = np.asarray([
        [pixel_spacing/2, pixel_spacing/2],
        [pixel_spacing/2, -pixel_spacing/2],
        [-pixel_spacing/2, pixel_spacing/2],
        [-pixel_spacing/2, -pixel_spacing/2]
    ])

    parallax = f * displacement / scene_depth
    return parallax

def pad_mask_fov(mask: np.ndarray, model_fov: int, mask_fov: int):
    """
    Add black padding where mask FOV extends beyond the model FOV 
    """
    if model_fov == mask_fov: 
        return mask

    mask_size = mask.shape
    mask_edge_r = mask_size[0] / 2
    mask_edge_c = mask_size[1] / 2
    mask_fov_rad = np.deg2rad(mask_fov)
    model_fov_rad = np.deg2rad(model_fov)

    pad_edge_r = int(np.round(mask_edge_r * np.sin(mask_fov_rad / 2) / np.sin(model_fov_rad / 2)) - mask_edge_r)
    pad_edge_c = int(np.round(mask_edge_c * np.sin(mask_fov_rad / 2) / np.sin(model_fov_rad / 2)) - mask_edge_c)
    
    mask_padded = np.pad(mask, ((pad_edge_r, pad_edge_r), (pad_edge_c, pad_edge_c)))
    return mask_padded

def mask_stack_to_img(mask_stack: np.ndarray, dpi: int, model_fov: int, mask_fov: int):
    assert mask_stack.ndim == 3
    N_masks = mask_stack.shape[0]
    assert N_masks == PROTOTYPE_NUM_PIXELS, f"Expected {PROTOTYPE_NUM_PIXELS} masks, got {N_masks}"

    mm_to_dots = (1 / 25.4) * dpi # (in / mm) * (dots / in)

    dest_mask_size = (
        int(MASK_WIDTH_MM * mm_to_dots), 
        int(MASK_WIDTH_MM * mm_to_dots))
    
    # Calculate full layout size for 2x2 grid
    full_img_size_mm = (MASK_WIDTH_MM * 2) + INTER_MASK_DIST_MM + (IMG_BORDER_WIDTH_MM * 2)
    full_img_size = (int(full_img_size_mm * mm_to_dots), int(full_img_size_mm * mm_to_dots))

    # Initialize with all 0's
    mask_img = np.zeros(full_img_size, dtype=np.float32)

    if HANDLE_PARALLAX:
        parallax_m = per_pixel_parallax(SCENE_DEPTH, FOCAL_LENGTH)

    i = 0
    # Clean 2x2 Layout mapping
    for r in range(2):
        for c in range(2):
            cell_r = int((r * (MASK_WIDTH_MM + INTER_MASK_DIST_MM) + IMG_BORDER_WIDTH_MM) * mm_to_dots)
            cell_c = int((c * (MASK_WIDTH_MM + INTER_MASK_DIST_MM) + IMG_BORDER_WIDTH_MM) * mm_to_dots)

            current_mask = mask_stack[i]

            current_mask = pad_mask_fov(current_mask, model_fov, mask_fov)
            current_mask = cv2.resize(current_mask, dest_mask_size, interpolation=cv2.INTER_NEAREST)

            if HANDLE_PARALLAX:
                shift_r_m, shift_c_m = -parallax_m[i]
                shift_r_px = int(np.round(shift_r_m * 1e3 * mm_to_dots))
                shift_c_px = int(np.round(shift_c_m * 1e3 * mm_to_dots))
                current_mask = shift_img(current_mask, shift_r_px, shift_c_px, pad_value=0)

            mask_img[cell_r:(cell_r+dest_mask_size[0]),
                     cell_c:(cell_c+dest_mask_size[1])] = current_mask
            i += 1

    mask_img = (mask_img * 255).astype(np.uint8) # cast to uint8
    return mask_img

def inv_printer_transfer_fn(mask_stack):
    D = np.load(utils.get_data_path() / "inkjet_transfer_fn.npz")
    m = D["m"]
    y = D["y"]
    return np.interp(mask_stack, y, m)

@torch.no_grad()
def mask_stack_from_model(args, config_path, checkpoint_name):
    model = load_model(args, config_path, checkpoint_name)
    
    model = model.to(torch.device("cpu"))
    model.simulate_pd_area_blur = False
    model._create_mask_blur_kernel()

    mask_stack = model.visualize_overlap_masks()[:,0].squeeze()

    print("Rotating masks by 90 degrees...")
    for i in range(mask_stack.shape[0]):
        mask_stack[i] = torch.rot90(mask_stack[i], k=1)

    model_fov = model.model_vert_fov
    assert model.model_horiz_fov == model.model_vert_fov

    return mask_stack, model_fov

def mask_stack_from_model_list(args, model_list: List[Tuple[str, int]]):
    model_fov = None 
    mask_stack = np.ones((PROTOTYPE_NUM_PIXELS, 128, 128))

    for label_select in model_list:
        pixel_ids = model_list[label_select]["pixel_ids"]
        if label_select == "all_black":
            mask_stack_curr = np.zeros((len(pixel_ids), 128, 128))
        elif label_select == "all_white":
            mask_stack_curr = np.ones((len(pixel_ids), 128, 128))
        elif label_select == "pinhole":
            mask_stack_curr = np.zeros((len(pixel_ids), 128, 128))
            mask_stack_curr[:,16:18,64:66] = 1
        else:
            config_path = Path(model_list[label_select]["config_path"])
            checkpoint_path = Path(model_list[label_select]["checkpoint_path"])

            mask_stack_curr, model_fov_curr = mask_stack_from_model(args, config_path, checkpoint_path)
            if model_fov is None:
                model_fov = model_fov_curr
            elif model_fov != model_fov_curr:
                print("Model FOV's should all be the same in the model list.")
                sys.exit(1)
            
            mask_stack = np.ones((PROTOTYPE_NUM_PIXELS, *mask_stack_curr.shape[1:]))

        for i, pi in enumerate(pixel_ids):
            # Ensure index bounds since we shrunk to 4 pixels (IDs 1-4)
            if pi - 1 < PROTOTYPE_NUM_PIXELS:
                mask_stack[pi-1] = mask_stack_curr[i]
    
    return mask_stack, model_fov

def parse_model_list(model_list_file: Path | str) -> Dict:
    with open(model_list_file, "r") as f:
        D = yaml.safe_load(f)
    for k in D.keys():
        D[k] = utils.list_of_dicts_to_dict(D[k])
    return D

def apply_multiplicative_mask(mask_stack: np.ndarray, multiplicative_mask: Path | str):
    m = cv2.imread(str(multiplicative_mask))
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.float32:
        m = m.astype(np.float32) / 255
    mask_stack = mask_stack * m[None,:,:]
    return mask_stack

def gen_mask_img(args):
    mask_type = args.mask_type
    if mask_type == "model":
        model_list = parse_model_list(args.model_list_file)
    mask_fov = args.mask_fov
    dpi = args.dpi
    output_name = args.output_name

    if mask_type == "model":
        mask_stack, model_fov = mask_stack_from_model_list(args, model_list)

        if not args.disable_printer_trans_fn:
            mask_stack = inv_printer_transfer_fn(mask_stack)
        mask_stack = np.flip(mask_stack, 2)

    elif mask_type == "linspace": 
        mask_stack = np.tile(np.linspace(0, 1, PROTOTYPE_NUM_PIXELS)[:,None,None], (1, 128, 128))
        model_fov = mask_fov

    elif mask_type == "checkerboard":
        r, c = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
        checkerboard_img = ((r + c) % 2 == 0).astype(np.float32)
        
        mask_stack = np.zeros((PROTOTYPE_NUM_PIXELS, 128, 128))
        mask_stack[:] = checkerboard_img[None,:,:]
        model_fov = mask_fov

    elif mask_type == "pinhole":
        mask_stack = np.zeros((PROTOTYPE_NUM_PIXELS, 128, 128))
        mask_stack[:, 63:65, 63:65] = 1 # Centered pinhole
        model_fov = mask_fov

    elif mask_type == "all_white":
        mask_stack = np.ones((PROTOTYPE_NUM_PIXELS, 128, 128))
        model_fov = mask_fov
    
    else:
        print("Invalid mask type:", mask_type)
        sys.exit(1)

    if args.multiplicative_mask is not None:
        mask_stack = apply_multiplicative_mask(mask_stack, args.multiplicative_mask)

    img = mask_stack_to_img(mask_stack, dpi, model_fov, mask_fov)

    img = Image.fromarray(img)
    img.save(output_name, dpi=(dpi, dpi))

def parse_args():
    parser = argparse.ArgumentParser(description="Create mask printout from trained model")
    parser.add_argument("-d", "--dpi", type=int, default=1200, help="Printer dots-per-inch (DPI)")
    parser.add_argument("-t", "--mask_type", type=str, default="model", help="Mask type (model, linspace, checkerboard, pinhole)")
    parser.add_argument("--mask_fov", type=int, required=True, help="Mask field-of-view (prototype-dependent)")
    parser.add_argument("--model_list_file", type=str, default=None, help="Text file specifying the models")
    parser.add_argument("--multiplicative_mask", type=str, default=None, help="Path to multiplicative mask (for occlusions in camera image)")
    parser.add_argument("-o", "--output_name", type=str, default="masks.png", help="Output file name")
    parser.add_argument("--disable_printer_trans_fn", action="store_true", default=False, help="Do not use the printer transfer function")
    return parser.parse_args()

def run_mask_printing():
    args = parse_args()
    gen_mask_img(args)

if __name__ == "__main__":
    run_mask_printing()