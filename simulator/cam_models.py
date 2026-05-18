import sys
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal.windows
import logging
import math

module_logger = logging.getLogger(__name__)
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import utils.constants as constants

def solid_angle(img_size, fov_deg, f=1.0, normalization='max'):
   
    H, W = img_size
    if np.isscalar(fov_deg):
        vFOV_deg = hFOV_deg = float(fov_deg)
    else:
        vFOV_deg, hFOV_deg = float(fov_deg[0]), float(fov_deg[1])

    x_max = f * np.tan(np.deg2rad(hFOV_deg) / 2.0)
    y_max = f * np.tan(np.deg2rad(vFOV_deg) / 2.0)

    # pixel edge coordinates
    x_edges = np.linspace(-x_max, x_max, W + 1, dtype=np.float64)
    y_edges = np.linspace(-y_max, y_max, H + 1, dtype=np.float64)

    # left/right and bottom/top for each pixel
    x1 = x_edges[:-1]   # (W,)
    x2 = x_edges[1:]    # (W,)
    y1 = y_edges[:-1]   # (H,)
    y2 = y_edges[1:]    # (H,)

    # make shapes broadcastable to (H, W)
    X1 = x1[None, :]  # (1, W) - Width varies along columns
    X2 = x2[None, :]
    Y1 = y1[:, None]  # (H, 1) - Height varies along rows
    Y2 = y2[:, None]

    def corner_term(x, y):
        denom = f * np.sqrt(f**2 + x**2 + y**2)
        return np.arctan2(x * y, denom)

    t22 = corner_term(X2, Y2)   # (H, W)
    t12 = corner_term(X1, Y2)
    t21 = corner_term(X2, Y1)
    t11 = corner_term(X1, Y1)

    sa = (t22 - t12 - t21 + t11).astype(np.float64)

    if normalization == 'max':
        sa = sa / float(sa.max())
    elif normalization == 'sum':
        sa = sa / float(sa.sum())
    elif normalization == 'none':
        pass

    return sa.astype(np.float32)

###################
# MINCAM NETWORKS #
###################

class GaborCam(nn.Module):
    def __init__(self,
                 img_size=(128,128),
                 seq_len=1000,
                 realistic_sensor=False,
                 sensor_gain=1,
                 sensor_n_bits=12,
                 sensor_saturation_val=1,
                 read_noise_std=1/255,
                 mask_init_method="gabors",
                 simulate_pd_area_blur=False,
                 mask_blur_kernel_sigma=0,
                 simulate_directivity=False,
                 simulate_solid_angle=False,
                 mask_min_value=0,
                 mask_max_value=1,
                 model_vert_fov=90,
                 model_horiz_fov=90,
                 n_gabors=1,
                 init_freqs=[8.0],
                 batch_size=8,
                 amplitude_floor=0.0025,
                 debug_mode=False,
                 height = 0.06, #m
                 camera_displacement = 0.019, #cm,
                 full_masks = False,
                 differential_overlap = False
                 ):
         
        super().__init__()
        
        if batch_size < 16:
            norm_type = 'group'
        else:
            norm_type = 'batch'

        self.img_size = img_size
        self.seq_len = seq_len
        self.n_gabors = n_gabors 
        self.amplitude_floor = amplitude_floor
        self.model_vert_fov = model_vert_fov
        self.model_horiz_fov = model_horiz_fov
        self.n_pairs = n_gabors * 2 
        self.simulate_pd_area_blur = simulate_pd_area_blur
        self.mask_blur_kernel_sigma = mask_blur_kernel_sigma
        self._create_mask_blur_kernel()
        self.realistic_sensor = realistic_sensor
        self.simulate_directivity = simulate_directivity
        self.register_buffer("sensor_gain", torch.tensor(sensor_gain, dtype=torch.float32))
        self.register_buffer("sensor_n_bits", torch.tensor(sensor_n_bits, dtype=torch.int64))
        self.register_buffer("sensor_saturation_val", torch.tensor(sensor_saturation_val, dtype=torch.float32))
        self.register_buffer("read_noise_std", torch.tensor(read_noise_std, dtype=torch.float32))
        self.register_buffer("mask_min_value", torch.tensor(mask_min_value, dtype=torch.float32))
        self.register_buffer("mask_max_value", torch.tensor(mask_max_value, dtype=torch.float32))
        if simulate_directivity:
            self.register_buffer("diode_directivity", self._load_diode_directivity())
        else:
            self.directivity_fn = None

        self.simulate_solid_angle = simulate_solid_angle
        if self.simulate_solid_angle:
            module_logger.info("Simulating solid angle per photodiode")
            sa_np = solid_angle(
                img_size=self.img_size,
                fov_deg=(self.model_vert_fov, self.model_horiz_fov),
                f=1.0,
                normalization='max')
            
            self.register_buffer("solid_angle", torch.from_numpy(sa_np))

        self.mask_init_method = mask_init_method
        self.debug_mode = debug_mode
        self.height = height
        H, W = img_size
        assert H == W, "This simplified implementation assumes H == W"
        ###################
        # ALIGNMENT LOGIC #
        ###################
        # Calculate physical shift 'd'
        f_px = H / (2.0 * math.tan(math.radians(model_vert_fov / 2.0)))
        d = int(round(f_px * ((camera_displacement / 2) / max(height, 1e-9))))
        d = max(0, min(d, H // 2))
        view_shifts = torch.tensor([
            [ d, -d], # View 0: BL 
            [ d,  d], # View 1: BR 
            [-d, -d], # View 2: TL 
            [-d,  d]  # View 3: TR 
        ], dtype=torch.int64)
        self.register_buffer("_view_shifts", view_shifts)

        # 2. Calculate Standard Shared ROI Geometry
        geo_roi_h = max(0, H - 2 * d)
        geo_roi_w = max(0, W - 2 * d)
        roi_starts = torch.zeros((4, 2), dtype=torch.int64)
        for v in range(4):
            dy, dx = int(view_shifts[v, 0]), int(view_shifts[v, 1])
            y0_l = d - dy
            x0_l = d - dx
            roi_starts[v] = torch.tensor([y0_l, x0_l])
            
        self.register_buffer("roi_starts", roi_starts)

        # 3. Grid Generation 
        # Aligned to the 'Shared Core' calculated above regardless of the blackening mode
        x = torch.linspace(-torch.pi, torch.pi, W) 
        y = torch.linspace(-torch.pi, torch.pi, H)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        self.register_buffer('X', X.clone())
        self.register_buffer('Y', Y.clone())
        dx_per_pixel = (2 * torch.pi) / W
        dy_per_pixel = (2 * torch.pi) / H
        
        X_shifted = []
        Y_shifted = []
        for v in range(4):
            y0_l, x0_l = int(roi_starts[v, 0]), int(roi_starts[v, 1])
            # Center of the shared ROI core
            roi_center_x = x0_l + geo_roi_w / 2.0
            roi_center_y = y0_l + geo_roi_h / 2.0
            # Shift coordinate system so ROI center is at origin (0, 0)
            X_v = X - (roi_center_x - W / 2.0) * dx_per_pixel
            Y_v = Y - (roi_center_y - H / 2.0) * dy_per_pixel
            X_shifted.append(X_v)
            Y_shifted.append(Y_v)
        
        self.register_buffer('X_shifted', torch.stack(X_shifted, dim=0))
        self.register_buffer('Y_shifted', torch.stack(Y_shifted, dim=0))

        # 4. Gate (Mask) Generation
        gates = torch.zeros((4, H, W), dtype=torch.float32)

        # Default reported HW (may be overridden by differential)
        final_roi_hw = [geo_roi_h, geo_roi_w]

        if differential_overlap:
            # Differential: Full Width, but vertically aligned to shared region
            final_roi_hw = [geo_roi_h, W]             
            if geo_roi_h > 0:
                # Views 0, 1 (Bottom): Keep top pixels (0 to roi_h)
                gates[0:2, 0:geo_roi_h, :] = 1.0 
                # Views 2, 3 (Top): Keep bottom pixels (2*d to H)
                # Note: 2*d is where the bottom region starts relative to the top shift
                gates[2:4, (2 * d):H, :] = 1.0 
                
        else:
            # DEFAULT MODE
            # Standard: Only the central shared rectangle
            if geo_roi_h > 0 and geo_roi_w > 0:
                for v in range(4):
                    y0, x0 = int(roi_starts[v, 0]), int(roi_starts[v, 1])
                    gates[v, y0 : y0 + geo_roi_h, x0 : x0 + geo_roi_w] = 1.0

        if full_masks:
            #Overrides everythin
            gates.fill_(1.0)

        self.register_buffer("roi_gates", gates)
        self.register_buffer("_roi_hw", torch.tensor(final_roi_hw, dtype=torch.int64))


        if self.mask_init_method == "gabors":
            # ALL differentiable Gabor masks parameters
            
            # Amplitude
            initial_raw_amplitudes = torch.full((self.n_gabors,), 6.0)
            self.gabor_raw_amplitudes = nn.Parameter(initial_raw_amplitudes)

            # Frequency
            nyquist_limit = self.img_size[0] / 2
            self.register_buffer("nyquist_limit", torch.tensor(nyquist_limit, dtype=torch.float32))

            assert len(init_freqs) == self.n_gabors, "Length of init_freqs must match n_gabors."
            initial_bounded_freqs = torch.tensor(init_freqs, dtype=torch.float32)
            initial_raw_freqs = torch.logit(initial_bounded_freqs / nyquist_limit)
            self.gabor_raw_frequencies = nn.Parameter(initial_raw_freqs)
            
            # Phases
            self.gabor_phases = nn.Parameter(torch.zeros(self.n_gabors))
            
            # STD
            target_std = 1.0
            target_raw_std = math.log(math.exp(target_std) - 1)
            initial_raw_stds = torch.full((self.n_gabors,), target_raw_std)
            self.gabor_raw_gaussian_stds = nn.Parameter(initial_raw_stds)

            if self.seq_len == 1000:
                self.tcn = tcnDecoder(in_channels = 2, norm_type='group')


    def _create_mask_blur_kernel(self):

        mask_kernel = None
            
        if self.simulate_pd_area_blur:
            
            f_ref = 8e-3 
            fov_ref = 90.0
            
            pd_size_m = .88e-3 # photodiode side lengrth
            
            # Calculate physical mask dimensions based on reference configuration
            mask_width_m = 2*f_ref*np.tan(np.deg2rad(fov_ref/2))
            mask_height_m = 2*f_ref*np.tan(np.deg2rad(fov_ref/2))
            
            pd_kernel_h = self.img_size[0] * pd_size_m / mask_height_m
            pd_kernel_w = self.img_size[1] * pd_size_m / mask_width_m

            print(f"Photodiode kernel size: {int(np.round(pd_kernel_h))}x{int(np.round(pd_kernel_w))}")
            # Box filter from photodiode's active area
            mask_kernel = np.ones((int(np.round(pd_kernel_h)), 
                                   int(np.round(pd_kernel_w))), 
                                  dtype=np.float32)
            mask_kernel /= mask_kernel.sum()


        if self.mask_blur_kernel_sigma is not None and \
            self.mask_blur_kernel_sigma > 0:
            # Gaussian smoothing for robustness to misalignment
            assert self.img_size == (128, 128)
            M = self.mask_blur_kernel_sigma * 4 + 1
            k_gaussian = 1 / np.sqrt(2 * np.pi * self.mask_blur_kernel_sigma**2) * \
                np.exp(
                    - np.arange(-np.floor(M/2), np.floor(M/2)+1)**2 / \
                        (2 * self.mask_blur_kernel_sigma**2))
            k_gaussian /= k_gaussian.sum()
            k_gaussian = k_gaussian[:,None] * k_gaussian[None,:]

            if mask_kernel is not None:
                mask_kernel = scipy.signal.convolve2d(
                    mask_kernel, k_gaussian, mode='full')
            else:
                mask_kernel = k_gaussian


        if mask_kernel is not None:
            mask_kernel = torch.from_numpy(mask_kernel).to(torch.float32)[None,None,:,:]

        self.register_buffer("mask_blur_kernel", mask_kernel)

    def _load_diode_directivity(self):
        p = scipy.io.loadmat(
            str(constants.DIODE_DIRECTIVITY_PATH))["p"].ravel()

        r_start = np.tan(np.deg2rad(self.model_vert_fov) / 2)
        c_start = np.tan(np.deg2rad(self.model_horiz_fov) / 2)
        r, c = np.meshgrid(
            np.linspace(-r_start, r_start, self.img_size[0]),
            np.linspace(-c_start, c_start, self.img_size[1]),
            indexing="ij")

        radius = np.sqrt(r**2 + c**2)
        theta = np.rad2deg(np.arctan(radius))

        v = np.polyval(p, theta)
        v /= v.max()

        return torch.from_numpy(v).to(torch.get_default_dtype())

    @torch.no_grad()
    def export_full_masks(self):
        """
        Returns the final full-size masks per view after transform and hard gating.
        Output order matches views: [0:+cos, 1:+sin, 2:-cos, 3:-sin].
        Shape: [4, H, W]
        """
        H, W = self.img_size
        
        # Build Gabors using shifted grids for each view
        amp = torch.sigmoid(self.gabor_raw_amplitudes)
        freq = torch.sigmoid(self.gabor_raw_frequencies) * self.nyquist_limit
        std = F.softplus(self.gabor_raw_gaussian_stds) + 1e-6

        masks = []
        
        for v in range(4):
            X_v = self.X_shifted[v]
            Y_v = self.Y_shifted[v]
            
            if v == 0:  # +cos
                M = amp[:, None, None] * torch.cos(freq[:, None, None] * X_v) * \
                    torch.exp(-X_v**2 / (2 * std[:, None, None]**2))
            elif v == 1:  # +sin
                M = amp[:, None, None] * torch.sin(freq[:, None, None] * X_v) * \
                    torch.exp(-X_v**2 / (2 * std[:, None, None]**2))
            elif v == 2:  # -cos
                M = -amp[:, None, None] * torch.cos(freq[:, None, None] * X_v) * \
                    torch.exp(-X_v**2 / (2 * std[:, None, None]**2))
            else:  # -sin
                M = -amp[:, None, None] * torch.sin(freq[:, None, None] * X_v) * \
                    torch.exp(-X_v**2 / (2 * std[:, None, None]**2))
            
            M_transformed = self._mask_param_transform(M.reshape(self.n_gabors, -1)).reshape(self.n_gabors, H, W)
            masks.append(M_transformed[0])  # Take first gabor
        
        masks = torch.stack(masks, dim=0)  # [4, H, W]
        
        # Apply gates
        gates = self.roi_gates
        inv_gates = 1.0 - gates
        min_val = self.mask_min_value
        
        masks = masks * gates + min_val * inv_gates
        
        return masks

    @torch.no_grad()
    def visualize_overlap_masks(self, to_cpu=True):
        """
        Convenience visualization tensor for plotting.
        Returns [4, 3, H, W] float in [0,1], ordered:
          0: +cos (view 0), 1: +sin (view 1), 2: −cos (view 2), 3: −sin (view 3)
        """
        masks = self.export_full_masks()  # [4,H,W]

        masks_rgb = torch.tile(masks[:, None, :, :], (1, 3, 1, 1))  # [4,3,H,W]


        return masks_rgb.cpu() if to_cpu else masks_rgb


    def _mask_param_transform(self, x: torch.Tensor):
        y = F.relu(x) * \
            (self.mask_max_value - self.mask_min_value) + self.mask_min_value

        # Convolve with mask blur kernel
        if self.mask_blur_kernel is not None:
            y = y.reshape(y.shape[0], 1, *self.img_size) # Nx1xHxW
            y = F.conv2d(y, self.mask_blur_kernel, padding="same")
            y = y.reshape(y.shape[0], -1)

        return y

    def _mask_param_inv_transform(self, y: torch.Tensor):
        x = torch.logit((y - self.mask_min_value) / \
            (self.mask_max_value - self.mask_min_value))

        return x


    def _forward_mask(self, imgs):
        """
        imgs: [C=4, B*S, 1, H, W] with ordering [0:+cos, 1:+sin, 2:-cos, 3:-sin]
        Returns: Tensor [B*S, 2]  -> [diff_cos, diff_sin]
        """
        C, BS, ch, H, W = imgs.shape
        assert C == 4 and ch == 1

        # Flatten spatial dims
        x = imgs.reshape(C, BS, H*W)  # [4, BS, HW]

        roi_h, roi_w = int(self._roi_hw[0]), int(self._roi_hw[1])
        if roi_h == 0 or roi_w == 0:
            return torch.zeros((BS, 2), device=imgs.device, dtype=imgs.dtype)

        # 1) Build Gabors using shifted grids for each view
        amp = torch.sigmoid(self.gabor_raw_amplitudes)
        freq = torch.sigmoid(self.gabor_raw_frequencies) * self.nyquist_limit
        std = F.softplus(self.gabor_raw_gaussian_stds) + 1e-6

        # Generate only the needed mask for each view (same logic as export_full_masks)
        M_views = []
        
        #Removed Phase parameter
        for v in range(4):
            X_v = self.X_shifted[v]  # [H, W]
            Y_v = self.Y_shifted[v]  # [H, W]
            
            if v == 0:  # +cos
                M = amp[:, None, None] * torch.cos(freq[:, None, None] * X_v ) * \
                    torch.exp(-X_v**2 / (2 * std[:, None, None]**2))
            elif v == 1:  # +sin
                M = amp[:, None, None] * torch.sin(freq[:, None, None] * X_v ) * \
                    torch.exp(-X_v**2 / (2 * std[:, None, None]**2))
            elif v == 2:  # -cos
                M = -amp[:, None, None] * torch.cos(freq[:, None, None] * X_v ) * \
                    torch.exp(-X_v**2 / (2 * std[:, None, None]**2))
            else:  # -sin (v == 3)
                M = -amp[:, None, None] * torch.sin(freq[:, None, None] * X_v) * \
                    torch.exp(-X_v**2 / (2 * std[:, None, None]**2))
            
            # Transform to mask values
            M_transformed = self._mask_param_transform(M.reshape(self.n_gabors, -1)).reshape(self.n_gabors, H, W)
            M_views.append(M_transformed)

        # 2) Apply ROI gates to masks (zero out non-overlapping regions)
        gates = self.roi_gates  # [4, H, W]
        inv_gates = 1.0 - gates
        min_val = self.mask_min_value
        
        # Gate each view's mask
        M_v0 = M_views[0] * gates[0] + min_val * inv_gates[0]  # +cos, view 0
        M_v1 = M_views[1] * gates[1] + min_val * inv_gates[1]  # +sin, view 1
        M_v2 = M_views[2] * gates[2] + min_val * inv_gates[2]  # -cos, view 2
        M_v3 = M_views[3] * gates[3] + min_val * inv_gates[3]  # -sin, view 3

        # 3) Project and form differential signals
        outs = []
        
        # diff_cos = v0 - v2
        diff_cos = self._apply_sensor_model(F.linear(x[0], M_v0.reshape(self.n_gabors, -1), bias=None)) - \
                self._apply_sensor_model(F.linear(x[2], M_v2.reshape(self.n_gabors, -1), bias=None))
        outs.append(diff_cos.unsqueeze(1))

        # diff_sin = v1 - v3
        diff_sin = self._apply_sensor_model(F.linear(x[1], M_v1.reshape(self.n_gabors, -1), bias=None)) - \
                self._apply_sensor_model(F.linear(x[3], M_v3.reshape(self.n_gabors, -1), bias=None))
        outs.append(diff_sin.unsqueeze(1))

        out = torch.cat(outs, dim=1)  # [BS, 2]
        return out

    
    
    def _apply_sensor_model(self, x):
        # Scale to the sensor's dynamic range
        x = x * self.sensor_gain
        if self.realistic_sensor:
            # Saturation
            x = torch.where(x < self.sensor_saturation_val,
                            x,
                            0.01 * (x - self.sensor_saturation_val) +
                            self.sensor_saturation_val)
            # Read Noise
            x = x + torch.randn_like(x) * self.read_noise_std
            # Quantization noise
            x = x + torch.rand_like(x) * \
                (self.sensor_saturation_val / 2**self.sensor_n_bits)
        
        return x


    def _sensor_fn(self, x):
        # Apply sensor directivity (vignetting)
        if self.simulate_directivity:
            #diode_directivity has shape (H, W)
            # x: (S,batch*N, 1, H, W)
            x = x * self.diode_directivity[None, None,None,:,:]
            
        if self.simulate_solid_angle:
            # solid_angle has shape (H, W)
            # x: (S,batch*N, 1, H, W)
            x = x * self.solid_angle[None, None,None,:,:]

        # Projection with masks, the sensor model is directly embedded in the mask
        x = self._forward_mask(x)

        return x

    def forward(self, imgs):
        B, C, S, Ch, H, W = imgs.shape #Batch, Camera, Sequence, Channel, Height, Width
        imgs = imgs.permute(1, 0, 2, 3, 4, 5).reshape(C, B*S, Ch, H, W) # (Cameras, Batch*Sequence, Ch, H, W)
        
        x = self._sensor_fn(imgs)         # (B*Seq, n_pairs)
        x = x.view(B, S, -1)              # (B, Seq, n_pairs)
        
        ## 0 mean normalization per channel ##
        mean = x.mean(dim=1, keepdim=True)

        x_norm = (x - mean)
        x_norm = x_norm.permute(0,2,1)                # (B, n_pairs, S)
        
        if self.debug_mode:
            output = self.tcn(x_norm)
            return output, x.permute(0,2,1), x_norm 
        else:
            output = self.tcn(x_norm)
            return output
        
    def sensor_step(self, frame):
        """
        frame: (B, C, S, Ch, H, W) arbitrary number of time steps
        Returns: (B, S, n_pairs) sensor measurement for these steps
        """
        B, C, S, Ch, H, W = frame.shape  
        
        imgs = frame.permute(1, 0, 2, 3, 4, 5).reshape(C, B*S, Ch, H, W)
        meas = self._sensor_fn(imgs)              # (B*S, n_pairs)
        
        return meas.view(B, S, -1)
    
    def forward_measurements(self, meas_buffer):
        """
        meas_buffer: (B, S, n_pairs) accumulated sensor measurements
        """
        x = meas_buffer  # (B, S, n_pairs)
        mean = x.mean(dim=1, keepdim=True)
        x_norm = x - mean
        x_norm = x_norm.permute(0, 2, 1)              # (B, n_pairs, S)
        output = self.tcn(x_norm)

        return output

class FreeformCam(nn.Module):
    def __init__(self,
                 img_size,
                 mincam_size=4,
                 realistic_sensor=False,
                 sensor_gain=1,
                 sensor_n_bits=12,
                 sensor_saturation_val=1,
                 read_noise_std=1/255,
                 mask_init_method="random",
                 simulate_pd_area_blur=False,
                 mask_blur_kernel_sigma=0,
                 simulate_directivity=False,
                 simulate_solid_angle=False,
                 mask_min_value=0,
                 mask_max_value=1,
                 model_vert_fov=70,
                 model_horiz_fov=70,
                 seq_len= 1000):
        super().__init__()

        self.mincam_size = mincam_size
        self.img_size = img_size
        self.seq_len = seq_len
        # Field of view of the model masks, unrelated to the prototype
        self.model_vert_fov = model_vert_fov
        self.model_horiz_fov = model_horiz_fov

        self.simulate_pd_area_blur = simulate_pd_area_blur
        self.mask_blur_kernel_sigma = mask_blur_kernel_sigma
        self._create_mask_blur_kernel()

        self.realistic_sensor = realistic_sensor
        self.simulate_directivity = simulate_directivity

        self.register_buffer("sensor_gain",
                             torch.tensor(sensor_gain, dtype=torch.float32))
        self.register_buffer("sensor_n_bits",
                             torch.tensor(sensor_n_bits, dtype=torch.int64))
        self.register_buffer("sensor_saturation_val",
                             torch.tensor(sensor_saturation_val,
                                          dtype=torch.float32))
        self.register_buffer("read_noise_std",
                             torch.tensor(read_noise_std, dtype=torch.float32))
        self.register_buffer("mask_min_value",
                             torch.tensor(mask_min_value, dtype=torch.float32))
        self.register_buffer("mask_max_value",
                             torch.tensor(mask_max_value, dtype=torch.float32))

        if simulate_directivity:
            self.register_buffer("diode_directivity",
                                 self._load_diode_directivity())
        else:
            self.directivity_fn = None

        self.simulate_solid_angle = simulate_solid_angle
        if self.simulate_solid_angle:
            module_logger.info("Simulating solid angle per photodiode")
            sa_np = solid_angle(
                img_size=self.img_size,
                fov_deg=(self.model_vert_fov, self.model_horiz_fov),
                f=1.0,
                normalization='max')
            
            self.register_buffer("solid_angle", torch.from_numpy(sa_np))

        self._mincam_init_masks(mask_init_method)

        self.tcn = tcnDecoder(in_channels = 4, norm_type='group')


    def _create_mask_blur_kernel(self):

        mask_kernel = None

        if self.simulate_pd_area_blur:
            f_ref = 8e-3 
            fov_ref = 90.0
            
            pd_size_m = .88e-3 # photodiode side lengrth
            
            # Calculate physical mask dimensions based on reference configuration
            mask_width_m = 2*f_ref*np.tan(np.deg2rad(fov_ref/2))
            mask_height_m = 2*f_ref*np.tan(np.deg2rad(fov_ref/2))
            
            pd_kernel_h = self.img_size[0] * pd_size_m / mask_height_m
            pd_kernel_w = self.img_size[1] * pd_size_m / mask_width_m

            print(f"Photodiode kernel size: {int(np.round(pd_kernel_h))}x{int(np.round(pd_kernel_w))}")
            # Box filter from photodiode's active area
            mask_kernel = np.ones((int(np.round(pd_kernel_h)), 
                                   int(np.round(pd_kernel_w))), 
                                  dtype=np.float32)
            mask_kernel /= mask_kernel.sum()

        if self.mask_blur_kernel_sigma is not None and \
            self.mask_blur_kernel_sigma > 0:
            # Gaussian smoothing for robustness to misalignment
            assert self.img_size == (128, 128)
            M = self.mask_blur_kernel_sigma * 4 + 1
            k_gaussian = 1 / np.sqrt(2 * np.pi * self.mask_blur_kernel_sigma**2) * \
                np.exp(
                    - np.arange(-np.floor(M/2), np.floor(M/2)+1)**2 / \
                        (2 * self.mask_blur_kernel_sigma**2))
            k_gaussian /= k_gaussian.sum()
            k_gaussian = k_gaussian[:,None] * k_gaussian[None,:]

            if mask_kernel is not None:
                mask_kernel = scipy.signal.convolve2d(
                    mask_kernel, k_gaussian, mode='full')
            else:
                mask_kernel = k_gaussian

        if mask_kernel is not None:
            mask_kernel = torch.from_numpy(mask_kernel).to(torch.float32)[None,None,:,:]

        self.register_buffer("mask_blur_kernel", mask_kernel)

    def _load_diode_directivity(self):
        p = scipy.io.loadmat(
            str(constants.DIODE_DIRECTIVITY_PATH))["p"].ravel()

        r_start = np.tan(np.deg2rad(self.model_vert_fov) / 2)
        c_start = np.tan(np.deg2rad(self.model_horiz_fov) / 2)
        r, c = np.meshgrid(
            np.linspace(-r_start, r_start, self.img_size[0]),
            np.linspace(-c_start, c_start, self.img_size[1]),
            indexing="ij")

        radius = np.sqrt(r**2 + c**2)
        theta = np.rad2deg(np.arctan(radius))

        v = np.polyval(p, theta)
        v /= v.max()

        return torch.from_numpy(v).to(torch.get_default_dtype())

    @torch.no_grad()
    def visualize_overlap_masks(self, num_masks=4):
        """
        Return the first N masks for visualization
        """
        if num_masks == -1:
            num_masks = self.masks.shape[0]

        masks_vis = self._mask_param_transform(
            self.masks[:num_masks,:]).reshape(
                num_masks, *self.img_size).detach()

        masks_vis = torch.tile(masks_vis[:,None,:,:], (1, 3, 1, 1))

        return masks_vis # Nx3xHxW

    def _mask_param_transform(self, x: torch.Tensor):
        y = torch.sigmoid(x) * \
            (self.mask_max_value - self.mask_min_value) + self.mask_min_value

        # Convolve with mask blur kernel
        if self.mask_blur_kernel is not None:
            y = y.reshape(y.shape[0], 1, *self.img_size) # Nx1xHxW
            y = F.conv2d(y, self.mask_blur_kernel, padding="same")
            y = y.reshape(y.shape[0], -1)

        return y

    def _mask_param_inv_transform(self, y: torch.Tensor):
        x = torch.logit((y - self.mask_min_value) / \
            (self.mask_max_value - self.mask_min_value))

        return x

    def _mincam_init_masks(self, mask_init_method):
        """
        Initialize the masks using uniform noise
        """
        mask_size = (self.mincam_size, self.img_size[0] * self.img_size[1])

        if mask_init_method == "random":
            # Initialize M(x,y) ~ Uniform(mid_point - half_range,
            #                             mid_point + half_range)
            mid_point = 0.15
            half_range = 0.13
            masks = self._mask_param_inv_transform(
                torch.rand(mask_size) * 2 * half_range + mid_point - half_range
            )
        
        elif mask_init_method == "flat":
            mid_point = 0.1 
            masks = self._mask_param_inv_transform(
                torch.full(mask_size, mid_point)
            )

        else:
            module_logger.error("Unsupported mask initialization method: %s" %
                                mask_init_method)
            sys.exit(1)

        self.masks = nn.Parameter(masks)

    def _forward_mask(self, imgs):
        C, BS, Ch, H, W = imgs.shape
        # Flatten spatial dims: [4, BS, H*W]
        x = imgs.view(C, BS, H*W)  
        
        # Get the 4 freeform masks: [4, H*W]
        M = self._mask_param_transform(self.masks)
        
        outs = []
        for i in range(4):
            
            proj = F.linear(x[i], M[i].unsqueeze(0), bias=None) # Shape: [BS, 1]
            outs.append(proj)
            
        # Concatenate them back together to [BS, 4]
        out = torch.cat(outs, dim=1)
        return out

    def _apply_sensor_model(self, x):
        # Scale to the sensor's dynamic range
        x = x * self.sensor_gain

        if self.realistic_sensor:
            # Saturation
            x = torch.where(x < self.sensor_saturation_val,
                            x,
                            0.01 * (x - self.sensor_saturation_val) +
                            self.sensor_saturation_val)

            # Read Noise
            x = x + torch.randn_like(x) * self.read_noise_std

            # Quantization noise
            x = x + torch.rand_like(x) * \
                (self.sensor_saturation_val / 2**self.sensor_n_bits)

        return x

    def _sensor_fn(self, x):
        # Apply sensor directivity (vignetting)
        if self.simulate_directivity:
            x = x * self.diode_directivity[None, None,None,:,:]

        if self.simulate_solid_angle:
            # solid_angle has shape (H, W)
            # x: (S,batch*N, 1, H, W)
            x = x * self.solid_angle[None, None,None,:,:]

        # Projection with masks
        x = self._forward_mask(x)

        # Sensor model
        x = self._apply_sensor_model(x)

        return x

    def forward(self, imgs):
        B, C, S, Ch, H, W = imgs.shape
        imgs = imgs.permute(1, 0, 2, 3, 4, 5).reshape(C, B*S, Ch, H, W) 
        x = self._sensor_fn(imgs)
        x = x.view(B, S, -1)
        mean = x.mean(dim=1, keepdim=True)
        x_norm = (x - mean)
        x_norm = x_norm.permute(0,2,1)   
        output = self.tcn(x_norm)
        return output

#########################################
#  Temporal Convolutional Networ (TCN)  #
#########################################

# UTILS FOR TCN #
class ChannelAffine(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
    def forward(self, x):
        return x * self.scale + self.bias

class ResidualDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1, norm_type='batch', groups=1):
        """
        norm_type: 'batch' | 'group' | 'channel_affine' | 'none'
        """
        super(ResidualDilatedBlock, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation, padding=padding,  groups=groups)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation, padding=padding)

        # Configure normalization
        self.norm_type = norm_type
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        
        elif norm_type == 'group':
            num_groups1 = 8 if out_channels % 8 == 0 else 4 if out_channels % 4 == 0 else 2 if out_channels % 2 == 0 else 1
            self.norm1 = nn.GroupNorm(num_groups=num_groups1, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups1, num_channels=out_channels)
        
        elif norm_type == 'none':
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        else:
            raise ValueError(f'Unsupported norm_type: {norm_type}')

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # If in/out channels differ, use 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out = out + residual
        out = self.relu2(out)
        return out
    
class ResidualHeadV2(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(ResidualHeadV2, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2) 
        
        self.activation = nn.GELU() 

        if in_channels != hidden_dim:
            self.shortcut = nn.Linear(in_channels, hidden_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Calculate the residual path first
        shortcut = self.shortcut(x)
        
        # Calculate the main path with two non-linear layers
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        
        # Add the residual connection and apply the final activation
        out = out + shortcut
        out = self.activation(out)
        
        # Final projection to the output
        out = self.fc3(out)
        return out

class tcnDecoder(nn.Module):

    def __init__(self, in_channels=4, norm_type='group'):  #Should probably consider weight normalization!
        super(tcnDecoder, self).__init__()
        
        self.tcn_block = nn.Sequential(
            ResidualDilatedBlock(in_channels, 16, kernel_size=3, dilation=1, dropout=0.05, norm_type=norm_type),
            ResidualDilatedBlock(16, 32, kernel_size=3, dilation=2, dropout=0.05, norm_type=norm_type),
            ResidualDilatedBlock(32, 64, kernel_size=3, dilation=4, dropout=0.05, norm_type=norm_type),
            ResidualDilatedBlock(64, 64, kernel_size=3, dilation=8, dropout=0.05, norm_type=norm_type),
            ResidualDilatedBlock(64, 64, kernel_size=3, dilation=16, dropout=0.05, norm_type=norm_type), #64
            ResidualDilatedBlock(64, 64, kernel_size=3, dilation=32, dropout=0.05, norm_type=norm_type), #127
            ResidualDilatedBlock(64, 64, kernel_size=3, dilation=64, dropout=0.05, norm_type=norm_type), #255
            ResidualDilatedBlock(64, 64, kernel_size=3, dilation=128, dropout=0.05, norm_type=norm_type), #511
            ResidualDilatedBlock(64, 64, kernel_size=3, dilation=256, dropout=0.05, norm_type=norm_type) #1023
        )
        
        self.speed_attention_pool = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

        self.speed_regressor_head = ResidualHeadV2(64, 32)

    def forward(self, x):

        speed_fmap = self.tcn_block(x)              # -> (batch, 64, seq_len)
        speed_w = self.speed_attention_pool(speed_fmap)
        x_speed_pooled = (speed_fmap * speed_w).sum(dim=-1) # -> (batch, 128)
        
        speed_out = self.speed_regressor_head(x_speed_pooled)
        speed, speed_log_var = speed_out[:,0].unsqueeze(1), speed_out[:,1].unsqueeze(1)

        return speed, speed_log_var


