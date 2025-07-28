import json 
import glob 
import os 
import random
from skimage import io

import numpy as np 
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch 

class VGGTDatasetRaw(Dataset):
    def __init__(self, 
                 root_path,
                 scaling_constant_range=(21.0, 23.0), 
                 sequence_range=(5, 15),
                 sampling_rate=3,
                 noise_alphas=[5.96, 3.13, 6.81],
                 noise_betas=[-3669, -1991, -4189], 
                 ds_factor=16, 
                 target_size=518,
                 use_clean_jpg=False,
                 vis_every=None, 
                 save_dir=None, 
                 max_size=None, 
                 subsample=None,
                 distortion_coefficents=None,
                 use_distorted=False,
                 ):
        self.root_path = root_path 

        self.clean_jpg_naming = "clean_jpg"
        self.clean_npy_naming = "clean_npy"
        self.clean_jpg_dir = os.path.join(self.root_path, self.clean_jpg_naming)
        self.clean_npy_dir = os.path.join(self.root_path, self.clean_npy_naming)
        self.use_clean_jpg = use_clean_jpg
        self.use_distorted = use_distorted

        if use_distorted and distortion_coefficents is not None:
            with open(distortion_coefficents, 'r') as f:
                intrinsics = json.load(f)
            fx, fy = intrinsics["fl_x"], intrinsics["fl_y"]
            cx, cy = intrinsics["cx"], intrinsics["cy"]
            k1, k2 = intrinsics["k1"], intrinsics["k2"]
            p1, p2 = intrinsics["p1"], intrinsics["p2"]
            w, h = intrinsics["w"], intrinsics["h"]
            self.image_size = (h, w)  # height, width

            self.camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)
            self.dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)  

        elif distortion_coefficents is None:
            raise ValueError("If use_distorted is True, distortion_coefficents must be provided.")

        self.ds_factor = ds_factor 
        self.target_size = target_size
        self.max_size = max_size
        self.subsample = subsample

        self.scaling_constant_range = scaling_constant_range

        self.sequence_range = sequence_range
        self.sampling_rate = sampling_rate

        self._setup_scenes()

        self.save_dir = save_dir
        if vis_every is None:
            self.vis_every = len(self.clean_npy_paths) // 5
        else:
            self.vis_every = vis_every
        # Noise calibration parameters 
        self.B_m1 = np.array([512, 512, 512]).reshape(1, 1, 3)   # Black level for capture 1
        self.B_m2 = np.array([1024, 1024, 1024]).reshape(1, 1, 3)   # Black level for capture 2
        self.alphas = np.array(noise_alphas).reshape(1, 1, 3)   # Per-channel slopes
        self.betas = np.array(noise_betas).reshape(1, 1, 3)   # Per-channel intercepts 

        self.raw_bit_depth = 14
        self.srgb_bit_depth = 8
        self.count = 0

    def _setup_scenes(self):
        scenes = sorted(os.listdir(self.clean_npy_dir))

        self.scene_scaling_info = {}

        images_folder_name = "images_undistorted" if self.use_distorted else "images"
        for scene in scenes:
            metadata_path = os.path.join(self.clean_npy_dir, scene, f"downsampled_{str(self.ds_factor).zfill(3)}", "scene_metadata.json")
            images_path = os.path.join(self.clean_npy_dir, scene, f"downsampled_{str(self.ds_factor).zfill(3)}", images_folder_name)

            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                scaling_constant = np.random.uniform(self.scaling_constant_range[0], self.scaling_constant_range[1])
                scaling_factor = self._get_scaling_factor(metadata["overall_mean"], scaling_constant)
                self.scene_scaling_info[images_path] = {
                    "scaling_factor": scaling_factor,
                    "scaling_constant": scaling_constant,
                    "total_images": metadata["total_images"],
                }
            
        # save image paths in single list 
        self.clean_npy_paths = []
        for scene in scenes:
            npy_dir = os.path.join(self.clean_npy_dir, scene, f"downsampled_{str(self.ds_factor).zfill(3)}", images_folder_name)
            self.clean_npy_paths += sorted(glob.glob(f"{npy_dir}/*.npy"))
        if self.subsample is not None:
            self.clean_npy_paths = self.clean_npy_paths[::self.subsample]
        elif self.max_size is not None:
            self.clean_npy_paths = random.sample(self.clean_npy_paths, self.max_size)

    def _get_scaling_factor(self, scene_global_mean, scene_scaling_constant):
        scaling_factor = (scene_global_mean - self.B_m1.mean()) / scene_scaling_constant

        return scaling_factor

    def __len__(self):
        return len(self.clean_npy_paths)
    
    def get_image_scene_idx(self, img_path):
        image_idx = int(os.path.basename(img_path).split(".")[0])

        return image_idx 

    def get_sequence_paths(self, img_path, sequence_length):
        scene_idx = self.get_image_scene_idx(img_path)

        window_size = sequence_length * self.sampling_rate
        num_images_scene = self.scene_scaling_info[os.path.dirname(img_path)]["total_images"]

        start_idx = max(0, scene_idx - window_size // 2)
        end_idx = min(num_images_scene - 1, scene_idx + (window_size - (scene_idx - start_idx + 1)))
        
        selected_indices = sorted(random.sample(range(start_idx, end_idx + 1), sequence_length))

        sequence_paths = []
        for i in selected_indices:
            sequence_paths.append(img_path.replace(f"{str(scene_idx).zfill(6)}.", f"{str(i).zfill(6)}."))

        return sequence_paths

    def _simulate_noise_raw(self, M1, scaling_factor):
        C = M1 - self.B_m1

        image_clean = C / scaling_factor + self.B_m2
        noise_variance = np.maximum(self.alphas * image_clean + self.betas, 0)
        noise_std = np.sqrt(noise_variance)
        
        noise = np.random.normal(0, 1, size=image_clean.shape) * noise_std

        M2 = image_clean + noise

        max_val = 2**self.raw_bit_depth - 1
        M2 = np.clip(M2, 0, max_val)
        M2 = np.round(M2).astype(np.uint16)

        return M2

    def __getitem__(self, idx):
        # pick start frame & sequence length
        start_path = self.clean_npy_paths[idx]
        sf = self.scene_scaling_info[os.path.dirname(start_path)]["scaling_factor"]
        T = random.randint(*self.sequence_range)
        seq_paths = self.get_sequence_paths(start_path, T)

        clean_list, noisy_list = [], []
        raw_max = 2**self.raw_bit_depth - 1
        jpg_max = 2**self.srgb_bit_depth - 1

        for p in seq_paths:
            raw = np.load(p)                                  # uint16 H×W×3

            # clean load + bit-depth max
            if self.use_clean_jpg:
                jpg = io.imread(p.replace(self.clean_npy_naming, self.clean_jpg_naming)
                                .replace(".npy", ".jpg"))
                clean_np = jpg
                maxc = jpg_max
            else:
                clean_np = raw
                maxc = raw_max

            noisy_np = self._simulate_noise_raw(raw, sf)      # uint16 H×W×3

            # normalize → float32 → Tensor(C,H,W)
            clean_t = torch.from_numpy(clean_np.astype(np.float32) / maxc) \
                          .permute(2, 0, 1)
            noisy_t = torch.from_numpy(noisy_np.astype(np.float32) / raw_max) \
                          .permute(2, 0, 1)

            clean_list.append(clean_t)
            noisy_list.append(noisy_t)

        # stack into (T, C, H, W)
        clean_seq = torch.stack(clean_list, dim=0)
        noisy_seq = torch.stack(noisy_list, dim=0)

        # compute center-pad amounts
        _, _, H, W = clean_seq.shape
        dh, dw = self.target_size - H, self.target_size - W
        pad = (dw//2, dw - dw//2, dh//2, dh - dh//2)  # (left, right, top, bottom)

        # pad with 1.0 (white after normalization)
        clean_seq = F.pad(clean_seq, pad, mode='constant', value=1.0)
        noisy_seq = F.pad(noisy_seq, pad, mode='constant', value=1.0)

        self.count += 1
        return clean_seq, noisy_seq
