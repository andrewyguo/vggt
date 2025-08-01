import json 
import glob 
import os 
import random
from skimage import io

import numpy as np 
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch 

class VGGTDatasetRawOverfit(Dataset):
    def __init__(self, 
                 root_path,
                 sequence_range=(5, 15),
                 sampling_rate=3,
                 ds_factor=16, 
                 target_size=518,
                 vis_every=None, 
                 save_dir=None, 
                 max_size=None, 
                 subsample=None,
                 ):
        self.root_path = root_path 

        self.clean_npy_naming = "iso006400_s0_00003_npy"
        self.noisy_npy_naming = "iso102400_s1_00080_npy"
        self.clean_npy_dir = os.path.join(self.root_path, self.clean_npy_naming)

        self.ds_factor = ds_factor 
        self.target_size = target_size
        self.max_size = max_size
        self.subsample = subsample

        self.sequence_range = sequence_range
        self.sampling_rate = sampling_rate

        self.raw_bit_depth = 14
        self.count = 0

        self._setup_scenes()

        self.save_dir = save_dir
        if vis_every is None:
            self.vis_every = len(self.clean_npy_paths) // 5
        else:
            self.vis_every = vis_every

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

    def __getitem__(self, idx):
        # pick start frame & sequence length
        start_path = self.clean_npy_paths[idx]
        if self.manual_scaling_factor is not None:
            sf = self.manual_scaling_factor
        else:
            sf = self.scene_scaling_info[os.path.dirname(start_path)]["scaling_factor"]
        T = random.randint(*self.sequence_range)
        seq_paths = self.get_sequence_paths(start_path, T)

        clean_list, noisy_list = [], []
        raw_max = 2**self.raw_bit_depth - 1

        for p in seq_paths:
            clean_np = np.load(p)                                  # uint16 H×W×3

            noisy_p = p.replace(self.clean_npy_naming, self.noisy_npy_naming)
            noisy_np = np.load(noisy_p)

            # normalize → float32 → Tensor(C,H,W)
            clean_t = torch.from_numpy(clean_np.astype(np.float32) / raw_max) \
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
