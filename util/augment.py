import random
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import numpy as np
import os
import cv2
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from io import BytesIO
import numpy as np
import random

def lighting_variation_augmentation(frames, strength=0.5):
    B, C, T, H, W = frames.shape
    device = frames.device

    # Generate a smooth random brightness curve over time (shared across batch)
    time_curve = torch.randn(T, device=device)
    time_curve = F.avg_pool1d(time_curve.unsqueeze(0).unsqueeze(0), kernel_size=15, stride=1, padding=7).squeeze()
    time_curve = (time_curve - time_curve.min()) / (time_curve.max() - time_curve.min() + 1e-6)
    time_curve = 1.0 + (time_curve - 0.5) * 2 * strength  # Range: [1-strength, 1+strength]

    # Expand to [B, T, 1, 1, 1]
    brightness_factors = time_curve.view(1, T, 1, 1, 1).repeat(B, 1, H, W, 1)

    frames = frames.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]

    # Set a fixed initial center for light source
    center_x = random.randint(W // 4, 3 * W // 4)
    center_y = random.randint(H // 4, 3 * H // 4)
    radius = random.randint(min(H, W) // 5, min(H, W) // 2)  # Larger light radius

    # Decide the fixed jitter movement (shared across batch)
    jitter_dx = random.randint(-W // 10, W // 10)
    jitter_dy = random.randint(-H // 10, H // 10)
    jitter_dr = random.randint(-radius // 5, radius // 5)

    for b in range(B):
        for t in range(T):

            # Apply the same jitter pattern scaled by frame index
            # Optional: make it progressive per frame
            jitter_x = center_x + (jitter_dx * t // max(T-1, 1))
            jitter_y = center_y + (jitter_dy * t // max(T-1, 1))
            jitter_radius = radius + (jitter_dr * t // max(T-1, 1))

            # Generate a soft Gaussian mask
            Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
            dist = ((X - jitter_x)**2 + (Y - jitter_y)**2).float()
            gaussian_mask = torch.exp(-dist / (4 * (jitter_radius**2)))

            # Apply brightness factor
            mask = 1.0 + (brightness_factors[b, t, :, :, 0] - 1.0) * gaussian_mask
            frames[b, t] = frames[b, t] * mask.unsqueeze(0)

    frames = torch.clamp(frames, 0, 1)
    frames = frames.permute(0, 2, 1, 3, 4)  # Restore [B, C, T, H, W]

    return frames

def nir_motion_blur_augmentation(frames, kernel_size_choices=[3,5], sigma_range=(0.1, 1.5)):
    B, C, T, H, W = frames.shape
    device = frames.device

    kernel_size = random.choice(kernel_size_choices)

    sigma_curve = torch.randn(T)
    sigma_curve = F.avg_pool1d(sigma_curve.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze()
    sigma_curve = (sigma_curve - sigma_curve.min()) / (sigma_curve.max() - sigma_curve.min() + 1e-6)
    sigma_curve = sigma_range[0] + sigma_curve * (sigma_range[1] - sigma_range[0])  # map to [sigma_min, sigma_max]

    frames = frames.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]

    blurred_frames = []
    for b in range(B):
        single_clip = []
        for t in range(T):
            frame = frames[b, t]  # (C, H, W)
            frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            #Apply Gaussian blur with time-varying sigma
            sigma_t = float(sigma_curve[t])

            blurred_np = cv2.GaussianBlur(frame_np, (kernel_size, kernel_size), sigmaX=sigma_t, sigmaY=sigma_t)

            blurred_tensor = torch.from_numpy(blurred_np).permute(2, 0, 1)  # (C, H, W)
            single_clip.append(blurred_tensor)

        single_clip = torch.stack(single_clip, dim=0)  # (T, C, H, W)
        blurred_frames.append(single_clip)

    blurred_frames = torch.stack(blurred_frames, dim=0).to(device)  # (B, T, C, H, W)
    blurred_frames = blurred_frames.permute(0, 2, 1, 3, 4)  # Restore [B, C, T, H, W]

    return blurred_frames


def jpeg_comp(frames, quality_range=(5, 20)):
   
    B, C, T, H, W = frames.shape
    compressed = torch.zeros_like(frames)

    #Sample one quality level for the whole batch
    quality = random.randint(*quality_range)

    for b in range(B):
        for t in range(T):
            frame = frames[b, :, t, :, :]  # [C, H, W]
            frame_np = (frame.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            frame_pil = to_pil_image(torch.from_numpy(frame_np))

            buffer = BytesIO()
            frame_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed_pil = Image.open(buffer).convert("RGB")
            compressed_tensor = torch.from_numpy(np.array(compressed_pil)).permute(2, 0, 1).float() / 255.0

            compressed[b, :, t] = compressed_tensor

    return compressed.to(frames.device)


def periodic_noise(frames, region_size=5, amplitude=60, offset=128, freq_range=(1.0, 2.5)):
    
    B, C, T, H, W = frames.shape
    device = frames.device

    # Random frequency (shared across batch)
    freq = random.uniform(*freq_range)
    t = torch.arange(T, device=device).float()
    signal = torch.sin(2 * np.pi * freq * t / 30.0)  # assuming 30fps
    signal = (signal * amplitude + offset).clamp(0, 255).to(dtype=torch.uint8)  # [T]

    # Region location (top-left corner)
    i_start, j_start = 3, 3
    i_end, j_end = i_start + region_size, j_start + region_size

    # Inject into frames
    frames = frames.clone() 
    for b in range(B):
        for t_idx in range(T):
            for c in range(3):  # R,G,B
                frames[b, c, t_idx, i_start:i_end, j_start:j_end] = signal[t_idx] / 255.0

    return frames.clamp(0, 1)