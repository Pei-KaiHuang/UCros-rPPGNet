import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util import *


class UncertaintyEstimationModule(nn.Module):
    def __init__(self, estimators_rgb, estimators_nir, S=2, K=10, dropout_rate=0.2, fs=30):
        """
        Args:
            estimators_rgb (dict): Dictionary of modules for RGB (decoder1, decoder2, and end).
            estimators_nir (dict): Dictionary of modules for NIR (decoder1, decoder2, and end).
            dropout_rate (float): Dropout probability.
            num_samples (int): Number of Monte Carlo Dropout samples.
        """
        super(UncertaintyEstimationModule, self).__init__()
        self.K = K
        self.S = S
        self.fs = fs  # Sampling frequency
        self.dropout = nn.Dropout3d(p=dropout_rate)
        
        # Use the provided dictionaries for RGB and NIR estimators
        self.rgb_estimators = estimators_rgb
        self.nir_estimators = estimators_nir
        
    def compute_hr_estimates(self, signals, fs, min_hr=40., max_hr=250., zero_pad=100):
        # signals: [K, B, N, T]
        K, B, N, T = signals.shape#[5, 4, 4, 300]

        signals = signals.reshape(K * B * N, T)  # Shape: [K*B*N, T]

        # Remove mean
        signals = signals - signals.mean(dim=1, keepdim=True)

        # Zero-pad signals in time domain to increase frequency resolution
        padded_length = T + zero_pad * T
        signals_padded = torch.nn.functional.pad(signals, (0, zero_pad * T), mode='constant', value=0)

        # Compute FFT
        #fft_vals = torch.fft.rfft(signals, dim=1)  # Shape: [K*B*N, T//2+1] --># Shape: [K*B*N, (T + zero_pad * T)//2 + 1]
        fft_vals = torch.fft.rfft(signals_padded, dim=1)  # Shape: [K*B*N, T//2+1] --># Shape: [K*B*N, (T + zero_pad * T)//2 + 1]

        # Frequencies in BPM
        #freqs = torch.fft.rfftfreq(T, d=1./fs) * 60.  # Shape: [T//2+1]--> # Shape: [(T + zero_pad * T)//2 + 1]
        freqs = torch.fft.rfftfreq(padded_length, d=1./fs) * 60.  # Shape: [(T + zero_pad * T)//2 + 1]

        # Compute power spectrum
        psd = torch.abs(fft_vals) ** 2  # Shape: [K*B*N, T//2+1]--># Shape: [K*B*N, (T + zero_pad * T)//2 + 1]

        # Select frequencies within HR range
        freq_mask = (freqs >= min_hr) & (freqs <= max_hr)
        freqs = freqs[freq_mask]  # Shape: [F]
        psd = psd[:, freq_mask]  # Shape: [K*B*N, F]

        # Move freqs to the same device as psd (or max_indices)
        freqs = freqs.to(psd.device)

        # Find peak frequency
        max_indices = torch.argmax(psd, dim=1)  # Shape: [K*B*N]
        hr_estimates = freqs[max_indices]  # Shape: [K*B*N]
        # Reshape back to [K, B, N]
        hr_estimates = hr_estimates.view(K, B, N)

        return hr_estimates  # Shape: [K, B, N]
    
    def forward(self, x, parity, modality=''):

        # Select estimators based on modality
        with torch.no_grad(): 
            if modality == 'rgb':
                estimators = self.rgb_estimators
            elif modality == 'nir':
                estimators = self.nir_estimators
        
        # Enable dropout layers temporarily, otherwise dropout will not be applied and variance will be zero
        self.dropout.train()

        # Calculate uncertainty signals
        signals_list = []
        B, C, T, H, W = x.size()
        for _ in range(self.K):
            # Apply dropout to generate a feature sample
            x_dropout = self.dropout(x)
            # Decoder pipeline for modality
            sample = self.decode_pipeline(x_dropout, parity, estimators)  # [B, self.S * self.S, 1, T] / [B, N, 1, T]
            signals_list.append(sample)
        
        # Stack over K
        signals = torch.stack(signals_list, dim=0)  # Shape: [K, B, N, 1, T]
        signals = signals.squeeze(3)  # Shape: [K, B, N, T]

        # Compute HR estimates
        hr_estimates = self.compute_hr_estimates(signals, self.fs)  # Shape: [K, B, N] [10, 4, 16]
        #==============================
        # Compute variance over K HR estimates
        variance = torch.var(hr_estimates, dim=0)  # Shape: [B, N]
        weight=variance

        mean = torch.mean(hr_estimates, dim=0)  # Shape: [B, N]
        
        return weight, mean, hr_estimates 

    def decode_pipeline(self, x, parity, estimators):
        """
        Args:
            x (Tensor): Input tensor after dropout, shape (B, C, T, H, W).
            parity (List[int]): Parity values used for padding.
            estimators (dict): Estimators for the given modality.

        Returns:
            sample (Tensor): Decoded tensor, shape [B, 1, T, self.S * self.S].
        """
        B = x.size(0)
        sample = estimators['decoder1'](x)
        sample = F.pad(sample, (0, 0, 0, 0, 0, parity[-1]), mode='replicate')
        sample = F.interpolate(sample, scale_factor=(2, 1, 1))
        sample = estimators['decoder2'](sample)
        sample = F.pad(sample, (0, 0, 0, 0, 0, parity[-2]), mode='replicate')
        sample = estimators['end'](sample)  # Shape: [B, 1, T, 2, 2]
        
        # Extract signals from spatial-temporal blocks
        sample = sample.view(B, 1, -1, self.S * self.S)  # Shape: [B, 1, T, self.S * self.S]
        sample = sample.permute(0, 3, 1, 2)  # Shape: [B, N, 1, T]
        
        return sample
        

class UncertaintyGuidedModule(nn.Module):
    def __init__(self, input_channels=64, spatial_dim=4):
        super(UncertaintyGuidedModule, self).__init__()
        self.spatial_dim = spatial_dim

        # RGB branch
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  
        )

        # NIR branch
        self.nir_branch = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, rgb_x, nir_x):
        B, C, T, S, S = rgb_x.shape
        
        # Aggregate temporal information to focus on spatial confidence estimation
        rgb_x = torch.mean(rgb_x, dim=2)  # Shape: [B, C, S, S]
        nir_x = torch.mean(nir_x, dim=2)  # Shape: [B, C, S, S]

        #Estimate logit confidence maps for each modality
        logits_rgb = self.rgb_branch(rgb_x)  # Shape: [B, 1, S, S]
        logits_nir = self.nir_branch(nir_x)  # Shape: [B, 1, S, S]

        # Stack logits into 2 channels and apply softmax across channel dim
        # This guarantees sum(U_rgb, U_nir) = 1 at each spatial location
        fused_logits = torch.cat([logits_rgb, logits_nir], dim=1)  # [B, 2, S, S]
        confidence_maps = F.softmax(fused_logits, dim=1)  # [B, 2, S, S]

        U_rgb = confidence_maps[:, 0, :, :]  # [B, S, S]
        U_nir = confidence_maps[:, 1, :, :]  # [B, S, S]

        return U_rgb , U_nir