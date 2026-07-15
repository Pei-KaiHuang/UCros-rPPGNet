import torch
import torch.nn as nn
tr = torch
import torch.nn.functional as F
import numpy as np
import torch.fft

class ContrastLoss(nn.Module):
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super(ContrastLoss, self).__init__()
        self.ST_sampling = ST_sampling(delta_t, K, Fs, high_pass, low_pass) # spatiotemporal sampler
        self.distance_func = nn.MSELoss(reduction = 'mean') # mean squared error for comparing two PSDs
        self.delta_t = delta_t
        self.K = K  # the number of rPPG samples at each spatial position

    def forward(self, X_rPPG, fps_list=None):
        samples = self.ST_sampling(X_rPPG, fps_list)
        batch_size = X_rPPG.size(0)
        sample_counts = [len(s) for s in samples]
        assert len(set(sample_counts)) == 1, "The number of samples in all groups must be the same"
        num_samples_per_group = sample_counts[0]
        total_size = len(samples)

        all_samples = torch.cat([torch.stack(group) for group in samples], dim=0)
        group_ids = torch.arange(total_size, device=all_samples.device).repeat_interleave(num_samples_per_group)
        person_ids = group_ids  # Single modal, group id is person id
        modality_ids = torch.zeros_like(group_ids)  # All samples are considered homomodal

        # Calculate pairwise mse
        diff = all_samples[:, None, :] - all_samples[None, :, :]
        mse = (diff ** 2).mean(dim=2)

        same_person = person_ids[:, None] == person_ids[None, :]
        diff_person = ~same_person
        same_modality = modality_ids[:, None] == modality_ids[None, :]
        diag_mask = ~torch.eye(len(all_samples), dtype=torch.bool, device=all_samples.device)

        # Positive samples (same person, same modality, different sample)
        mask_pos = same_person & same_modality & diag_mask
        # Negative samples (different person)
        mask_neg = diff_person & same_modality  # Single modal, no diff_modality

        eps = 1e-8
        loss_pos = mse[mask_pos].mean() if mask_pos.any() else eps
        loss_neg = -mse[mask_neg].mean() if mask_neg.any() else eps

        total_loss = loss_pos + loss_neg
        return total_loss

    def forward_rp(self, X_orig, X_adapted, fps_list=None):
        B, P, T = X_orig.shape
        assert X_orig.shape == X_adapted.shape

        # Concatenate for one-time ST sampling
        X_all = torch.cat([X_orig, X_adapted], dim=0)  # (2B, P, T)

        # Shared offsets
        shared_offsets = {
            (b, i): torch.randint(0, T - self.delta_t + 1, (1,), device=X_orig.device)
            for b in range(B) for i in range(self.K)
        }

        offsets_dict = {}
        for b in range(B):
            for c in range(P):
                for i in range(self.K):
                    offset = shared_offsets[(b, i)]
                    offsets_dict[(b, c, i)] = offset
                    offsets_dict[(b + B, c, i)] = offset

        # ST sampling
        z_all = self.ST_sampling(X_all, fps_list, offsets_dict=offsets_dict)  # list of (2B, P*K, D)
        z_all = torch.stack([torch.stack(p) for p in z_all])  # (2B, P*K, D)
        
        z_orig = z_all[:B]   # (B, N, D)
        z_adpt = z_all[B:]   # (B, N, D)
        N = z_orig.shape[1]  # N = P*K

        # === Fully pairwise MSE ===
        # Expand z_orig and z_adpt to get all (p,t) vs (p',t') within each b
        z1 = z_orig.unsqueeze(2)  # (B, N, 1, D)
        z2 = z_adpt.unsqueeze(1)  # (B, 1, N, D)
        
        mse = ((z1 - z2) ** 2).mean(dim=-1)  # (B, N, N)
        loss_rp = mse.mean()  # average over all B*N*N

        return loss_rp

    def forward_multi(self, x_rgb, x_nir, x_rgb_aug=None, x_nir_aug=None, fps_list=None):
        B, P, T = x_rgb.shape
        device = x_rgb.device

        # ===== Combine =====
        X_all = torch.cat([x_rgb, x_nir, x_rgb_aug, x_nir_aug], dim=0)  # (4B, P, T)
        M = 4  # now 4 modalities

        # ===== Shared offsets =====
        shared_offsets = {
            (b, i): torch.randint(0, T - self.delta_t + 1, (1,), device=device)
            for b in range(B) for i in range(self.K)
        }
        offsets_dict = {
            (b + m * B, c, i): shared_offsets[(b, i)]
            for m in range(M) for b in range(B) for c in range(P) for i in range(self.K)
        }

        # ===== ST sampling =====
        samples = self.ST_sampling(X_all, fps_list, offsets_dict=offsets_dict)
        all_samples = torch.stack([torch.stack(p) for p in samples])  # (4B, P*K, D)
        z_all = all_samples.view(4, B, P * self.K, -1)  # (4, B, P*K, D)
        z_rgb, z_nir, z_rgb_aug, z_nir_aug = z_all[0], z_all[1], z_all[2], z_all[3]

        # ===== Spatial loss (NIR vs RGB_aug) =====
        loss_spatial = torch.tensor(0., device=device)
        num_spatial = torch.tensor(0, device=device)
        z_nir_sp = z_nir.view(B, self.K, P, -1).permute(0, 1, 2, 3).reshape(B * self.K, P, -1)
        z_rgb_aug_sp = z_rgb_aug.view(B, self.K, P, -1).permute(0, 1, 2, 3).reshape(B * self.K, P, -1)

        diff_n2n = (z_nir_sp.unsqueeze(2) - z_nir_sp.unsqueeze(1)).pow(2).sum(dim=-1)
        diff_n2r = (z_nir_sp.detach().unsqueeze(2) - z_rgb_aug_sp.unsqueeze(1)).pow(2).sum(dim=-1)
        mask = ~torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0).expand(B * self.K, P, P)

        loss_spatial = diff_n2n[mask].sum() + diff_n2r[mask].sum()
        num_spatial = mask.sum() * 2

        # ===== Temporal loss (RGB vs NIR_aug) =====
        loss_temporal = torch.tensor(0., device=device)
        num_temporal = torch.tensor(0, device=device)
        z_rgb_tmp = z_rgb.view(B, P, self.K, -1).permute(0, 1, 3, 2).reshape(B * P, -1, self.K)
        z_nir_aug_tmp = z_nir_aug.view(B, P, self.K, -1).permute(0, 1, 3, 2).reshape(B * P, -1, self.K)

        diff_rgb2rgb = (z_rgb_tmp.unsqueeze(2) - z_rgb_tmp.unsqueeze(3)).pow(2).sum(dim=1)
        diff_rgb2nir = (z_rgb_tmp.detach().unsqueeze(3) - z_nir_aug_tmp.unsqueeze(2)).pow(2).sum(dim=1)

        mask_t = ~torch.eye(self.K, dtype=torch.bool, device=device).unsqueeze(0).expand(B * P, self.K, self.K)
        loss_temporal = diff_rgb2rgb[mask_t].sum() + diff_rgb2nir[mask_t].sum()
        num_temporal = mask_t.sum() * 2

        # ===== ID assignment for HR + dissim loss =====
        z_hr = z_all[:2].reshape(2 * B * P * self.K, -1)
        diff = z_hr[:, None, :] - z_hr[None, :, :]
        mse = (diff ** 2).mean(dim=2)

        person_ids = torch.arange(B, device=device).repeat_interleave(P * self.K).repeat(2)
        diag = ~torch.eye(person_ids.shape[0], dtype=torch.bool, device=device)
        same_person = person_ids[:, None] == person_ids[None, :]
        diff_person = ~same_person

        loss_HR = mse[same_person & diag].sum()
        loss_dissim = -mse[diff_person & diag].sum()
        num_HR = (same_person & diag).sum()
        num_dissim = (diff_person & diag).sum()

        # ===== Final loss =====
        eps = 1e-8
        factor = 0.1
        basic_loss = (loss_HR + loss_dissim) / (num_HR + num_dissim + eps)
        spatial_loss = loss_spatial / (num_spatial + eps)
        temporal_loss = loss_temporal / (num_temporal + eps)
        #print(f"spatial_loss: {spatial_loss.item()}, temporal_loss: {temporal_loss.item()}")
        total_loss = basic_loss + factor * (spatial_loss + temporal_loss)
        return total_loss
    
    def forward_multi_pretrain(self, x_rgb, x_nir, fps_list=None):
        B, P, T = x_rgb.shape
        device = x_rgb.device

        # ===== Combine =====
        X_all = torch.cat([x_rgb, x_nir], dim=0)  # (2B, P, T)
        M = 2  # now 4 modalities

        # ===== Shared offsets =====
        shared_offsets = {
            (b, i): torch.randint(0, T - self.delta_t + 1, (1,), device=device)
            for b in range(B) for i in range(self.K)
        }
        offsets_dict = {
            (b + m * B, c, i): shared_offsets[(b, i)]
            for m in range(M) for b in range(B) for c in range(P) for i in range(self.K)
        }

        # ===== ST sampling =====
        samples = self.ST_sampling(X_all, fps_list, offsets_dict=offsets_dict)
        all_samples = torch.stack([torch.stack(p) for p in samples])  # (4B, P*K, D)
        z_all = all_samples.view(2, B, P * self.K, -1)  # (4, B, P*K, D)


        # ===== ID assignment for HR + dissim loss =====
        z_hr = z_all[:2].reshape(2 * B * P * self.K, -1)
        diff = z_hr[:, None, :] - z_hr[None, :, :]
        mse = (diff ** 2).mean(dim=2)

        person_ids = torch.arange(B, device=device).repeat_interleave(P * self.K).repeat(2)
        diag = ~torch.eye(person_ids.shape[0], dtype=torch.bool, device=device)
        same_person = person_ids[:, None] == person_ids[None, :]
        diff_person = ~same_person

        loss_HR = mse[same_person & diag].sum()
        loss_dissim = -mse[diff_person & diag].sum()
        num_HR = (same_person & diag).sum()
        num_dissim = (diff_person & diag).sum()
        # ===== Final loss =====
        eps = 1e-8
        basic_loss = (loss_HR + loss_dissim) / (num_HR + num_dissim + eps)
        total_loss = basic_loss 
        return total_loss



class ST_sampling(nn.Module):
    # spatiotemporal sampling on ST-rPPG block.
    
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t # time length of each rPPG sample
        self.K = K # the number of rPPG samples at each spatial position
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)

    def forward(self, input, fps_list=None, offsets_dict=None):# input: (2, N, T) [4, 17, 300]
        samples = []
        min_fps = None if fps_list is None else min(fps_list)
        
        for b in range(input.shape[0]): # loop over videos (totally 2 videos) 
            cur_fps = None if fps_list is None else fps_list[b]
            samples_per_video = []
            for c in range(input.shape[1]): # loop for sampling over spatial dimension
                for i in range(self.K): # loop for sampling K samples with time length delta_t along temporal dimension
                    #offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device) # randomly sample along temporal dimension
                    if offsets_dict is not None and (b, c, i) in offsets_dict:
                        offset = offsets_dict[(b, c, i)]
                        #print(f"Using provided offset: {offset} for video {b}, spatial {c}, sample {i}")
                    else:
                        offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device)
                    zero_pad = 0 if min_fps is None else (cur_fps/min_fps)-1.0
                    x = self.norm_psd(input[b, c, offset:offset + self.delta_t], 
                                      zero_pad=zero_pad,
                                      cur_fps=cur_fps)
                    
                    samples_per_video.append(x)
            samples.append(samples_per_video)
        return samples


class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0, cur_fps=None):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        if cur_fps is not None:
            Fn = cur_fps / 2
        else:
            Fn = self.Fs / 2
            
            
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        
        # print(zero_pad, cur_fps, Fn, x.shape)
        
        return x

def get_rPPG(path):

    f = open(path, 'r')
    lines = f.readlines()
    PPG = [float(ppg) for ppg in lines[0].split()]
    # hr = [float(ppg) for ppg in lines[1].split()[:100]]
    # no = [float(ppg) for ppg in lines[2].split()[:100]]
    f.close()

    return PPG

