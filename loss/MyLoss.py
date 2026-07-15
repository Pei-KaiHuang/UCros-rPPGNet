import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class AdaINLoss(nn.Module):
    def __init__(self):
        super(AdaINLoss, self).__init__()
        return

    def forward(self, x, y):
        # Calculate the mean and standard deviation
        mu_x = torch.mean(x, dim=1, keepdim=True)
        mu_y = torch.mean(y, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True)
        std_y = torch.std(y, dim=1, keepdim=True)
        
        mean_loss = torch.mean((mu_x - mu_y) ** 2)
        std_loss = torch.mean((std_x - std_y) ** 2)
        
        return mean_loss + std_loss

class NoiseLearningLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, n_rgb: dict, n_nir: dict, f_rgb: torch.Tensor, f_nir: torch.Tensor):
        """
        Perform positive/negative sampling according to the noise type.
        """

        B = f_rgb.size(0)
        # Flatten to [B, D]
        def flatten_dict(n):
            return {k: n[k].view(B, -1) for k in n if n[k] is not None}
        
        n_rgb_flat = flatten_dict(n_rgb)
        n_nir_flat = flatten_dict(n_nir)

        # Merge modalities, same noise type
        n_all = {}
        for k in set(n_rgb_flat) | set(n_nir_flat):
            rgb = n_rgb_flat.get(k, None)
            nir = n_nir_flat.get(k, None)
            if rgb is not None and nir is not None:
                n_all[k] = torch.cat([rgb, nir], dim=0)
            elif rgb is not None:
                n_all[k] = rgb
            else:
                n_all[k] = nir

        # Merge all rPPG features to [2B, D]
        f_all = torch.cat([f_rgb.view(B, -1), f_nir.view(B, -1)], dim=0)

        device = f_all.device

        noise_types = list(n_all.keys())
        loss = 0

        for t in noise_types:
            n_t = n_all[t]  
            N = n_t.size(0)

            # === Positive samples: same noise type, different samples ===
            sim_matrix = F.cosine_similarity(n_t.unsqueeze(1), n_t.unsqueeze(0), dim=-1)  
            exp_sim = torch.exp(sim_matrix / self.temperature)
            mask = ~torch.eye(N, dtype=torch.bool, device=device)
            pos = exp_sim.masked_select(mask).view(N, N - 1).sum(dim=1)  

            # === Negative samples ===
            neg = torch.zeros(N, device=device)

            # 1. Different noise types
            for t_neg in noise_types:
                if t_neg == t:
                    continue
                n_neg = n_all[t_neg]  
                sim_neg = F.cosine_similarity(n_t.unsqueeze(1), n_neg.unsqueeze(0), dim=-1) 
                neg += torch.exp(sim_neg / self.temperature).sum(dim=1)

            # 2. rPPG features
            sim_nf = F.cosine_similarity(n_t.unsqueeze(1), f_all.unsqueeze(0), dim=-1) 
            neg += torch.exp(sim_nf / self.temperature).sum(dim=1)

            # === Loss ===
            loss += (-torch.log(pos / (neg + 1e-8))).mean()

        return loss / (len(noise_types) + 1e-8)
    
    
class NoiseResilientLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, f_rgb, f_nir, f_rgb_aug, f_nir_aug, n_rgb: dict, n_nir: dict):
        B = f_rgb.size(0)
        f_rgb = f_rgb.view(B, -1)
        f_nir = f_nir.view(B, -1)
        f_rgb_aug = f_rgb_aug.view(B, -1)
        f_nir_aug = f_nir_aug.view(B, -1)

        D = f_rgb.size(1)
        device = f_rgb.device

        # Filter out None noise, flatten and concatenate
        valid_rgb_noise = [n.view(B, -1) for n in n_rgb.values() if n is not None]
        valid_nir_noise = [n.view(B, -1) for n in n_nir.values() if n is not None]

        n_rgb_all = torch.cat(valid_rgb_noise, dim=0) if valid_rgb_noise else torch.empty(0, D).to(device)
        n_nir_all = torch.cat(valid_nir_noise, dim=0) if valid_nir_noise else torch.empty(0, D).to(device)

        n_all = torch.cat([n_rgb_all, n_nir_all], dim=0)  # shape: [?, D]

        loss = 0
        for f, f_aug in [(f_rgb, f_rgb_aug), (f_nir, f_nir_aug)]:
            pos = torch.exp(F.cosine_similarity(f, f_aug, dim=-1) / self.temperature)
            
            sim_all = F.cosine_similarity(f.unsqueeze(1), n_all.unsqueeze(0), dim=-1) 
            neg = torch.exp(sim_all / self.temperature).sum(dim=1)
            loss += (-torch.log(pos / (neg + 1e-8))).mean()

        return loss

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()
        return

    def forward(self, x, y):
        # for i in range(x.shape[0]):
        vx = x - torch.mean(x, dim = 1, keepdim = True)
        vy = y - torch.mean(y, dim = 1, keepdim = True)
        r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        cost = 1 - r
        return cost

    
class Cos_Sim_loss(nn.Module):
    
    def __init__(self):
        super(Cos_Sim_loss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output, label):
        return 1 - torch.mean(self.cos(output, label))