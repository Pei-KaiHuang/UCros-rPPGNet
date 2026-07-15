import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util import *
from Uncertainty_module import UncertaintyEstimationModule, UncertaintyGuidedModule

# -------------------------------------------------------------------------------------------------------------------
# PhysNet fusion model
# 
# the output is an ST-rPPG block rather than a rPPG signal.
# -------------------------------------------------------------------------------------------------------------------
# FeatureExtractor-->start-->loop1-->encoder1-->encoder2-->loop4
def build_encoder_block(in_ch, out_ch, conv3x3x3, downsample=(2, 2, 2)):
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=downsample, stride=downsample, padding=0),
            conv3x3x3(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(out_ch),
            nn.ELU(),
            conv3x3x3(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(out_ch),
            nn.ELU()
        )
class SharedEncoder(nn.Module):
    def __init__(self, in_channels, conv3x3x3=nn.Conv3d):
        super().__init__()

        self.loop1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        self.encoder1 = build_encoder_block(64, 64, conv3x3x3, downsample=(2, 2, 2))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / stds  # Normalize input
        parity = []
        x = self.loop1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x)
        parity.append(x.size(2) % 2)
        #x = self.encoder2(x)
        #x = self.loop4(x)
        #x = F.interpolate(x, scale_factor=(2, 1, 1))
        return x , parity

class Head(nn.Module):
    def __init__(self, conv3x3x3=nn.Conv3d):
        super().__init__()
        self.encoder2 = build_encoder_block(64, 64, conv3x3x3, downsample=(2, 2, 2))
        self.loop4 = build_encoder_block(64, 64, conv3x3x3, downsample=(1, 2, 2))
        

    def forward(self, x):
        x = self.encoder2(x)
        x = self.loop4(x)
        x = F.interpolate(x, scale_factor=(2, 1, 1))
        return x
    
#Estimator-->decoder1-->decoder2-->end
class Estimator(nn.Module):
    def __init__(self, S, conv3x3x3=nn.Conv3d):
        super().__init__()
        self.S = S
        self.decoder1 = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )
    
    def forward(self, x, parity):
        x = self.decoder1(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder2(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-2]), mode='replicate')
        x = self.end(x)

        x_list = []
        for a in range(self.S):
            for b in range(self.S):

                x_list.append(x[:, :, :, a, b])  # Extract signals from ST blocks

        x_avg = sum(x_list) / (self.S * self.S)  # Average signal
        X = torch.cat(x_list + [x_avg], dim=1)  # Concatenate signals
        return X  # Output shape: (B, N, T), where N = S*S + 1
    
        #return torch.cat([x[:, :, :, a, b] for a in range(x.shape[-2]) for b in range(x.shape[-1])], dim=1)
    
class _PhysNetFuse(nn.Module):
    def __init__(self, S=4, in_ch=3, conv3x3x3=nn.Conv3d, pretrain_epoch=0, modality=""):
        super().__init__()

        self.S = S  # Spatial dimension of ST-rPPG block
        self.pretrain_epoch = pretrain_epoch

        self.rppg_head_rgb = Head()
        self.rppg_head_nir = Head()
        self.noise_head = Head()

        self.shared_encoder_rgb = SharedEncoder(in_ch)
        self.shared_encoder_nir = SharedEncoder(in_ch) 

        self.rgb_estimator = Estimator(S=S)
        self.nir_estimator = Estimator(S=S)
        self.fuse_estimator = Estimator(S=S)
        self.uncertainty_estimation = UncertaintyEstimationModule(
            {'decoder1': self.rgb_estimator.decoder1, 'decoder2': self.rgb_estimator.decoder2, 'end': self.rgb_estimator.end},
            {'decoder1': self.nir_estimator.decoder1, 'decoder2': self.nir_estimator.decoder2, 'end': self.nir_estimator.end},
            S=S)
        
        self.uncertainty_module = UncertaintyGuidedModule(input_channels=64, spatial_dim=S)
        
        self.adapter = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=128, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)), 
            nn.BatchNorm3d(128),
            nn.ELU(),
            conv3x3x3(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),  
            nn.BatchNorm3d(128),
            nn.ELU(),
            nn.Dropout3d(p=0.2),
            conv3x3x3(in_channels=128, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)), 
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),  
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

    
    def set_train_stage(self, stage: str):
        """
        stage:
          'pretrain' : θ_F_R , θ_F_N , θ_E_R , θ_E_N
          'nl'       : θ_NL
          'rppg'     : θ_F_R , θ_F_N , θ_E_R , θ_E_N   (同 pretrain，但有 L_nc)
          'fusion'   : θ_F_R , θ_F_N , θ_E_C , θ_E_R , θ_E_N, θ_A
        """
        # 先全部鎖住
        for p in self.parameters():
            p.requires_grad = False

        if stage == 'pretrain':
            modules = [self.shared_encoder_rgb, self.shared_encoder_nir, self.rppg_head_rgb,self.rppg_head_nir,
                       self.rgb_estimator, self.nir_estimator]
        elif stage == 'nl':            # noise-learning 只有 noise_learner
            modules = [self.noise_head]
        elif stage == 'fusion':
            modules = [self.shared_encoder_rgb, self.shared_encoder_nir, self.rppg_head_rgb,self.rppg_head_nir,
                       self.rgb_estimator, self.nir_estimator,
                       self.fuse_estimator,  self.adapter, self.uncertainty_module] 
        else:
            raise ValueError(f"Unknown stage: {stage}")

        for m in modules:
            for p in m.parameters():
                p.requires_grad = True

    def _compute_uncertainty(self, x1, x2, parity, m1='', m2='', temp=2.0):
        """Compute uncertainty """
        Var1, mean_1, hr_pre1 = self.uncertainty_estimation(x1, parity, m1)  # Var1 → [B, N], hr_pre1 → [K, B, N]
        Var2, mean_2, hr_pre2 = self.uncertainty_estimation(x2, parity, m2)
        K, B, N = hr_pre1.shape

        Var1 = Var1.view(B, self.S, self.S)
        Var2 = Var2.view(B, self.S, self.S)

        Var1_flat = Var1.view(B, -1)  # [B, N]
        Var2_flat = Var2.view(B, -1)  # [B, N]

        Variance_combined = torch.stack([Var1_flat, Var2_flat], dim=2)  # [B, N, 2]
        Uncertainty_combined = F.softmax(-Variance_combined / temp, dim=2)  # [B, N, 2]

        U_1 = Uncertainty_combined[:, :, 0].view(B, self.S, self.S)
        U_2 = Uncertainty_combined[:, :, 1].view(B, self.S, self.S)

        return U_1, U_2
    
    def _apply_uncertainty(self, x, U):
        """ Apply uncertainty """
        return x * U.unsqueeze(1).unsqueeze(2)  # Shape: [B, S, S] -> [B, 1, 1, S, S]
    
    def _compute_l1_loss(self, U_generated, U_guided):
        """Compute L1 loss between generated uncertainty maps and guided maps."""
        # [B,S,S] -> [B,N]
        return F.l1_loss(U_generated, U_guided) #/ U_generated.shape[0]  #, reduction='sum'

    def forward(self, rgb_input, nir_input, modality, epoch, rgb_aug_input=None, nir_aug_input=None, rgb_aug_input_jpeg=None, nir_aug_input_jpeg=None, rgb_aug_input_period=None, nir_aug_input_period=None): #, x_rgb_aug, x_nir_aug,
        
        if rgb_input is not None:
            rgb_x, parity_rgb = self.shared_encoder_rgb(rgb_input)
            rgb_x = self.rppg_head_rgb(rgb_x)  # rPPG head

        else:
            rgb_x, parity_rgb = None, None

        if nir_input is not None:
            #nir_x, parity_nir = self.shared_encoder_nir(nir_input)
            nir_x, parity_nir = self.shared_encoder_nir(nir_input)
            nir_x = self.rppg_head_nir(nir_x)  # rPPG head
        else:
            nir_x, parity_nir = None, None

        rgb_aug_x, nir_aug_x, rgb_noise, nir_noise = None, None, None, None
        rgb_noise_jpeg, nir_noise_jpeg = None, None
        rgb_noise_period, nir_noise_period = None, None

        #noise-resistant
        if rgb_aug_input is not None:
            rgb_aug_g_x, _ = self.shared_encoder_rgb(rgb_aug_input)
            rgb_aug_x = self.rppg_head_rgb(rgb_aug_g_x)  # rPPG head
            rgb_noise = self.noise_head(rgb_aug_g_x)  # Noise head

        if nir_aug_input is not None:
            nir_aug_g_x, _ = self.shared_encoder_nir(nir_aug_input)
            nir_aug_x = self.rppg_head_nir(nir_aug_g_x)  # rPPG head
            nir_noise = self.noise_head(nir_aug_g_x)  # Noise head

        if rgb_aug_input_jpeg is not None:
            rgb_aug_g_jpeg, _ = self.shared_encoder_rgb(rgb_aug_input_jpeg)
            rgb_noise_jpeg = self.noise_head(rgb_aug_g_jpeg)

        if nir_aug_input_jpeg is not None:
            nir_aug_g_jpeg, _ = self.shared_encoder_nir(nir_aug_input_jpeg)
            nir_noise_jpeg = self.noise_head(nir_aug_g_jpeg)

        if rgb_aug_input_period is not None:
            rgb_aug_g_period, _ = self.shared_encoder_rgb(rgb_aug_input_period)
            rgb_noise_period = self.noise_head(rgb_aug_g_period)
           
        if nir_aug_input_period is not None:
            nir_aug_g_period, _ = self.shared_encoder_nir(nir_aug_input_period)
            nir_noise_period = self.noise_head(nir_aug_g_period)

        #FUSION SHARE THE SAME PARITY
        parity=parity_rgb if parity_rgb is not None else parity_nir

        F_rgb = rgb_x if rgb_x is not None else None
        F_nir = nir_x if nir_x is not None else None
        F_rgb_aug = rgb_aug_x if rgb_aug_x is not None else None
        F_nir_aug = nir_aug_x if nir_aug_x is not None else None
        N_rgb = rgb_noise if rgb_noise is not None else None
        N_nir = nir_noise if nir_noise is not None else None
        N_rgb_jpeg = rgb_noise_jpeg if rgb_noise_jpeg is not None else None
        N_nir_jpeg = nir_noise_jpeg if nir_noise_jpeg is not None else None
        N_rgb_period = rgb_noise_period if rgb_noise_period is not None else None
        N_nir_period = nir_noise_period if nir_noise_period is not None else None

        loss_map = 0

        U_rgb, U_nir, diff_map_rgb, diff_map_nir, rgb_adapted_nir_x, nir_adapted_rgb_x, map_dif = None, None, None, None, None, None, 0
        rgb_X, nir_X, fuse_X, rgb2nir_X, nir2rgb_X, F_r2n, F_n2r = None, None, None, None, None, None, None
        rgb_aug_X, nir_aug_X = None, None
        if rgb_x is not None:
            rgb_adapted_nir_x = self.adapter(rgb_x)
            F_r2n = rgb_adapted_nir_x
        if nir_x is not None:
            nir_adapted_rgb_x = self.adapter(nir_x)
            F_n2r = nir_adapted_rgb_x

        temperature = 2.0
        #print("rgb_x", rgb_x.shape)#rgb_x torch.Size([4, 64, 150, 4, 4])
        if epoch>=self.pretrain_epoch:
            #==========element-wise addition==========
            if modality=="SFUSE":
                fuse_x = rgb_x + nir_x
                #fusion estimator
                fuse_X = self.fuse_estimator(fuse_x, parity)
            #==========handcraft weighted fusion==========
            elif modality=="UFUSE":
                if rgb_x is not None and nir_x is not None:# we have both modality-> do enhancement
                    if self.training:  
                        U_rgb_g, U_nir_g = self._compute_uncertainty(rgb_x, nir_x, parity, 'rgb', 'nir', temperature)
                    U_rgb, U_nir = self.uncertainty_module(rgb_x, nir_x)
                    #U_nir = 1-U_rgb
                    if self.training:
                        map_dif_r = self._compute_l1_loss(U_rgb, U_rgb_g)
                        map_dif_n = self._compute_l1_loss(U_nir, U_nir_g)
                        loss_map = (map_dif_r + map_dif_n)/2

                    weighted_rgb_x = self._apply_uncertainty(rgb_x, U_rgb)
                    weighted_nir_x = self._apply_uncertainty(nir_x, U_nir)
                    

                if rgb_x is not None and nir_x is None: # only rgb modality
                    #U_rgb, U_nir = self._compute_uncertainty(rgb_x, rgb_adapted_nir_x, parity, 'rgb', 'nir', temperature)
                    U_rgb, U_nir = self.uncertainty_module(rgb_x, rgb_adapted_nir_x)
                    #U_nir = 1-U_rgb

                    weighted_rgb_x = self._apply_uncertainty(rgb_x, U_rgb)
                    weighted_nir_x = self._apply_uncertainty(rgb_adapted_nir_x, U_nir)
                    
                if rgb_x is None and nir_x is not None: # only nir modality
                    #U_rgb, U_nir= self._compute_uncertainty(nir_adapted_rgb_x, nir_x, parity, 'rgb', 'nir', temperature)
                    U_rgb, U_nir = self.uncertainty_module(nir_adapted_rgb_x, nir_x)
                    #U_nir = 1-U_rgb
                        
                    weighted_rgb_x = self._apply_uncertainty(nir_adapted_rgb_x, U_rgb)
                    weighted_nir_x = self._apply_uncertainty(nir_x, U_nir)

                fuse_x = weighted_rgb_x + weighted_nir_x
                #fusion estimator
                fuse_X = self.fuse_estimator(fuse_x, parity)
                
        if rgb_x is not None:
            rgb_X = self.rgb_estimator(rgb_x, parity_rgb)
            rgb2nir_X = self.nir_estimator(rgb_adapted_nir_x, parity)
            
        if nir_x is not None:
            nir_X = self.nir_estimator(nir_x, parity_nir)
            nir2rgb_X = self.rgb_estimator(nir_adapted_rgb_x, parity)

        if rgb_aug_input is not None:
            rgb_aug_X = self.rgb_estimator(rgb_aug_x, parity)

        if nir_aug_input is not None:
            nir_aug_X = self.nir_estimator(nir_aug_x, parity)
            
        # signal  --> rgb, nir, nir2rgb, rgb2nir, fuse
        #return rgb_X, nir_X, fuse_X, nir2rgb_X, rgb2nir_X, F_rgb, F_nir, F_n2r, F_r2n, U_rgb, U_nir, diff_map_rgb, diff_map_nir, map_dif
        return {
                "rgb_X": rgb_X,
                "nir_X": nir_X,
                "fuse_X": fuse_X,
                "nir2rgb_X": nir2rgb_X,
                "rgb2nir_X": rgb2nir_X,
                "F_rgb": F_rgb,
                "F_nir": F_nir,
                "F_n2r": F_n2r,
                "F_r2n": F_r2n,
                "U_rgb": U_rgb,
                "U_nir": U_nir,
                "diff_map_rgb": diff_map_rgb,
                "diff_map_nir": diff_map_nir,
                "map_dif": map_dif,
                "F_rgb_aug": F_rgb_aug,
                "F_nir_aug": F_nir_aug,
                "N_rgb": N_rgb,
                "N_nir": N_nir,
                "N_rgb_compression": N_rgb_jpeg,
                "N_nir_compression": N_nir_jpeg,
                "N_rgb_period": N_rgb_period,
                "N_nir_period": N_nir_period,
                "loss_map": loss_map,
                "rgb_aug_X": rgb_aug_X,
                "nir_aug_X": nir_aug_X,
            }
    
  
class PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3, conv3x3x3=nn.Conv3d, pretrain_epoch="", modality=""):
        super().__init__()
        self.S = S
        #multimodal modality
        self.fuse_branch = _PhysNetFuse(S=S, in_ch=in_ch, conv3x3x3=conv3x3x3, pretrain_epoch=pretrain_epoch, modality=modality)
        
    def set_train_stage(self, stage: str):
        self.fuse_branch.set_train_stage(stage)

    def forward(self, x_rgb, x_nir, modality, epoch, do_not_en=False, x_rgb_aug=None, x_nir_aug=None, x_rgb_aug_jpeg=None, x_nir_aug_jpeg=None, x_rgb_aug_period=None, x_nir_aug_period=None):

        if modality =="SFUSE" or modality =="UFUSE": 
            output = self.fuse_branch(
                                        x_rgb, x_nir, modality, epoch,
                                        rgb_aug_input=x_rgb_aug,
                                        nir_aug_input=x_nir_aug,
                                        rgb_aug_input_jpeg=x_rgb_aug_jpeg,
                                        nir_aug_input_jpeg=x_nir_aug_jpeg,
                                        rgb_aug_input_period=x_rgb_aug_period,
                                        nir_aug_input_period=x_nir_aug_period
                                    )
            return output
        else:
            raise ValueError(f"Unsupported modality: {modality}")

