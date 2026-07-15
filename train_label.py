import torch
from PhysNetModel import PhysNet
from train_loss_sub import ContrastLoss
from loss import *
#from IrrelevantPowerRatio import IrrelevantPowerRatio
import torch.nn
from torch import optim
from util import *
from dataloader import get_loader

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

import logging
from pytz import timezone
from datetime import datetime
import timm
import sys
import torchvision.transforms as T
import os

from einops import rearrange
from torchinfo import summary
import traceback

#================================================================================================
# parameter setting
args = get_args()

train_dataset = args.train_dataset
total_epoch = args.epoch  # total number of epochs for training the model
modality=args.modality
fps=args.fps
seq_len = args.train_T*fps # fs * 10  # temporal dimension of ST-rPPG block, default is 10 seconds.
model_S = args.model_S  # spatial dimension of model
high_pass = args.high_pass
low_pass = args.low_pass
pretrain_epoch=args.pretrain_epoch
preload = args.do_preload
is_cross = args.is_cross
print(f"preload: {preload}")
result_dir = f"/shared/xax1001/Multimodal_TCSVT/multimodal_submit/results/{train_dataset}/{modality}_S{model_S}_T{args.train_T}/"
os.makedirs(f"{result_dir}/weight", exist_ok=True)
alpha = 0.1

#================================================================================================
# Load all subject names
all_subjects, kfold_splits = load_kfold_splits(train_dataset=train_dataset, is_cross=is_cross)
for i, split in enumerate(kfold_splits):
    print(f"i: {i}, kfold_splits[i]: {split} len: {len(split)}")
print(f"all_subjects: {all_subjects}, len: {len(all_subjects)}")

def build_optimizer(model, lr):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

#================================================================================================
# K-fold Training loop
try:
    for fold_idx, split_videos in enumerate(kfold_splits):
        if is_cross:
            train_videos = all_subjects  # Use all subjects for cross-dataset training
        else:
            train_videos = [s for s in all_subjects if s not in split_videos]
            
        #if fold_idx not in [0]:
            #continue
        print(f"Starting Fold {fold_idx+1}/{len(kfold_splits)}: Process videos {train_videos}", len(train_videos))
        """
        intra:
        Starting Fold 1/5: Process videos [('MD_GarSti', 'subject10_garage_still_940'), ('MD_GarSti', 'subject11_garage_still_940'), ('MD_GarSti', 'subject13_garage_still_940'), ('MD_GarSti', 'subject14_garage_still_940'), ('MD_GarSti', 'subject15_garage_still_940'), ('MD_GarSti', 'subject17_garage_still_940'), ('MD_GarSti', 'subject18_garage_still_940'), ('MD_GarSti', 'subject3_garage_still_940'), ('MD_GarSti', 'subject4_garage_still_940'), ('MD_GarSti', 'subject6_garage_still_940'), ('MD_GarSti', 'subject7_garage_still_940'), ('MD_GarSti', 'subject8_garage_still_940'), ('MD_GarSti', 'subject9_garage_still_940'), ('MD_GarSti', 'subject10_garage_still_975')] 14
        cross:
        Starting Fold 1/1: Process videos [('Tokyo', '01'), ('Tokyo', '02'), ('Tokyo', '03'), ('Tokyo', '04'), ('Tokyo', '05'), ('Tokyo', '06'), ('Tokyo', '07'), ('Tokyo', '08'), ('Tokyo', '09'), ('MI', 'Subject1_still_940'), ('MI', 'Subject1_motion_940'), ('MI', 'Subject2_still_940'), ('MI', 'Subject2_motion_940'), ('MI', 'Subject3_still_940'), ('MI', 'Subject3_motion_940'), ('MI', 'Subject4_still_940'), ('MI', 'Subject4_motion_940'), ('MI', 'Subject5_still_940'), ('MI', 'Subject6_still_940'), ('MI', 'Subject6_motion_940'), ('MI', 'Subject7_still_940'), ('MI', 'Subject7_motion_940'), ('MI', 'Subject8_still_940'), ('MI', 'Subject8_motion_940')] 24
        """
        
        # Logger setup
        train_name, _ = get_name(args, modality, model_name=PhysNet.__name__)
        log = get_logger(f"logger/train_label/{train_dataset}/{train_dataset}_fold{fold_idx}", train_name)
        log.info(f"Training configuration: {args} \n")
        log.info(f"Train videos: {train_videos}, \n Train videos: {len(train_videos)}")

        # Load data
        train_loader = get_loader(train_dataset, seq_len, train_videos, batch_size=args.bs, train=True, if_preload=preload)
        #test_loader = get_loader(train_dataset, seq_len, test_videos, batch_size=1, train=False, if_preload=preload)

        # Initialize model
        model_fg = PhysNet(S=args.model_S, in_ch=3, modality=modality, pretrain_epoch=pretrain_epoch).to(device).train()
        opt_fg = optim.AdamW(model_fg.parameters(), lr=args.lr)

        # Calculating the number of model parameters
        total_params = sum(p.numel() for p in model_fg.parameters() if p.requires_grad)
        log.info(f"Total Trainable Parameters: {total_params}")
        # define loss function
        loss_func_rPPG = ContrastLoss(delta_t=seq_len//2, K=4, Fs=args.fps, high_pass=high_pass, low_pass=low_pass)
        adain_loss = AdaINLoss()
        
        noise_learning_loss = NoiseLearningLoss(temperature=0.5)
        noise_resilient_loss = NoiseResilientLoss(temperature=0.5)
        
        # Training loop
        for epoch in range(total_epoch):
            print(f"epoch_train: {epoch}/{total_epoch}:")
            #count=0
            for step, (RGB_frames, NIR_frames, _, subjects) in enumerate(train_loader):
                if NIR_frames.shape[0] < 2:
                    continue

                #(B,C,T,H,W)
                _RGB_frames = RGB_frames.to(device)
                _NIR_frames = NIR_frames.to(device)
                _NIR_frames = _NIR_frames.repeat(1, 3, 1, 1, 1)
                #_ppg_label = ppg.to(device)

                # get augmented data
                _RGB_frames_aug = lighting_variation_augmentation(_RGB_frames.clone(), strength=0.4)
                _NIR_frames_aug = nir_motion_blur_augmentation(_NIR_frames.clone())
                _RGB_frames_aug_jpeg = jpeg_comp(_RGB_frames.clone())
                _NIR_frames_aug_jpeg = jpeg_comp(_NIR_frames.clone())
                _RGB_frames_aug_period = periodic_noise(_RGB_frames.clone())
                _NIR_frames_aug_period = periodic_noise(_NIR_frames.clone())

                _RGB_frames_aug = _RGB_frames_aug.to(device)
                _NIR_frames_aug = _NIR_frames_aug.to(device)
                _RGB_frames_aug_jpeg = _RGB_frames_aug_jpeg.to(device)
                _NIR_frames_aug_jpeg = _NIR_frames_aug_jpeg.to(device)
                _RGB_frames_aug_period = _RGB_frames_aug_period.to(device)
                _NIR_frames_aug_period = _NIR_frames_aug_period.to(device)

                # Assert that the length of _imgs along the temporal axis (dim=2) is seq_len
                assert _RGB_frames.shape[2] == seq_len, f"Expected RGB_frames of length {seq_len}, but got {RGB_frames.shape[2]} instead."
                assert _NIR_frames.shape[2] == seq_len, f"Expected NIR_frames of length {seq_len}, but got {NIR_frames.shape[2]} instead."

                if modality == "UFUSE" or modality == "SFUSE":
                    # ────────────────────────────────────────────────────────────
                    # A.  Pre-train phase  (epoch < pretrain_epoch)
                    # ────────────────────────────────────────────────────────────
                    if epoch < pretrain_epoch: #pretrain_epoch:
                        model_fg.set_train_stage('pretrain')
                        opt_fg = build_optimizer(model_fg, args.lr)
                        out = model_fg(x_rgb=_RGB_frames, x_nir=_NIR_frames, modality=modality, epoch=epoch)   
                        X_rgb, X_nir = out["rgb_X"], out["nir_X"]

                        loss_multi = loss_func_rPPG.forward_multi_pretrain(x_rgb=X_rgb, x_nir=X_nir) #, X_rgb_aug=X_rgb_aug, X_nir_aug=X_nir_aug

                        total_loss = loss_multi

                        opt_fg.zero_grad()
                        total_loss.backward()
                        opt_fg.step()
                        log_message = f"[epoch {epoch} step {step} =pretrain=]" + "".join(
                        f" {name} {value.item():.5f}" for name, value in {
                            "loss_multi": loss_multi,
                        }.items() if value != 0
                        )
                        log.info(log_message)

                    # ────────────────────────────────────────────────────────────
                    # B.  Alternating 3-step update  (epoch ≥ pretrain_epoch)
                    # ────────────────────────────────────────────────────────────
                    else:# epoch >= pretrain_epoch
                        # -------- step 1 : Noise learning (θ_NL) --------
                        model_fg.set_train_stage('nl')
                        opt_fg = build_optimizer(model_fg, args.lr)
                        out = model_fg(x_rgb=_RGB_frames, x_nir=_NIR_frames, modality=modality, epoch=epoch, x_rgb_aug=_RGB_frames_aug, x_nir_aug=_NIR_frames_aug,
                                x_rgb_aug_jpeg=_RGB_frames_aug_jpeg, x_nir_aug_jpeg=_NIR_frames_aug_jpeg, x_rgb_aug_period=_RGB_frames_aug_period, x_nir_aug_period=_NIR_frames_aug_period)    # forward 不用 aug
                                #x_rgb_aug_jpeg=None, x_nir_aug_jpeg=None, x_rgb_aug_period=None, x_nir_aug_period=None)    # forward 不用 aug
                        F_rgb, F_nir, N_rgb, N_nir  = out["F_rgb"], out["F_nir"], out["N_rgb"], out["N_nir"]

                        n_rgb = {
                            'lighting': N_rgb,
                            'motion': out.get("N_rgb_motion", None),
                            'compression': out.get("N_rgb_compression", None),
                            'period': out.get("N_rgb_period", None),
                        }

                        n_nir = {
                            'lighting': out.get("N_nir_lighting", None),
                            'motion': N_nir,
                            'compression': out.get("N_nir_compression", None),
                            'period': out.get("N_nir_period", None),
                        }
                        loss_nl  = noise_learning_loss(n_rgb, n_nir, F_rgb.detach(), F_nir.detach())
                        
                        total_loss = loss_nl * alpha
                        opt_fg.zero_grad()
                        total_loss.backward()
                        opt_fg.step()
                        log_message = f"[epoch {epoch} step {step}] =step1=" + "".join(
                        f" {name} {value.item():.5f}" for name, value in {
                            "loss_nl": loss_nl,
                        }.items() if value != 0
                        )
                        log.info(log_message)
                        # -------- step 3 : Fusion + Adapter  --------
                        model_fg.set_train_stage('fusion')
                        opt_fg = build_optimizer(model_fg, args.lr)
                        out = model_fg(x_rgb=_RGB_frames, x_nir=_NIR_frames, modality=modality, epoch=epoch, x_rgb_aug=_RGB_frames_aug, x_nir_aug=_NIR_frames_aug,
                                x_rgb_aug_jpeg=_RGB_frames_aug_jpeg, x_nir_aug_jpeg=_NIR_frames_aug_jpeg, x_rgb_aug_period=_RGB_frames_aug_period, x_nir_aug_period=_NIR_frames_aug_period) 
                        X_rgb, X_nir, X_fuse = out["rgb_X"], out["nir_X"], out["fuse_X"]
                        X_n2r, X_r2n = out["nir2rgb_X"], out["rgb2nir_X"]
                        F_rgb, F_nir, F_n2r, F_r2n = out["F_rgb"], out["F_nir"], out["F_n2r"], out["F_r2n"]
                        X_rgb_aug, X_nir_aug = out["rgb_aug_X"], out["nir_aug_X"]
                        F_rgb_aug, F_nir_aug, N_rgb, N_nir = out["F_rgb_aug"], out["F_nir_aug"], out["N_rgb"], out["N_nir"]
                        loss_map=out["loss_map"]
                        #print("X_rgb_aug.shape, X_nir_aug.shape", X_rgb_aug.shape, X_nir_aug.shape)
                        n_rgb = {
                            'lighting': N_rgb,
                            'motion': out.get("N_rgb_motion", None),
                            'compression': out.get("N_rgb_compression", None),
                            'period': out.get("N_rgb_period", None),
                        }

                        n_nir = {
                            'lighting': out.get("N_nir_lighting", None),
                            'motion': N_nir,
                            'compression': out.get("N_nir_compression", None),
                            'period': out.get("N_nir_period", None),
                        }
                        # loss_nr  = noise_resilient_loss(F_rgb, F_nir, F_rgb_aug, F_nir_aug, N_rgb, N_nir)
                        n_rgb_detach = {k: v.detach() if v is not None else None for k, v in n_rgb.items()}
                        n_nir_detach = {k: v.detach() if v is not None else None for k, v in n_nir.items()}
                        loss_nr  = noise_resilient_loss(F_rgb, F_nir, F_rgb_aug, F_nir_aug, n_rgb_detach, n_nir_detach)
                        
                        loss_multi, loss_con_n2r, loss_con_r2n = 0, 0, 0
                        loss_multi = loss_func_rPPG.forward_multi(x_rgb=X_rgb, x_nir=X_nir, x_rgb_aug=X_rgb_aug, x_nir_aug=X_nir_aug)
                        # loss_multi = loss_func_rPPG.forward_multi(x_rgb=X_rgb, x_nir=X_nir, x_rgb_aug=X_rgb, x_nir_aug=X_nir)

                        loss_fuse=loss_func_rPPG(X_fuse)
                        loss_con_n2r = loss_func_rPPG.forward_rp(X_rgb.detach(), X_n2r)
                        loss_con_r2n = loss_func_rPPG.forward_rp(X_nir.detach(), X_r2n)
                        
                        #loss_rgb_n2r = 1 - F.cosine_similarity(F_rgb.view(F_rgb.shape[0], -1), F_n2r.view(F_n2r.shape[0], -1), dim=1).mean()
                        #loss_nir_r2n = 1 - F.cosine_similarity(F_nir.view(F_nir.shape[0], -1), F_r2n.view(F_r2n.shape[0], -1), dim=1).mean()
                        loss_rgb_n2r = adain_loss(F_rgb.detach().view(F_rgb.shape[0], -1), F_n2r.view(F_n2r.shape[0], -1))
                        loss_nir_r2n = adain_loss(F_nir.detach().view(F_nir.shape[0], -1), F_r2n.view(F_r2n.shape[0], -1))

                        total_loss = loss_multi  + loss_con_n2r + loss_con_r2n + loss_rgb_n2r + loss_nir_r2n + loss_fuse + loss_map + loss_nr * alpha

                        opt_fg.zero_grad()
                        total_loss.backward()
                        opt_fg.step()

                        log_message = f"[epoch {epoch} step {step}] =fusion=" + "".join(
                        f" {name} {value.item():.5f}" for name, value in {
                            "loss_multi": loss_multi , "loss_fuse": loss_fuse,
                            "loss_con_n2r": loss_con_n2r, "loss_con_r2n": loss_con_r2n,
                            "loss_rgb_n2r": loss_rgb_n2r, "loss_nir_r2n": loss_nir_r2n,
                            "loss_map": loss_map,
                            "loss_nr": loss_nr,
                        }.items() if value != 0
                        )
                        log.info(log_message)
            torch.save(model_fg.state_dict(), f"{result_dir}/weight/fold{fold_idx}_epoch{epoch:03d}.pt")
except Exception as e:
    err_msg = traceback.format_exc()
    if 'log' in locals():
        log.error("An exception occurred:\n" + err_msg)
    else:
        print("An exception occurred:\n" + err_msg)
    sys.exit(1)