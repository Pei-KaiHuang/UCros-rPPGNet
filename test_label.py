import torch
from PhysNetModel import PhysNet
#from train_loss_sub import ContrastLoss
from loss import *
#from IrrelevantPowerRatio import IrrelevantPowerRatio
from torchprofile import profile_macs  # FLOPs
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
import shutil

from collections import defaultdict

# Calaulate FLOPs
class FLOPsWrapper(nn.Module):
    def __init__(self, model, modality, epoch):
        super().__init__()
        self.model = model
        self.modality = modality
        self.epoch = epoch

    def forward(self, *inputs):  
        x_rgb, x_nir = inputs     
        return self.model(x_rgb, x_nir, modality=self.modality, epoch=self.epoch)
    
# parameter setting
args = get_args()

train_dataset = args.train_dataset
total_epoch = args.epoch  # total number of epochs for training the model
modality=args.modality
fps=args.fps
is_cross = args.is_cross
model_S = args.model_S  # spatial dimension of model
high_pass = args.high_pass
low_pass = args.low_pass
pretrain_epoch=args.pretrain_epoch
preload = args.do_preload
print(f"preload: {preload}")

test_dataset=args.test_dataset
batch_size = args.bs
seq_len = args.test_T*fps # fs * 10  # temporal dimension of ST-rPPG block, default is 10 seconds.
# path
model_path = f"/shared/xax1001/Multimodal_TCSVT/multimodal_submit/results/{train_dataset}/{modality}_S{model_S}_T{args.train_T}/weight/"

#================================================================================================
# Load all subject names
all_subjects, kfold_splits = load_kfold_splits(test_dataset, is_cross=is_cross)
#================================================================================================
test_epoch = pretrain_epoch
#================================================================================================
scenario = "BOTH"
trainName, testName = get_name(args, modality, model_name=PhysNet.__name__)
all_subjects_log = get_logger(f"logger/test_label_FUSE/{test_dataset}/{testName}", f"{test_dataset}_all")
all_subjects_log.info("testName: " + testName)
all_subjects_log .info("Testing configuration:\n" + "\n".join([f"  {key}: {value}" for key, value in vars(args).items()]))
all_subjects_log.info(f"=================={scenario}==================")
# Storage for all subjects' best performances
all_best_mae, all_best_rmse, all_best_R = [], [], []
#================================================================================================

for fold_idx, split_videos in enumerate(kfold_splits):
    #if fold_idx != 3:
        #continue
    if is_cross:
        test_videos = all_subjects  # Use all subjects for cross-dataset training
    else:
        test_videos = [s for s in all_subjects if s in split_videos]
    
    print(f"Starting Fold {fold_idx+1}/{len(kfold_splits)}: Process videos {split_videos}")
    
    # Logger setup
    
    log = get_logger(f"logger/test_label_FUSE/{test_dataset}/{testName}", f"{test_dataset}_fold{fold_idx}")
    log.info(f"=================={scenario}==================")
    
    log.info(f"Testing configuration: {args}\n")
    log.info(f"Test videos: {test_videos}, \n length: {len(test_videos)}")
    
    # Load data
    test_loader = get_loader(test_dataset, seq_len, test_videos, batch_size=args.bs, shuffle=False, train=False, if_preload=preload)
    
    # Initialize model
    model_fg = PhysNet(S=model_S, in_ch=3, modality=modality, pretrain_epoch=pretrain_epoch).to(device).eval()

    total_params = sum(p.numel() for p in model_fg.parameters() if p.requires_grad)
    log.info(f"Total Trainable Parameters: {total_params:,}")
    # FLOPs
    wrapped_model = FLOPsWrapper(model_fg, modality="UFUSE", epoch=40)

    dummy_rgb = torch.randn(1, 3, 300, 64, 64).to("cuda")
    dummy_nir = torch.randn(1, 3, 300, 64, 64).to("cuda")
    flops = profile_macs(wrapped_model, (dummy_rgb, dummy_nir))
    log.info(f"Estimated FLOPs: {flops / 1e9:.2f} GFLOPs")

    # **testing**
    with torch.no_grad():
        best_mae, best_rmse, best_R, best_epoch = float('inf'), float('inf'), float('-inf'), float('inf')

        for epoch in range(test_epoch, total_epoch):
            model_state = torch.load(model_path + f'fold{fold_idx}_epoch{epoch:03d}.pt', map_location=device)
            model_fg.load_state_dict(model_state)

            all_hr_predicts, all_hr_labels = [], []
            all_video_predictions = defaultdict(list)
            all_video_gt = defaultdict(list)

            for step, (RGB_frames, NIR_frames, ppg_label, video_id) in enumerate(test_loader):
                RGB_frames, NIR_frames = RGB_frames.to(device), NIR_frames.to(device)

                NIR_frames = NIR_frames.repeat(1, 3, 1, 1, 1)
                hr_gt=predict_heart_rate_batch(ppg_label.numpy(), fs=fps)
                hr_gt = torch.tensor(hr_gt, dtype=torch.float32).to(device)
                
                if modality in ["SFUSE", "UFUSE"]:
                    out = model_fg(x_rgb=RGB_frames, x_nir=NIR_frames, modality=modality, epoch=epoch)
                    X_fuse = out['fuse_X']
                    rppg = X_fuse[:, -1]
 
                rppg = rppg.detach().cpu().numpy()
                rppg = butter_bandpass_batch(rppg, lowcut=0.6, highcut=3, fs=fps)
                ppg_label = butter_bandpass_batch(ppg_label.cpu().numpy(), lowcut=0.6, highcut=3, fs=fps)

                # Save the clip to the corresponding video
                all_video_predictions[video_id].append(rppg)
                all_video_gt[video_id].append(ppg_label)

            all_pred_signals = []
            all_gt_signals = []
            # Convert all video predictions and GT data into complete time series signals
            for video_id in all_video_predictions:
                all_pred_signals.extend([np.array(clip).reshape(-1) for clip in all_video_predictions[video_id]])
                all_gt_signals.extend([np.array(clip).reshape(-1) for clip in all_video_gt[video_id]])

            log.info(f"Total number of videos (GT signals): {len(all_gt_signals)}")

            # **Compute HR evaluation metrics at once**
            mae, rmse, pearson = compute_hr(all_pred_signals, all_gt_signals, fps=fps)
            log.info(f"[Epoch {epoch}] Overall MAE: {mae:.5f}, RMSE: {rmse:.5f}, pearson: {pearson:.5f}")

            # Update best indicators
            if mae < best_mae:
                best_mae, best_rmse, best_R, best_epoch = mae, rmse, pearson, epoch

        log.info(f"[Best] epoch {best_epoch} - MAE: {best_mae:.5f}, RMSE: {best_rmse:.5f}, pearson: {best_R:.5f}")
        all_subjects_log.info(f"[Best fold{fold_idx}] epoch {best_epoch} - MAE: {best_mae:.5f}, RMSE: {best_rmse:.5f}, pearson: {best_R:.5f}")
        all_best_mae.append(best_mae)
        all_best_rmse.append(best_rmse)
        all_best_R.append(best_R)

# Calculate the overall average indicators
avg_mae, avg_rmse, avg_R = np.mean(all_best_mae), np.mean(all_best_rmse), np.mean(all_best_R)
all_subjects_log.info(f"[Overall] Avg Best - MAE: {avg_mae:.5f}, RMSE: {avg_rmse:.5f}, pearson: {avg_R:.5f}")