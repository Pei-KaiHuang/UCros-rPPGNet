from __future__ import print_function, division
import os
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms
import glob
import random

from io import BytesIO

import argparse
from util import *
from tqdm import tqdm
import threading
import multiprocessing

import matplotlib.pyplot as plt


def process_images(paths, transform_face, gray, results, index):
    """Processes a batch of images and stores the results in the given list."""
    frames = []

    for path in paths:
        try:
            frame = Image.open(path)
            if gray:
                frame = frame.convert('L')
            processed_frame = transform_face(frame)
            frames.append(processed_frame)
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}: {e}")

    results[index] = frames

def preload_frames_multithreaded(paths, transform_face, gray=False, num_threads=8):
    '''Loads and processes frames using multiple threads.'''
    thread_list = []
    results = [None] * num_threads  # List to store results from each thread
    n = len(paths)
    images_per_thread = n // num_threads

    for i in range(num_threads):
        start_index = i * images_per_thread
        end_index = start_index + images_per_thread
        if i == num_threads - 1:  # Handle the last batch
            end_index = n

        thread = threading.Thread(
            target=process_images,
            args=(paths[start_index:end_index], transform_face, gray, results, i)
        )
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    # Combine the results from all threads
    all_frames = []

    for frames in results:
        all_frames.extend(frames)

    return all_frames


args = get_args()

def get_rPPG(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    f = open(path, 'r')
    lines = f.readlines()
    PPG = [float(ppg) for ppg in lines[0].split()]
    f.close()

    return PPG


class rPPG_Dataset(Dataset):
    def __init__(self, datasets, seq_length, process_videos, train, RGB_path, NIR_path, if_preload=False, data_version='Default'):

        self.train = train
        self.RGB_path = RGB_path
        self.NIR_path = NIR_path
        datasets = datasets.split(',')
        self.datasets = datasets
        self.data_version = data_version
        self.if_preload = if_preload
        prefix_root="/shared/xax1001"
        self.preload_root = {
            "Default": f"{prefix_root}/Multimodal/Default",
            "JPEG": f"{prefix_root}/Multimodal/JPEG",
            "Periodic": f"{prefix_root}/Multimodal/Periodic",
        }[self.data_version]
        
        NIR_face_folder = "NIR_crop"
        RGB_face_folder = "RGB_crop"
        GT_folder = "GT"
        self.root_dir = {
            "Default": {
                "MD_DriSti" : f"/shared/rPPG_dataset/MR-NIRP_Car",
                "MD_GarSti" : f"/shared/rPPG_dataset/MR-NIRP_Car",
                "MD_DriSma" : f"/shared/rPPG_dataset/MR-NIRP_Car",
                "MD_GarSma" : f"/shared/rPPG_dataset/MR-NIRP_Car",
                "VIPL" : f"/shared/rPPG_dataset/VIPL_multimodal",
                # for cross-dataset training
                "MI" : f"/shared/rPPG_dataset/MR-NIRP_indoor",
                "Tokyo" : f"/shared/rPPG_dataset/Tokyo",
                "MD" : f"/shared/rPPG_dataset/MR-NIRP_Car",
                
            },
            "JPEG": {
                "MI": "/shared/rPPG_dataset/MR-NIRP_indoor_JPEG",
                
            },
            "Periodic": {
                "MI": "/shared/rPPG_dataset/MR-NIRP_indoor_Periodic",
                
            }
        }[self.data_version]

        if self.train:
            #_h, _w = 64, 64
            _h, _w = 64, 64
        else:
            _h, _w = 64, 64

        self.transform_face = transforms.Compose([
            transforms.Resize((_h, _w)),
            transforms.ToTensor()
        ])
        self.seq_length = seq_length
        
        self.video_data = {}  # Store RGB, NIR, PPG signals for each video
        self.video_keys = []  # Save the video ID for easy indexing
        self.index_map = []   # The precomputed index corresponds to {idx: (video_id, clip_index)}
        self.bad_sample = 0
        self.good_sample = 0
        
        # load the dataset
        #total_clips = 0
        for key, video_name in tqdm(process_videos):
            assert key in self.root_dir, f"{key} not in root_dir"       

            def get_files(path, face_folder, subj):
                path=os.path.join(self.root_dir[key], face_folder, subj)
                files = []
                for ext in ('*.png', '*.jpg'):
                    files.extend(sorted(glob.glob(os.path.join(path, ext))))
                return files
            
            RGB_image_paths = get_files(self.root_dir[key], RGB_face_folder, video_name)
            NIR_image_paths = get_files(self.root_dir[key], NIR_face_folder, video_name)
            
            _key = f"{key}_{video_name}"
    
            ground_truth = os.path.join(self.root_dir[key], GT_folder, video_name, "ground_truth.txt")
            gt_signal = get_rPPG(ground_truth)

           
            min_length = min(len(RGB_image_paths), len(NIR_image_paths), len(gt_signal))
            
            RGB_image_paths = RGB_image_paths[:min_length]
            NIR_image_paths = NIR_image_paths[:min_length]
            
            if self.if_preload:
                # Preload images and store them into the dictionary
                #prefix = "/shared/xax1001/paper2/preprocessed_data_clips"
                #prefix = "/shared/xax1001/paper2/preprocessed_data_clips_0423_all/"
                prefix = self.preload_root
                if not os.path.exists(f'{prefix}/{_key}_RGB_fg.pt'):
                    RGBimages = preload_frames_multithreaded(RGB_image_paths, self.transform_face, gray=False)
                    os.makedirs(prefix, exist_ok=True)
                    torch.save(RGBimages, f'{prefix}/{_key}_RGB_fg.pt')
                else:
                    RGBimages = torch.load(f'{prefix}/{_key}_RGB_fg.pt')

                if not os.path.exists(f'{prefix}/{_key}_NIR_fg.pt'):
                    NIRimages = preload_frames_multithreaded(NIR_image_paths, self.transform_face, gray=True)
                    os.makedirs(prefix, exist_ok=True)
                    torch.save(NIRimages, f'{prefix}/{_key}_NIR_fg.pt')
                else:
                    NIRimages = torch.load(f'{prefix}/{_key}_NIR_fg.pt')
            else:
                RGBimages = RGB_image_paths
                NIRimages = NIR_image_paths

            # Clip-based segmentation
            clips = []
            stride = self.seq_length
            for start in range(0, min_length - (self.seq_length + 30), stride):
                # Following Gideon et al. (2021), who noted the presence of long stretches of zero values in the PPG signal, we excluded these non-informative regions to ensure data quality and model robustness.
                gt_clip = torch.FloatTensor(gt_signal[start: start + self.seq_length]).unsqueeze(0)  # shape: [1, T]
                irr_ratio = IrrelevantPowerRatio(Fs=30, high_pass=40, low_pass=250)(gt_clip, None)  # shape: [1]
                if irr_ratio.item() >= 0.90: # 0.85 bad:527/good:597 | 0.90 bad:29 good:1095
                    self.bad_sample += 1
                    continue
                clips.append((start, start + self.seq_length))
                self.good_sample += 1

            self.video_data[_key] = {
                "RGB": RGBimages,
                "NIR": NIRimages,
                "GT": gt_signal,
                "clips": clips
            }
            self.video_keys.append(_key)

            for clip_idx in range(len(clips)):
                self.index_map.append((_key, clip_idx))

    def __getitem__(self, idx):
        """
        Returns:
            RGB_clip: [C, T, H, W]
            NIR_clip: [C, T, H, W]
            GT_clip: [T]
            video_id: str   #MD_DriSti_subject12_driving_still_940
        """
        
        # Find the corresponding video ID and clip number directly from index_map
        #video_id, clip_idx = self.index_map[idx]
        if self.train:
            # Corresponding video_id
            video_id = self.video_keys[idx % len(self.video_keys)]
            clips = self.video_data[video_id]["clips"]
            if not clips:
                raise RuntimeError(f"No valid clips for video {video_id}")
            clip_idx = random.randint(0, len(clips) - 1)
        else:
            video_id, clip_idx = self.index_map[idx]
        
        start, end = self.video_data[video_id]["clips"][clip_idx]

        #print(f"{idx=}, {video_id=}, {clip_idx=}, {start=}, {end=}")

        if clip_idx >= len(self.video_data[video_id]["clips"]):
            raise IndexError(f"clip_idx {clip_idx} out of range for video {video_id}")

        if self.if_preload:
            RGB_clip = self.video_data[video_id]["RGB"][start:end]
            NIR_clip = self.video_data[video_id]["NIR"][start:end]
        else:
            RGB_clip = preload_frames_multithreaded(self.video_data[video_id]["RGB"][start:end], self.transform_face, gray=False)
            NIR_clip = preload_frames_multithreaded(self.video_data[video_id]["NIR"][start:end], self.transform_face, gray=True)
        
        GT_clip = torch.FloatTensor(self.video_data[video_id]["GT"][start:end])

        _RGB_face_frame = torch.stack(RGB_clip).transpose(0, 1)
        _NIR_face_frame = torch.stack(NIR_clip).transpose(0, 1)

        # Z-score normalization for PPG
        mean_ppg = GT_clip.mean()
        std_ppg = GT_clip.std()
        GT_clip = (GT_clip - mean_ppg) / (std_ppg + 1e-8)

        return _RGB_face_frame, _NIR_face_frame, GT_clip, video_id

    def __len__(self):
        if self.train:
            return len(self.video_keys)
        else:
            return sum(len(self.video_data[k]["clips"]) for k in self.video_keys)
    
    
def get_loader(_datasets, _seq_length, process_videos, batch_size=1, shuffle=True, train=True, RGB_path=True, NIR_path=True, if_preload=False, data_version='Default'):#Periodic, JPEG
    
    _dataset = rPPG_Dataset(datasets=_datasets, 
                            seq_length=_seq_length,
                            process_videos=process_videos,
                            train=train,
                            RGB_path=RGB_path,
                            NIR_path=NIR_path,
                            if_preload=if_preload,
                            data_version=data_version)

    # num_id, num_domain = _dataset.get_id_domain_num()
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)



if __name__ == "__main__":
    
    from einops import rearrange
    import numpy as np
    
    def saveFrames(frames):
        
        os.makedirs("test", exist_ok=True)
        # save tensor to images
        for i in range(frames.shape[2]):
            img = frames[0, :, i, :, :]
            # print(img.shape)
            # print(torch.max(img), torch.min(img))
            
            img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
            img = (img*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f"test/{i}.jpg", img)

    
    train_loader = get_loader(_datasets=list("UPC"),
                              _seq_length=10*30,
                              batch_size=3,
                              train=True)
    test_loader = get_loader(_datasets=list("UPC"),
                              _seq_length=10*30,
                              batch_size=1,
                              train=False)

    print(f"{len(train_loader)=}")
    for step, (face_frames, bg_frames, ppg_label, subjects) in enumerate(train_loader):
        print(f"{face_frames.shape=}")
        print(f"{bg_frames.shape=}")
        print(f"{ppg_label.shape=}")
        print(subjects)
        # saveFrames(face_frames)
        break
    
    print(f"{len(test_loader)=}")
    for step, (face_frames, bg_frames, ppg_label, subjects) in enumerate(test_loader):
        print(f"{face_frames.shape=}")
        print(f"{ppg_label.shape=}")
        print(subjects)
        # saveFrames(face_frames)
        break