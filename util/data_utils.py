import random
import numpy as np
import os

def load_kfold_splits(train_dataset: str, is_cross: bool, config_dir="./config/"):
    """
    Loads all subjects and returns kfold splits based on is_cross.
    """
    # 1. Parse dataset(s)
    split_datasets = train_dataset.split(",")
    all_subjects = []

    for ds in split_datasets:
        with open(os.path.join(config_dir, f"{ds}_all.txt")) as f:
            subjects = [line.strip() for line in f]
            all_subjects.extend([(ds, s) for s in subjects])

    # 2. Return full set as one fold in cross-dataset setting
    if is_cross:
        kfold_splits = [all_subjects]
        return all_subjects, kfold_splits

    # 3. Intra-dataset k-fold
    kfold_file = os.path.join(config_dir, f"{train_dataset}_kfold.txt")
    with open(kfold_file) as f:
        raw_splits = [line.strip().split() for line in f]  # Each line is a fold's subject IDs

    kfold_splits = []
    for fold_subjects in raw_splits:
        videos_in_fold = []
        for subj in fold_subjects:
            matched_videos = [
                vid for vid in all_subjects
                if vid[1].split('_')[0] == subj  # According to the subject prefix corresponding
            ]
            videos_in_fold.extend(matched_videos)
        kfold_splits.append(videos_in_fold)

    return all_subjects, kfold_splits