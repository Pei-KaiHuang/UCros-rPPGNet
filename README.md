# UCros-rPPGNet

Official PyTorch implementation of **UCros-rPPGNet: Unsupervised Cross-Modal rPPG Estimation via Noise-Resistant Learning, Reliability-Aware Fusion, and Cross-Spectral Translation**.

UCros-rPPGNet is a fully unsupervised framework for estimating remote photoplethysmography (rPPG) signals from paired RGB and near-infrared (NIR) facial videos. It combines:

- Noise-resistant feature learning
- Reliability-aware RGB-NIR fusion
- Cross-spectral translation for missing-modality inference

The paper also introduces the **DG-CMrPPG** benchmark, which evaluates intra-domain and cross-domain performance on MR-NIRP-Car, MR-NIRP-Indoor, and TokyoTech Remote PPG.

## Requirements

- Python 3.9+
- PyTorch and torchvision
- OpenCV, NumPy, SciPy, Pillow, timm, einops, tqdm, matplotlib, pytz, torchinfo, and torchprofile
- An NVIDIA GPU is recommended

Install the main dependencies with:

```bash
pip install torch torchvision opencv-python numpy scipy pillow timm einops tqdm matplotlib pytz torchinfo torchprofile
```

## Data preparation

Download the datasets from their official sources and preprocess each video into the following structure:

```text
dataset_root/
  RGB_crop/<video_name>/*.png
  NIR_crop/<video_name>/*.png
  GT/<video_name>/ground_truth.txt
```

Update the dataset paths in `dataloader.py`. Also update the checkpoint/output paths in `train_label.py` and `test_label.py` for your environment.

## Training

Example for MR-NIRP-Indoor:

```bash
python train_label.py \
  --train_dataset MI \
  --bs 2 \
  --epoch 60 \
  --modality UFUSE \
  --model_S 4 \
  --lr 1e-4 \
  --train_T 10 \
  --pretrain_epoch 15 \
  --do_preload
```

To run the experiment script:

```bash
bash train_all.sh
```

## Evaluation

```bash
python test_label.py \
  --train_dataset MI \
  --test_dataset MI \
  --bs 1 \
  --epoch 60 \
  --modality UFUSE \
  --model_S 4 \
  --train_T 10 \
  --test_T 10 \
  --pretrain_epoch 15 \
  --do_preload
```

Add `--is_cross` for cross-dataset experiments.

## Citation

If this work is useful for your research, please cite the paper. The BibTeX entry will be added after publication.
