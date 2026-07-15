import numpy as np
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt

import torch
import torch.nn.functional as F

def Pearson_np(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    if np.sqrt(np.sum(vx ** 2)) == 0 and np.sqrt(np.sum(vy ** 2)) == 0:
        r = 1
    elif min(np.sqrt(np.sum(vx ** 2)), np.sqrt(np.sum(vy ** 2))) == 0:
        r = 0
    else:
        r = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return r

def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter
    
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    y = filtfilt(b, a, sig)
    return y


def butter_bandpass_batch(sig_list, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter (batch version)
    # signals are in the sig_list

    y_list = []
    
    for sig in sig_list:
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        y_list.append(y)
    return np.array(y_list)


def predict_heart_rate(signal, Fs, min_hr=40., max_hr=250., method='fast_ideal'):

    if method == 'ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        cs = Akima1DInterpolator(freqs, ps)
        max_val = -np.Inf
        interval = 0.1
        min_bound = max(min(freqs), min_hr)
        max_bound = min(max(freqs), max_hr) + interval
        for bpm in np.arange(min_bound, max_bound, interval):
            cur_val = cs(bpm)
            if cur_val > max_val:
                max_val = cur_val
                max_bpm = bpm
        return max_bpm

    elif method == 'fast_ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        if 0 < max_ind < len(ps)-1:
            inds = [-1, 0, 1] + max_ind
            x = ps[inds]
            f = freqs[inds]
            d1 = x[1]-x[0]
            d2 = x[1]-x[2]
            offset = (1 - min(d1,d2)/max(d1,d2)) * (f[1]-f[0])
            if d2 > d1:
                offset *= -1
            max_bpm = f[1] + offset
        elif max_ind == 0:
            x0, x1 = ps[0], ps[1]
            f0, f1 = freqs[0], freqs[1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        elif max_ind == len(ps) - 1:
            x0, x1 = ps[-2], ps[-1]
            f0, f1 = freqs[-2], freqs[-1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        return max_bpm

    elif method == 'fast_ideal_bimodal_filter':
        """ Same as above but check for secondary peak around 1/2 of first
        (to break the tie in case of occasional bimodal PS)
        Note - this may make metrics worse if the power spectrum is relatively flat
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        max_freq = freqs[max_ind]
        max_ps = ps[max_ind]

        # check for a second lower peak at 0.45-0.55f and >50% power
        freqs_valid = np.logical_and(freqs >= max_freq * 0.45, freqs <= max_freq * 0.55)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        if len(freqs) > 0:
            max_ind_lower = np.argmax(ps)
            max_freq_lower = freqs[max_ind_lower]
            max_ps_lower = ps[max_ind_lower]
        else:
            max_ps_lower = 0

        if max_ps_lower / max_ps > 0.50:
            return max_freq_lower
        else:
            return max_freq
    else:
        raise NotImplementedError 
    

def predict_heart_rate_batch(signals, fs, min_hr=40., max_hr=250., method='fast_ideal'):
    return np.array([predict_heart_rate(s, fs, min_hr, max_hr, method) for s in signals])


from scipy.interpolate import Akima1DInterpolator

def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)
        signal = np.pad(signal, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant')
    freqs = np.fft.fftfreq(len(signal), 1 / Fs) * 60  # in bpm
    ps = np.abs(np.fft.fft(signal))**2
    cutoff = len(freqs)//2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps

def reform_data_from_dict(data_dict, flatten=True):
    """
    calculate metrics: reformat predictions and labels from dicts.
    """
    sorted_data = []

    # Sort by video name to ensure the video clips are in the same order
    for video_id in sorted(data_dict.keys()):
        clips = data_dict[video_id]  # This is a list, each element is the rPPG signal of the clip
        clips = [torch.tensor(c) if not isinstance(c, torch.Tensor) else c for c in clips] 
        video_signal = torch.cat(clips, dim=0)  # Merge all clips to form a complete video signal

        if flatten:
            video_signal = video_signal.view(-1).cpu().numpy()
        
        sorted_data.append((video_id, video_signal))

    return sorted_data  # list of (video_id, full video rPPG signal)

def compute_hr(predictions, labels, fps=30):
    hr_pred = predict_heart_rate_batch(predictions, fs=fps, min_hr=40., max_hr=250.)
    hr_gt = predict_heart_rate_batch(labels, fs=fps, min_hr=40., max_hr=250.)

    #print(f"hr_pred: {hr_pred}, hr_gt: {hr_gt}")
    #print(f"hr_pred.shape: {hr_pred.shape}, hr_gt.shape: {hr_gt.shape}")

    mae = np.mean(np.abs(hr_pred - hr_gt))
    rmse = np.sqrt(np.mean((hr_pred - hr_gt) ** 2))
    pearson = np.corrcoef(hr_pred, hr_gt)[0, 1]
    #r=Pearson_np(hr_pred, hr_gt)

    return mae, rmse, pearson #, r