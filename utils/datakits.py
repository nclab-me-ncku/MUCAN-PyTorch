import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def get_dataset(data_dir: str, file: str, label_type: str) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load and preprocess a neural dataset.
    """
    data = np.load(os.path.join(data_dir, file), allow_pickle=True)
        
    spikes = list(data['spike'])
    spikes = smooth_binned_spikes(spikes)
    
    labels = list(data[label_type])
    labels = label_preprocessing(labels)
    
    return spikes, labels


def get_dataset_info(data_dir: str, file: str, key: str) -> list:
    """
    Get info from npz file by given key
    """
    data = np.load(os.path.join(data_dir, file), allow_pickle=True)
    info = list(data[key])
    return info


def smooth_binned_spikes(spikes: list[np.ndarray], bin_size=0.05, kernel_SD=0.1) -> list[np.ndarray]:
    """
    Smooths binned spike data using a Gaussian kernel.
    """
    sections = [len(item) for item in spikes]
    indices = [sum(sections[:i]) for i, _ in enumerate(sections, 1)][:-1]
    spike = np.concatenate(spikes)
    smoothed_spike = np.empty_like(spike, dtype=float)
    
    norm = stats.norm(0, kernel_SD)  # normal distribution
    kernel_hl = 3 * int(kernel_SD / bin_size)
    kernel = norm.pdf(np.arange(-kernel_hl, kernel_hl + 1) * bin_size)
    n_sample = len(spike)
    conv_range = slice(kernel_hl, kernel_hl + n_sample)
    nm_factor = np.convolve(kernel, np.ones(n_sample)).T[conv_range]
    
    for i, each in enumerate(spike.T):
        conv = np.convolve(kernel, each)
        smoothed_spike[:, i] = conv[conv_range] / nm_factor    
    return np.split(smoothed_spike, indices)


def label_preprocessing(labels: list[np.ndarray]) -> list[np.ndarray]:
    """
    Preprocess labels by clipping and normalizing each dimension.
    """
    merged = np.concatenate(labels)
    
    # clipping
    outlier = np.mean(merged, axis=0) + 6*np.std(merged, axis=0)
    for i, label in enumerate(labels):
        for j, val in enumerate(outlier):
            labels[i][:, j] = label[:, j].clip(max=val)

    # normalization (min-max)
    baseline = np.percentile(merged, 2, axis=0)
    merged_max = np.percentile(merged - baseline, 90, axis=0)
    for i, label in enumerate(labels):
        labels[i] = (label - baseline) / merged_max
  
    return labels

