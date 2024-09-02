import numpy as np
import pandas as pd
import torch
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader
import torchaudio
import os

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_data_path, trans_type, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_data_path = audio_data_path
        self.device = device
        self.trans_type = trans_type
        self.transformation = transformation.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        if(self.trans_type == "mel_spec"):
            mel_spec = self.transformation(signal)
            final = torchaudio.transforms.AmplitudeToDB(top_db = 80)(mel_spec)
        else: 
            ggram = self.transformation(signal)
            ggram_rms = torch.sqrt(ggram**2)
            ggram_db = torchaudio.functional.amplitude_to_DB(ggram_rms, multiplier=10, amin=1e-10, db_multiplier = torch.log10(max(ggram_rms.max(), 1e-10)), top_db = 80)
            res = ggram_db.clone()
            res = res.view(ggram_db.size(0), -1) 
            res -= res.min(1, keepdim=True)[0] 
            res /= res.max(1, keepdim=True)[0] 
            final = res.view(ggram_db.size(0), ggram_db.size(1), ggram_db.size(2))
        return final, label
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_data_path, fold, self.annotations.iloc[index, 0])
        return path 
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).cuda()
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal 
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal 
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # [1, 1, 1] -> [1, 1, 1, 0, 0, 0]
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal