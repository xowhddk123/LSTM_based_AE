import numpy as np

import torch
from torch.utils.data import Dataset

class AD_Dataset(Dataset):
    def __init__(self, data:np.ndarray, config): 
        assert isinstance(data, np.ndarray), 'The data is not numpy ndarray.'
        
        self.data = data
        self.input_size = config.input_size
        self.normal_mean = data.mean(axis=1)
        self.normal_std = data.std(axis=1)
        self.window_size = config.window_size
        
        
        # 정규화
        self.data = (self.data - self.normal_mean) / self.normal_std
        print('정규화 완료')
        
        
        self.input_idx = self.data.reshape(-1,self.window_size)
        self.var_data = torch.tensor(self.data, dtype=torch.float)
        
    def __len__(self):
        return len(self.input_idx)
    
    def __getitem__(self, item):
        temp_input_idx = self.input_idx[item]
        input_values = self.var_data[temp_input_idx]
        return input_values