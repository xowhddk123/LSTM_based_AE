import numpy as np
import torch
import torch.nn as nn

import tqdm

def get_reconstrcution_error(config, model, test_loader):
    loss = nn.L1Loss(reduce=False)
    test_iterator = enumerate(test_loader)
    reconstrcution_error = []
    
    with torch.no_grad():
        for i, batch_data in tqdm(test_iterator):
            
            batch_data = batch_data.to(config.device)
            predict_values = model(batch_data)
            
            # MAE loss 
            loss_value = loss(predict_values, batch_data)
            reconstrcution_error.append(loss_value.mean(dim=0).cpu().numpy())

    reconstrcution_error = np.concatenate(reconstrcution_error, axis=0)
    return reconstrcution_error

class Anomaly_Calculator:
    def __init__(self, mean:np.array, std:np.array):
        assert mean.shape[0] == std.shape[0] and mean.shape[0] == std.shape[1]  # 평균과 분산의 차원이 같아야함
        self.mean = mean
        self.std = std
    
    def __call__(self, reconstruction_error:np.array):
        x = reconstruction_error - self.mean
        return np.matmul(np.matmul(x, self.std), x.T)