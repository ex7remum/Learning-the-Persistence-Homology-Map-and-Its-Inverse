import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

# set-to-vec -> z -> vec-to-set AE
class AutoEncoderModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(AutoEncoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output = nn.Softplus()
        
    def forward(self, X: Tensor, mask: Tensor):
        z_enc = self.encoder(X, mask)
        z = self.decoder(z_enc, X.shape[1], mask)
        coords = self.output(z[:, :, :2]) # to be non-negative
        dims = z[:, :, 2:]
        res = torch.cat((coords, dims), axis = 2)
        full_mask = mask.unsqueeze(2).repeat(1, 1, res.shape[2])
        return res * full_mask