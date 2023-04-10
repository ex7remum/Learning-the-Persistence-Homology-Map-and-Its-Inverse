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
        
    def forward(self, X: Tensor, mask: Tensor):
        z_enc = self.encoder(X, mask)
        z = self.decoder(z_enc, X.shape[1], mask)
        return z