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
    
# full model
class PDNetOrbit5k(nn.Module):
    def __init__(self, encoder_data: nn.Module, decoder_data: nn.Module, encoder_pds: nn.Module, decoder_pds: nn.Module):
        super(PDNetOrbit5k, self).__init__()
        self.encoder_data = encoder_data
        self.decoder_data = decoder_data
        self.encoder_pds = encoder_pds
        self.decoder_pds = decoder_pds
        self.output = nn.Softplus()
        
    def forward(self, X: Tensor, pds: Tensor, mask: Tensor):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        dumb_mask = torch.zeros_like(X, )[:, :, 0].long().to(device) + 1
        z_enc_pds = self.encoder_pds(pds, mask)
        z_enc_data = self.encoder_data(X, dumb_mask)
        
        latent = z_enc_pds
        
        z_pds = self.decoder_pds(latent, pds.shape[1], mask)
        coords = self.output(z_pds[:, :, :2]) # to be non-negative
        dims = z_pds[:, :, 2:]
        res = torch.cat((coords, dims), axis = 2)
        full_mask = mask.unsqueeze(2).repeat(1, 1, res.shape[2])
        
        z_pic = self.decoder_data(latent, X.shape[1], dumb_mask)
        
        return z_pic, res * full_mask, z_enc_pds, z_enc_data

# only getting 
class DataToPd(nn.Module):
    def __init__(self, encoder_data: nn.Module, decoder_pd: nn.Module):
        super(DataToPd, self).__init__()
        self.encoder_data = encoder_data
        self.decoder_pd = decoder_pd
        self.output = nn.Softplus()
        
    def forward(self, X: Tensor, mask: Tensor):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dumb_mask = (torch.zeros((X.shape[0], X.shape[1])) + 1.).to(torch.long).to(device)
        z_enc = self.encoder_data(X, dumb_mask)
        z = self.decoder_pd(z_enc, mask.shape[1], mask)
        coords = self.output(z[:, :, :2]) # to be non-negative
        logits = z[:, :, 2:]
        res = torch.cat((coords, logits), axis = 2)
        full_mask = mask.unsqueeze(2).repeat(1, 1, res.shape[2])
        return res * full_mask