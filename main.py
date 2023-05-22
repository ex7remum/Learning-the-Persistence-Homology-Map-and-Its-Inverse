import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch

import generators
import modules
from orbit_dataset import get_datasets

from torch.nn import MSELoss

from model import AutoEncoderModel, PDNetOrbit5k
from utils import ChamferLoss, HungarianLoss, SlicedWasserstein

from torch.optim import Adam, AdamW

from model_train import train_epoch_ae, train_epoch_full

from calculate_metrics import pd_to_pd_ae_metrics, orbit5k_metrics

from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR


if __name__ == "__main__":
    
    n_points = 300
    dataset_train, dataset_test, dataloader_train, dataloader_test, n_max = get_datasets(1000, 32, 3500, 1500, n_points)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #define your model here and hyperparameters
    #TODO: add console input
    
    lr = 0.001
    set_channels = 3
    warmup_iters = 30
    n_epochs = 300 + warmup_iters
    criterion = SlicedWasserstein(n_projections=200)
    crit_chamfer = ChamferLoss()
    criterion_hungarian = HungarianLoss()
    mse = MSELoss(reduction='mean')

    encoder_pd = modules.TransformerEncoder(n_in=set_channels, embed_dim=64, fc_dim=128, num_heads=4, num_layers=5, n_out_enc=512,
                            dropout=0.1, reduction="attention", use_skip=True)
    
    generator = generators.TopNGenerator(set_channels=3, cosine_channels=32, max_n=n_max, latent_dim=512)
    
    decoder_pd = modules.TransformerDecoder(n_in=3, latent_dim=512, fc_dim=128, num_heads=8, num_layers=5, n_out=3, generator=generator, 
                     n_out_lin=128, n_hidden=256, num_layers_lin = 3, dropout = 0.1, use_conv=True)

    encoder_data = modules.TransformerEncoder(n_in=2, embed_dim=512, fc_dim=1024, num_heads=8, num_layers=5, n_out_enc=512,
                                dropout=0.1, reduction="attention", use_skip=True)
    
    decoder_data = modules.MLPDecoder(n_in=512, n_hidden=1024, n_out=n_points * 2, num_layers=3, set_channels=2)

    model = PDNetOrbit5k(encoder_data, decoder_data, encoder_pd, decoder_pd).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler1 = LinearLR(optimizer, start_factor=0.0000001, total_iters=warmup_iters)
    
    model_classificator = modules.CustomPersformer(n_in=3, embed_dim=64, fc_dim=128, num_heads=4, num_layers=5, n_out_enc=5, dropout=0.0, 
                           reduction="attention", use_skip=False).to(device)
    checkpoint = torch.load("pretrained_models/persformer_orbit5k_77_test_acc_only_one_dim.pt", map_location=device)
    model_classificator.load_state_dict(checkpoint)
    
    loss_train, loss_test = [], []
    gamma = 1
    alpha = 0.05

    for epoch_idx in range(n_epochs):
        if epoch_idx % 100 == 0:
            gamma *= 10
        train_loss, test_loss = train_epoch_full(model, dataloader_train, dataloader_test, optimizer, 
                                                 criterion, crit_chamfer, progress=False, alpha=alpha, dif_lambda=gamma)
        loss_train.append(train_loss)
        loss_test.append(test_loss)

        test_acc_approx, w2, chamfer = orbit5k_metrics(model, model_classificator, dataloader_train, dataloader_test)
        print(test_acc_approx, w2, chamfer)
        if epoch_idx < warmup_iters:
            scheduler1.step()
        else:
            if epoch_idx == warmup_iters:
                scheduler2 = ReduceLROnPlateau(optimizer, patience=20, min_lr=1e-6)
            scheduler2.step(test_loss)
    