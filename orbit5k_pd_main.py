import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch

import generators
import modules
from orbit_dataset import get_datasets

from torch.nn import MSELoss
import wandb

from model import AutoEncoderModel, PDNetOrbit5k, DataToPd
from utils import ChamferLoss, HungarianLoss, SlicedWasserstein

from torch.optim import Adam, AdamW

from model_train import train_epoch_ae, train_epoch_full, train_epoch_data_to_pd

from calculate_metrics import pd_to_pd_ae_metrics, orbit5k_metrics, data_to_pd_metrics

from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR


if __name__ == "__main__":
    wandb.login()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_points = 1000
    
    # bigger version
    model_classificator = modules.CustomPersformer(n_in=3, embed_dim=64, fc_dim=128, num_heads=8, num_layers=5, n_out_enc=5, dropout=0.0,
                                                   reduction="attention").to(device)
    checkpoint = torch.load("data_to_pd/Learning-the-Persistence-Homology-Map-and-Its-Inverse/pretrained_models/persformer_orbit5k_90acc.pt", map_location=device)
    model_classificator.load_state_dict(checkpoint)
    model_classificator.eval()

    dataset_train, dataset_test, dataloader_train, dataloader_test, n_max = get_datasets(1000, 32, 3500, 1500, n_points)
    # smaller version

    #model_classificator = modules.CustomPersformer(n_in=3, embed_dim=64, fc_dim=128, num_heads=4, num_layers=5, n_out_enc=5, dropout=0.0,
    #                          reduction="attention", use_skip=False).to(device)
    #checkpoint = torch.load("pretrained_models/persformer_orbit5k_77_test_acc_only_one_dim.pt", map_location=device)

    # maybe try adding sampling n from distribution
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lr = 0.001
    n_epochs = 100
    model_size = modules.MLPEncoder(n_in=2, n_hidden=2048, n_out=1, num_layers=3).to(device)
    mse_crit = MSELoss()
    optimizer = AdamW(model_size.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=20, min_lr=1e-5, factor=0.5)

    run = wandb.init(
            # Set the project where this run will be logged
            project="Data to pd v2",
            # Track hyperparameters and run metadata
    )


    for epoch_idx in range(n_epochs):
        # train
        model_size.train()

        loss = 0
        for batch in dataloader_train:
            src_pd, mask, labels, src_data = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            sizes = (torch.sum(mask, dim=1)) * 1.
            dumb_mask = (torch.zeros((src_data.shape[0], src_data.shape[1])) + 1).to(torch.long).to(device)
            pred = model_size(src_data, dumb_mask).squeeze()
            loss_batch = mse_crit(pred, sizes)
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss += loss_batch.detach().cpu()

        loss_train = loss / len(dataloader_train.dataset)

        model_size.eval()
        for batch in dataloader_test:
            with torch.no_grad():
                src_pd, mask, labels, src_data = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                sizes = torch.sum(mask, dim=1) * 1.
                dumb_mask = (torch.zeros((src_data.shape[0], src_data.shape[1])) + 1.).to(torch.long).to(device)
                pred = model_size(src_data, dumb_mask).squeeze()
                loss_batch = mse_crit(pred, sizes)
                loss += loss_batch.detach().cpu()

        loss_test = loss / len(dataloader_test.dataset)
        wandb.log({"train loss size": loss_train, "test loss size": loss_test})
        scheduler.step(loss_test)

    model_size.eval()
    torch.save(model_size.state_dict(), "size_predictor.pt")

    # MAIN PART
    # encoders
    point_net = modules.PointNet(emb_dims=256, dim_in=2, output_channels=1024).to(device)
    mlp_encoder = modules.MLPEncoder(n_in=2, n_hidden=512, n_out=1024, num_layers=4).to(device)
    transformer_encoder = modules.TransformerEncoder(n_in=2, embed_dim=512, fc_dim=1024, num_heads=8,
                                                     num_layers=5, n_out_enc=1024, dropout = 0.1, reduction="attention").to(device)
    DCGNN = modules.DGCNN(k=10, dim_in=2, emb_dims=512, dropout=0.1, output_channels=1024).to(device)
    # generators
    top_n = generators.TopNGenerator(set_channels=3, cosine_channels=32, max_n=n_max + 10, latent_dim=1024).to(device)
    mlp_gen = generators.MLPGenerator(set_channels=3, max_n=n_max + 10, mlp_gen_hidden=1024, n_layers=2, latent_dim=1024).to(device)
    random_gen = generators.RandomSetGenerator(set_channels=3).to(device)
    firstk_gen = generators.FirstKSetGenerator(set_channels=3, max_n=n_max + 10).to(device)

    encoders = [(point_net, 'PointNet'), (mlp_encoder, 'MLPEnc'), (transformer_encoder, 'TransformerEnc'), (DCGNN, 'DCGNN')]
    generators = [(top_n, 'Top-n'), (mlp_gen, 'MLPGen'), (random_gen, 'Random'), (firstk_gen, 'FirstK')]

    for encoder, encoder_name in encoders:
        for generator, generator_name in generators:
            model_name = encoder_name + '_' + generator_name

            decoder = modules.TransformerDecoder(n_in=3, latent_dim=1024, fc_dim=128, num_heads=8, num_layers=5, n_out=3,
                                            generator=generator,  n_out_lin=128, n_hidden=256, num_layers_lin=3,
                                            dropout = 0.1, use_conv=True).to(device)
            model = DataToPd(encoder, decoder).to(device)

            train_epoch_data_to_pd(model, model_size, model_classificator, dataloader_train,
                                   dataloader_test, model_name, n_max)