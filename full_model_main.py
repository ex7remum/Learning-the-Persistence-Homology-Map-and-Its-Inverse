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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    task_type = 'data_to_pd'
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

    # DEFINE HYPERPARAMETERS
    
    lr = 0.001
    set_channels = 3
    warmup_iters = 30
    n_epochs = 300 + warmup_iters
    criterion_pd = SlicedWasserstein(n_projections=200)
    criterion_data = ChamferLoss()
    criterion_hungarian = HungarianLoss()
    mse = MSELoss(reduction='mean')

    # DEFINE MODEL
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

    # TRAIN AND EVAL MODEL   
    loss_train, loss_test = [], []
    gamma = 1
    alpha = 0.05
    
    for epoch_idx in range(n_epochs):
        if epoch_idx % 100 == 0:
            gamma *= 10
        # train
        model.train()

        loss = 0
        for batch in tqdm(dataloader_train):
            src_pd = batch[0].to(device)
            mask = batch[1].to(device)

            src_data = batch[3].to(device)

            tgt_data, tgt_pd, z_enc_pd, z_enc_data = model(src_data, src_pd, mask)
            loss_batch = alpha * criterion_pd(src_pd.to(torch.float), tgt_pd) + \
                        (1-alpha) * criterion_data(src_data.to(torch.float), tgt_data) + \
                         dif_lambda * mse(z_enc_pd, z_enc_data)
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss += loss_batch.detach().cpu()

        train_loss = loss / len(dataloader_train.dataset)

        # test
        model.eval()
        loss = 0
        for batch in tqdm(dataloader_test):
            src = batch[0].to(device)
            mask = batch[1].to(device)

            with torch.no_grad():
                src_pd = batch[0].to(device)
                mask = batch[1].to(device)

                src_data = batch[3].to(device)

                tgt_data, tgt_pd, _, _ = model(src_data, src_pd, mask)
                loss_batch = alpha * criterion_pd(src_pd.to(torch.float), tgt_pd) + \
                        (1-alpha) * criterion_data(src_data.to(torch.float), tgt_data) + \
                         dif_lambda * mse(z_enc_pd, z_enc_data)
                loss += loss_batch


        test_loss = loss / len(dataloader_test.dataset)
        
        
        loss_train.append(train_loss)
        loss_test.append(test_loss)
        
        # PRINT METRICS AND A PROGRESS ON FIRST ELEMENT FROM TEST
        print("Train loss {:.10f}\t Val loss {:.10f}".format(loss_train, loss_test))

        for batch in dataloader_train:
            src_pd = batch[0].to(device)
            mask = batch[1].to(device)

            src_data = batch[3].to(device)

            with torch.no_grad():
                tgt_data, tgt_pd, z1, z2 = model(src_data, src_pd, mask)

            print("Data: {:.4f}, Sliced: {:.4f}, Latent diff: {:.4f}, W2: {:.4f}".format(
                    criterion_data(src_data, tgt_data), criterion_pd(tgt_pd, src_pd),
                    mse(z1, z2), criterion_hungarian(tgt_pd, src_pd)))
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(src_pd[0, :, 0].detach().cpu().numpy(), src_pd[0, :, 1].detach().cpu().numpy(), label = "True", c = "b")
            plt.scatter(tgt_pd[0, :, 0].detach().cpu().numpy(), tgt_pd[0, :, 1].detach().cpu().numpy(),
                        label = 'Model', alpha = 0.5, c = "r")

            plt.grid()
            plt.legend()
            plt.title("Progress pd")

            plt.subplot(1, 2, 2)
            plt.scatter(src_data[0, :, 0].detach().cpu().numpy(), src_data[0, :, 1].detach().cpu().numpy(), label = "True", c = "b")
            plt.scatter(tgt_data[0, :, 0].detach().cpu().numpy(), tgt_data[0, :, 1].detach().cpu().numpy(),
                        label = 'Model', alpha = 0.5, c = "r")

            plt.grid()
            plt.legend()
            plt.title("Progress data")
            plt.show()
            break
        
        test_acc_approx, w2, chamfer = orbit5k_metrics(model, model_classificator, dataloader_train, dataloader_test)
        print(test_acc_approx, w2, chamfer)
        if epoch_idx < warmup_iters:
            scheduler1.step()
        else:
            if epoch_idx == warmup_iters:
                scheduler2 = ReduceLROnPlateau(optimizer, patience=20, min_lr=1e-6)
            scheduler2.step(test_loss)
            
    torch.save(model.state_dict(), "full_model.pt")