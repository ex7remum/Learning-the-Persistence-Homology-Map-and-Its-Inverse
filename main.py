import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch

import generators
import modules
from orbit_dataset import get_datasets

from torch.nn import MSELoss

from model import AutoEncoderModel, PDNetOrbit5k, DataToPd
from utils import ChamferLoss, HungarianLoss, SlicedWasserstein

from torch.optim import Adam, AdamW

from model_train import train_epoch_ae, train_epoch_full, train_epoch_data_to_pd

from calculate_metrics import pd_to_pd_ae_metrics, orbit5k_metrics, data_to_pd_metrics

from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR


if __name__ == "__main__":
    task_type = 'data_to_pd'
    n_points = 1000
    dataset_train, dataset_test, dataloader_train, dataloader_test, n_max = get_datasets(1000, 32, 3500, 1500, n_points)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # smaller version
    
    #model_classificator = modules.CustomPersformer(n_in=3, embed_dim=64, fc_dim=128, num_heads=4, num_layers=5, n_out_enc=5, dropout=0.0, 
    #                          reduction="attention", use_skip=False).to(device)
    #checkpoint = torch.load("pretrained_models/persformer_orbit5k_77_test_acc_only_one_dim.pt", map_location=device)
    
    # bigger version
    
    model_classificator = modules.CustomPersformer(n_in=3, embed_dim=64, fc_dim=128, num_heads=8, num_layers=5, n_out_enc=5, dropout=0.0, 
                                                   reduction="attention").to(device)
    checkpoint = torch.load("pretrained_models/persformer_orbit5k_90acc.pt", map_location=device)
    model_classificator.load_state_dict(checkpoint)
    model_classificator.eval()
    
    if task_type == 'full':
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
                
    elif task_type == 'data_to_pd':
        wandb.login()
        # train size predictor here
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
                project="Pd size predict",
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
            wandb.log({"train loss": loss_train, "test loss": loss_test})
            scheduler.step(loss_test)
            
        model_size.eval()
        torch.save(model_size.state_dict(), "size_predictor.pt")
        
        # MAIN PART
        # encoders
        point_net = modules.PointNet(emb_dims=512, dim_in=2, output_channels=1024).to(device)
        mlp_encoder = modules.MLPEncoder(n_in=2, n_hidden=2048, n_out=1024, num_layers=4).to(device)
        transformer_encoder = modules.TransformerEncoder(n_in=2, embed_dim=1024, fc_dim=2048, num_heads=8, 
                                                         num_layers=5, n_out_enc=1024, dropout = 0.1, reduction="attention").to(device)
        DCGNN = modules.DGCNN(k=20, dim_in=2, emb_dims=2048, dropout=0.1, output_channels=1024).to(device)
        # generators
        top_n = generators.TopNGenerator(set_channels=3, cosine_channels=64, max_n=n_max + 10, latent_dim=1024).to(device)
        mlp_gen = generators.MLPGenerator(set_channels=3, max_n=n_max + 10, mlp_gen_hidden=2048, n_layers=2, latent_dim=1024).to(device)
        random_gen = generators.RandomSetGenerator(set_channels=3).to(device)
        firstk_gen = generators.FirstKSetGenerator(set_channels=3, max_n=n_max + 10).to(device)
        
        encoders = [(point_net, 'PointNet'), (mlp_encoder, 'MLPEnc'), (transformer_encoder, 'TransformerEnc'), (DCGNN, 'DCGNN')]
        generators = [(top_n, 'Top-n'), (mlp_gen, 'MLPGen'), (random_gen, 'Random'), (firstk_gen, 'FirstK')]
        
        for encoder, encoder_name in encoders:
            for generator, generator_name in generators:
                model_name = encoder_name + '_' + generator_name
                
                run = wandb.init(
                    # Set the project where this run will be logged
                    project=model_name,
                    # Track hyperparameters and run metadata
                )
                
                decoder = modules.TransformerDecoder(n_in=3, latent_dim=1024, fc_dim=256, num_heads=8, num_layers=5, n_out=3, 
                                                generator=generator,  n_out_lin=256, n_hidden=512, num_layers_lin=3, 
                                                dropout = 0.1, use_conv=True).to(device)
                model = DataToPd(encoder, decoder).to(device)
                
                # define learning hyperparameters
                lr = 0.001
                warmup_iters = 30
                n_epochs = 300 + warmup_iters
                criterion = SlicedWasserstein(n_projections=200)
                optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                scheduler1 = LinearLR(optimizer, start_factor=0.00000001, total_iters=warmup_iters)
                
                for i in range(n_epochs):
                    loss_train, loss_test = train_epoch_data_to_pd(model, dataloader_train, dataloader_test, optimizer, criterion)
                    if epoch_idx < warmup_iters:
                        scheduler1.step()
                    else:
                        if epoch_idx == warmup_iters:
                            scheduler2 = ReduceLROnPlateau(optimizer, patience=25, min_lr=1e-5, factor=0.5)
                        scheduler2.step(loss_test)
                        
                    wandb.log({"train loss": loss_train, "test loss": loss_test})    
                    data_to_pd_metrics(model, model_size, model_classificator, dataloader, name, n_max)    
                torch.save(model.state_dict(), model_name + ".pt")   
                
    else:
        raise NotImplementedError