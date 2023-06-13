import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import HungarianLoss
from torch.nn import MSELoss

def train_epoch_ae(model, dataloader_train, dataloader_test, optimizer, criterion, progress = True):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # train
    model.train()
        
    loss = 0
    for batch in tqdm(dataloader_train):
        src = batch[0].to(device)
        mask = batch[1].to(device)
        tgt = model(src, mask)
        loss_batch = criterion(src.to(torch.float), tgt)
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss += loss_batch.detach().cpu()
        
    loss_train = loss / len(dataloader_train.dataset)
        
    # test
    model.eval()
    loss = 0
    for batch in tqdm(dataloader_test):
        src = batch[0].to(device)
        mask = batch[1].to(device)
        
        with torch.no_grad():
            tgt = model(src, mask)
            loss_batch = criterion(src.to(torch.float), tgt)
            loss += loss_batch
                       
    loss_test = loss / len(dataloader_test.dataset) 
    
    #############################
    
    if progress:
        print("Train loss {:.10f}\t Val loss {:.10f}".format(loss_train, loss_test))
        
        for batch in dataloader_train:
            src = batch[0].to(device)
            mask = batch[1].to(device)

            with torch.no_grad():
                tgt = model(src, mask)

            plt.figure(figsize=(10, 10))

            plt.scatter(src[0, :, 0].detach().cpu().numpy(), src[0, :, 1].detach().cpu().numpy(), label = "True", c = "b")
            plt.scatter(tgt[0, :, 0].detach().cpu().numpy(), tgt[0, :, 1].detach().cpu().numpy(), 
                        label = 'Model', alpha = 0.5, c = "r")

            plt.grid()
            plt.legend()
            plt.title("Progress")
            plt.show()
            break
            
    return loss_train, loss_test

def train_epoch_full(model, dataloader_train, dataloader_test, optimizer, criterion_pd, criterion_data, 
                     progress = True, alpha  = 0.05, dif_lambda = 1):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    criterion_hungarian = HungarianLoss()
    mse = MSELoss(reduction='mean')
    
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
        
    loss_train = loss / len(dataloader_train.dataset)
        
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
            
            
    loss_test = loss / len(dataloader_test.dataset)
    
    #############################

    if progress:
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
            
    return loss_train, loss_test


def train_epoch_data_to_pd(model, dataloader_train, dataloader_test, optimizer, criterion):
    model.train()
    loss = 0
    for batch in dataloader_train:
        src_pd, mask, labels, src_data = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        tgt_pd = model(src_data, mask)
        loss_batch = criterion(src_pd.to(torch.float), tgt_pd)
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss += loss_batch.detach().cpu()
    loss_train = loss / len(dataloader_train.dataset)
    
    loss = 0
    model.eval()
    for batch in dataloader_test:    
        with torch.no_grad():
            src_pd, mask, labels, src_data = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            tgt_pd = model(src_data, mask)
            loss_batch = criterion(src_pd.to(torch.float), tgt_pd)
            loss += loss_batch
    loss_test = loss / len(dataloader_test.dataset)
            
    return loss_train, loss_test