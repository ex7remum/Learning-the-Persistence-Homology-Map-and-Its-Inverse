import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

def train_epoch(model, dataloader_train, dataloader_test, optimizer, criterion, progress = True):
    
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
    
    print("Train loss {:.10f}\t Val loss {:.10f}".format(loss_train, loss_test))
    #############################
    if progress:
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