import torch
import torch.nn as nn
from tqdm import tqdm
from utils import HungarianLoss, ChamferLoss
import ot

def pd_to_pd_ae_metrics(model_approximator, model_classificator, dataloader_train, dataloader_test):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    W2 = HungarianLoss()
    model_classificator.eval()
    model_approximator.eval()

    correct, correct_approx, w2_dist_train = 0, 0, 0
    for batch in tqdm(dataloader_train):
        batch[0][:, :, 1] += batch[0][:, :, 0]
        y_hat = model_classificator(batch[0].to(device), batch[1].to(device)).argmax(dim=1)
        correct += int((y_hat == batch[2].to(device)).sum())
        batch[0][:, :, 1] -= batch[0][:, :, 0]
        
        predicted_diagrams = model_approximator(batch[0].to(device), batch[1].to(device))
        predicted_diagrams[:, :, 2] = predicted_diagrams[:, :, 2] > 0.5
        predicted_diagrams[:, :, 1] += predicted_diagrams[:, :, 0]

        y_hat_approx = model_classificator(predicted_diagrams, batch[1].to(device)).argmax(dim=1)
        correct_approx += int((y_hat_approx == batch[2].to(device)).sum())
        predicted_diagrams[:, :, 1] -= predicted_diagrams[:, :, 0]
        
        with torch.no_grad():
            w2_dist_train += W2(predicted_diagrams[:, :, :2], batch[0].to(device)[:, :, :2]) / batch.shape[0]

    accuracy_train = correct / len(dataloader_train.dataset)
    accuracy_train_approx = correct_approx / len(dataloader_train.dataset)
    w2_dist_train /= len(dataloader_train.dataset)

    correct, correct_approx, w2_dist_test = 0, 0, 0
    for batch in tqdm(dataloader_test):
        batch[0][:, :, 1] += batch[0][:, :, 0]
        y_hat = model_classificator(batch[0].to(device), batch[1].to(device)).argmax(dim=1)
        correct += int((y_hat == batch[2].to(device)).sum())
        batch[0][:, :, 1] -= batch[0][:, :, 0]
        
        predicted_diagrams = model_approximator(batch[0].to(device), batch[1].long().to(device))
        predicted_diagrams[:, :, 2] = predicted_diagrams[:, :, 2] > 0.5
        predicted_diagrams[:, :, 1] += predicted_diagrams[:, :, 0]

        y_hat_approx = model_classificator(predicted_diagrams, batch[1].to(device)).argmax(dim=1)
        correct_approx += int((y_hat_approx == batch[2].to(device)).sum())
        predicted_diagrams[:, :, 1] -= predicted_diagrams[:, :, 0]
        
        with torch.no_grad():
            w2_dist_test += W2(predicted_diagrams[:, :, :2], batch[0].to(device)[:, :, :2]) / batch.shape[0]

    accuracy_test = correct / len(dataloader_test.dataset)
    accuracy_test_approx = correct_approx / len(dataloader_test.dataset)
    w2_dist_test /= len(dataloader_test.dataset)
    
    return accuracy_train, accuracy_train_approx, accuracy_test, accuracy_test_approx, w2_dist_train, w2_dist_test


def orbit5k_metrics(model, model_classificator, dataloader_train, dataloader_test):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_classificator.eval()
    model.eval()
    W2 = HungarianLoss()
    Chamfer = ChamferLoss()
    sp = nn.Softplus()

    correct_approx, w2, chamfer = 0, 0, 0
    for batch in dataloader_test:
        src_data = batch[3].to(device)
        src_pd = batch[0].to(device)
        labels = batch[2].to(device)
        mask = batch[1].to(device)

        dumb_mask = torch.zeros_like(src_data, )[:, :, 0].long().to(device) + 1
        predicted_diagrams = model.decoder_pds(model.encoder_data(src_data, dumb_mask), src_pd.shape[1], mask)
        predicted_data = model.decoder_data(model.encoder_pds(src_pd, mask), src_data.shape[1], dumb_mask)

        predicted_diagrams[:, :, 2] = predicted_diagrams[:, :, 2] > 0.5
        coords = sp(predicted_diagrams[:, :, :2]) # to be non-negative
        dims = predicted_diagrams[:, :, 2:]
        res = torch.cat((coords, dims), axis = 2)
        full_mask = mask.unsqueeze(2).repeat(1, 1, res.shape[2])
        predicted_diagrams = res * full_mask

        predicted_diagrams[:, :, 1] += predicted_diagrams[:, :, 0]

        y_hat_approx = model_classificator(predicted_diagrams, mask).argmax(dim=1)
        correct_approx += int((y_hat_approx == labels).sum())

        predicted_diagrams[:, :, 1] -= predicted_diagrams[:, :, 0]
        
        with torch.no_grad():
            w2 += W2(predicted_diagrams[:, :, :2], batch[0].to(device)[:, :, :2]) / batch[0].shape[0]
            chamfer += Chamfer(predicted_data, src_data)

    accuracy_test_approx = correct_approx / len(dataloader_test.dataset)
    w2 /= len(dataloader_test.dataset)
    chamfer /= len(dataloader_test.dataset)
    return accuracy_test_approx, w2.item(), chamfer.item()


def data_to_pd_metrics(model, model_size, model_classificator, dataloader, name, n_max):
    model.eval()
    model_size.eval()
    model_classificator.eval()
    
    correct_orig, correct_pred = 0, 0
    W2_pred, W2_orig = 0, 0
    
    for batch in dataloader:
        with torch.no_grad():
            src_pd, mask, labels, src_data = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            dumb_mask = (torch.zeros((src_data.shape[0], src_data.shape[1])) + 1.).to(torch.long).to(device)
            pred = model_size(src_data, dumb_mask)
            sizes = torch.ceil(pred)
            real_sizes = torch.sum(mask, dim=1)
            pred_mask = torch.zeros(mask.shape).to(torch.long).to(device)
            for i in range(mask.shape[0]):
                if sizes[i] <= 0:
                    sizes[i] = 1
                if sizes[i] > n_max:
                    sizes[i] = n_max              
                pred_mask[i, :int(sizes[i])] = 1
                
            tgt_pd_pred = model(src_data, pred_mask)
            y_hat = model_classificator(tgt_pd_pred, pred_mask).argmax(dim=1)
            correct_pred += int((y_hat == labels).sum())
            
            tgt_pd_orig = model(src_data, mask)
            y_hat = model_classificator(tgt_pd_orig, mask).argmax(dim=1)
            correct_orig += int((y_hat == labels).sum())
            
            #calculate W2
            for i in range(src_pd.shape[0]):
                M = ot.dist(tgt_pd_orig[i][:real_sizes[i]], src_pd[i][:real_sizes[i]])
                W2_orig += ot.emd2(torch.tensor([]), torch.tensor([]), M)
                M = ot.dist(tgt_pd_pred[i][:int(sizes[i])], src_pd[i][:real_sizes[i]])
                W2_pred += ot.emd2(torch.tensor([]), torch.tensor([]), M)
                
    #add wandb logging
    correct_pred /= len(dataloader.dataset)
    correct_orig /= len(dataloader.dataset)
    W2_orig /= len(dataloader.dataset)
    W2_pred /= len(dataloader.dataset)
    wandb.log({"correct orig": correct_orig, "correct pred": correct_pred})
    wandb.log({"W2 orig": W2_orig.item(), "W2 pred": W2_pred.item()})
    return