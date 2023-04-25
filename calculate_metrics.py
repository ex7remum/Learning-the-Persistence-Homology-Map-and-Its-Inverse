import torch
from tqdm import tqdm
from utils import HungarianLoss
import matplotlib.pyplot as plt

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
            w2_dist_train += W2(predicted_diagrams[:, :, :2], batch[0].to(device)[:, :, :2])

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
            w2_dist_test += W2(predicted_diagrams[:, :, :2], batch[0].to(device)[:, :, :2])

    accuracy_test = correct / len(dataloader_test.dataset)
    accuracy_test_approx = correct_approx / len(dataloader_test.dataset)
    w2_dist_test /= len(dataloader_test.dataset)
    
    return accuracy_train, accuracy_train_approx, accuracy_test, accuracy_test_approx, w2_dist_train, w2_dist_test