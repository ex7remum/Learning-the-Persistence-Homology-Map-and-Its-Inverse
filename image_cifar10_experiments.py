import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np

from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

from ripser import Rips
from persim import PersistenceImager

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
import generators, modules
from utils import SlicedWasserstein
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from model import DataToPd

from modules import CustomPersformer
from utils import HungarianLoss
import time
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class PI_Net(nn.Module):
    def __init__(self, in_channels = 3):
        super(PI_Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same'),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 'same'),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 'same'), 
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2) 
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 'same'), 
            nn.BatchNorm2d(num_features = 1024),
            nn.ReLU(),
            nn.AvgPool2d(4)
        )
        self.dense = nn.Sequential(
            nn.Linear(1024, 2500),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1, out_channels = in_channels, kernel_size = 51, padding = 25),
            nn.BatchNorm2d(num_features = in_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dense(x)
        x = x.reshape((-1, 1, 50, 50))
        x = self.decoder(x)
        return x.reshape((-1, 50 * 50 * x.shape[1]))


class ImagePds(Dataset):

    def __init__(self, X, y, X_orig, PI, transform=None):
        self.X = X
        self.y = y
        self.X_orig = X_orig
        self.PI = PI

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.X_orig[idx], self.PI[idx]
    
    
def collate_fn(data):

    tmp_pd, _, image, pimg = data[0]

    n_image = image.shape[0]

    n_batch = len(data)
    n_features_pd = tmp_pd.shape[1]
    n_points_pd = max(len(pd) for pd, _, _, _ in data)
    inputs_pd = np.zeros((n_batch, n_points_pd, n_features_pd), dtype=float)

    images = np.zeros((n_batch, n_image, n_image), dtype=float)

    mask = np.zeros((n_batch, n_points_pd))
    labels = np.zeros(len(data))
    PI = np.zeros((n_batch, pimg.shape[0], pimg.shape[1]))

    for i, (pd, label, image, pimg) in enumerate(data):
        inputs_pd[i][:len(pd)] = pd
        labels[i] = label
        mask[i][:len(pd)] = 1
        images[i] = image
        PI[i] = pimg
    return torch.Tensor(inputs_pd), torch.Tensor(mask).long(), torch.Tensor(labels).long(), \
            torch.Tensor(images), torch.Tensor(PI)

def conv_pd(diagrams):
    pd = np.zeros((0, 3))

    for k, diagram_k in enumerate(diagrams):
        if k != 0:
            diagram_k = diagram_k[~np.isinf(diagram_k).any(axis=1)] # filter infs
            diagram_k = np.concatenate((diagram_k, k * np.ones((diagram_k.shape[0], 1))), axis=1)
            if len(diagram_k) > 0:
                pd = np.concatenate((pd, diagram_k))

    return pd


def get_datasets_pinet(dataloader, batch_size, is_train, filtration = '3D-rips', pimgr = None):
    all_diagrams, all_images, all_labels, pis = [], [], [], []
    vr = Rips()
    
    for batch in dataloader:
        images, labels = batch
        for label in labels:
            all_labels.append(label)
            if filtration == '3D-rips' or filtration == 'sublevel':
                for _ in range(2):
                    all_labels.append(label)      
        for image in images:
            
            if filtration == '3D-rips':
                for channel in range(3):
                    layer = image[channel]
                    # (32, 32) image -> (32 * 32, 3) where coords are (x, y, density)
                    cur_diagram = np.zeros((0, 3))
                    nx, ny = layer.shape
                    x = torch.linspace(0, 1, nx)
                    y = torch.linspace(0, 1, ny)
                    xv, yv = torch.meshgrid(x, y)
                    xv = xv.unsqueeze(2)
                    yv = yv.unsqueeze(2)
                    layer = layer.unsqueeze(2)
                    res = torch.cat((xv, yv, layer), 2)
                    res = res.reshape(-1, 3)
                    diagram = conv_pd(vr.fit_transform(res))                    
                    diagram[:, 1] -= diagram[:, 0]
                    cur_diagram = np.concatenate((cur_diagram, diagram))
                    all_images.append(layer)
                    all_diagrams.append(diagram)
                    
            elif filtration == 'sublevel':
                import gudhi as gd
                
                for channel in range(3):
                    layer = image[channel]
                    cc_density_crater = gd.CubicalComplex(
                        dimensions = [32 , 32], 
                        top_dimensional_cells = layer.flatten()
                    )
                    cc_density_crater.compute_persistence()
                    diagram = cc_density_crater.persistence()

                    pd = np.zeros((0, 3))
                    
                    for k, pair in diagram:
                        if k == 1 and not np.isinf(pair[1]):
                            cur = np.zeros((1, 3))
                            cur[0, 0], cur[0, 1], cur[0, 2] = pair[0], pair[1] - pair[0], k
                            pd = np.concatenate((pd, cur))
                    all_images.append(layer)
                    all_diagrams.append(pd)
            elif filtration == '5D-rips':

                layer1, layer2, layer3 = image[0], image[1], image[2]
                nx, ny = layer1.shape
                x, y = torch.linspace(0, 1, nx), torch.linspace(0, 1, ny)
                xv, yv = torch.meshgrid(x, y)
                xv, yv, layer1, layer2, layer3 = xv.unsqueeze(2), yv.unsqueeze(2), layer1.unsqueeze(2), layer2.unsqueeze(2), \
                                                 layer3.unsqueeze(2)
                res = torch.cat((xv, yv, layer1, layer2, layer3), 2).reshape(-1, 5)
                pd = conv_pd(vr.fit_transform(res))
                
                pd[:, 1] -= pd[:, 0]
                all_images.append(image)
                all_diagrams.append(pd)
            
            else:
                raise NotImplementedError
    n_max = 0

    for x in all_diagrams:
        n_max = max(n_max, len(x))
    pimgs = pimgr.transform(all_diagrams, skew=False)
    dataset = ImagePds(all_diagrams, all_labels, all_images, pimgs)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, collate_fn=collate_fn)
    
    return dataset, dataloader, n_max



def train_classifier_on_pds(dataloader_train, dataloader_test, n_max, real_pd, model = None):
    if real_pd:
        add = 'real pds'
    else:
        add = 'pred pds'
    print('Training classifier on ', add)
    model_pd = CustomPersformer(n_in = 3, embed_dim = 128, fc_dim = 256, num_heads = 8, num_layers = 5, n_out_enc = 10).to(device)
    crit = CrossEntropyLoss()
    n_epochs = 200
    lr = 1e-3
    optimizer = AdamW(model_pd.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(n_epochs):
        model_pd.train()
        loss, correct = 0.0, 0.0
        for batch in dataloader_train:
            src_pd, mask, labels, src_data, PI = batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                                                batch[3].to(device), batch[4].to(device)

            with torch.no_grad():
                if real_pd:
                    pred_pds = src_pd.detach().clone()
                else:
                    pred_pds = model(src_data.unsqueeze(1).repeat(1, 3, 1, 1), mask)


            logits = model_pd(pred_pds, mask)
            loss_batch = crit(logits, labels)
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss += loss_batch.detach().cpu()
            correct += (labels == torch.argmax(logits, axis=1)).sum()
        loss /= len(dataloader_train.dataset)
        correct /= len(dataloader_train.dataset)
        print("Train loss : {}\t Train acc : {}".format(loss, correct))

        model_pd.eval()
        loss, correct = 0.0, 0.0
        for batch in dataloader_test:
            with torch.no_grad():
                src_pd, mask, labels, src_data, PI = batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                                                    batch[3].to(device), batch[4].to(device)

                with torch.no_grad():
                    if real_pd:
                        pred_pds = src_pd.detach().clone()
                    else:
                        pred_pds = model(src_data.unsqueeze(1).repeat(1, 3, 1, 1), mask)

                logits = model_pd(pred_pds, mask)
                loss_batch = crit(logits, labels)
                loss += loss_batch
                correct += (labels == torch.argmax(logits, axis=1)).sum()
                
        loss /= len(dataloader_test.dataset)
        correct /= len(dataloader_test.dataset)
        print("Test loss : {}\t Test acc : {}".format(loss, correct))
        
        
    return model_pd


def train_full_image_model(dataloader_train, dataloader_test, n_max):
    print('Training full model')
    # if want pretrained
    encoder = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT) 
    encoder.fc = Identity()
    #encoder = torchvision.models.resnet18()
    
    # if want freeze all layers in encoder
    for param in encoder.parameters():
        param.requires_grad = False
                
    #encoder.fc = nn.Linear(in_features=512, out_features=256)
    
    generator = generators.TopNGenerator(set_channels=3, cosine_channels=32, max_n=n_max + 5, latent_dim=512)
    decoder = modules.TransformerDecoder(n_in=3, latent_dim=512, fc_dim=1024, num_heads=8, num_layers=5, n_out=3,
                                                generator=generator, n_out_lin=128, n_hidden=256, num_layers_lin=1,
                                                dropout = 0.1, use_conv=True)
    model = DataToPd(encoder, decoder, False).to(device)

    lr = 0.001
    warmup_iters = 10
    n_epochs = 200 + warmup_iters
    criterion = SlicedWasserstein(n_projections=200)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler1 = LinearLR(optimizer, start_factor=0.0001, total_iters=warmup_iters)

    for epoch_idx in range(n_epochs):
        model.train()
        loss = 0
        for batch in dataloader_train:
            src_pd, mask, labels, src_data, PI = batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                                                batch[3].to(device), batch[4].to(device)
            tgt_pd = model(src_data.unsqueeze(1).repeat(1, 3, 1, 1), mask)
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
                src_pd, mask, labels, src_data, PI = batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                                                    batch[3].to(device), batch[4].to(device)
                tgt_pd = model(src_data.unsqueeze(1).repeat(1, 3, 1, 1), mask)
                loss_batch = criterion(src_pd.to(torch.float), tgt_pd)
                loss += loss_batch
        loss_test = loss / len(dataloader_test.dataset)

        if epoch_idx < warmup_iters:
            scheduler1.step()
        else:
            if epoch_idx == warmup_iters:
                scheduler2 = ReduceLROnPlateau(optimizer, patience=25, min_lr=1e-5, factor=0.5)
            scheduler2.step(loss_test)

        print("Train loss: {}\t Test loss: {}".format(loss_train, loss_test))
        
    return model


def train_pinet(dataloader_train, dataloader_test):
    print('Train PI-Net')
    pinet_model = PI_Net(in_channels = 1).to(device)

    crit = CrossEntropyLoss()
    n_epochs = 200
    lr = 1e-3
    optimizer = AdamW(pinet_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=25, min_lr=1e-6, factor=0.5)

    for _ in range(n_epochs):
        pinet_model.train()
        loss = 0.0
        for batch in dataloader_train:
            src_pd, mask, labels, src_data, PI = batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                                                batch[3].to(device), batch[4].to(device)


            out = pinet_model(PI.unsqueeze(1))
            loss_batch = crit(out, PI.reshape((-1, 2500)))
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss += loss_batch.detach().cpu()
        loss /= len(dataloader_train.dataset)
        print("Train loss : {}".format(loss))

        pinet_model.eval()
        loss = 0.0
        for batch in dataloader_test:
            with torch.no_grad():
                src_pd, mask, labels, src_data, PI = batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                                                        batch[3].to(device), batch[4].to(device)

                out = pinet_model(PI.unsqueeze(1))
                loss_batch = crit(out, PI.reshape((-1, 2500)))
                loss += loss_batch
        loss /= len(dataloader_test.dataset)
        print("Test loss : {}".format(loss))
        scheduler.step(loss)
        
    return pinet_model

def compute_metrics(dataloader_test, model, on_pds = True, pimgr = None):
    hung = HungarianLoss()
    
    total_time, total_mse, W2 = 0, 0, 0
    for batch in dataloader_test:
        with torch.no_grad():
            src_pd, mask, labels, src_data, PI = batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                                                batch[3].to(device), batch[4].to(device)
            t1 = time.time()
            if on_pds:
                tgt = model(src_data.unsqueeze(1).repeat(1, 3, 1, 1), mask)
            else:
                approx_pi = model(PI.unsqueeze(1)).squeeze(1).cpu().numpy().reshape((-1, 50, 50))
                
            t2 = time.time()
            total_time += t2 - t1
            
            if on_pds:
                pred_pds = tgt.cpu().numpy()  
                W2 += hung(src_pd.to(torch.float), tgt)           
                approx_pi = np.array(pimgr.transform(pred_pds, skew = False))
                
            for i in range(len(PI)):
                total_mse += mean_squared_error(approx_pi[i].flatten(), PI[i].cpu().numpy().flatten())
                
    W2 = total_mse / len(dataloader_test.dataset)
    total_time /= len(dataloader_test.dataset)
    total_mse /= len(dataloader_test.dataset)
    
    if on_pds:
        print('W2: {}'.format(W2))
        
    print('Avg MSE: {}'.format(total_mse))
    print('Avg time on sample : {} seconds on CPU'.format(total_time))
    
    return


def compute_accuracy(dataloader_train, dataloader_test, model_pd, model):
    print('Accuracy on predicted diagrams evaluated on real PDs classificator')
    model_pd.eval()
    model.eval()

    # compare acc on pred pds and real
    # (works if trained on real pds)

    correct_orig, correct_pred = 0.0, 0.0
    for batch in dataloader_train:
        with torch.no_grad():
            src_pd, mask, labels, src_data = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

            pred_pd = model(src_data.unsqueeze(1).repeat(1, 3, 1, 1), mask)

            logits_orig = model_pd(src_pd, mask)
            logits_pred = model_pd(pred_pd, mask)

            correct_orig += (labels == torch.argmax(logits_orig, axis=1)).sum()
            correct_pred += (labels == torch.argmax(logits_pred, axis=1)).sum()

    correct_orig /= len(dataloader_train.dataset)
    correct_pred /= len(dataloader_train.dataset)
    print("Orig train acc : {}\t Pred train acc : {}".format(correct_orig, correct_pred))

    correct_orig, correct_pred = 0.0, 0.0
    for batch in dataloader_test:
        with torch.no_grad():
            src_pd, mask, labels, src_data = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

            pred_pd = model(src_data.unsqueeze(1).repeat(1, 3, 1, 1), mask)

            logits_orig = model_pd(src_pd, mask)
            logits_pred = model_pd(pred_pd, mask)

            correct_orig += (labels == torch.argmax(logits_orig, axis=1)).sum()
            correct_pred += (labels == torch.argmax(logits_pred, axis=1)).sum()

    correct_orig /= len(dataloader_test.dataset)
    correct_pred /= len(dataloader_test.dataset)
    print("Orig test acc : {}\t Pred test acc : {}".format(correct_orig, correct_pred))
    
    return
    
    
def logreg_and_rfc_acc(dataloader_train, dataloader_test, pi_type = 'default', model = None, pimgr = None):
    print(pi_type)
    X_train, X_test = [], []
    y_train, y_test = [], []

    for batch in dataloader_train:
        src_pd, mask, labels, src_data, PI = batch[0], batch[1], batch[2], batch[3], batch[4]
        
        if pi_type == 'default':
            approx_pi = PI.clone()
        elif pi_type == 'pi-net':
            with torch.no_grad():
                out = model(PI.to(device).unsqueeze(1))
                approx_pi = out.squeeze(1).cpu()
        elif pi_type == 'from_pd':
            with torch.no_grad():
                pred_pds = model(src_data.to(device).unsqueeze(1).repeat(1, 3, 1, 1), mask.to(device))
            pred_pds = pred_pds.cpu()
    
            approx_pi = np.array(pimgr.transform(pred_pds, skew = False)).reshape((-1, 50*50))
        else:
            raise NotImplementedError
        
        for img in approx_pi:
            if pi_type != 'from_pd':
                X_train.append(img.numpy())
            else:
                X_train.append(img)
        for label in labels:
            if pi_type != 'from_pd':
                y_train.append(label.numpy())
            else:
                y_train.append(label)

    for batch in dataloader_test:
        src_pd, mask, labels, src_data, PI = batch[0], batch[1], batch[2], batch[3], batch[4]
         
        if pi_type == 'default':
            approx_pi = PI.clone()
        elif pi_type == 'pi-net':
            with torch.no_grad():
                out = model(PI.to(device).unsqueeze(1))
                approx_pi = out.squeeze(1).cpu()
        elif pi_type == 'from_pd':
            with torch.no_grad():
                pred_pds = model(src_data.to(device).unsqueeze(1).repeat(1, 3, 1, 1), mask.to(device))
            pred_pds = pred_pds.cpu().numpy()
    
            approx_pi = np.array(pimgr.transform(pred_pds, skew = False)).reshape((-1, 50*50))
        else:
            raise NotImplementedError
        
        
        for img in approx_pi:
            if pi_type != 'from_pd':
                X_test.append(img.numpy())
            else:
                X_test.append(img)
        for label in labels:
            if pi_type != 'from_pd':
                y_test.append(label.numpy())
            else:
                y_test.append(label)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))
    
    accuracies = []

    for _ in range(5):
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(X_train, y_train)
        accuracies.append(rfc.score(X_test, y_test))

    print("Accuracy random forest {}: {:.4f} ± {:.4f}".format(pi_type, np.mean(accuracies), np.std(accuracies)))
    
    accuracies = []

    for _ in range(5):
        log_reg = LogisticRegression(C=10.0, max_iter = 1000)
        log_reg.fit(X_train, y_train)
        accuracies.append(log_reg.score(X_test, y_test))

    print("Accuracy log reg {}: {:.4f} ± {:.4f}".format(pi_type, np.mean(accuracies), np.std(accuracies)))
       
    return

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
       
    pimgr = PersistenceImager(pixel_size=0.02)
    pimgr.birth_range = (0, 1)
    pimgr.pers_range = (0, 1)
    pimgr.kernel_params['sigma'] = 0.0005 
    
    dataset_train, dataloader_train, n_max_train = get_datasets_pinet(batch_size=64, dataloader=trainloader, filtration='sublevel',
                                                           is_train = True, pimgr = pimgr)
    dataset_test, dataloader_test, n_max_test = get_datasets_pinet(batch_size=64, dataloader=testloader, filtration='sublevel',
                                                           is_train = False, pimgr = pimgr)
    n_max = max(n_max_train, n_max_test)
  
    model = train_full_image_model(dataloader_train, dataloader_test, n_max)
    model_pd_real = train_classifier_on_pds(dataloader_train, dataloader_test, n_max, real_pd = True)
    model_pd_pred = train_classifier_on_pds(dataloader_train, dataloader_test, n_max, real_pd = False, model = model)
    model_pinet = train_pinet(dataloader_train, dataloader_test)
    
    compute_accuracy(dataloader_train, dataloader_test, model_pd_real, model)
    
    logreg_and_rfc_acc(dataloader_train, dataloader_test, 'default')
    logreg_and_rfc_acc(dataloader_train, dataloader_test, 'pi-net', model = model_pinet)
    logreg_and_rfc_acc(dataloader_train, dataloader_test, 'from_pd', model = model, pimgr = pimgr)
    
    print('Metrics on PDs for full model')
    compute_metrics(dataloader_test, model, on_pds = True, pimgr = pimgr)
    print('Metrics on PIs for PI-Net model')
    compute_metrics(dataloader_test, model_pinet, on_pds = False, pimgr = pimgr) 