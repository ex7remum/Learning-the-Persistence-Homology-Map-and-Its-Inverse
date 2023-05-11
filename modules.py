import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.functional import relu
import torch.nn.functional as F
from torch import Tensor
import math
from layers import TransformerLayer, MLP, create_padding_mask, MultiHeadAttention
import numpy as np

##########
#Encoders#
##########

class TransformerEncoder(nn.Module):
    def __init__(self, n_in, embed_dim, fc_dim, num_heads, num_layers, n_out_enc, 
                 dropout = 0.0, reduction="mean", use_skip=True):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.reduction = reduction
        self.embedding = nn.Sequential(
            nn.Linear(n_in, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # (batch, length, 3) -> (batch, length, emb_dim)
        
        
        self.query = nn.parameter.Parameter(
            torch.Tensor(1, embed_dim), requires_grad=True
        )
        self.scaled_dot_product_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, fc_dim, num_heads, dropout, 'relu', use_skip) \
                                     for _ in range(num_layers)])
        
        self.output = nn.Sequential(
            nn.Linear(embed_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, n_out_enc)
        )
        
    def forward(self, X, mask):
        outputs = self.embedding(X) * math.sqrt(self.embed_dim)
        
        #attention_scores = []
        padding_mask = create_padding_mask(mask)
        for layer in self.layers:
            outputs, attention_score = layer(outputs, padding_mask)
        
        if self.reduction == "mean":
            lengths = mask.sum(dim=1).detach()
            outputs = (outputs * mask.unsqueeze(2)).sum(dim=1) / lengths.unsqueeze(1)
        elif self.reduction == "attention":
            outputs, _ = self.scaled_dot_product_attention(
                self.query.expand(outputs.shape[0], -1, -1),
                outputs,
                outputs,
                mask = padding_mask[:, 0, :].unsqueeze(1),
            )
            outputs = outputs.squeeze(dim=1)
        else:
            raise NotImplementedError
            
        # outputs : (batch_size, emb_dim)
        
        z = self.output(outputs)
        # z : (batch_size, n_out_enc)
        
        return z
    
    
class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, num_layers):
        super(MLPEncoder, self).__init__()
        self.mlp = MLP(dim_in=n_in, width=n_hidden, dim_out=n_out, nb_layers=num_layers)
        
    def forward(self, X, mask):
        # X : (batch_size, max_len_batch, set_channels)
        # mask : (batch_size, max_len_batch) 1, if real, 0, if padding
        X = self.mlp(X)
        lengths = mask.sum(dim=1).detach()
        outputs = (X * mask.unsqueeze(2)).sum(dim=1) / lengths.unsqueeze(1) # aggregating
        return outputs # (batch_size, latent_dim)

    
# Persformer from https://arxiv.org/abs/2112.15210
class CustomPersformer(nn.Module):
    def __init__(self, n_in, embed_dim, fc_dim, num_heads, num_layers, n_out_enc, 
                 dropout = 0.1, reduction="mean", use_skip=True):
        super(CustomPersformer, self).__init__()
        self.embed_dim = embed_dim
        self.reduction = reduction
        self.embedding = nn.Sequential(
            nn.Linear(n_in, embed_dim),
            nn.GELU(),
        )
        # (batch, length, 3) -> (batch, length, emb_dim)
        
        
        self.query = nn.parameter.Parameter(
            torch.Tensor(1, embed_dim), requires_grad=True
        )
        self.scaled_dot_product_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, fc_dim, num_heads, dropout, 'gelu', use_skip) \
                                     for _ in range(num_layers)])
        
        self.output = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_out_enc),
        )
        
    def forward(self, X, mask):
        outputs = self.embedding(X) * math.sqrt(self.embed_dim)
        
        #attention_scores = []
        #print(mask.shape)
        padding_mask = create_padding_mask(mask)
        for layer in self.layers:
            outputs, attention_score = layer(outputs, padding_mask)
        
        if self.reduction == "mean":
            lengths = mask.sum(dim=1).detach()
            outputs = (outputs * mask.unsqueeze(2)).sum(dim=1) / lengths.unsqueeze(1)
        elif self.reduction == "attention":
            #print(padding_mask.shape)
            outputs, _ = self.scaled_dot_product_attention(
                self.query.expand(outputs.shape[0], -1, -1),
                outputs,
                outputs,
                mask = padding_mask[:, 0, :].unsqueeze(1),
            )
            outputs = outputs.squeeze(dim=1)
        else:
            raise NotImplementedError
            
        # outputs : (batch_size, emb_dim)
        
        z = self.output(outputs)
        # z : (batch_size, n_out_enc)
        
        return z
    
    
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


#adapted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
class PointNet(nn.Module):
    def __init__(self, emb_dims, dim_in=3, output_channels=40):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(dim_in, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x, mask=None):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

    
#adapted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
class DGCNN(nn.Module):
    def __init__(self, k, dim_in, emb_dims, dropout, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(dim_in * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, mask=None):
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


##########
#Decoders#
##########


def _get_full_mask(X, mask):
    full_mask = np.zeros(X.shape)
    for i in range(X.shape[0]):
        full_mask[i, :mask[i].sum()] = np.ones(X.shape[-1])
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    full_mask = torch.tensor(full_mask).long().to(device)
    return full_mask


class TransformerDecoder(nn.Module):
    def __init__(self, n_in, latent_dim, fc_dim, num_heads, num_layers, n_out, generator, 
                 n_out_lin, n_hidden, num_layers_lin, dropout = 0.1, use_conv=False, last_mlp_width=32, last_mlp_layers=2):
        super(TransformerDecoder, self).__init__()
        self.set_generator = generator
        
        self.film_mlp = MLP(dim_in=n_in, width=n_hidden, dim_out=n_out_lin, nb_layers=num_layers_lin, 
                            dim_in_2=latent_dim, modulation='film')
        
        self.layers = nn.ModuleList([TransformerLayer(n_out_lin, fc_dim, num_heads, dropout) \
                                     for _ in range(num_layers)])
        
        self.use_conv = use_conv
        if use_conv:
            self.output = nn.Conv1d(n_out_lin, n_out, 1)
        else:
            self.output = MLP(dim_in=n_out_lin, width=last_mlp_width, dim_out=n_out, nb_layers=last_mlp_layers)
        
    def forward(self, latent, n_max, mask):
        ref_set = self.set_generator(latent, n_max, mask)
        
        full_mask = _get_full_mask(ref_set, mask)
        
        # concat version
        #outputs = torch.cat((ref_set, latent.unsqueeze(1).repeat(1, n_max, 1)), 2)
        # (batch_size, n_max_batch, set_channels + latent_dim) 
        #outputs_full_mask = get_full_mask(outputs, mask)
        #outputs = outputs * outputs_full_mask + (1 - outputs_full_mask) * (-1)
        
        # film version
        outputs = self.film_mlp(ref_set, latent.unsqueeze(1).repeat(1, n_max, 1))
        # (batch_size, n_max_batch, n_out)
        
        #attention_scores = []
        padding_mask = create_padding_mask(full_mask[:, :, 0])
        for layer in self.layers:
            outputs, attention_score = layer(outputs, padding_mask)
        
        if self.use_conv:
            z = self.output(outputs.transpose(1, 2)).transpose(1, 2)
        else:
            z = self.output(outputs)
        # z : (batch_size, n_max_batch, set_channels)
        
        return z * full_mask
    
    
class MLPDecoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, num_layers, set_channels):
        super(MLPDecoder, self).__init__()
        self.mlp = MLP(dim_in=n_in, width=n_hidden, dim_out=n_out, nb_layers=num_layers)
        self.set_channels = set_channels
        
    def forward(self, latent, max_batch_len, mask):
        z = self.mlp(latent)
        z = z.reshape(latent.shape[0], -1, self.set_channels)
        full_mask = _get_full_mask(z[:, :max_batch_len, :], mask)
        return full_mask * z[:, :max_batch_len, :]
    
    
class GeneratorMLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, num_layers, generator):
        super(GeneratorMLP, self).__init__()
        self.mlp = MLP(dim_in=n_in, width=n_hidden, dim_out=n_out, nb_layers=num_layers)
        self.generator = generator
        
    def forward(self, latent, max_batch_len, mask):
        latent = self.generator(latent, max_batch_len, mask)
        # (bs, n_batch, n_in)
        z = self.mlp(latent)
        full_mask = _get_full_mask(z, mask)
        return full_mask * z
    