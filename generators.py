from layers import MLP
import torch
import torch.nn as nn
from torch import Tensor

class TopNGenerator(nn.Module):
    def __init__(self, set_channels: int, cosine_channels: int, max_n: int, latent_dim: int):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.set_channels = set_channels
        self.cosine_channels = cosine_channels
        self.points = nn.Parameter(torch.randn(max_n, set_channels).float()).to(self.device)

        angles = torch.randn(max_n, cosine_channels).float()
        angles = angles / (torch.norm(angles, dim=1)[:, None] + 1e-5)
        self.angles_params = nn.Parameter(angles).to(self.device)

        self.angle_mlp = MLP(latent_dim, self.cosine_channels, 32, 2).to(self.device)

        self.lin1 = nn.Linear(1, set_channels).to(self.device)
        self.lin2 = nn.Linear(1, set_channels).to(self.device)

    def forward(self, latent: Tensor, n: int = None):
        """ latent: batch_size x d
            self.points: max_points x d"""
        
        batch_size = latent.shape[0]

        angles = self.angle_mlp(latent)
        angles = angles / (torch.norm(angles, dim=1)[:, None] + 1e-5)

        cosine = (self.angles_params[None, ...] @ angles[:, :, None]).squeeze(dim=2)
        cosine = torch.softmax(cosine, dim=1)
        # cosine = cosine / (torch.norm(set_angles, dim=1)[None, ...] + 1)        # 1 is here to avoid instabilities
        # Shape of cosine: bs x max_points
        srted, indices = torch.topk(cosine, n, dim=1, largest=True, sorted=True)  # bs x n

        indices = indices[:, :, None].expand(-1, -1, self.points.shape[-1])  # bs, n, set_c
        batched_points = self.points[None, :].expand(batch_size, -1, -1)  # bs, n_max, set_c

        selected_points = torch.gather(batched_points, dim=1, index=indices)

        alpha = self.lin1(selected_points.shape[1] * srted[:, :, None])
        beta = self.lin2(selected_points.shape[1] * srted[:, :, None])
        modulated = alpha * selected_points + beta
        return modulated
    
class MLPGenerator(nn.Module):
    def __init__(self, set_channels : int, max_n : int, mlp_gen_hidden : int, n_layers : int, latent_dim : int):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mlp_gen_hidden = mlp_gen_hidden
        self.latent_channels = latent_dim
        self.set_channels = set_channels
        self.max_n = max_n
        self.mlp = MLP(self.latent_channels, self.max_n * self.set_channels, self.mlp_gen_hidden, nb_layers=n_layers).to(self.device)

    def forward(self, latent: Tensor, n: int):
        batch_size = latent.shape[0]
        points = self.mlp(latent).reshape(batch_size, self.max_n, self.set_channels)
        points = points[:, :n, :]
        return points


class RandomSetGenerator(nn.Module):
    def __init__(self, set_channels : int):
        super().__init__()
        self.set_channels = set_channels
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, latent: Tensor, n: int):
        batch_size = latent.shape[0]
        points = torch.randn(batch_size, n, self.set_channels, dtype=torch.float).to(self.device)
        points = points / math.sqrt(n)
        return points


class FirstKSetGenerator(nn.Module):
    def __init__(self, set_channels : int, max_n : int):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.points = nn.Parameter(torch.randn(max_n, set_channels).float()).to(self.device)

    def forward(self, latent: Tensor, n: int):
        batch_size = latent.shape[0]
        points = self.points[:n].unsqueeze(0).expand(batch_size, -1, -1)
        return points