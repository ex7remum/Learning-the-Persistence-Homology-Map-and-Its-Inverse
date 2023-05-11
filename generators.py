from layers import MLP
import torch
import torch.nn as nn
from torch import Tensor

class TopNGenerator(nn.Module):
    def __init__(self, set_channels: int, cosine_channels: int, max_n: int, latent_dim: int):
        super().__init__()
        self.set_channels = set_channels
        self.cosine_channels = cosine_channels
        self.points = nn.Parameter(torch.randn(max_n, set_channels).float())

        angles = torch.randn(max_n, cosine_channels).float()
        angles = angles / (torch.norm(angles, dim=1)[:, None] + 1e-5)
        self.angles_params = nn.Parameter(angles)

        self.angle_mlp = MLP(latent_dim, self.cosine_channels, 32, 2)

        self.lin1 = nn.Linear(1, set_channels)
        self.lin2 = nn.Linear(1, set_channels)

    def forward(self, latent: Tensor, n: int = None, mask: Tensor = None):
        """ latent: batch_size x d
            self.points: max_points x d"""
        
        # TODO:
        # during train we have n fixed but while inference we have to predict n somehow
        
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