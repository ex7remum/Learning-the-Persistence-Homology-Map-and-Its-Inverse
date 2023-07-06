import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np
import ot
from ot.sliced import sliced_wasserstein_distance


class SlicedWasserstein(nn.Module):
    def __init__(self, n_projections = 100, projs = None):
        super().__init__()
        if projs is not None:
            self.projs = projs
        else:
            self.n_projections = n_projections
            self.projs = None

    def forward(self, set1, set2) -> torch.Tensor:
        """ set1, set2: (bs, N, C)"""
        loss_batch = 0
        for val, pred in zip(set1, set2):
            a = ((val[:, 1] + val[:, 0])**2 / torch.sum((val[:, 1] + val[:, 0])**2 + 1e-8).unsqueeze(0)).clone().detach()
            b = ((pred[:, 1] + pred[:, 0])**2 / torch.sum((pred[:, 1] + pred[:, 0])**2 + 1e-8).unsqueeze(0)).clone().detach()

            if self.projs is not None:
                loss_batch += sliced_wasserstein_distance(val, pred, a, b, projections=self.projs)
            else:
                loss_batch += sliced_wasserstein_distance(val, pred, a, b, n_projections=self.n_projections)

        return loss_batch

class HungarianLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, set1, set2) -> torch.Tensor:
        """ set1, set2: (bs, N, C)"""
        batch_dist = torch.cdist(set1, set2, 2)
        numpy_batch_dist = batch_dist.detach().cpu().numpy()            # bs x n x n
        numpy_batch_dist[np.isnan(numpy_batch_dist)] = 1e6
        indices = map(linear_sum_assignment, numpy_batch_dist)
        indices = list(indices)
        loss = [dist[row_idx, col_idx].mean() for dist, (row_idx, col_idx) in zip(batch_dist, indices)]
        # Sum over the batch (not mean, which would reduce the importance of sets in big batches)
        total_loss = torch.sum(torch.stack(loss))
        return total_loss

class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, set1, set2) -> torch.Tensor:
        """ set1, set2: (bs, N, C)"""
        dist = torch.cdist(set1, set2, 2)
        out_dist, _ = torch.min(dist, dim=2)
        out_dist2, _ = torch.min(dist, dim=1)
        total_dist = (torch.mean(out_dist) + torch.mean(out_dist2)) / 2
        return total_dist