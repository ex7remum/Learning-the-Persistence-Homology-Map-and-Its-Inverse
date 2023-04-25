import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np
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

class HungarianLossDimensionMatching(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, set1, set2) -> torch.Tensor:
        """ set1 is source, set2 is predicted """
        labels = torch.argmax(set2[:, :, 2:], axis = 2) # bs x n
        max_dim = set2.shape[2] - 2
        
        total_loss = 0
        
        for dim in range(max_dim):
            mask = (labels == dim).unsqueeze(2).repeat(1, 1, set2.shape[2])
            set2_cur = set2 * mask
            set1_cur = set1 * (set1[:, :, 2] == dim).unsqueeze(2).repeat(1, 1, set1.shape[2])
         
            set2_cur = set2_cur[:, :, :2]
            set1_cur = set1_cur[:, :, :2]   
                
            set2_cur_projected = set2_cur
            set2_cur_projected[:, :, 0] += torch.sqrt(set2_cur_projected[:, :, 1] + 1e-8) / 2
            set2_cur_projected[:, :, 1] = 0
            
            set1_cur_projected = set1_cur
            set1_cur_projected[:, :, 0] += torch.sqrt(set1_cur_projected[:, :, 1] + 1e-8) / 2
            set1_cur_projected[:, :, 1] = 0
            
            #concat
            set1_res = torch.cat((set1_cur, set2_cur_projected), axis=0)
            set2_res = torch.cat((set2_cur, set1_cur_projected), axis=0)
            
            batch_dist = torch.cdist(set1_res, set2_res, 2)
        
            numpy_batch_dist = batch_dist.detach().cpu().numpy()            # bs x n x n
            numpy_batch_dist[np.isnan(numpy_batch_dist)] = 1e6

            indices = map(linear_sum_assignment, numpy_batch_dist)
            indices = list(indices)

            loss = [dist[row_idx, col_idx].mean() for dist, (row_idx, col_idx) in zip(batch_dist, indices)]
            total_loss += torch.sum(torch.stack(loss))
        
        return total_loss

class HungarianLossCustom(nn.Module):
    def __init__(self, ce_coeff=10, use_weight=False, distance_penalty=10):
        super().__init__()
        self.ce_coeff = ce_coeff
        self.weight = use_weight
        self.distance_penalty = distance_penalty

    def forward(self, set1, set2) -> torch.Tensor:
        """ set1 is source, set2 is predicted """
        batch_dist = torch.cdist(set1[:, :, :2], set2[:, :, :2], 2)
        
        if self.weight:
            weights = torch.exp(self.distance_penalty * set1[:, :, 1].clone().detach()) # bs x n
            weights = weights.unsqueeze(2).repeat(1, 1, set2.shape[1]) # bs x n x n
            
            batch_dist = torch.mul(batch_dist, weights)
        
        numpy_batch_dist = batch_dist.detach().cpu().numpy()            # bs x n x n
        numpy_batch_dist[np.isnan(numpy_batch_dist)] = 1e6
       
        ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        indices = map(linear_sum_assignment, numpy_batch_dist)
        indices = list(indices)
        
        loss = [dist[row_idx, col_idx].mean() for dist, (row_idx, col_idx) in zip(batch_dist, indices)]
        
        loss2 = 0
        if self.ce_coeff != 0:
            cur = [ce_loss(set2[i, :, 2:][col_idx], set1[i, :, 2].to(torch.long)) for i, (row_idx, col_idx) in enumerate(indices)]
            loss2 = self.ce_coeff * torch.sum(torch.stack(cur)) 
        
        # Sum over the batch (not mean, which would reduce the importance of sets in big batches)
        total_loss = torch.sum(torch.stack(loss)) + loss2
        
        return total_loss

    
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

    
class ChamferLossWeighted(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, set1, set2, w1, w2) -> torch.Tensor:
        """ set1, set2: (bs, N, C)"""
        dist = torch.cdist(set1, set2, 2)
        out_dist, _ = torch.min(dist, dim=2)
        out_dist2, _ = torch.min(dist, dim=1)
        total_dist = (torch.sum(w1 * out_dist) + torch.sum(w2 * out_dist2)) / 2
        return total_dist    
    

class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, set1, set2) -> torch.Tensor:
        """ set1, set2: (bs, N, C)"""
        dist = torch.cdist(set1, set2, 2)
        out_dist, _ = torch.min(dist, dim=2)
        out_dist2, _ = torch.min(dist, dim=1)
        total_dist = (torch.sum(out_dist) + torch.sum(out_dist2)) / 2
        return total_dist

    
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # x, y : (bs, n_max, set_channels)
        mask = (x != -1)
        x = x * mask
        y = y * mask
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        a = (x[:, :, 1]**2) / torch.sum(x[:, :, 1]**2, axis=1).unsqueeze(1)
        
        b = (y[:, :, 1]**2) / torch.sum(y[:, :, 1]**2, axis=1).unsqueeze(1)
        # both marginals are fixed with equal weights
        
        # mu, nu : (bs, n_max)
        #mu = torch.empty(batch_size, x_points, dtype=torch.float,
        #                 requires_grad=False).fill_(1.0 / x_points).squeeze()
        #nu = torch.empty(batch_size, y_points, dtype=torch.float,
        #                 requires_grad=False).fill_(1.0 / y_points).squeeze()

        mu = a.clone().detach()
        #print(torch.sum(mu, axis=1))
        #print(mu)
        nu = b.clone().detach()
        
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-8

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1