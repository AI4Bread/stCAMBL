import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class HypersphericalLoss(nn.Module):
    def __init__(self):
        super(HypersphericalLoss, self).__init__()  
    def forward(self, embeddings):
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        cosine_similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        loss = 1 - cosine_similarity_matrix.mean()
        return loss
    
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss
    
def semi_grad_function(z1, z2, neg_matrix, ex_r, tau):
    f = lambda x: torch.exp(x / tau)
    sim_t = F.normalize(z1) @ F.normalize(z2).T
    sim1 = f(sim_t)
    sim2 = f(F.normalize(z1) @ F.normalize(z1).T)
    sim3 = f(F.normalize(z2) @ F.normalize(z2).T)
    l1 = -torch.log(sim1.diag() / (((sim1*neg_matrix).sum(dim=1)+(sim2*neg_matrix).sum(dim=1)) * ex_r +sim1.diag() ) )
    l2 = -torch.log(sim1.diag() / (((sim1*neg_matrix).sum(dim=1)+(sim3*neg_matrix).sum(dim=1)) * ex_r +sim1.diag() ) )
    
    return ((l1 + l2)/2).mean() - sim_t.diag().mean() * (1-1/ex_r)


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn

def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    preds = preds.to(mu.device)
    labels = labels.to(mu.device)
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD
