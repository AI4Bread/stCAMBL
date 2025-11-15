import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import partial
from .loss_func import sce_loss

def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )

# GCN Layer
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, mask):
        col = mask.coalesce().indices()[0]
        row = mask.coalesce().indices()[1]
        result = self.act(torch.sum(z[col] * z[row], axis=1))

        return result


class stCAMBL_Lib(nn.Module):
    def __init__(
            self,
            dataset,
            cell_dim,
            gene_dim,
            feat_hidden1=64,
            feat_hidden2=32,
            gcn_hidden1=64,
            gcn_hidden2=32,
            p_drop=0.2,
            alpha=1.0,
            dec_clsuter_n=10,
    ):
        super(stCAMBL_Lib, self).__init__()
        self.dataset = dataset
        self.gene_dim = gene_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_clsuter_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.gene_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        self.decoder = GraphConvolution(self.latent_dim, self.gene_dim, self.p_drop, act=lambda x: x)

        # GCN layers
        self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)

        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.gcn_hidden2 + self.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.enc_mask_token = nn.Parameter(torch.zeros(cell_dim, gene_dim))
        if self.dataset in ['lymph','human_breast']:
            self._mask_rate = 0.5
        else :
            self._mask_rate = 0.7
        self.criterion = self.setup_loss_fn(loss_fn='mse')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return criterion

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        adj, x, mask_matrix = self.encoding_mask_noise(adj, x, self._mask_rate)

        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z, adj)

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()


        x_init = x * mask_matrix
        x_rec = de_feat * mask_matrix 

        loss = self.criterion(x_rec, x_init)  

        return z, mu, logvar, de_feat, q, feat_x, gnn_z, loss
    
    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        num_genes = x.shape[1]
        
        perm_nodes = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm_nodes[:num_mask_nodes]
        
        mask_matrix = torch.zeros(num_nodes, num_genes, device=x.device)
        mask_matrix[mask_nodes, :] = 1.0   
        # mask_pre = mask_matrix.clone()
        
        for col in range(num_genes):
            if self.dataset in ['human_breast']:
                if col < 49:
                    mask_prob = 0.5
                elif 49 <= col < 99:
                    mask_prob = 0.3
                elif 99 <= col < 199:
                    mask_prob = 0.1 
                else:
                    continue
            else:
                if col < 49:
                    mask_prob = 0.1
                elif 49 <= col < 99:
                    mask_prob = 0.05  
                elif 99 <= col < 199:
                    mask_prob = 0.01 
                else:
                    continue
            
              
            candidate_rows = torch.where(mask_matrix[:, col] == 1)[0]
            
            if len(candidate_rows) == 0:
                continue

            num_to_mask = int(len(candidate_rows) * mask_prob)
            num_to_mask = max(1, num_to_mask)  

            selected = torch.randperm(len(candidate_rows), device=x.device)[:num_to_mask]
            rows_to_mask = candidate_rows[selected]
            
            mask_matrix[rows_to_mask, col] = 0
        
        out_x = x.clone()
        out_x += mask_matrix * self.enc_mask_token
        
        return adj.clone(), out_x, mask_matrix
