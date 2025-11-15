import torch
import torch.nn.functional as F
from .stCAMBL_Lib import stCAMBL_Lib
from tqdm import tqdm
import torch.nn as nn
from .utils_func import WeightInit, drop_edges, drop_feature, isin
from .loss_func import HypersphericalLoss, semi_grad_function, reconstruction_loss, gcn_loss

class stCAMBL:
    def __init__(
            self,
            dataset,
            X,
            graph_dict,
            rec_w=10,   
            gcn_w=0.5,  
            self_w=4,   
            csl_w=5,    
            hsl_w=2,           
            device = 'cuda:0',
            ignore_rate=0.05,   #0.05

    ):
        self.dataset = dataset
        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.self_w = self_w
        self.csl_w = csl_w
        self.hsl_W = hsl_w
        self.device = device
        self.init_param = 64
        self.latent_dim = 64

        if 'mask' in graph_dict:
            self.mask = True
            self.adj_mask = graph_dict['mask'].to(self.device)
        else:
            self.mask = False

        self.cell_num = len(X)

        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]
        self.neg_matrix = torch.ones(self.X.shape[0], self.X.shape[0]).to(self.device)
        self.ex_r = 1.0
        self.ignore_rate = ignore_rate
        self.tau = 6.0

        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)
        self.wi =WeightInit(self.init_param).to(self.device)
 
        self.fc1 = torch.nn.Linear(self.latent_dim, 2 * self.latent_dim).to(self.device)
        self.fc2 = torch.nn.Linear(2 * self.latent_dim, self.latent_dim).to(self.device)

        self.norm_value = graph_dict["norm_value"]
       
        self.model = stCAMBL_Lib(self.dataset, self.cell_num, self.input_dim).to(self.device)
        self.loss_HSL = HypersphericalLoss()

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        z = F.elu(self.fc1(z))
        z = self.fc2(z)
        return z

    def mask_generator(self, N=1):
        self.adj_label = self.adj_label.coalesce()  # 合并稀疏张量
        idx = self.adj_label.indices()

        list_non_neighbor = []
        for i in range(0, self.cell_num):
            neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
            n_selected = len(neighbor) * N

            total_idx = torch.range(0, self.cell_num-1, dtype=torch.float32).to(self.device)
            non_neighbor = total_idx[~isin(total_idx, neighbor)]
            indices = torch.randperm(len(non_neighbor), dtype=torch.float32).to(self.device)
            random_non_neighbor = indices[:n_selected]
            list_non_neighbor.append(random_non_neighbor)

        x = torch.repeat_interleave(self.adj_label.indices()[0], N).to(self.device)
        y = torch.cat(list_non_neighbor).to(self.device)

        self.adj_label = self.adj_label.to(self.device)
        indices = indices.to(self.device)
        indices = torch.stack([x, y])
        indices = torch.cat([self.adj_label.indices(), indices], axis=1)

        value = torch.cat([self.adj_label.values(), torch.zeros(len(x), dtype=torch.float32).to(self.device)])
        adj_mask = torch.sparse_coo_tensor(indices, value)

        return adj_mask

    def train_model(
            self,
            epochs=200,
            lr=0.01,
            decay=0.01,
            N=1,
    ):
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr,
            weight_decay=decay)

        self.model.train()
        if self.dataset in ['lymph']:
            self.rec_w = 10
            self.gcn_w = 2
            self.self_w = 5
            self.hsl_W = 2
            self.csl_w = 2
        if self.dataset in ['human_breast']:
            self.rec_w = 10
            self.gcn_w = 0.5
            self.self_w = 3
            self.hsl_W = 5
            self.csl_w = 2
            self.ignore_rate = 0.5

        if self.dataset in ['fov8', 'fov10', 'fov11', 'fov12', 'fov14', 'fov16', 'fov17']:
            epochs = 200
        elif self.dataset in ['fov1', 'fov2', 'fov3', 'fov4', 'fov5', 'fov6', 'fov7', 'fov9', 'fov13', 'fov15', 'fov18', 'fov19', 'fov20']:
            epochs = 150

        if self.dataset in ['151670','151675','151676']:
            epochs = 100
        elif self.dataset in ['151671']:
            epochs = 600
        elif self.dataset in ['151669']:
            epochs = 200

        elif self.dataset in ['151508', '151507', '151510']:
            epochs = 300
        elif self.dataset in ['151509', '151672', '151673', '151674']:
            epochs = 400
            
        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, _, feat_x, _, loss_self = self.model(self.X, self.adj_norm)

            if self.mask:
                pass
            else:
                adj_mask = self.mask_generator(N=1)
                self.adj_mask = adj_mask
                self.mask = True

            self.edge_index = self.adj_norm.coalesce().indices()
            edge_index_1 = drop_edges(self.adj_norm, drop_prob=0.4) #0.4， 0.5， 0.4， 0.5
            edge_index_2 = drop_edges(self.adj_norm, drop_prob=0.5)
            
            x_1 = drop_feature(self.X, 0.4)
            x_2 = drop_feature(self.X, 0.5)
            latent_z1, mu1, logvar1, _, _, _, _, loss_self1 = self.model(x_1, edge_index_1)
            latent_z2, mu2, logvar2, _, _, _, _, loss_self2 = self.model(x_2, edge_index_2)

            loss_self = (loss_self1 + loss_self2)/2
            
            loss_csl = semi_grad_function(self.projection(latent_z1), self.projection(latent_z2), self.neg_matrix, self.ex_r, self.tau)

            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )
            
            loss_rec = reconstruction_loss(de_feat, self.X)
            loss_hsl = self.loss_HSL(latent_z)

            loss = self.rec_w * loss_rec + self.gcn_w * loss_gcn + self.self_w * loss_self +\
                self.csl_w * loss_csl + self.hsl_W * loss_hsl
            loss.backward()
            self.optimizer.step()

            if epoch == 0:
                    ori_sim = F.normalize(latent_z) @ F.normalize(latent_z).T

            if epoch % 40 == 0:
                with torch.no_grad():  
                    sim = F.normalize(latent_z) @ F.normalize(latent_z).T
                    sim_grad = sim - ori_sim
                    te = sim_grad.reshape(-1).sort()[0][int(sim_grad.size()[0]*sim_grad.size()[1]*self.ignore_rate)]
                    self.neg_matrix = torch.zeros_like(sim_grad).to(self.device)
                    self.neg_matrix[sim_grad>=te] = 1
                    self.ex_r = sim.mean() / ((sim * self.neg_matrix).mean() )
                    ori_sim = sim.clone()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        latent_z, _, _, de_feat, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        
        latent_z = latent_z.data.cpu().numpy()
        de_feat = de_feat.data.cpu().numpy()
        q = q.data.cpu().numpy()
        feat_x = feat_x.data.cpu().numpy()
        gnn_z = gnn_z.data.cpu().numpy()
        
        return latent_z,de_feat, q, feat_x, gnn_z

    def recon(self):
        self.model.eval()
        latent_z, _, _, de_feat, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        de_feat = de_feat.data.cpu().numpy()

        # revise std and mean
        from sklearn.preprocessing import StandardScaler
        out = StandardScaler().fit_transform(de_feat)

        return out
