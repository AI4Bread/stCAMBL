import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch
import scipy
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import stCAMBL
import os
os.environ['R_HOME'] = '/data3/wkcui/env/anaconda3/envs/stCAMBL/lib/R'
random_seed = 2050
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
stCAMBL.set_seed(random_seed)
##########################################################
import matplotlib as mpl
import matplotlib.font_manager as fm
from pathlib import Path
font_path = Path('/data3/wkcui/Arial.ttf')         
fm.fontManager.addfont(str(font_path))
plt.rcParams['font.family'] = 'Arial'

mpl.rcParams['font.size']        = 12  
mpl.rcParams['axes.labelsize']   = 10  
mpl.rcParams['axes.titlesize']   = 12   
mpl.rcParams['xtick.labelsize']  = 10   
mpl.rcParams['ytick.labelsize']  = 10  
mpl.rcParams['legend.fontsize']  = 10
mpl.rcParams['figure.titlesize'] = 15  
###########################################################
# the number of clusters
n_clusters = 26
# read data
file_fold = '/data3/yfchen/stCAMBL/data/mouse_anterior_posterior_brain_merged.h5ad'
adata = sc.read_h5ad(file_fold)
adata.var_names_make_unique()

sc.pp.filter_genes(adata, min_cells=50)
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)

adata = adata[:, adata.var['highly_variable'] == True]

sc.pp.scale(adata)

from sklearn.decomposition import PCA
adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
adata.obsm['X_pca'] = adata_X

graph_dict = stCAMBL.graph_construction(adata, 12)
model = stCAMBL.stCAMBL(adata.obsm['X_pca'], graph_dict, device=device , rec_w=6, gcn_w=6, self_w=4, hsl_w=4, csl_w=1)
model.train_model(epochs=400)
mapgcl_feat, defeat, _, _, _ = model.process()
adata.obsm['emb'] = mapgcl_feat
radius = 50
tool = 'mclust' 
from stCAMBL.clust_func import clustering
clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
# plotting spatial clustering result
adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]
import seaborn as sns
rgb_values = sns.color_palette("tab20", len(adata.obs['mclust'].unique()))
color_fine = dict(zip(list(adata.obs['mclust'].unique()), rgb_values))

plt.rcParams["figure.figsize"] = (17, 6)
sc.pl.embedding(adata, basis="spatial",
                color="mclust",
                s=100,
                palette=color_fine,
                show=False,
                title='Mouse Anterior & Posterior Brain (Section 1)')
file_path = os.path.join('/data3/yfchen/stCAMBL/results', "Mouse anterior and posterior brain_louvain.pdf") 
plt.savefig(file_path) 
