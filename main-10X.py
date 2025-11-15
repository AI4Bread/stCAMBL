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
#Please change this path to your local R environment path
os.environ['R_HOME'] = '/data3/wkcui/env/anaconda3/envs/stCAMBL/lib/R'
# 0.52661, 0.62050 ,0.50143 ,0.71868,0.71445,0.69403, 0.71035, 0.66576,0.54958,0.59229,0.46019,0.45463
###################################
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
random_seed = 2050
stCAMBL.set_seed(random_seed)
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
# You can change the dataset to other 12 DLPFC slices
dataset = '151676'
n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7
file_name = dataset
# Please change this path to your local data path
file_fold = '/data3/yfchen/stCAMBL/data/10X/' + file_name 
adata = sc.read_visium(file_fold, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
adata.obs['layer_guess'] = df_meta['layer_guess']
adata.layers['count'] = adata.X.toarray()
# Data preprocessing
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)

if scipy.sparse.issparse(adata.X):
    adata.X = adata.X.toarray()
    
from sklearn.decomposition import PCA
adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)

adata.obsm['X_pca'] = adata_X
graph_dict = stCAMBL.graph_construction(adata, 12)
model = stCAMBL.stCAMBL(dataset, adata.obsm['X_pca'], graph_dict, device=device)
# Begin to train the model
model.train_model(epochs=300)
stCAMBL_feat, defeat, _, _, _ = model.process()
adata.obsm['emb'] = stCAMBL_feat

radius = 50
tool = 'mclust' 
from stCAMBL.clust_func import clustering
clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
# Calculate ARI
ARI = metrics.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['domain'])
adata.uns['ARI'] = ARI
print('Dataset:', dataset)
print('ARI:', ARI)   

# # plotting spatial clustering result
# sc.pl.spatial(adata,
#           img_key="hires",
#           color=["layer_guess", "domain"],
#           title=["Ground truth", "ARI=%.4f"%ARI],
#           show=False)

# file_path1 = os.path.join('/data3/yfchen/stCAMBL/results/10X/clustering', dataset+"_clustering.pdf") 
# plt.savefig(file_path1) 

# # plotting predicted labels by UMAP
# sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10)
# sc.tl.umap(adata)
# sc.pl.umap(adata, color='domain', title=['Predicted labels'], show=False)

# file_path2 = os.path.join('/data3/yfchen/stCAMBL/results/10X/UMAP', dataset+"_UMAP.pdf")
# plt.savefig(file_path2)  
