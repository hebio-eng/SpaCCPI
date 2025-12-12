# SpaCCPI is a method for analyzing spatial cell communication, which requires scRNA-seq and spatial transcriptome data as input (matched or unmatched).
## Install：
```
conda env create -f environment.yml
```
## example：
```
import os
os.environ["OMP_NUM_THREADS"] = "4"   # OpenMP 线程数
os.environ["MKL_NUM_THREADS"] = "4"   # MKL 线程数
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # NumExpr 线程数
import mi
import scanpy as sc
import pandas as pd
import numpy as np
```
### Using the dataset of aging mice as an example
```
data_name = 'mouse_old_sub'
sc_data = sc.read('../scripts/data/public_mouse_aging/scRNA/mouse_old_sub.h5ad')
st_data = sc.read('../scripts/data/public_mouse_aging/ST/Old_mouse_brain_A1-2/mouse_age.h5ad')
st_data.obs['x'] = st_data.obs['pxl_row']
st_data.obs['y'] = st_data.obs['pxl_col']
sc_data.var_names_make_unique()
st_data.var_names_make_unique()
celltype = 'MajorType'
value_counts = sc_data.obs[celltype].value_counts()
valid_subclasses = value_counts[value_counts > 20].index.tolist()
sc_data = sc_data[sc_data.obs[celltype].isin(valid_subclasses)]
print(sc_data)
print(st_data)
```
### Perform data preprocessing
```
sc.pp.filter_genes(sc_data, min_cells=20)
sc.pp.filter_genes(st_data, min_cells=20)
sc.pp.normalize_total(sc_data, target_sum=1e4)
sc.pp.normalize_total(st_data, target_sum=1e4)
sc.pp.log1p(sc_data)
sc.pp.log1p(st_data)
sc.pp.highly_variable_genes(sc_data, flavor='seurat', n_top_genes=5000)
sc.pp.highly_variable_genes(st_data, flavor='seurat', n_top_genes=5000)
sc_data = sc_data[:, sc_data.var['highly_variable'] == True]
st_data = st_data[:, st_data.var['highly_variable'] == True]
```
### Adjust the size of the image reconstruction site according to the resolution of ST.
```
path = f'./results/{data_name}_MajorType/'
os.makedirs(path, exist_ok=True)
if 'spatial' not in st_data.obsm:
    spatial_coords = st_data.obs[['x', 'y']].values
    st_data.obsm['spatial'] = spatial_coords
    print("spatial location has been added to obsm.")
else:
    print("obsm 中已存在 spatial 数据，跳过添加。")
st_data.obs['uniform_color'] = 1
sc.pl.spatial(
    st_data,
    color='uniform_color', 
    color_map='gray', 
    spot_size=150,
    frameon=False,  
    show=False             
)
```
### Cellular spatial co-localization calculation
```
mi.get_cca(sc_data, st_data, path, GPU = True)
mi.get_scc(sc_data, path, GPU = True)
perc = 0.005
mi.get_mapping(path, perc)
n_samples = 20 #采样数量
min_cell = 20 #至少存在的细胞数量
spot_size = 150 #画图的位点大小
mi.build_image(sc_data, st_data, celltype, spot_size, n_samples, min_cell, inhere = 'recluster', path = path)
mi.distance_score(path)
```
### Inference of cellular communication
#### Here, we use cell2cell to calculate the cell communication relationship of lr (this can be replaced with other cell communication methods based on lr)
```
weight = 0.7 #spatial weight
df = pd.read_csv(f'./results/{data_name}_MajorType/CCI_cell2cell.csv', index_col=0)
img_dis = np.load(path + 'distance/image/image_similarity.npy', allow_pickle=True)
dis_ct = pd.read_csv(path + 'st_score.csv')
CCC = pd.read_csv(f'./results/{data_name}_MajorType/lr_cell2cell.csv')
CCC_pval = pd.read_csv(f'./results/{data_name}_MajorType/lr_value_cell2cell.csv')
mi.cal_CCC(sc_data = sc_data,
           CCC = CCC,
           CCC_pval = CCC_pval,
           dis = img_dis,
           dis_ct = dis_ct,
           threshold = 0.05,
           path = path)
```
