import scanpy as sc
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spaotsc import SpaOTsc
import seaborn as sns
import os, sys
import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import svds
import anndata as ad
from sklearn.cross_decomposition import CCA
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier  # 解决NameError的关键导入
from scipy.stats import entropy
import umap
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.spatial import KDTree
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import median_abs_deviation
from libpysal.weights import DistanceBand
from esda.geary import Geary
from libpysal.weights import KNN
from scipy.sparse import issparse
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree

#画出细胞类型空间分布
def plot_cell_annotation_sc(
    adata_sp, 
    annotation_list,
    save_dir,
    x="x", 
    y="y", 
    spot_size=None, 
    scale_factor=None, 
    perc=0,
    alpha_img=1.0,
    bw=False,
    ax=None
):
        
    # remove previous df_plot in obs
    adata_sp.obs.drop(annotation_list, inplace=True, errors="ignore", axis=1)
    df = adata_sp.obsm["tangram_ct_pred"][annotation_list]
    construct_obs_plot(df, adata_sp, perc=perc)
    
    #non visium data 
    if 'spatial' not in adata_sp.obsm.keys():
        #add spatial coordinates to obsm of spatial data 
        coords = [[x,y] for x,y in zip(adata_sp.obs[x].values,adata_sp.obs[y].values)]
        adata_sp.obsm['spatial'] = np.array(coords)
    
    if 'spatial' not in adata_sp.uns.keys() and spot_size == None and scale_factor == None:
        raise ValueError("Spot Size and Scale Factor cannot be None when ad_sp.uns['spatial'] does not exist")
    
    #REVIEW
    if 'spatial' in adata_sp.uns.keys() and spot_size != None and scale_factor != None:
        raise ValueError("Spot Size and Scale Factor should be None when ad_sp.uns['spatial'] exists")
    
    # sc.pl.spatial(
    #     adata_sp, color=annotation_list, cmap="viridis", show=False, frameon=False, spot_size=spot_size,
    #     scale_factor=scale_factor, alpha_img=alpha_img, bw=bw, ax=ax
    # )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果目录不存在，创建目录

    for annotation in annotation_list:
        # prefix = extract_prefix(annotation)  # 提取前缀
        # prefix_dir = os.path.join(save_dir, prefix.replace("/", "_"))  # 替换非法字符并创建子目录
        # os.makedirs(prefix_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))  # 确保每次创建独立的 fig 和 ax
        sc.pl.spatial(
            adata_sp, color=annotation, cmap="Greens", show=False, frameon=False, spot_size=spot_size,
            scale_factor=scale_factor, alpha_img=alpha_img, bw=bw, ax=ax
        )

        # 替换非法文件名字符
        sanitized_name = annotation.replace("/", "_")
        file_name = f"{sanitized_name}_plot.png"
        # save_path = os.path.join(prefix_dir, file_name)

        save_path = os.path.join(save_dir, file_name)

        # 保存并关闭图像
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)  # 关闭当前图像，释放内存

    adata_sp.obs.drop(annotation_list, inplace=True, errors="ignore", axis=1)

def one_hot_encoding(l, keep_aggregate=False):
    df_enriched = pd.DataFrame({"cl": l})
    for i in l.unique():
        df_enriched[i] = list(map(int, df_enriched["cl"] == i))
    if not keep_aggregate:
        del df_enriched["cl"]
    return df_enriched

def project_cell_annotations(
    adata_map, adata_sp, annotation="cell_type", threshold=0.5
):

    df = one_hot_encoding(adata_map.obs[annotation])
    if "F_out" in adata_map.obs.keys():
        df_ct_prob = adata_map[adata_map.obs["F_out"] > threshold]

    df_ct_prob = adata_map.X.T @ df
    print(df_ct_prob.shape)
    df_ct_prob.index = adata_map.var.index

    adata_sp.obsm["tangram_ct_pred"] = df_ct_prob
    logging.info(
        f"spatial prediction dataframe is saved in `obsm` `tangram_ct_pred` of the spatial AnnData."
    )

#scc相关性矩阵计算
def compute_pairwise_scc(X1, X2):
    X1 = X1.argsort(axis=1).argsort(axis=1)
    X2 = X2.argsort(axis=1).argsort(axis=1)
    X1 = (X1-X1.mean(axis=1, keepdims=True))/X1.std(axis=1, keepdims=True)
    X2 = (X2-X2.mean(axis=1, keepdims=True))/X2.std(axis=1, keepdims=True)
    sccmat = np.empty([X1.shape[0], X2.shape[0]], float)
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            c = np.dot( X1[i,:], X2[j,:]) / float(X1.shape[1])
            sccmat[i,j] = c
        if i%10000 == 0: print(i)
    return sccmat

def compute_pairwise_scc_gpu(X1, X2, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转为 GPU tensor
    X1 = torch.tensor(X1, dtype=torch.float32, device=device)
    X2 = torch.tensor(X2, dtype=torch.float32, device=device)

    # Step 1: rank transform
    X1_rank = X1.argsort(dim=1).argsort(dim=1).float()
    X2_rank = X2.argsort(dim=1).argsort(dim=1).float()

    # Step 2: 标准化
    X1_rank = (X1_rank - X1_rank.mean(dim=1, keepdim=True)) / (X1_rank.std(dim=1, keepdim=True) + 1e-8)
    X2_rank = (X2_rank - X2_rank.mean(dim=1, keepdim=True)) / (X2_rank.std(dim=1, keepdim=True) + 1e-8)

    n1, n2 = X1_rank.shape[0], X2_rank.shape[0]
    sccmat = torch.empty((n1, n2), device=device, dtype=torch.float32)

    # Step 3: 分块计算避免显存爆掉
    for i in range(0, n1, batch_size):
        end_i = min(i + batch_size, n1)
        sccmat[i:end_i] = torch.matmul(X1_rank[i:end_i], X2_rank.T) / X1_rank.shape[1]
        if i % 2000 == 0:
            print(f"Processed {i}/{n1} rows")

    return sccmat.detach().cpu().numpy()

#cca主成分分析
def run_cca(object1, object2, standardize=True, num_cc=100, seed_use=42, verbose=False):
    if seed_use is not None:
        np.random.seed(seed_use)

    if standardize:
        object1 = (object1 - object1.mean(axis=1, keepdims=True)) / object1.std(axis=1, keepdims=True)
        object2 = (object2 - object2.mean(axis=1, keepdims=True)) / object2.std(axis=1, keepdims=True)

    mat3 = object1.T @ object2

    try:
        U, s, Vt = svds(mat3, k=num_cc, which='LM')  # 'LM' 表示最大模数
        # svds返回的结果默认按升序排列，所以我们需要反转它们
        U = U[:, ::-1]
        s = s[::-1]
        Vt = Vt[::-1]
    except Exception as e:
        print(f"An error occurred during SVD: {e}")
        return None

    cca_data = np.vstack((U, Vt.T))

    def adjust_column_signs(x):
        # 检查列的第一个值，如果为负，则翻转符号
        if np.sign(x[0]) == -1:
            x *= -1
        return x
    cca_data = np.apply_along_axis(adjust_column_signs, axis=0, arr=cca_data)

    return cca_data

def run_cca_gpu(object1, object2, standardize=True, num_cc=100, seed_use=42, verbose=False):
    """
    GPU 加速版 CCA (基于 PyTorch)
    object1, object2: numpy 数组，形状为 (features, samples)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if seed_use is not None:
        torch.manual_seed(seed_use)
        np.random.seed(seed_use)

    # 转换为 GPU tensor
    x1 = torch.tensor(object1, dtype=torch.float32, device=device)
    x2 = torch.tensor(object2, dtype=torch.float32, device=device)

    if standardize:
        x1 = (x1 - x1.mean(dim=1, keepdim=True)) / (x1.std(dim=1, keepdim=True) + 1e-8)
        x2 = (x2 - x2.mean(dim=1, keepdim=True)) / (x2.std(dim=1, keepdim=True) + 1e-8)

    # 矩阵乘法
    mat3 = torch.matmul(x1.T, x2)

    # SVD分解
    try:
        # torch.linalg.svd 比 torch.svd 稳定且更快
        U, S, Vh = torch.linalg.svd(mat3, full_matrices=False)
    except RuntimeError as e:
        print(f"SVD failed on GPU: {e}, falling back to CPU...")
        mat3_cpu = mat3.cpu().numpy()
        from scipy.sparse.linalg import svds
        U, s, Vt = svds(mat3_cpu, k=num_cc, which='LM')
        U, s, Vt = U[:, ::-1], s[::-1], Vt[::-1]
        cca_data = np.vstack((U, Vt.T))
        return cca_data

    # 截取前 num_cc 个成分
    U = U[:, :num_cc]
    V = Vh[:num_cc, :].T

    # 拼接并调整符号
    cca_data = torch.cat([U, V], dim=0)

    # 调整符号一致性
    first_row = cca_data[0, :]
    sign_mask = torch.sign(first_row)
    sign_mask[sign_mask == 0] = 1
    cca_data = cca_data * sign_mask

    # 返回 numpy 数组
    cca_data = cca_data.detach().cpu().numpy()
    return cca_data

# 保存cca结果
def get_cca(sc_data, st_data, path, GPU = False):
    if os.path.exists(os.path.join(path, "cca.npy")):
        print(f"文件 {os.path.join(path, 'cca.npy')} 已存在，跳过计算")
        return
    overloap_genes = list(set(sc_data.var_names).intersection(set(st_data.var_names)))
    print("overloap_genes数量：", len(overloap_genes))
    a = sc_data[:, overloap_genes]
    b = st_data[:, overloap_genes]
    if isinstance(a.X, np.ndarray):  # 已经是密集矩阵
        data1 = a.X
    else:
        data1 = a.X.toarray()
    if isinstance(b.X, np.ndarray):  # 已经是密集矩阵
        data2 = b.X
    else:
        data2 = b.X.toarray()

    if not isinstance(data1, np.ndarray):
        data1 = np.array(data1)
    if not isinstance(data2, np.ndarray):
        data2 = np.array(data2)

    if GPU:
        try:
            from torch import cuda
            if not cuda.is_available():
                print("⚠️ 检测到 GPU 参数为 True，但当前环境无 GPU，将使用 CPU 模式。")
                cca = run_cca(data1.T, data2.T)
            else:
                print("使用 GPU 计算 CCA ...")
                cca = run_cca_gpu(data1.T, data2.T)
        except Exception as e:
            print(f"GPU 计算失败 ({e})，回退到 CPU 模式。")
            cca = run_cca(data1.T, data2.T)
    else:
        print("使用 CPU 计算 CCA ...")
        cca = run_cca(data1.T, data2.T)
    print("cca形状", cca.shape)
    for i in range(0,99):
        if(cca[0,i]<=0):
            print('exist:-')
    np.save(os.path.join(path, "cca.npy"), cca)
    print(f"结果已保存到 {os.path.join(path, 'cca.npy')}")

# 保存scc结果
def get_scc(sc_data, path, GPU = False, batch_size=256):
    if os.path.exists(os.path.join(path, "sccmat_scc.npy")):
        print(f"文件 {os.path.join(path, 'sccmat_scc.npy')} 已存在，跳过计算")
        return
    cca = np.load(os.path.join(path, "cca.npy"))
    X_cca = cca
    X1 = X_cca[:sc_data.shape[0],:]
    X2 = X_cca[sc_data.shape[0]:,:]

    print(f"开始计算 SCC 矩阵，GPU={GPU}")
    if GPU:
        try:
            if not torch.cuda.is_available():
                print("⚠️ GPU 不可用，使用 CPU 版本计算。")
                sccmat = compute_pairwise_scc(X1, X2)
            else:
                print("使用 GPU 计算 Spearman SCC ...")
                sccmat = compute_pairwise_scc_gpu(X1, X2, batch_size=batch_size)
        except Exception as e:
            print(f"GPU 计算失败 ({e})，回退到 CPU。")
            sccmat = compute_pairwise_scc(X1, X2)
    else:
        print("使用 CPU 计算 Spearman SCC ...")
        sccmat = compute_pairwise_scc(X1, X2)
    
    print(sccmat.shape)
    np.save(os.path.join(path, "sccmat_scc.npy"), sccmat)
    print(f"结果已保存到 {os.path.join(path, 'sccmat_scc.npy')}")


def lof_suppress(prob_dist, ad_map, k=10, suppress_factor=0.3):
    
    coords = ad_map.var[['x', 'y']].values  # shape: [n_spots, 2]
    tree = KDTree(coords)

    # 构造每个 spot 的邻域 profile（局部空间中该细胞的分布）
    local_profiles = []
    for j in range(len(prob_dist)):
        dists, idx = tree.query([coords[j]], k=k+1)  # 包括自己
        local_vals = prob_dist[idx[0]]  # 该点及邻居点的值
        local_profiles.append(local_vals)

    local_profiles = np.array(local_profiles)  # shape: [n_spots, k+1]

    # LOF 检测离群点
    lof = LocalOutlierFactor(n_neighbors=k+1, novelty=False)
    lof_score = -lof.fit_predict(local_profiles)  # +1 (inlier), -1 (outlier)

    suppressed_prob = prob_dist.copy()
    outliers = np.where(lof_score == -1)[0]
    suppressed_prob[outliers] *= suppress_factor

    return suppressed_prob


def process_mapping(matrix, thre=0, perc=0, suffix=None):
    
    df_plot = pd.DataFrame(matrix.T)
    
    all_values = df_plot.values.flatten()
    global_quantiles = np.quantile(all_values, [1, 0.99, 0.98, 0.95, 0.9, 0.5])
    print("全局分位数统计:")
    for q, val in zip([0, 1, 2, 5, 10, 50], global_quantiles):
        print(f"{q}%分位数: {val}")
    
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)
    print(df_plot.shape)
    # normalize
    # df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
    df_plot = df_plot / df_plot.sum()
    return df_plot.T


def get_mapping(path, perc = 0):
    if os.path.exists(os.path.join(path, "gamma.subset.npy")):
        print(f"文件 {os.path.join(path, 'gamma.subset.npy')} 已存在，跳过计算")
        return

    sccmat = np.load(os.path.join(path, "sccmat_scc.npy"))
    issc = SpaOTsc.spatial_sc()
    gamma = issc.transport_plan(cost_matrix=np.exp(1-sccmat)**2,
                                cor_matrix=sccmat,
                                alpha=0.0,
                                rho=100.0,
                                epsilon=1.0)
    gamma = process_mapping(gamma, perc)
    print("ganmma的形状：", gamma.shape)
    np.save(os.path.join(path, "gamma.subset.npy"), gamma)
    print(f"结果已保存到 {os.path.join(path, 'gamma.subset.npy')},形状为{gamma.shape}")

def construct_obs_plot(df_plot, perc=0, suffix=None):
    # clip
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)
    print(df_plot.shape)

    # normalize
    df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
    print(df_plot)
    return df_plot


def clip_spatial_prob(spatial_prob, lower_perc=0.05, upper_perc=0.95):
    # 计算分位数
    lower_bound = np.quantile(spatial_prob, lower_perc)
    upper_bound = np.quantile(spatial_prob, upper_perc)
    
    # 执行裁剪
    clipped_prob = np.clip(spatial_prob, lower_bound, upper_bound)
    
    return clipped_prob

def plot_cell(ct, sampled_idx, ad_map, st_data, spot_size, output_dir):
    # 4.3 创建子目录
    ct_dir = os.path.join(output_dir, ct.replace('/','%'))
    os.makedirs(ct_dir, exist_ok=True)
    # 4.4 绘制每个采样细胞
    for idx in sampled_idx:
        prob_dist = ad_map[idx, :].X.flatten()
        prob_dist = lof_suppress(prob_dist, ad_map, k=10, suppress_factor=0.2)
        prob_dist = clip_spatial_prob(prob_dist, lower_perc=0.003, upper_perc=0.997)
        st_data.obs['temp_prob'] = prob_dist
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sc.pl.spatial(
            st_data,
            color='temp_prob',
            cmap='Greens',
            spot_size=spot_size,
            colorbar_loc=None,
            ax=ax,  # 指定绘图的Axes对象
            show=False
        )
        
        # 隐藏所有边框和坐标轴
        ax.set_axis_off()
        ax.set_title('')  # 可选：移除标题（若不需要）
        plt.tight_layout(pad=0)  # 去除所有边距
        
        # 保存图像（需在plt.close之前）
        output_path = os.path.join(ct_dir, f"{idx}.png")
        plt.savefig(
            output_path,
            bbox_inches='tight',
            pad_inches=0,  # 去除保存时的额外边距
            transparent=True  # 透明背景（可选）
        )
        # 清理临时列和图形
        del st_data.obs['temp_prob']
        plt.close(fig)


def cal_spatio(sampled_idx, ad_map, st_data, ct, kde_bandwidth):
    # 空间概率
    if len(sampled_idx) == 0:
        print("采样细胞数量为0")
        return
    combined_prob = np.zeros(ad_map.shape[1])
    for cell_id in sampled_idx:
        spatial_prob = ad_map[cell_id, :].X.toarray().flatten()
        # spatial_prob = lof_suppress(spatial_prob, ad_map, k=10, suppress_factor=0.2)
        combined_prob += spatial_prob

    # prob_series = pd.Series(combined_prob, index=ad_map.var_names, name=f"{ct}_prob")
    # col_name = f"{ct}_prob"
    # st_data.obs[col_name] = prob_series.reindex(st_data.obs_names).values
    # frac_col = f'frac_{ct}'
    # mask = st_data.obs[frac_col] > 0
    # total_prob = prob_series.sum()
    # spatial_prob = (prob_series[mask] * st_data.obs.loc[mask, frac_col]).sum()
    # spatial_ratio = spatial_prob / total_prob if total_prob > 0 else 0

    spatial_ratio = 1

    coords = st_data.obsm['spatial']

    # ===============================
    # KDE 熵（越小越聚集）
    kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth)
    kde.fit(coords, sample_weight=combined_prob)
    log_dens = kde.score_samples(coords)
    dens = np.exp(log_dens)
    dens /= dens.sum()
    kde_entropy = entropy(dens)
    
    return spatial_ratio, kde_entropy

def calculate_cell_entropy(prob_dist, coords, kde_bandwidth):
    """计算单个细胞概率分布在给定坐标上的KDE熵"""
    kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth)
    kde.fit(coords, sample_weight=prob_dist)
    log_dens = kde.score_samples(coords)
    dens = np.exp(log_dens)
    dens /= dens.sum()  # 归一化为概率分布
    return entropy(dens)


def plot_umap(ct, ad_ct, sampled_idx, spatial_ratio, kde_entropy, out_dir, method):

    umap_coords = ad_ct.obsm['X_umap']  # shape: (n_cells, 2) 
    clusters = ad_ct.obs['leiden'].astype(str)
    unique_clusters = sorted(clusters.unique())

    save_dir = os.path.join(out_dir, f"{ct}_umap")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存 UMAP 坐标和聚类标签
    df_umap = pd.DataFrame(
        data=umap_coords,
        columns=['UMAP1', 'UMAP2'],
        index=ad_ct.obs_names
    )
    df_umap['cluster'] = clusters
    df_umap.to_csv(os.path.join(save_dir, f'{ct}_{method}_umap_coords.csv'))
    
    # 2. 保存采样细胞的索引
    pd.Series(sampled_idx).to_csv(
        os.path.join(save_dir, f'{ct}_{method}_sampled_cells.csv'),
        index=False,
        header=['cell_id']
    )
    
    # 可视化
    # 设置颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    # 创建画布
    plt.figure(figsize=(8, 5))
    
    # 逐簇绘图
    for i, clus in enumerate(unique_clusters):
        mask = clusters == clus
        coords = umap_coords[mask.values]
        plt.scatter(coords[:, 0], coords[:, 1],
                    s=10, color=colors[i], label=f'Cluster {clus}', alpha=0.7)
    
    # 标记采样的细胞（黑色星号）
    sampled_coords = ad_ct.obsm['X_umap'][ad_ct.obs_names.isin(sampled_idx)]
    plt.scatter(sampled_coords[:, 0], sampled_coords[:, 1],
                s=50, color='black', marker='*', label='Sampled', zorder=10)
    
    # 图形美化
    plt.title(f"{method} of {ct}", fontsize=14)
    main_legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clusters")
    plt.gca().add_artist(main_legend)  # 添加第一个图例
    
    # 添加指标图例（放在右下角）
    metric_legend = [
        mpatches.Patch(color='white', label=f"spatial_ratio = {spatial_ratio:.2f}"),
        mpatches.Patch(color='white', label=f"kde_entropy = {kde_entropy:.2f}"),
    ]
    plt.legend(handles=metric_legend, 
               bbox_to_anchor=(1.05, 0), 
               loc='lower left', 
               frameon=False,
               handlelength=0,  # 隐藏图例中的颜色块
               handletextpad=0)  # 减少文本和颜色块之间的间距
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{ct}_{method}.svg'))
    # plt.show()


def plot_cluster_spatial(
    ad_ct,
    ad_map,
    st_data,
    cell_type,
    cluster_ids,  # 支持列表，例如：[largest_cluster, smallest_cluster]
    resolution,
    iteration,
    kde_bandwidth,
    output_dir,
    vmin_value=0,
    vmax_value=0.5,
    label_dict=None  # 例如：{largest_cluster: "largest_cluster", smallest_cluster: "smallest_cluster"}
):
    if not pd.api.types.is_categorical_dtype(ad_ct.obs['leiden']):
        ad_ct.obs['leiden'] = ad_ct.obs['leiden'].astype('category')
        print("类别属性不为category")

    cluster_labels = ad_ct.obs['leiden']
    for cluster_id in cluster_ids:
        label = label_dict.get(cluster_id, f"cluster_{cluster_id}" if label_dict else str(cluster_id))

        # 获取当前簇的细胞
        core_cells_mask = ad_ct.obs['leiden'] == cluster_id
        core_cells = ad_ct.obs.index[core_cells_mask]
        core_coords = ad_ct.obsm['X_umap'][core_cells_mask]

        if len(core_cells) <= 20:
            sampled_idx = core_cells
        else:
            center = np.mean(core_coords, axis=0)
            dists = np.linalg.norm(core_coords - center, axis=1)
            top_idx = np.argsort(dists)[:20]
            sampled_idx = np.array(core_cells)[top_idx]

        spatial_ratio, kde_entropy = cal_spatio(sampled_idx, ad_map, st_data, cell_type, kde_bandwidth)
        
        # UMAP 图
        coords = ad_ct.obsm['X_umap']
        coords_df = pd.DataFrame(coords, columns=['UMAP1', 'UMAP2'], index=ad_ct.obs_names)

        # 使用原始的 leiden 标签（字符串），不要转为编码
        unique_clusters = cluster_labels.cat.categories
        n_clusters = len(unique_clusters)
        cmap = plt.get_cmap('tab20') if n_clusters > 10 else plt.get_cmap('tab10')
        color_map = {cluster: cmap(i % cmap.N) for i, cluster in enumerate(unique_clusters)}
        
        # 直接使用列表推导式创建颜色数组
        colors = [color_map[cluster] for cluster in cluster_labels]
        
        plt.figure(figsize=(8, 5))
        plt.scatter(coords_df['UMAP1'], coords_df['UMAP2'],
                   c=colors, s=10, alpha=0.5)
        
        # 高亮采样的细胞
        image_cells_in_sub = [i for i in sampled_idx if i in coords_df.index]
        if image_cells_in_sub:
            highlight_df = coords_df.loc[image_cells_in_sub]
            plt.scatter(highlight_df['UMAP1'], highlight_df['UMAP2'],
                        c='black', marker='*', s=80, label='sample')
        
        # 创建图例
        legend_elements = []
        for cluster in unique_clusters:
            legend_elements.append(
                mpatches.Patch(color=color_map[cluster], label=f"Cluster {cluster}")
            )

        metric_legend = [
            mpatches.Patch(color='white', label=f"spatial_ratio = {spatial_ratio:.2f}"),
            mpatches.Patch(color='white', label=f"kde_entropy = {kde_entropy:.2f}"),
        ]

        plt.legend(handles=legend_elements + metric_legend, loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.title(f"{cell_type}_iter{iteration}_{label}")
        plt.tight_layout()
        umap_dir = os.path.join(output_dir, "umap")
        os.makedirs(umap_dir, exist_ok=True)
        plt.savefig(os.path.join(umap_dir, f'{cell_type}_iter{iteration}_{label}.svg'))
    
    return spatial_ratio


def find_resolution_by_entropy(
    ad_ct,
    ad_map,
    st_data,
    cell_type,
    kde_bandwidth,
    output_dir,
    verbose=True
):
    best_res = None
    best_entropy = float("inf")
    best_cluster = None

    resolution_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

    all_clusters_results = []  # 存储所有簇的结果
    best_per_res = []          # 每个分辨率的最佳簇

    for i, res in enumerate(resolution_list, start=1):
        # 聚类
        sc.tl.leiden(ad_ct, resolution=res, key_added='leiden', random_state=0)
        sc.tl.umap(ad_ct, random_state=0)

        cluster_counts = ad_ct.obs['leiden'].value_counts()
        cluster_ids = cluster_counts.index.tolist()

        max_entropy_cluster = None
        max_entropy_value = float("inf")
        max_entropy_spatial = None
        found_valid_cluster = False

        for cluster_id in cluster_ids:
            # 当前簇的细胞
            coords = ad_ct.obsm["X_umap"][ad_ct.obs['leiden'] == cluster_id]
            idx_in_cluster = ad_ct.obs.index[ad_ct.obs['leiden'] == cluster_id]
            cell_count = len(idx_in_cluster)

            if cell_count < 20:
                continue

            found_valid_cluster = True
            # 计算簇中心 & 最近的 20 个细胞
            center = coords.mean(axis=0)
            distances = np.linalg.norm(coords - center, axis=1)
            closest_idx = np.argsort(distances)[:20]
            core_cells_idx = idx_in_cluster[closest_idx]

            # 计算两个指标
            spatial_ratio, kde_entropy = cal_spatio(core_cells_idx, ad_map, st_data, cell_type, kde_bandwidth)

            # 保存所有簇的结果
            all_clusters_results.append((res, cluster_id, kde_entropy, spatial_ratio))
            cell_count = len(idx_in_cluster)  # 当前簇的细胞数量
            
            if kde_entropy < max_entropy_value:
                max_entropy_value = kde_entropy
                max_entropy_cluster = cluster_id
                max_entropy_spatial = spatial_ratio
        
        # 保存当前分辨率的最佳簇
        if found_valid_cluster:
            best_per_res.append((res, max_entropy_cluster, max_entropy_value, max_entropy_spatial))
            
            # 全局最佳分辨率
            if max_entropy_value < best_entropy:
                best_entropy = max_entropy_value
                best_res = res
                best_cluster = max_entropy_cluster
        else:
            best_per_res.append((res, None, None, None))  # 如果没有符合条件的簇，记录为None
    
        if verbose:
            if found_valid_cluster:
                print(f"[{i}] resolution={res:.3f}, 最佳簇={max_entropy_cluster}, "
                      f"kde_entropy={max_entropy_value:.4f}, spatial_ratio={max_entropy_spatial:.4f}")
            else:
                print(f"[{i}] resolution={res:.3f}, 无符合条件的簇（细胞数≥10）")

        # 全局最佳分辨率
        if max_entropy_value < best_entropy:
            best_entropy = max_entropy_value
            best_res = res
            best_cluster = max_entropy_cluster

    # 计算相关性（基于所有簇）
    res_arr = np.array(all_clusters_results, dtype=object)
    kde_vals = res_arr[:, 2].astype(float)
    spatial_vals = res_arr[:, 3].astype(float)
    corr = np.corrcoef(kde_vals, spatial_vals)[0, 1]

    print(f"\n最终选择分辨率：{best_res} (最佳簇={best_cluster}, kde_entropy={best_entropy:.4f})")
    print(f"所有簇的 kde_entropy 与 spatial_ratio 的相关系数: {corr:.4f}")

    # df_results = pd.DataFrame(all_clusters_results, 
    #                      columns=['resolution', 'cluster', 'kde_entropy', 'spatial_ratio'])
    # df_results.to_csv(f"{output_dir}/spa_rat_{cell_type}.csv", index=False)
    # print(f"结果已保存到: {output_dir}/spa_rat_{cell_type}.csv")

    # # 绘制相关性散点图
    # plt.figure(figsize=(5, 4))
    # plt.scatter(kde_vals, spatial_vals, alpha=0.6)
    # plt.xlabel("kde_entropy")
    # plt.ylabel("spatial_ratio")
    # plt.title(f"{cell_type}: {corr:.4f}")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/kde_spatial_correlation_{cell_type}.svg", dpi=600)
    plt.close()

    return best_res

    
def random(st_data, sc_data, ad_map, ad_ct, ct, ct_names, output_dir):
    print('随机采样')
    spatial_coords = st_data.obsm['spatial']
    n, d = spatial_coords.shape
    kde_bandwidth = np.power(n, -1.0/(d+4))  # Scott's Rule
    sampled_idx = np.random.choice(
        ct_names, 
        size=min(20, len(ct_names)), 
        replace=False
    )
    
    # spatial_ratio, kde_entropy = cal_spatio(sampled_idx, ad_map, st_data, ct, kde_bandwidth)
    # plot_umap(ct, ad_ct, sampled_idx, spatial_ratio, kde_entropy, output_dir, method = 'random')
    
    return sampled_idx

def find_cluster(ad_ct, ad_map, st_data, ct, kde_bandwidth, n_core_cells=20):
    """根据空间指标选择最优簇（使用每个簇的中心n_core_cells个细胞计算）"""
    clusters = ad_ct.obs['leiden'].astype(str).unique()
    cluster_metrics = []
    
    for cluster in clusters:
        # 获取当前簇的所有细胞
        cluster_mask = ad_ct.obs['leiden'] == cluster
        all_cells = ad_ct.obs_names[cluster_mask]
        
        # 如果簇细胞数<=n_core_cells，使用全部细胞
        if len(all_cells) < n_core_cells:
            continue
        else:
            # 计算UMAP坐标中心
            umap_coords = ad_ct.obsm['X_umap'][cluster_mask]
            center = np.mean(umap_coords, axis=0)
            
            # 计算每个细胞到中心的距离
            dists = np.linalg.norm(umap_coords - center, axis=1)
            
            # 选择距离最近的n_core_cells个细胞
            closest_idx = np.argsort(dists)[:n_core_cells]
            core_cells = np.array(all_cells)[closest_idx]
        
        # 计算空间指标（仅使用核心细胞）
        spatial_ratio, kde_entropy = cal_spatio(core_cells, ad_map, st_data, ct, kde_bandwidth)
        
        # 存储指标
        cluster_metrics.append({
            'cluster': cluster,
            'kde_entropy': kde_entropy,
            'spatial_ratio': spatial_ratio,
        })
    
    # 转换为DataFrame便于分析
    metrics_df = pd.DataFrame(cluster_metrics)
    
    # 选择KDE熵最小的簇（熵越小表示空间分布越聚集）
    best_cluster = metrics_df.loc[metrics_df['kde_entropy'].idxmin(), 'cluster']
    
    # 打印各簇指标
    print("\n各簇空间指标比较（基于中心{}个细胞计算）:".format(n_core_cells))
    print(metrics_df.sort_values('kde_entropy'))
    
    return best_cluster


# def plot_probability_spatial(st_data, prob_dist, output_path, title=None, cmap='Greens', spot_size=40):
#     """辅助函数：绘制概率分布的空间图并保存"""
#     # 临时存储概率到adata.obs
#     st_data.obs['temp_prob'] = prob_dist

#     # 创建画布
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
#     # 绘制空间分布
#     sc.pl.spatial(
#         st_data,
#         color='temp_prob',
#         cmap=cmap,
#         spot_size=spot_size,
#         colorbar_loc='right',
#         ax=ax,
#         show=False
#     )
    
#     # 隐藏坐标轴和边框
#     ax.set_axis_off()
#     if title:
#         ax.set_title(title, fontsize=12)  # 可选标题
    
#     # 保存图像
#     plt.savefig(
#         output_path,
#         bbox_inches='tight',
#         pad_inches=0,
#         transparent=True
#     )
    
#     # 清理临时列和图形
#     del st_data.obs['temp_prob']
#     plt.close(fig)


def recluster(st_data, sc_data, ad_map, ad_ct, ct, ct_names, output_dir):
    print("重聚类采样（迭代）")
    
    spatial_coords = st_data.obsm['spatial']
    n, d = spatial_coords.shape
    kde_bandwidth = np.power(n, -1.0/(d+4))  # Scott's Rule
    
    # # Leiden聚类：寻找最佳分辨率，并可视化每次迭代的空间分布
    best_res = find_resolution_by_entropy(
        ad_ct,
        ad_map,
        st_data,
        ct,
        kde_bandwidth,
        output_dir,
        verbose=True
    )
    # celltype_to_resolution = {
    #     "ct1": 1.2,
    #     "ct2": 1.2,
    #     "ct3": 1.0,
    #     "ct4": 0.4,
    # }
    # best_res = celltype_to_resolution.get(ct)
    sc.tl.leiden(ad_ct, resolution=best_res, key_added='leiden', random_state=0)
    sc.tl.umap(ad_ct, random_state=0)
            
    # 6. 获取选定簇的细胞
    best_cluster = find_cluster(ad_ct, ad_map, st_data, ct, kde_bandwidth, n_core_cells=20)
    
    core_cells_mask = ad_ct.obs['leiden'] == best_cluster
    core_cells = ad_ct.obs[core_cells_mask].index
    core_coords = ad_ct.obsm['X_umap'][core_cells_mask]
    
    print(f"{ct} 的最佳子簇: {best_cluster}，包含 {len(core_cells)} 个细胞（占比 {len(core_cells)/len(ad_ct):.1%}）")
    
    # 7. 采样：可以选择全部，或部分采样
    if len(core_cells) <= 20:
        sampled_idx = core_cells
    else:
        # 计算簇中心
        center = np.mean(core_coords, axis=0)  # shape: (2,)
        dists = np.linalg.norm(core_coords - center, axis=1)  # shape: [n_cells_in_cluster]
        top_idx = np.argsort(dists)[:20]
        sampled_idx = np.array(core_cells)[top_idx]

    
    ##仿真数据集验证f
    # spatial_ratio, kde_entropy = cal_spatio(sampled_idx, ad_map, st_data, ct, kde_bandwidth)
    # plot_umap(ct, ad_ct, sampled_idx, spatial_ratio, kde_entropy, output_dir, method = 'recluster')
    
    return sampled_idx



def plot_celltype_probability(ad_map, st_data, sc_data, celltype_col, spot_size, n_samples=20, min_cell=20, inhere = "random", remapped_image_dict = None, output_dir=None):

    if os.path.exists(output_dir):
        print(f"文件 {output_dir} 已存在，跳过计算")
        return

    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 检查输入数据
    assert celltype_col in ad_map.obs, f"细胞类型列 {celltype_col} 不存在"
    
    # 4. 按细胞类型采样
    celltypes = ad_map.obs[celltype_col].unique()
    st_data.obsm['spatial_original'] = st_data.obsm['spatial'].copy()

    # 对 y 轴坐标取反
    st_data.obsm['spatial'][:, 1] = -st_data.obsm['spatial'][:, 1]
    
    for ct in celltypes:
        
        # recluster
        # 1. 获取该细胞类型的所有细胞索引
        ct_mask = sc_data.obs[celltype_col] == ct
        ct_indices = np.where(ct_mask)[0]
        ct_names = sc_data.obs_names[ct_indices]
        if len(ct_indices) < min_cell:
            continue
        
        ad_ct = sc_data[ct_names].copy()
        
        # 3. 特征处理：标准流程
        sc.pp.normalize_total(ad_ct, target_sum=1e4)
        sc.pp.log1p(ad_ct)
        sc.pp.highly_variable_genes(ad_ct, n_top_genes=2000, subset=True)
        sc.tl.pca(ad_ct, n_comps=20)

        sc.pp.neighbors(ad_ct, n_neighbors=10, n_pcs=20)
        sc.tl.leiden(ad_ct, resolution=0.5, key_added='leiden', random_state=0)
        sc.tl.umap(ad_ct, random_state=0)
        
        # # 4.2 随机采样
        # np.random.seed(42) 
        if inhere == "random":
            random_dir = output_dir
            os.makedirs(random_dir, exist_ok=True)
            random_idx = random(st_data, sc_data, ad_map, ad_ct, ct, ct_names, random_dir)
            plot_cell(ct, random_idx, ad_map, st_data, spot_size, random_dir)

        elif inhere == "inhere": 
            # 获取当前细胞类型的细胞索引
            sampled_idx = remapped_image_dict.get(ct, [])
            plot_cell(ct, sampled_idx, ad_map, st_data, spot_size, output_dir)
  
        elif inhere == "recluster":
            recluster_dir = output_dir
            os.makedirs(recluster_dir, exist_ok=True)
            recluster_idx = recluster(st_data, sc_data, ad_map, ad_ct, ct, ct_names, recluster_dir)
            plot_cell(ct, recluster_idx, ad_map, st_data, spot_size, recluster_dir)

        elif inhere == "randomforest":
            randomforest_dir = output_dir + '/randomforest'
            os.makedirs(randomforest_dir, exist_ok=True)
            randomforest_idx = randomforest(st_data, sc_data, ad_map, ad_ct, ct, ct_names, randomforest_dir)
            plot_cell(ct, randomforest_idx, ad_map, st_data, spot_size, randomforest_dir)

        # recluster_dir = output_dir + '/recluster'
        # os.makedirs(recluster_dir, exist_ok=True)
        # recluster_idx = recluster(st_data, sc_data, ad_map, ad_ct, ct, ct_names, recluster_dir)
        # plot_cell(ct, recluster_idx, ad_map, st_data, spot_size, recluster_dir)
        # random_dir = output_dir + '/random'
        # os.makedirs(random_dir, exist_ok=True)
        # random_idx = random(st_data, sc_data, ad_map, ad_ct, ct, ct_names, random_dir)
        # plot_cell(ct, random_idx, ad_map, st_data, spot_size, random_dir)

    # 恢复原始坐标
    st_data.obsm['spatial'] = st_data.obsm['spatial_original'].copy()


def build_image(sc_data, st_data, celltype, spot_size, n_samples, min_cell = 20, ratio = 0.1, inhere = None, remapped_image_dict = None, path = None):
    
    gamma = np.load(os.path.join(path, "gamma.subset.npy"))
    
    sign = sc_data.obs[celltype]
    obs_df = pd.DataFrame({
        celltype: sign.values
    },index = sc_data.obs_names)
    
    ad_map = sc.AnnData(X=gamma,obs=obs_df,var=pd.DataFrame(index=st_data.obs_names))
    ad_map.var['x'] = st_data.obs.loc[ad_map.var.index, 'x'].values
    ad_map.var['y'] = st_data.obs.loc[ad_map.var.index, 'y'].values
    print(ad_map)

    if 'spatial' not in st_data.obsm:
        # 如果不存在，添加 spatial 数据
        spatial_coords = st_data.obs[['x', 'y']].values
        st_data.obsm['spatial'] = spatial_coords
        print("spatial 数据已添加到 obsm 中。")
    else:
        print("obsm 中已存在 spatial 数据，跳过添加。")

    plot_celltype_probability(
        ad_map=ad_map,
        st_data = st_data,
        sc_data = sc_data,
        celltype_col= celltype,
        spot_size = spot_size,
        n_samples=n_samples,
        min_cell = min_cell,
        inhere = inhere,
        remapped_image_dict = remapped_image_dict,
        output_dir = path+'image'
    )
