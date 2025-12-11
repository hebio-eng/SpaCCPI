import os
os.environ["OMP_NUM_THREADS"] = "4"   # OpenMP 线程数
os.environ["MKL_NUM_THREADS"] = "4"   # MKL 线程数
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # NumExpr 线程数
from scipy.spatial.distance import pdist, squareform
import scanpy as sc
import libpysal
from esda.moran import Moran_BV
from libpysal.weights import KNN
import concurrent.futures
from functools import partial
import pandas as pd
import lpips
import torch
from PIL import Image
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import itertools
from torch.cuda.amp import autocast
import pickle
import numpy as np
from scipy import stats
import csv
from scipy.spatial import KDTree
from sklearn.neighbors import LocalOutlierFactor


# 全局变量
GLOBAL_KNN = None
GLOBAL_AD_MAP = None
GLOBAL_CELL_DICT = None

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(image_path).convert('RGB') 
    img = transform(img).unsqueeze(0)  # 添加batch维度
    return img

def image_similarity(path):

    if os.path.exists(os.path.join(path, "distance/image/image_similarity.npy")):
        print(f"文件 {os.path.join(path, 'distance/image/image_similarity.npy')} 已存在，跳过计算")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net='alex').to(device)  # 可以选择 'alex', 'vgg', 或 'squeeze'
    root_dir = path + 'image'
    output_dir = path + 'distance/image'
    
    subdirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d)) and d != '.ipynb_checkpoints']
    print(subdirs)
    # 创建一个字典来保存每个子目录的图片路径
    all_subdirs = list(set(subdirs))
    image_dict = {d: [os.path.join(root_dir, d, f) for f in os.listdir(os.path.join(root_dir, d)) 
                      if f.endswith(('.png', '.jpg'))] for d in all_subdirs}
    
    distances = []
    all_distances = []
    distances_both = []
    start_time = time.time()
    # 遍历所有可能的子目录组合
    for i in range(len(subdirs)):
        for j in range(i+1,len(subdirs)):
            group1_name = subdirs[i]
            group2_name = subdirs[j]
            print(f'{group1_name}-{group2_name}')
            group1_images = image_dict[group1_name]
            group2_images = image_dict[group2_name]
    
            image_pairs = list(itertools.product(group1_images, group2_images))
            group_distances = [] 
            
            # for pair in tqdm(image_pairs, desc=f"Calculating distances between {group1_name} and {group2_name}"):
            for pair in image_pairs:
                img1_path, img2_path = pair
                img1 = load_image(img1_path).to(device)
                img2 = load_image(img2_path).to(device)
                cell1 = os.path.basename(img1_path).replace('.png', '')
                cell2 = os.path.basename(img2_path).replace('.png', '')
    
                with torch.no_grad():  # 禁用梯度计算
                    distance = model(img1, img2).squeeze().item()  # 单独计算距离
                group_distances.append((cell1, cell2, distance)) 
            mean_distance = sum(d[2] for d in group_distances) / len(group_distances) if group_distances else 0
            distances_both.append((f"{group1_name.replace('%', '/')} vs {group2_name.replace('%', '/')}", group_distances, mean_distance))
            print(len(distances))
            distances.extend([d[2] for d in group_distances])
    
    all_mean_distance = sum(distances) / len(distances) if distances else 0
    all_distances.append(("based", distances, all_mean_distance))
    all_distances.extend(distances_both)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "image_similarity.npy")
    np.save(output_path, all_distances)
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")


def load_image_batch(image_paths, device):
    """批量加载图片到GPU"""
    batch_imgs = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        batch_imgs.append(img)
    return torch.stack(batch_imgs).to(device)

def batch_image_similarity(path, batch_size=32):
    """批量计算图像相似度"""
    output_path = os.path.join(path, "distance/image/image_similarity.npy")
    if os.path.exists(output_path):
        print(f"文件已存在，跳过计算")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net='alex').to(device)
    model.eval()

    root_dir = os.path.join(path, 'image')
    output_dir = os.path.join(path, 'distance/image')
    os.makedirs(output_dir, exist_ok=True)

    subdirs = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d)) and d != '.ipynb_checkpoints']

    # 创建图片路径字典
    image_dict = {}
    for d in subdirs:
        img_dir = os.path.join(root_dir, d)
        image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_dict[d] = image_paths

    all_distances = []
    distances = []
    distances_both = []
    start_time = time.time()

    # 遍历所有组别组合
    for i in range(len(subdirs)):
        for j in range(i + 1, len(subdirs)):
            group1_name = subdirs[i]
            group2_name = subdirs[j]
            print(f'计算 {group1_name} vs {group2_name}')

            group1_paths = image_dict[group1_name]
            group2_paths = image_dict[group2_name]

            group_distances = []

            for batch1_start in range(0, len(group1_paths), batch_size):
                batch1_paths = group1_paths[batch1_start: batch1_start + batch_size]
                batch1_imgs = load_image_batch(batch1_paths, device)

                for batch2_start in range(0, len(group2_paths), batch_size):
                    batch2_paths = group2_paths[batch2_start: batch2_start + batch_size]
                    batch2_imgs = load_image_batch(batch2_paths, device)

                    # 按照 batch1 x batch2 循环计算，避免显存过大
                    with torch.no_grad(), autocast():
                        for idx1, img1 in enumerate(batch1_imgs):
                            img1_expand = img1.unsqueeze(0).expand(batch2_imgs.size(0), -1, -1, -1)
                            sim = model(img1_expand, batch2_imgs)
                            sim = sim.view(batch2_imgs.size(0)).cpu().numpy()

                            cell1 = os.path.splitext(os.path.basename(batch1_paths[idx1]))[0]
                            for idx2, distance in enumerate(sim):
                                cell2 = os.path.splitext(os.path.basename(batch2_paths[idx2]))[0]
                                group_distances.append((cell1, cell2, float(distance)))

            mean_distance = np.mean([d[2] for d in group_distances]) if group_distances else 0
            distances_both.append((f"{group1_name.replace('%', '/')} vs {group2_name.replace('%', '/')}", group_distances, mean_distance))
            distances.extend([d[2] for d in group_distances])
            print(len(distances))
            print(f'完成 {group1_name} vs {group2_name}')
    all_mean_distance = sum(distances) / len(distances) if distances else 0
    all_distances.append(("based", distances, all_mean_distance))
    all_distances.extend(distances_both)     
    output_path = os.path.join(output_dir, "image_similarity.npy")
    np.save(output_path, all_distances)
    total_time = time.time() - start_time
    print(f"总运行时间: {total_time:.2f} 秒")


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

def maran_bv(var1, var2, knn_weights):
    # print(var1)
    # print(var2)

    # 双变量 Moran's I
    moran_bv = Moran_BV(var1, var2, knn_weights)
    return (moran_bv)

def process_cell_pair(cell_type_pair, knn_weights, ad_map, cell_dict):
    type1, type2 = cell_type_pair
    print(f"Processing {type1} vs {type2}")
    
    # 从cell_df获取两种类型的细胞索引
    cells_type1 = cell_dict.get(type1, [])  # 如果类型不存在，返回空列表
    cells_type2 = cell_dict.get(type2, [])
    
    # 在ad_map中找到这些细胞的索引位置和名称
    idx_name_pairs1 = [(i, name) for i, name in enumerate(ad_map.obs_names) if name in cells_type1]
    idx_name_pairs2 = [(i, name) for i, name in enumerate(ad_map.obs_names) if name in cells_type2]
    
    if not idx_name_pairs1 or not idx_name_pairs2:
        print(f"No matching cells found for {type1} vs {type2}")
        return (f"{type1} vs {type2}", np.array([]), 0)
    
    # 计算Moran's I并保存细胞索引
    a = 0
    group_moran = []
    for (i, name1) in idx_name_pairs1:  # 限制计算数量
        var1 = ad_map.X[i, :]
        var1 = lof_suppress(var1, ad_map, k=10, suppress_factor=0.2)
        for (j, name2) in idx_name_pairs2:
            var2 = ad_map.X[j, :]
            var2 = lof_suppress(var1, ad_map, k=10, suppress_factor=0.2)
            moran = Moran_BV(var1, var2, knn_weights)
            group_moran.append((name1, name2, moran.I))
            a = a+1
    print(a)
    mean_moran = sum(d[2] for d in group_moran) / len(group_moran) if group_moran else 0
    return (f"{type1.replace('%', '/')} vs {type2.replace('%', '/')}", group_moran, mean_moran)



def init_worker(knn_weights, ad_map, cell_dict):
    """初始化子进程，加载大对象到全局变量"""
    global GLOBAL_KNN, GLOBAL_AD_MAP, GLOBAL_CELL_DICT
    GLOBAL_KNN = knn_weights
    GLOBAL_AD_MAP = ad_map
    GLOBAL_CELL_DICT = cell_dict

def process_cell_pair_global(cell_type_pair):
    """子进程函数：直接用全局变量"""
    type1, type2 = cell_type_pair
    print(f"Processing {type1} vs {type2}")

    cells_type1 = GLOBAL_CELL_DICT.get(type1, [])
    cells_type2 = GLOBAL_CELL_DICT.get(type2, [])

    idx_name_pairs1 = [(i, name) for i, name in enumerate(GLOBAL_AD_MAP.obs_names) if name in cells_type1]
    idx_name_pairs2 = [(i, name) for i, name in enumerate(GLOBAL_AD_MAP.obs_names) if name in cells_type2]

    if not idx_name_pairs1 or not idx_name_pairs2:
        print(f"No matching cells found for {type1} vs {type2}")
        return (f"{type1} vs {type2}", np.array([]), 0)

    group_moran = []
    a = 0
    for (i, name1) in idx_name_pairs1:
        var1 = GLOBAL_AD_MAP.X[i, :]
        var1 = lof_suppress(var1, GLOBAL_AD_MAP, k=10, suppress_factor=0.2)
        for (j, name2) in idx_name_pairs2:
            var2 = GLOBAL_AD_MAP.X[j, :]
            var2 = lof_suppress(var2, GLOBAL_AD_MAP, k=10, suppress_factor=0.2)
            moran = Moran_BV(var1, var2, GLOBAL_KNN)
            group_moran.append((name1, name2, moran.I))
            a += 1
    print(f"{type1} vs {type2}: {a} pairs")

    mean_moran = sum(d[2] for d in group_moran) / len(group_moran) if group_moran else 0
    return (f"{type1.replace('%', '/')} vs {type2.replace('%', '/')}", group_moran, mean_moran)


def moran_similarity(celltype: np.ndarray,
                     sc_index: np.ndarray,
                     spatial_coords: np.ndarray,
                     st_index: np.ndarray,
                     path,
                     k = 8):
    if os.path.exists(os.path.join(path, "distance/moran/moran_similarity.npy")):
        print(f"文件 {os.path.join(path, 'distance/moran/moran_similarity.npy')} 已存在，跳过计算")
        return
    root_dir = path + 'image'
    subdirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d)) and d != '.ipynb_checkpoints']
    all_subdirs = list(set(subdirs))
    image_dict = {d: [os.path.join(root_dir, d, f) for f in os.listdir(os.path.join(root_dir, d)) 
                      if f.endswith(('.png', '.jpg'))] for d in all_subdirs}
    cell_dict = {}
    
    for cell_type in subdirs:
        image_paths = image_dict[cell_type]
        cell_indices = [os.path.splitext(os.path.basename(img))[0] for img in image_paths]
        print(cell_type,len(cell_indices))
        cell_dict[cell_type] = cell_indices
    
    gamma = np.load(os.path.join(path, "gamma.subset.npy"))
    obs_df = pd.DataFrame({
        'subclass': celltype.astype(str)
    },index = sc_index)
    ad_map = sc.AnnData(X=gamma,obs=obs_df,var=pd.DataFrame({
            'x': spatial_coords[:, 0],  # 空间X坐标
            'y': spatial_coords[:, 1]   # 空间Y坐标
        }, index = st_index))
    
    outpath = path + 'distance/moran'
    
    cell_types = list(cell_dict.keys()) 
    print(cell_types)
    cell_type_pairs = list(itertools.combinations(cell_types, 2))
    
    k = k
    print('KNN构建')
    knn_weights = KNN(spatial_coords, k=k)
    morans = []
    moran_both = []
    all_moran = []
    
    start_time = time.time()

    ## KNN图每次传递子进程
    # process_cell_pair_with_weights = partial(process_cell_pair, knn_weights=knn_weights, ad_map=ad_map, cell_dict = cell_dict) 
    # with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    #     results = list(executor.map(process_cell_pair_with_weights, cell_type_pairs))


    # KNN设为全局变量
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4,  # 可以调大
        initializer=init_worker,
        initargs=(knn_weights, ad_map, cell_dict)
    ) as executor:
        results = list(executor.map(process_cell_pair_global, cell_type_pairs))
    
    # 收集结果
    for result in results:
        moran_both.append(result)
        morans.extend(result[1]) 
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    
    morans = np.array(morans)
    all_mean_moran = np.mean(morans[:,2].astype(float))
    all_moran.append(("based", morans[:,2].astype(float), all_mean_moran))
    all_moran.extend(moran_both)
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = 'moran_similarity.npy'
    file_path = os.path.join(outpath, filename)
    np.save(file_path, all_moran)

def image_distance(path):
    
    file_path = path + 'distance/image/image_similarity.npy'
    
    if os.path.exists(file_path):
        loaded_data = np.load(file_path, allow_pickle=True)
    else:
        print(f"{file_path} 不存在，跳过image计算。")
        return 1
    
    all_distances = []
    for item in loaded_data[1:]:
        group_pair = item[0] 
        distance_matrix = np.array(item[1])
        distances = [float(x) for x in distance_matrix[:, 2]]
        mean_distance = item[2]
        all_distances.append((group_pair, distances, mean_distance))
    
    first_item = loaded_data[0] 
    all_distances.append((first_item[0],first_item[1],first_item[2]))
    all_distances.sort(key=lambda x: x[2])

    based_element = None
    for item in all_distances:
        group_pair, _, _ = item  
        if "based" in group_pair:  
            based_element = item
            print('based存在')
            break 
    mad_results = {}
    _, based_distances, _ = based_element
    based_median = np.median(based_distances)

    for i, (group_pair, distances, mean_distance) in enumerate(all_distances):
        # print(len(distances))
        if group_pair == based_element[0]: 
            continue  # 跳过 based_element

        distances = np.array(distances, dtype=np.float32)  # 显式转为 NumPy 数组
        # Q1 = np.percentile(distances, 25)
        # Q3 = np.percentile(distances, 75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - 1.5 * IQR
        # upper_bound = Q3 + 1.5 * IQR
        # filtered_distances = distances[(distances >= lower_bound) & (distances <= upper_bound)]
        
        group_median = np.median(distances)
        value = - group_median
    
        # # 将结果存入字典，保存实际的差异值
        # group_mean = np.mean(filtered_distances)
        # value = - group_mean
        
        mad_results[group_pair] = value
        
    mad_results_list = [(key, value) for key, value in mad_results.items()]

    return mad_results_list

def moran_distance(path):
    
    file_path = path + 'distance/moran/moran_similarity.npy'
    
    if os.path.exists(file_path):
        loaded_data = np.load(file_path, allow_pickle=True)
    else:
        print(f"{file_path} 不存在，跳过moran计算。")
        return 1
    
    all_distances = []
    #moran数据
    for item in loaded_data[1:]:
        group_pair = item[0] 
        distance_matrix = np.array(item[1])
        distances = [float(x) for x in distance_matrix[:, 2]]
        mean_distance = item[2]
        all_distances.append((group_pair, distances, mean_distance))
    
    first_item = loaded_data[0]
    all_distances.append((first_item[0],first_item[1],first_item[2]))
    all_distances.sort(key=lambda x: x[2])
    
    based_element = None
    for item in all_distances:
        group_pair, _, _ = item  
        if "based" in group_pair:  
            based_element = item
            print('based存在')
            break 
    mad_results = {}
    _, based_distances, _ = based_element
    based_median = np.median(based_distances.astype(float))

    for i, (group_pair, distances, mean_distance) in enumerate(all_distances):
        # print(len(distances))
        if group_pair == based_element[0]: 
            continue  # 跳过 based_element
            
        distances = np.array(distances, dtype=np.float32)  # 显式转为 NumPy 数组
        # Q1 = np.percentile(distances, 25)
        # Q3 = np.percentile(distances, 75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - 1.5 * IQR
        # upper_bound = Q3 + 1.5 * IQR
        # filtered_distances = distances[(distances >= lower_bound) & (distances <= upper_bound)]
    
        group_median = np.median(distances)
        value = group_median

        # group_mean = np.mean(filtered_distances)
        # value = group_mean
        
        mad_results[group_pair] = value

    mad_results_list = [(key, value) for key, value in mad_results.items()]
    return mad_results_list

def distance_score(path):
    if os.path.exists(path + 'st_score.csv'):
        print(f"文件 {path + 'st_score.csv'} 已存在，跳过计算")
        return

    image_dis = image_distance(path)
    moran_dis = moran_distance(path)

    def standardize_label(label):
        """对标签进行排序标准化，忽略顺序"""
        return " vs ".join(sorted(label.split(" vs ")))

    # 生成 image_dict
    if image_dis != 1:
        image = []
        for item in image_dis:
            group_pair = standardize_label(item[0])
            mad_image = float(item[1])
            image.append((group_pair, mad_image))
        image_dict = dict(image)
    else:
        image_dict = {}

    # 生成 moran_dict
    if moran_dis != 1:
        moran = []
        for item in moran_dis:
            group_pair = standardize_label(item[0].replace(".h5ad", ""))
            mad_moran = float(item[1])
            moran.append((group_pair, mad_moran))
        moran_dict = dict(moran)
    else:
        moran_dict = {}

    # 组合数据
    combined_data = []
    a = 0
    # 遍历所有可能的 group_pair（以 moran_dict 为主）
    for group_pair in set(moran_dict.keys()).union(set(image_dict.keys())):
        mad_image = image_dict.get(group_pair, None)
        mad_moran = moran_dict.get(group_pair, None)

        if moran_dis != 1 and image_dis != 1:
            combined_data.append((group_pair, mad_image, mad_moran))
        elif moran_dis != 1 and image_dis == 1:
            combined_data.append((group_pair, None, mad_moran))
        elif moran_dis == 1 and image_dis != 1:
            combined_data.append((group_pair, mad_image, None))
        else:
            continue  # 都是1，不需要计算

        a += 1

    print("细胞类型对数量：", a)

    if not combined_data:
        print("无数据需要写入，退出。")
        return

    st_correlation = [list(row) for row in combined_data]

    # 分别归一化
    if image_dis != 1:
        mad_image_array = np.array([float(row[1]) for row in st_correlation if row[1] is not None])
        mad_image_min = mad_image_array.min()
        mad_image_max = mad_image_array.max()
        mad_image_normalized = (mad_image_array - mad_image_min) / (mad_image_max - mad_image_min)

    if moran_dis != 1:
        mad_moran_array = np.array([float(row[2]) for row in st_correlation if row[2] is not None])
        mad_moran_min = mad_moran_array.min()
        mad_moran_max = mad_moran_array.max()
        mad_moran_normalized = (mad_moran_array - mad_moran_min) / (mad_moran_max - mad_moran_min)

    st_dis = []
    img_idx, mor_idx = 0, 0
    for row in st_correlation:
        if "%" in row[0]:
            row[0] = row[0].replace("%", "/")
        if image_dis != 1 and row[1] is not None:
            row[1] = mad_image_array[img_idx]
            img_idx += 1
        if moran_dis != 1 and row[2] is not None:
            row[2] = mad_moran_array[mor_idx]
            mor_idx += 1
        st_dis.append(row)

    with open(path + 'st_dis.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Group Pair", "MAD Image", "MAD Moran"])
        writer.writerows(st_dis)

    print(f"已写入：{path + 'st_dis.csv'}")

    # 写入 st_score 列表
    st_score = []
    img_idx, mor_idx = 0, 0
    for row in st_correlation:
        if "%" in row[0]:
            row[0] = row[0].replace("%", "/")
        if image_dis != 1 and row[1] is not None:
            row[1] = mad_image_normalized[img_idx]
            img_idx += 1
        if moran_dis != 1 and row[2] is not None:
            row[2] = mad_moran_normalized[mor_idx]
            mor_idx += 1
        st_score.append(row)

    with open(path + 'st_score.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Group Pair", "MAD Image", "MAD Moran"])
        writer.writerows(st_score)

    print(f"已写入：{path + 'st_score.csv'}")