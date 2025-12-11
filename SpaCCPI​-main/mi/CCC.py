import numpy as np
import pandas as pd
import scanpy as sc
import csv
import ast
from scipy import sparse, stats
import re
from scipy.sparse import issparse

def normalize(df):
    current_min = df.min().min()
    current_max = df.max().max()
    
    # 特殊情况处理
    if current_min == current_max:
        print('所有细胞类型对得分相同，请检查')
        return df  # 全等数据直接返回
    
    # 检查是否已归一化
    if 0.0 <= current_min <= 1.0 and 0.0 <= current_max <= 1.0:
        print('df已经归一化，跳过')
        return df
    
    # 执行归一化
    return (df - current_min) / (current_max - current_min)

def cal_CCI(df, weight, path):
    # df：原始的CCI得分
    
    st_path = path + 'st_score.csv' 
    
    with open(st_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader) 
        st_correlation = list(reader)   


    valid_cells = set()
    for group_pair, _, _ in st_correlation:
        cells = group_pair.split(" vs ")
        valid_cells.update(cells)

    rows_to_drop = [cell for cell in df.index if cell not in valid_cells]
    cols_to_drop = [cell for cell in df.columns if cell not in valid_cells]
    
    print("被删除的行（细胞类型）:", rows_to_drop)
    print("被删除的列（细胞类型）:", cols_to_drop)
    
    # 保留 df_moran 中的行和列只包含 valid_cells
    df = df.loc[df.index.isin(valid_cells), df.columns.isin(valid_cells)]

    
    df = normalize(df)
    
    # # 判断 st_correlation 的 moran 列是否全为空或全为 None
    # moran_values = [item[2] for item in st_correlation if item[2] not in ["", None]]
    # if len(moran_values) == 0:
    #     print("st_correlation 中的 Moran 列全为空，跳过计算。")
    # else:
    #     df_moran = df.copy()
    
    #     # 遍历 df 的每一个元素
    #     for row in df_moran.index:
    #         for col in df_moran.columns:
    #             if row == col:
    #                 df_moran.at[row, col] = float('nan')
    #                 continue
    
    #             matched = False
    #             for group_pair, mad_distance, mad_i in st_correlation:
    #                 keywords = group_pair.split(" vs ")
    #                 if (row in keywords) and (col in keywords):
    #                     matched = True
    #                     df_moran.at[row, col] = (1-weight)*df_moran.at[row, col] + weight*float(mad_i)
    
    #             if not matched:
    #                 print(f"{row}-{col}的空间距离为空")
    #                 df_moran.at[row, col] *= weight
    
    #     # 归一化
    #     min_moran = df_moran.min().min()
    #     max_moran = df_moran.max().max()
    #     final_moran = (df_moran - min_moran) / (max_moran - min_moran)
        
    #     final_moran.to_csv(path + f'moran_{method}_CCI.csv')
    #     print(f"已保存：{path + f'moran_{method}_CCI.csv'}")

    
    image_values = [item[1] for item in st_correlation if item[1] not in ["", None]]
    if len(image_values) == 0:
        print("st_correlation 中的 image列全为空，跳过计算。")
    else:
        df_image = df.copy()
        for row in df_image.index:
            for col in df_image.columns:
                if row == col:
                    df_image.at[row, col] = float('nan')
                    continue
    
                matched = False
                # 检查当前元素的行列名称是否与组别中的两个关键词一致
                for group_pair, mad_distance, mad_i in st_correlation:
                    keywords = group_pair.split(" vs ")
                    if (row in keywords) and (col in keywords):
                        matched = True
                        df_image.at[row, col] = (1-weight)*df_image.at[row, col] + weight*float(mad_distance)
    
                if not matched:
                    df_image.at[row, col] *= weight   
        final_image = df_image
        min_image = final_image.min().min()  # 全局最小值
        max_image = final_image.max().max()  # 全局最大值
        final_image = (final_image - min_image) / (max_image - min_image)
        final_image.to_csv(path + 'CCI.csv')
        print(f"已保存：{path + 'CCI.csv'}")

def receptors_processe(lr_list, gene_to_idx):
    """处理包含复合受体的配受体对（支持+、_等分隔符）"""
    filtered_lr_pairs = []
    separator_pattern = re.compile(r'[+_]')  # 匹配+或_作为分隔符
    
    for lr_pair in lr_list:
        try:
            ligand, receptor = lr_pair.split(" - ")
        except ValueError:
            print(f"警告：跳过格式错误的配受体对 - {lr_pair}")
            continue
        
        # 检查是否为复合受体（包含分隔符）
        if re.search(separator_pattern, receptor):
            single_receptors = re.split(separator_pattern, receptor)
            # 验证所有基因都存在
            if all(gene in gene_to_idx for gene in [ligand] + single_receptors):
                for single_rec in single_receptors:
                    filtered_lr_pairs.append(f"{ligand} - {single_rec}")
        else:
            # 常规单受体情况
            if ligand in gene_to_idx and receptor in gene_to_idx:
                filtered_lr_pairs.append(lr_pair)
    
    return filtered_lr_pairs

def splide_lr(s):
    import ast
    # 尝试当作列表字符串解析
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
    except:
        pass
    # 如果不是列表字符串，那就当作逗号分隔字符串处理
    return [item.strip() for item in s.split(',') if item.strip()]

def preprocess_lr_pairs(lr_string):
    raw_pairs = splide_lr(lr_string)
    processed_pairs = []
    
    for pair in raw_pairs:
        # 清理特殊字符（如开头结尾的[ ]）
        pair = re.sub(r'^[\s\[\'\"]+|[\s\]\'\"]+$', '', pair.strip())
        if not pair or " - " not in pair:
            continue
        
        # 拆分配体和受体部分
        ligand, receptor = map(str.strip, pair.split(" - ", 1))
        
        # 进一步清理受体中的特殊字符
        receptor = re.sub(r'[\'\"\[\]]', '', receptor)
        
        # 拆分复合受体（支持+ _ &等分隔符）
        single_receptors = re.split(r'[_+&]', receptor)
        single_receptors = [r.strip() for r in single_receptors if r.strip()]
        
        if not single_receptors:  # 如果受体为空则跳过
            continue
            
        for rec in single_receptors:
            processed_pairs.append(f"{ligand.upper()} - {rec.upper()}")
    
    return processed_pairs

def filter_CCC(dis, CCC, gene_to_idx, cell_to_idx, X_dense):

    cell_to_idx = {k: int(v) for k, v in cell_to_idx.items()}
    gene_to_idx = {k: int(v) for k, v in gene_to_idx.items()}
    # 保存所有细胞对的结果
    results_dict = {}
    for idx, row in CCC.iterrows():
        type_pair = row['cell_pair']
        lr_string = row['lr_pairs']
        try:
            lr_list = preprocess_lr_pairs(lr_string)
        except (SyntaxError, ValueError):
            print(f"无法解析LR对字符串: {lr_string}")
            continue
        cell1, cell2 = type_pair.split('|')
        target_parts = {cell1, cell2}  # 用于筛选 img_dis 中的细胞对
        print(target_parts)
    
        
        mask = np.array([
            (" vs " in row[0]) and
            (set(row[0].split(" vs ")) == target_parts)
            for row in dis
        ])
        
        # 如果没有符合的距离数据，跳过
        if not mask.any():
            continue
    
        filtered = dis[mask, 1]
        flat = filtered[0]
        df = pd.DataFrame(flat, columns=['cell1', 'cell2', 'dis'])
    
        df['dis_norm'] = (df['dis'] - df['dis'].min()) / (df['dis'].max() - df['dis'].min())
    
        # 提取表达数据
        cell1_indices = df['cell1'].map(cell_to_idx).values
        cell2_indices = df['cell2'].map(cell_to_idx).values

    
        # 只保留能匹配的 ligand-receptor 对
        filtered_lr_list = receptors_processe(lr_list, gene_to_idx)
        print("匹配的配受体对数量", len(filtered_lr_list))
    
        results = []
        for lr in filtered_lr_list:
            ligand, receptor = lr.split(" - ")
    
            # 提取表达量并计算 sqrt(product)
            cell1_expr = X_dense[cell1_indices, gene_to_idx[ligand]] + X_dense[cell1_indices, gene_to_idx[receptor]]
            cell2_expr = X_dense[cell2_indices, gene_to_idx[ligand]] + X_dense[cell2_indices, gene_to_idx[receptor]]
            expr = np.sqrt(cell1_expr * cell2_expr)
    
            # 距离
            distances = df['dis_norm'].values
    
            if len(expr) > 1 and len(distances) > 1:
                corr, pvalue = stats.spearmanr(expr, distances)
            else:
                corr, pvalue = np.nan, np.nan
    
            results.append({
                'LR_pair': lr,
                'correlation': corr,
                'pvalue': pvalue,
            })
    
        # 保存每一个细胞对的结果
        results_dict[type_pair] = results
    return results_dict

from scipy.stats import combine_pvalues
def combine_with_fisher(row):
    # 条件：若 spearman_corr >= 0 或 spearman_pval 缺失，就保留原始 pval
    if (row['spearman_corr'] >= 0) or np.isnan(row['spearman_pval']):
        return row['pval']
    else:
        # Fisher’s method 合并两个独立 p 值
        try:
            _, p_combined = combine_pvalues([row['pval'], row['spearman_pval']], method='fisher')
            return p_combined
        except Exception:
            return np.nan  # 避免异常（如非法值）


def filter_CCC_pval(dis, CCC_pval, gene_to_idx, cell_to_idx, X_dense, corr_threshold=0.0):
    cell_to_idx = {k: int(v) for k, v in cell_to_idx.items()}
    gene_to_idx = {k: int(v) for k, v in gene_to_idx.items()}

    spearman_corrs = []
    spearman_pvals = []
    pearson_corrs = []
    pearson_pvals = []
    print(f'筛选前数量：{len(CCC_pval)}')

    for idx, row in CCC_pval.iterrows():
        type_pair = row['cell_pair']
        ligand = row['ligand']
        receptor = row['receptor']

        cell1_type, cell2_type = type_pair.split('|')
        target_parts = {cell1_type, cell2_type}

        # 筛选 dis 中对应的细胞类型对
        mask = np.array([
            (" vs " in r[0]) and (set(r[0].split(" vs ")) == target_parts)
            for r in dis
        ])
        if not mask.any():
            spearman_corrs.append(np.nan)
            spearman_pvals.append(np.nan)
            pearson_corrs.append(np.nan)
            pearson_pvals.append(np.nan)
            continue

        filtered = dis[mask, 1]
        flat = filtered[0]
        df_dis = pd.DataFrame(flat, columns=['cell1', 'cell2', 'dis'])
        df_dis['dis_norm'] = (df_dis['dis'] - df_dis['dis'].min()) / (df_dis['dis'].max() - df_dis['dis'].min())

        # 获取表达矩阵索引
        cell1_indices = df_dis['cell1'].map(cell_to_idx).values
        cell2_indices = df_dis['cell2'].map(cell_to_idx).values

        # 检查基因是否在表达矩阵里
        if ligand not in gene_to_idx or receptor not in gene_to_idx:
            # print(f'{ligand}-{receptor}不存在')
            spearman_corrs.append(np.nan)
            spearman_pvals.append(np.nan)
            pearson_corrs.append(np.nan)
            pearson_pvals.append(np.nan)
            continue

        cell1_expr = X_dense[cell1_indices, gene_to_idx[ligand]] + X_dense[cell1_indices, gene_to_idx[receptor]]
        cell2_expr = X_dense[cell2_indices, gene_to_idx[ligand]] + X_dense[cell2_indices, gene_to_idx[receptor]]
        expr = np.sqrt(cell1_expr * cell2_expr)

        distances = df_dis['dis_norm'].values

        if len(expr) > 1 and len(distances) > 1:
            spearman_corr, spearman_p = stats.spearmanr(expr, distances)
            pearson_corr, pearson_p = stats.pearsonr(expr, distances)
            
        else:
            spearman_corr, spearman_p = np.nan, np.nan
            pearson_corr, pearson_p = np.nan, np.nan

        spearman_corrs.append(spearman_corr)
        spearman_pvals.append(spearman_p)
        pearson_corrs.append(pearson_corr)
        pearson_pvals.append(pearson_p)
            
    CCC_pval = CCC_pval.copy()
    CCC_pval['spearman_corr'] = spearman_corrs
    CCC_pval['spearman_pval'] = spearman_pvals
    CCC_pval['pearson_corr'] = pearson_corrs
    CCC_pval['pearson_pval'] = pearson_pvals

    # CCC_pval['pvalue'] = np.where(
    #     CCC_pval['spearman_corr'] >= 0,  # 条件：如果spearman_corr >= 0
    #     CCC_pval['spearman_corr'].fillna(1) * CCC_pval['pval'],  # 正值的计算
    #     (1 - abs(CCC_pval['spearman_corr'].fillna(0))) * CCC_pval['pval']  # 负值的计算
    # )

    # CCC_pval['pvalue'] = np.where(
    #     CCC_pval['spearman_corr'] >= 0,  # 条件：如果spearman_corr >= 0
    #     (1+CCC_pval['spearman_pval'].fillna(1)) * CCC_pval['pval'],  # 正值的计算
    #     CCC_pval['spearman_pval'].fillna(1) * CCC_pval['pval']  # 负值的计算
    # )

    # CCC_pval['pvalue'] = np.where(
    #     CCC_pval['spearman_pval'].isna() | (CCC_pval['spearman_corr'] >= 0),
    #     np.log(CCC_pval['pval'] + 1e-10),
    #     -(np.log(CCC_pval['pval'] + 1e-10)) * (np.log(CCC_pval['spearman_pval'] + 1e-10))
    # )

    CCC_pval['pvalue'] = CCC_pval.apply(combine_with_fisher, axis=1)

    # modified_count = 0
    # mask = CCC_pval['spearman_corr'] > corr_threshold
    # modified_count = mask.sum()

    # # 将对应行的 spearman_pval 设为 1
    # CCC_pval.loc[mask, 'pvalue'] = 1

    # # 配受体对在细胞类型中排名
    # CCC_filtered = CCC_pval.reset_index(drop=True)
    # CCC_filtered['pvalue'] = CCC_filtered.groupby('cell_pair')['spearman_pval'].rank(pct=True)

    # print(f'满足条件并被设为 pval=1 的行数：{modified_count}')
    # print(f'筛选后数量：{len(CCC_filtered)}')

    return CCC_pval


def safe_mean(arr):
    if issparse(arr):
        return arr.mean().item()
    return float(np.mean(arr))

def filter_CCC_ct(sc_data, dis, CCC_pval, gene_to_idx, corr_threshold=0.0, distance_col="MAD Image"):
    # 1. subclass → numpy 行号
    subclass_to_rows = {
        subclass: sc_data.obs_names.get_indexer(df.index.values)
        for subclass, df in sc_data.obs.groupby('subclass')
    }

    CCC_filtered = CCC_pval.copy()
    LR_pairs = CCC_filtered['LR_pair'].unique()

    # 预取表达矩阵（dense）
    X = sc_data.X.A if hasattr(sc_data.X, "A") else sc_data.X

    # 预拆分 dis
    valid_dis = dis[dis[distance_col].notna()]
    pairs = valid_dis['Group Pair'].str.split(" vs ", expand=True).values
    dist_list = valid_dis[distance_col].values

    spearman_dict, pearson_dict = {}, {}

    for lr in LR_pairs:
        subset = CCC_filtered[CCC_filtered['LR_pair'] == lr]
        ligand, receptor = subset['ligand'].iloc[0], subset['receptor'].iloc[0]

        if ligand not in gene_to_idx or receptor not in gene_to_idx:
            spearman_dict[lr] = pearson_dict[lr] = (np.nan, np.nan)
            continue

        ligand_idx, receptor_idx = gene_to_idx[ligand], gene_to_idx[receptor]
        expr_list = []

        # 向量化计算
        for (cellA, cellB), dist in zip(pairs, dist_list):
            rowsA, rowsB = subclass_to_rows.get(cellA), subclass_to_rows.get(cellB)
            if rowsA is None or rowsB is None or (rowsA < 0).any() or (rowsB < 0).any():
                continue

            exprA = X[rowsA[:, None], [ligand_idx, receptor_idx]].mean()
            exprB = X[rowsB[:, None], [ligand_idx, receptor_idx]].mean()
            expr_list.append(exprA + exprB)

        if len(expr_list) > 1:
            sr, sp = stats.spearmanr(expr_list, dist_list[:len(expr_list)])
            pr, pp = stats.pearsonr(expr_list, dist_list[:len(expr_list)])
        else:
            sr, sp = pr, pp = np.nan, np.nan

        spearman_dict[lr] = (sr, sp)
        pearson_dict[lr] = (pr, pp)
        if sr > corr_threshold:
            print(f"LR对 {lr} 删除原因: spearman_corr({sr:.4f}) > 阈值({corr_threshold})")

    # 回填
    for col, d in [('spearman_corr', spearman_dict), ('spearman_pval', spearman_dict),
                   ('pearson_corr', pearson_dict), ('pearson_pval', pearson_dict)]:
        CCC_filtered[col] = CCC_filtered['LR_pair'].map(lambda x: d[x][0 if 'corr' in col else 1])

    # 删除 > 阈值
    drop_mask = CCC_filtered['spearman_corr'] > corr_threshold
    dropped_cnt = drop_mask.sum()
    dropped_lrs = CCC_filtered.loc[drop_mask, 'LR_pair'].unique().tolist()

    CCC_filtered = CCC_filtered[~drop_mask]
    print(f'被删掉的配受体对数量：{dropped_cnt}')
    if dropped_lrs:
        print('被删 LR 对：', ', '.join(dropped_lrs))
    return CCC_filtered, dropped_lrs

def sig_CCC(results_dict, threshold):

    filtered_results = []
    detailed_results = []
    # 遍历每个细胞类型对
    for type_pair, lr_list in results_dict.items():
        # 筛选显著相关的LR对（p值小于阈值）
        significant_lrs = [
            lr_info['LR_pair'] 
            for lr_info in lr_list 
            if lr_info['pvalue'] < threshold and lr_info['correlation'] < 0
        ]
        
        # 添加到结果列表
        filtered_results.append({
            'cell_pair': type_pair,
            'lr_pairs': significant_lrs,
            'lr_count': len(significant_lrs)
        })

        for lr_info in lr_list:
            if lr_info['pvalue'] < 0.9 and lr_info['correlation'] < 0:
                detailed_results.append({
                    'cell_pair': type_pair,
                    'LR_pair': lr_info['LR_pair'],
                    # 'pvalue': 0 if lr_info['pvalue'] < 1e-3 else lr_info['pvalue'],
                    'pvalue': lr_info['pvalue'],
                })
    
    # 转换为DataFrame
    result_df = pd.DataFrame(filtered_results)
    detailed_df = pd.DataFrame(detailed_results)
    
    return result_df, detailed_df
    

def cal_CCC(sc_data, CCC, CCC_pval, dis, dis_ct, threshold = 0.05, path = None):
    # threshold: 显著性阈值
    
    gene_to_idx = {gene.upper(): idx for idx, gene in enumerate(sc_data.var_names)}
    cell_to_idx = {cell: idx for idx, cell in enumerate(sc_data.obs_names)}
    
    # dense 表达矩阵
    X_dense = sc_data.X.toarray() if sparse.issparse(sc_data.X) else sc_data.X

    CCC = filter_CCC(dis, CCC, gene_to_idx, cell_to_idx, X_dense)
    
    sig, _ = sig_CCC(CCC, threshold)

    sig.to_csv(path + 'CCC.csv')
    
    # if score is not None:
    #     print("显著性相加")

    
    # sig_value_ct, dropped_lrs = filter_CCC_ct(sc_data, dis_ct, CCC_pval, gene_to_idx, corr_threshold=0.0, distance_col="MAD Image")    
    # sig_value_ct.to_csv(path + f'{choose}_{method}_CCC_value_ct.csv')


    
    sig_value = filter_CCC_pval(dis, CCC_pval, gene_to_idx, cell_to_idx, X_dense, corr_threshold=0.0)

    # if dropped_lrs:
    #     drop_mask = sig_value['LR_pair'].isin(dropped_lrs)
    #     sig_value = sig_value[~drop_mask]
    #     print(f'额外删除来自 filter_CCC_ct 的配受体对数量：{drop_mask.sum()}')
    #     if drop_mask.sum() > 0:
    #         print('额外删除的 LR 对：', ', '.join(sig_value.loc[drop_mask, 'LR_pair'].unique().tolist()))   
    sig_value.to_csv(path + 'CCC_value.csv')