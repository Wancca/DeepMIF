# --- START OF FILE preprocess_data_to_pt.py (Modified for Detailed Logging) ---

import torch
import pandas as pd
from tqdm import tqdm
import os
import sys
import numpy as np
from collections import Counter

print("--- 数据集离线预处理脚本 (含详细日志) ---")
print("正在导入必要的模块...")

try:
    from rna_dta_dataset import (
        smiles_to_pyg_graph,
        parse_st_file_enhanced,
        seq_to_local_eskmer_sequence,
        rna_struct_to_graph_fragment_enhanced,
        RNA_FM_EMBEDDING_DIR,
        RNA_FM_MAX_LEN,
        RNA_FM_PAD_VALUE,
        RNA_BASE_VOCAB,
        LOCAL_ESKMER_K,
        LOCAL_ESKMER_WINDOW_SIZE,
        ESKMER_VOCAB_MAP,
        ST_FILE_DIR
    )
    from finetune_rna_dta_Regression_cv import ScriptArgs

    print("模块导入成功。")
except ImportError as e:
    print(f"\n错误: 无法导入必要的模块: {e}")
    sys.exit(1)


def find_core_region_by_structure(sequence, dot_bracket, target_len):
    original_len = len(sequence)
    if original_len <= target_len:
        return 0, original_len
    complexity_scores = np.zeros(original_len)
    stack = []
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                complexity_scores[i] += 1;
                complexity_scores[j] += 1
    max_score, best_start_index = -1, 0
    current_score = np.sum(complexity_scores[0:target_len])
    max_score = current_score
    for i in range(1, original_len - target_len + 1):
        current_score = current_score - complexity_scores[i - 1] + complexity_scores[i + target_len - 1]
        if current_score > max_score:
            max_score, best_start_index = current_score, i
    return best_start_index, best_start_index + target_len


def preprocess_and_save():
    args = ScriptArgs()

    original_csv_file = 'RSM_data/All_sf_dataset_v1_2D.csv'
    full_data_path = os.path.join(args.data_dir, original_csv_file)
    output_dir = os.path.join(args.data_dir, "processed_pt_files_multiscale")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    print(f"\n读取原始数据从: {full_data_path}")
    print(f"处理后的 .pt 文件将保存到: {output_dir}")

    try:
        sep = '\t' if '\t' in open(full_data_path, encoding='utf-8').readline() else ','
        df = pd.read_csv(full_data_path, sep=sep, engine='python')
        # ... (与之前版本相同的CSV加载和清洗代码) ...
        required_cols = [args.smiles_col, args.seq_col, args.affinity_col, args.target_id_col]
        df = df.dropna(subset=required_cols)
        for col in [args.smiles_col, args.seq_col, args.target_id_col]: df[col] = df[col].astype(str)
        df[args.affinity_col] = pd.to_numeric(df[args.affinity_col], errors='coerce')
        df = df.dropna(subset=[args.affinity_col])
        print(f"成功加载 {len(df)} 条原始数据记录。")
    except Exception as e:
        print(f"错误: 加载或解析CSV文件失败: {e}")
        return

    print("\n开始处理每个样本...")
    success_count = 0
    failure_count = 0
    failure_log = []  # <--- 新增日志列表

    progress_bar = tqdm(df.iterrows(), total=len(df), desc="预处理进度", ncols=100)
    for original_csv_index, row in progress_bar:
        # --- 将所有处理逻辑包裹在一个大的try/except中，以捕获任何未知错误 ---
        try:
            smiles_str, affinity_val, target_rna_id_str = row[args.smiles_col], row[args.affinity_col], row[
                args.target_id_col]

            # 1. 药物处理
            drug_graph_obj = smiles_to_pyg_graph(smiles_str)
            if drug_graph_obj is None:
                failure_log.append({'idx': original_csv_index, 'id': target_rna_id_str, 'reason': 'Invalid SMILES'})
                failure_count += 1
                continue

            # 2. RNA文件检查
            seq_st_full, struct_db_st_full, modules_sixth_st_full, defined_elements_st = parse_st_file_enhanced(
                target_rna_id_str, ST_FILE_DIR)
            if seq_st_full is None:
                failure_log.append(
                    {'idx': original_csv_index, 'id': target_rna_id_str, 'reason': 'ST file missing or corrupt'})
                failure_count += 1
                continue

            embedding_file_path = os.path.join(RNA_FM_EMBEDDING_DIR, f"{target_rna_id_str}.npy")
            if not os.path.exists(embedding_file_path):
                failure_log.append(
                    {'idx': original_csv_index, 'id': target_rna_id_str, 'reason': 'NPY embedding file missing'})
                failure_count += 1
                continue

            raw_fm_embeddings_np = np.load(embedding_file_path)
            if raw_fm_embeddings_np.shape[0] == 0:
                failure_log.append(
                    {'idx': original_csv_index, 'id': target_rna_id_str, 'reason': 'NPY embedding file is empty'})
                failure_count += 1
                continue

            # 3. 多尺度特征生成
            global_summary_vector = torch.tensor(raw_fm_embeddings_np, dtype=torch.float).mean(dim=0)
            start, end = find_core_region_by_structure(seq_st_full, struct_db_st_full, RNA_FM_MAX_LEN)
            seq_st_core, struct_db_st_core, modules_sixth_st_core = seq_st_full[start:end], struct_db_st_full[
                                                                                            start:end], modules_sixth_st_full[
                                                                                                        start:end]
            len_core = len(seq_st_core)
            fm_embeddings_core_processed = raw_fm_embeddings_np[start:end, :]

            padding_to_max = RNA_FM_MAX_LEN - len_core
            fm_embeddings_cnn_np = np.pad(fm_embeddings_core_processed, ((0, padding_to_max), (0, 0)), mode='constant',
                                          constant_values=RNA_FM_PAD_VALUE)
            fm_mask_cnn_np = np.concatenate(
                [np.ones(len_core, dtype=np.int64), np.zeros(padding_to_max, dtype=np.int64)])

            rna_fm_embeddings_for_cnn = torch.tensor(fm_embeddings_cnn_np, dtype=torch.float)
            rna_fm_mask_for_cnn = torch.tensor(fm_mask_cnn_np, dtype=torch.long)

            local_eskmer_feat_seq, _ = seq_to_local_eskmer_sequence(seq_st_core.upper().replace('T', 'U'),
                                                                    LOCAL_ESKMER_K, LOCAL_ESKMER_WINDOW_SIZE,
                                                                    ESKMER_VOCAB_MAP, RNA_FM_MAX_LEN)

            node_features_for_gnn = torch.tensor(fm_embeddings_core_processed, dtype=torch.float)
            rna_graph_obj = rna_struct_to_graph_fragment_enhanced(seq_st_core, struct_db_st_core, modules_sixth_st_core,
                                                                  defined_elements_st, node_features_for_gnn,
                                                                  RNA_BASE_VOCAB)
            if rna_graph_obj is None:
                failure_log.append(
                    {'idx': original_csv_index, 'id': target_rna_id_str, 'reason': 'RNA graph generation failed'})
                failure_count += 1
                continue

            # 4. 打包保存
            processed_sample = {
                'drug_graph': drug_graph_obj, 'original_smiles': smiles_str,
                'rna_fm_embeddings': rna_fm_embeddings_for_cnn, 'rna_fm_mask': rna_fm_mask_for_cnn,
                'rna_local_eskmer_sequence': local_eskmer_feat_seq, 'rna_graph': rna_graph_obj,
                'rna_global_summary': global_summary_vector,
                'affinity': torch.tensor([affinity_val], dtype=torch.float),
                'idx': original_csv_index, 'target_rna_id': target_rna_id_str,
            }
            save_path = os.path.join(output_dir, f"sample_{original_csv_index}.pt")
            torch.save(processed_sample, save_path)
            success_count += 1

        except Exception as e:
            failure_log.append({'idx': original_csv_index, 'id': row.get(args.target_id_col, 'N/A'),
                                'reason': f'Unexpected Error: {str(e)}'})
            failure_count += 1
            continue

    # --- 步骤 5: 打印详细的最终报告 ---
    print("\n\n--- 数据集预处理详细报告 (多尺度特征) ---")
    print("=" * 45)
    print(f"总共尝试处理 {len(df)} 条记录。")
    print(f"成功处理并保存: {success_count} 个样本。")
    print(f"处理失败并跳过: {failure_count} 个样本。")
    print("-" * 45)

    if failure_count > 0:
        print("\n过滤原因统计:")
        reason_counts = Counter(log['reason'] for log in failure_log)
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count} 个样本")

        print("\n被过滤的样本详情 (最多显示前20条):")
        for i, sample_info in enumerate(failure_log[:20]):
            print(
                f"  {i + 1}. CSV行号: {str(sample_info['idx']).ljust(5)}, RNA_ID: {str(sample_info['id']).ljust(15)}, 原因: {sample_info['reason']}")
        if failure_count > 20:
            print("  ...")

        # 自动将所有被过滤的样本信息保存到CSV文件
        try:
            report_df = pd.DataFrame(failure_log)
            report_path = 'filtered_samples_report.csv'
            report_df.to_csv(report_path, index=False)
            print(f"\n注意: 已将被过滤样本的完整列表自动保存到 '{report_path}' 文件，以便详细排查。")
        except Exception as e:
            print(f"保存报告失败: {e}")
    else:
        print("\n恭喜！所有样本都通过了检查。")

    print(f"\n预处理完成！.pt 文件已保存在 '{output_dir}' 目录中。")
    print("=" * 45)


if __name__ == '__main__':
    preprocess_and_save()

# --- END OF FILE preprocess_data_to_pt.py (Modified for Detailed Logging) ---