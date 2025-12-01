# # --- START OF FILE finetune_rna_dta_Regression_cv.py (Standard MSE Version) ---
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# import os
# import sys
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.preprocessing import KBinsDiscretizer
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
#
# try:
#     from rna_dta_dataset import RNADataset, rna_dta_collate_fn, RNA_FM_DIM, RNA_BASE_VOCAB_SIZE, NUM_EDGE_FEATURES, \
#         ST_FILE_DIR as DEFAULT_ST_DIR, ESKMER_DIM, ESKMER_VOCAB_LIST, LOCAL_ESKMER_K, BASE_EMBED_DIM
#     from rna_dta_fm_cross_attention_model import RNADTA_FM_CrossAttentionModel
# except ImportError as e:
#     print(f"错误: 导入模块失败: {e}");
#     sys.exit(1)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import warnings
#
# warnings.filterwarnings("ignore")
#
#
# def train_epoch(model, data_loader, optimizer, loss_criterion, grad_scaler=None):
#     model.train();
#     total_loss = 0.0
#     progress_bar = tqdm(data_loader, desc="训练中", leave=False, ncols=100, file=sys.stdout)
#     for batch_data in progress_bar:
#         if batch_data is None: continue
#         try:
#             drug_graph, smiles, rna_fm, rna_mask, rna_eskmer, rna_graph, labels = [batch_data[k] for k in
#                                                                                    ['drug_graph', 'original_smiles',
#                                                                                     'rna_fm_embeddings', 'rna_fm_mask',
#                                                                                     'rna_local_eskmer_sequence',
#                                                                                     'rna_graph', 'affinity']]
#             drug_graph, rna_fm, rna_mask, rna_eskmer, rna_graph, labels = drug_graph.to(device), rna_fm.to(
#                 device), rna_mask.to(device), rna_eskmer.to(device), rna_graph.to(device), labels.to(device)
#         except (KeyError, AttributeError) as e:
#             print(f"训练批次错误: {e}");
#             continue
#         optimizer.zero_grad()
#         use_amp = grad_scaler is not None
#         with torch.cuda.amp.autocast(enabled=use_amp):
#             predictions = model(drug_graph, rna_fm, rna_mask, rna_eskmer, rna_graph, smiles)
#             loss = loss_criterion(predictions.squeeze(), labels.squeeze())
#         if torch.isnan(loss): print("警告: 损失为NaN"); optimizer.zero_grad(); continue
#         if use_amp:
#             grad_scaler.scale(loss).backward(); grad_scaler.step(optimizer); grad_scaler.update()
#         else:
#             loss.backward(); optimizer.step()
#         total_loss += loss.item()
#         progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
#     return total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
#
#
# def evaluate_epoch(model, data_loader, loss_criterion, desc_prefix="评估中", return_details_for_vis=False):
#     model.eval();
#     all_preds, all_trues = [], []
#     total_loss = 0.0
#     with torch.no_grad():
#         progress_bar = tqdm(data_loader, desc=desc_prefix, leave=False, file=sys.stdout, ncols=100)
#         for batch_data in progress_bar:
#             if batch_data is None: continue
#             try:
#                 drug_graph, smiles, rna_fm, rna_mask, rna_eskmer, rna_graph, labels = [batch_data[k] for k in
#                                                                                        ['drug_graph', 'original_smiles',
#                                                                                         'rna_fm_embeddings',
#                                                                                         'rna_fm_mask',
#                                                                                         'rna_local_eskmer_sequence',
#                                                                                         'rna_graph', 'affinity']]
#                 drug_graph, rna_fm, rna_mask, rna_eskmer, rna_graph, labels = drug_graph.to(device), rna_fm.to(
#                     device), rna_mask.to(device), rna_eskmer.to(device), rna_graph.to(device), labels.to(device)
#             except (KeyError, AttributeError) as e:
#                 print(f"评估批次错误: {e}");
#                 continue
#             with torch.cuda.amp.autocast(enabled=True):
#                 predictions = model(drug_graph, rna_fm, rna_mask, rna_eskmer, rna_graph, smiles)
#                 loss = loss_criterion(predictions.squeeze(), labels.squeeze())
#             if not torch.isnan(loss): total_loss += loss.item() * labels.size(0)
#             all_preds.append(predictions.cpu().numpy());
#             all_trues.append(labels.cpu().numpy())
#     if not all_trues:
#         empty_return = 0.0, float('inf'), float('inf'), 0.0, 0.0, -1.0
#         return empty_return + (None,) if return_details_for_vis else empty_return
#     avg_loss = total_loss / len(np.concatenate(all_trues))
#     all_preds, all_trues = np.concatenate(all_preds).flatten(), np.concatenate(all_trues).flatten()
#     pcc, scc, rmse, r2 = 0.0, 0.0, float('inf'), -1.0
#     try:
#         if np.std(all_preds) > 1e-8 and np.std(all_trues) > 1e-8:
#             pcc, _ = pearsonr(all_trues, all_preds);
#             scc, _ = spearmanr(all_trues, all_preds)
#         rmse = np.sqrt(mean_squared_error(all_trues, all_preds));
#         r2 = r2_score(all_trues, all_preds)
#     except Exception as e:
#         print(f"计算指标时出错: {e}")
#     metrics = [avg_loss, rmse, pcc, scc, r2]
#     if return_details_for_vis: return tuple(metrics) + ({'predictions': all_preds, 'trues': all_trues},)
#     return tuple(metrics)
#
#
# def visualize_results(history, test_details, model, output_dir, fold_num, model_name):
#     print(f"\n  [可视化] 开始为第 {fold_num} 折生成图表...")
#     if not os.path.exists(output_dir): os.makedirs(output_dir)
#     history_df = pd.DataFrame(history)
#     plt.figure(figsize=(10, 6));
#     sns.lineplot(data=history_df, x='epoch', y='train_loss', label='Train Loss');
#     sns.lineplot(data=history_df, x='epoch', y='test_loss', label='Test Loss');
#     plt.title(f'Fold {fold_num}: Training & Test Loss ({model_name})', fontsize=16);
#     plt.xlabel('Epoch');
#     plt.ylabel('MSE Loss');
#     plt.legend();
#     plt.grid(True);
#     plt.savefig(os.path.join(output_dir, f'fold_{fold_num}_loss_curve.png'), dpi=300);
#     plt.close();
#     print("    - 损失曲线图已保存。")
#     fig, axs = plt.subplots(2, 2, figsize=(16, 12));
#     fig.suptitle(f'Fold {fold_num}: Test Metrics over Epochs ({model_name})', fontsize=18)
#     metrics_to_plot = [('pcc', 'PCC'), ('rmse', 'RMSE'), ('r2', 'R-squared'), ('scc', 'Spearman')]
#     for i, (metric, title) in enumerate(metrics_to_plot):
#         ax = axs[i // 2, i % 2];
#         sns.lineplot(data=history_df, x='epoch', y=metric, ax=ax, marker='.');
#         ax.set_title(title);
#         ax.set_xlabel('Epoch');
#         ax.set_ylabel(metric.upper());
#         ax.grid(True)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
#     plt.savefig(os.path.join(output_dir, f'fold_{fold_num}_metrics_curves.png'), dpi=300);
#     plt.close();
#     print("    - 性能指标曲线图已保存。")
#     if test_details:
#         preds, trues = test_details['predictions'], test_details['trues']
#         pcc, rmse = pearsonr(trues, preds)[0], np.sqrt(mean_squared_error(trues, preds))
#         g = sns.jointplot(x=trues, y=preds, kind="reg", color="royalblue", scatter_kws={'s': 15, 'alpha': 0.5},
#                           line_kws={'color': 'red'}, marginal_kws=dict(bins=40, fill=True, edgecolor='none'), height=8)
#         g.ax_joint.plot([trues.min(), trues.max()], [trues.min(), trues.max()], '--', color='purple', linewidth=2,
#                         label='y=x')
#         text_str = f'PCC = {pcc:.4f}\nRMSE = {rmse:.4f}'
#         g.ax_joint.text(0.05, 0.95, text_str, transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top',
#                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#         g.fig.suptitle(f'Fold {fold_num}: Prediction vs. True Affinity ({model_name})', fontsize=16, y=1.02)
#         g.set_axis_labels('True Affinity (pKd)', 'Predicted Affinity (pKd)', fontsize=14)
#         g.savefig(os.path.join(output_dir, f'fold_{fold_num}_joint_scatter_plot.png'), dpi=300, bbox_inches='tight');
#         plt.close();
#         print("    - 预测-真实值联合分布图已保存。")
#
#
# def visualize_cv_summary(all_folds_metrics_df, output_dir, model_name):
#     if all_folds_metrics_df.empty: return
#     print("\n  [可视化] 开始生成交叉验证汇总图表...")
#     if not os.path.exists(output_dir): os.makedirs(output_dir)
#     metrics_to_plot = ['pearson', 'rmse', 'spearman', 'r2']
#     plt.figure(figsize=(12, 8));
#     df_melted = all_folds_metrics_df.melt(id_vars=['fold'], value_vars=metrics_to_plot, var_name='Metric',
#                                           value_name='Value')
#     sns.boxplot(data=df_melted, x='Metric', y='Value', palette='pastel');
#     sns.stripplot(data=df_melted, x='Metric', y='Value', color=".25", alpha=0.7)
#     plt.title(f'Performance Across All Folds ({model_name})', fontsize=18);
#     plt.xlabel('Metric');
#     plt.ylabel('Value')
#     metric_labels = [m.upper().replace('PEARSON', 'PCC').replace('SPEARMAN', 'SCC') for m in metrics_to_plot]
#     plt.xticks(ticks=range(len(metrics_to_plot)), labels=metric_labels);
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout();
#     plt.savefig(os.path.join(output_dir, f'cv_summary_boxplot_{model_name}.png'), dpi=300);
#     plt.close();
#     print("    - 交叉验证汇总箱形图已保存。")
#
#
# def main(cli_args):
#     model_name = "HD_WFAN_FV_MSE"
#     print(f"主程序: 开始执行RNA-DTA交叉验证 ({model_name} - 标准MSE版)...")
#     print(f"主程序: 设备检测为: {device}")
#     processed_data_dir = os.path.join(cli_args.data_dir, cli_args.input_file)
#     try:
#         complete_dataset = RNADataset(data_path=processed_data_dir)
#         print(f"主程序: 预处理数据集加载成功，总样本数: {len(complete_dataset)}")
#     except (FileNotFoundError, ValueError) as e:
#         print(f"\n错误: {e}");
#         sys.exit(1)
#     affinity_labels = np.array(complete_dataset.labels)
#     binned_labels = KBinsDiscretizer(n_bins=cli_args.num_folds, encode='ordinal', strategy='uniform').fit_transform(
#         affinity_labels.reshape(-1, 1))[:, 0]
#     skf = RepeatedStratifiedKFold(n_splits=cli_args.num_folds, n_repeats=1, random_state=cli_args.seed)
#     all_folds_metrics = []
#     for fold_num, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(complete_dataset)), binned_labels)):
#         print(f"\n===== 开始处理第 {fold_num + 1}/{cli_args.num_folds} 折 =====")
#         train_loader = DataLoader(Subset(complete_dataset, train_idx), batch_size=cli_args.batch_size, shuffle=True,
#                                   collate_fn=rna_dta_collate_fn, num_workers=cli_args.num_workers, pin_memory=True,
#                                   drop_last=True, persistent_workers=cli_args.num_workers > 0)
#         test_loader = DataLoader(Subset(complete_dataset, test_idx), batch_size=cli_args.batch_size, shuffle=False,
#                                  collate_fn=rna_dta_collate_fn, num_workers=cli_args.num_workers, pin_memory=True,
#                                  persistent_workers=cli_args.num_workers > 0)
#         model = RNADTA_FM_CrossAttentionModel(**vars(cli_args)).to(device)
#         optimizer = optim.AdamW(model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
#         loss_fn = nn.MSELoss()
#         scaler = torch.cuda.amp.GradScaler(enabled=cli_args.use_amp)
#         history, best_pcc, epochs_no_improve, best_epoch = [], -1.0, 0, -1
#         output_dir = os.path.dirname(cli_args.save_path)
#         model_path = os.path.join(output_dir, f"{model_name}_fold_{fold_num + 1}.pth")
#         vis_dir = os.path.join(output_dir, f'visualizations/{model_name}_fold_{fold_num + 1}')
#         print(f"  开始训练... 最佳模型将保存至: {model_path}, 可视化图表将保存至: {vis_dir}")
#         for epoch in range(1, cli_args.epochs + 1):
#             train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler)
#             test_loss, rmse, pcc, scc, r2 = evaluate_epoch(model, test_loader, loss_fn, f"Epoch {epoch} Test")
#             history.append(
#                 {'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss, 'pcc': pcc, 'rmse': rmse, 'r2': r2,
#                  'scc': scc})
#             epoch_log_summary = f"    Epoch {epoch}/{cli_args.epochs}: TrainLoss={train_loss:.4f}, TestLoss={test_loss:.4f} | TestMetrics: RMSE={rmse:.4f} PCC={pcc:.4f} R2={r2:.4f}"
#             if pcc > best_pcc:
#                 best_pcc, best_epoch, epochs_no_improve = pcc, epoch, 0
#                 torch.save(model.state_dict(), model_path)
#                 epoch_log_summary += " * (新最佳PCC! 模型已保存)"
#             else:
#                 epochs_no_improve += 1
#                 epoch_log_summary += f" (无改善 {epochs_no_improve}/{cli_args.early_stopping_patience})"
#             print(epoch_log_summary)
#             if epochs_no_improve >= cli_args.early_stopping_patience: print(f"\n    早停触发。"); break
#         print(f"\n  当前折训练结束。最佳PCC出现在第 {best_epoch} 轮。");
#         print(f"  加载最佳模型并进行最终评估和可视化...")
#         if os.path.exists(model_path):
#             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
#             final_loss, final_rmse, final_pcc, final_scc, final_r2, final_details = evaluate_epoch(model, test_loader,
#                                                                                                    loss_fn,
#                                                                                                    "Final Eval",
#                                                                                                    return_details_for_vis=True)
#             visualize_results(pd.DataFrame(history), final_details, model, vis_dir, fold_num + 1, model_name)
#             all_folds_metrics.append(
#                 {'fold': fold_num + 1, 'best_test_epoch': best_epoch, 'pearson': final_pcc, 'rmse': final_rmse,
#                  'spearman': final_scc, 'r2': final_r2})
#         else:
#             print("  警告：未找到已保存的最佳模型，跳过最终评估和可视化。")
#         del model, optimizer, train_loader, test_loader;
#         torch.cuda.empty_cache()
#     results_df = pd.DataFrame(all_folds_metrics)
#     csv_path = os.path.join(output_dir, f"{model_name}_all_folds_results.csv")
#     results_df.to_csv(csv_path, index=False)
#     print(f"\n所有折的详细结果已保存到: {csv_path}")
#     vis_summary_dir = os.path.join(os.path.dirname(cli_args.save_path), 'visualizations')
#     visualize_cv_summary(results_df, vis_summary_dir, model_name)
#     print(f"\n===== 交叉验证平均性能指标 ({model_name}版本) =====");
#     print(results_df.drop(columns=['fold', 'best_test_epoch']).mean());
#     print("==============================================")
#
#
# if __name__ == "__main__":
#     class ScriptArgs:
#         data_dir = 'RNA_DTA_data'
#         input_file = 'processed_pt_files_simple_truncate'
#         target_id_col, smiles_col, seq_col, affinity_col = 'Target_RNA_ID', 'SMILES', 'Target_RNA_sequence', 'pKd'
#         st_file_dir = DEFAULT_ST_DIR
#         pretrain_path = 'save/model_pre20.pth'
#         save_path = 'save_rna_cv_results/rna_dta_model_final.pth'
#         num_folds, seed, epochs, batch_size = 5, 42, 500, 32
#         lr, weight_decay = 5e-5, 1e-5
#         freeze_drug_encoder, use_amp, early_stopping_patience, num_workers = True, True, 100, 2
#         deep_fusion_layers = 2
#         cross_attention_embed_dim, cross_attention_num_heads, cross_attention_dropout = 128, 4, 0.1
#         rna_fm_embed_dim = RNA_FM_DIM
#         cnn_num_filters, cnn_kernel_sizes, rna_dropout = 128, [3, 5, 7], 0.3
#         gnn_hidden_dim, gnn_layers, gnn_type, gnn_heads, gnn_pooling = 128, 3, 'GAT', 4, 'mean'
#         rna_base_vocab_size, rna_base_embed_dim = RNA_BASE_VOCAB_SIZE, BASE_EMBED_DIM
#         use_feature_gating, gating_mlp_hidden_dim = False, 32
#         eskmer_dim, eskmer_projection_dim = ESKMER_DIM, 32
#         mlp_pred_hidden_dims, mlp_pred_dropout = [512, 256], 0.3
#
#
#     args_instance = ScriptArgs()
#     # ... (打印参数和检查目录的代码) ...
#     try:
#         main(args_instance)
#     except Exception as e_main:
#         print(f"\n主程序执行过程中发生未捕获的严重错误: {e_main}");
#         import traceback;
#
#         traceback.print_exc()
#     print("\n交叉验证及可视化流程执行完毕。")
#
# # --- END OF FILE finetune_rna_dta_Regression_cv.py (Standard MSE Version) ---



# --- START OF FILE finetune_rna_dta_Regression_cv.py (Single Fold Execution Version) ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

try:
    from rna_dta_dataset import RNADataset, rna_dta_collate_fn
    from rna_dta_fm_cross_attention_model import RNADTA_FM_CrossAttentionModel
    from rna_dta_dataset import RNA_FM_DIM, RNA_BASE_VOCAB_SIZE, NUM_EDGE_FEATURES, ST_FILE_DIR as DEFAULT_ST_DIR, \
        ESKMER_DIM, ESKMER_VOCAB_LIST, LOCAL_ESKMER_K, BASE_EMBED_DIM
except ImportError as e:
    print(f"错误: 导入模块失败: {e}");
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings

warnings.filterwarnings("ignore")


# --- 训练和评估函数 (与您的MSE版本完全相同) ---
def train_epoch(model, data_loader, optimizer, loss_criterion, grad_scaler=None):
    model.train();
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc="训练中", leave=False, ncols=100, file=sys.stdout)
    for batch_data in progress_bar:
        if batch_data is None: continue
        try:
            drug_graph, smiles, rna_fm, rna_mask, rna_eskmer, rna_graph, labels = [batch_data[k] for k in
                                                                                   ['drug_graph', 'original_smiles',
                                                                                    'rna_fm_embeddings', 'rna_fm_mask',
                                                                                    'rna_local_eskmer_sequence',
                                                                                    'rna_graph', 'affinity']]
            drug_graph, rna_fm, rna_mask, rna_eskmer, rna_graph, labels = drug_graph.to(device), rna_fm.to(
                device), rna_mask.to(device), rna_eskmer.to(device), rna_graph.to(device), labels.to(device)
        except (KeyError, AttributeError) as e:
            print(f"训练批次错误: {e}")
            continue
        optimizer.zero_grad()
        use_amp = grad_scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(drug_graph, rna_fm, rna_mask, rna_eskmer, rna_graph, smiles)
            loss = loss_criterion(predictions.squeeze(), labels.squeeze())
        if torch.isnan(loss): print("警告: 损失为NaN"); optimizer.zero_grad(); continue
        if use_amp:
            grad_scaler.scale(loss).backward(); grad_scaler.step(optimizer); grad_scaler.update()
        else:
            loss.backward(); optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(data_loader) if len(data_loader) > 0 else 0.0


def evaluate_epoch(model, data_loader, loss_criterion, desc_prefix="评估中", return_details_for_vis=False):
    model.eval();
    all_preds, all_trues = [], []
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=desc_prefix, leave=False, file=sys.stdout, ncols=100)
        for batch_data in progress_bar:
            if batch_data is None: continue
            try:
                drug_graph, smiles, rna_fm, rna_mask, rna_eskmer, rna_graph, labels = [batch_data[k] for k in
                                                                                       ['drug_graph', 'original_smiles',
                                                                                        'rna_fm_embeddings',
                                                                                        'rna_fm_mask',
                                                                                        'rna_local_eskmer_sequence',
                                                                                        'rna_graph', 'affinity']]
                drug_graph, rna_fm, rna_mask, rna_eskmer, rna_graph, labels = drug_graph.to(device), rna_fm.to(
                    device), rna_mask.to(device), rna_eskmer.to(device), rna_graph.to(device), labels.to(device)
            except (KeyError, AttributeError) as e:
                print(f"评估批次错误: {e}")
                continue
            with torch.cuda.amp.autocast(enabled=True):
                predictions = model(drug_graph, rna_fm, rna_mask, rna_eskmer, rna_graph, smiles)
                loss = loss_criterion(predictions.squeeze(), labels.squeeze())
            if not torch.isnan(loss): total_loss += loss.item() * labels.size(0)
            all_preds.append(predictions.cpu().numpy())
            all_trues.append(labels.cpu().numpy())
    if not all_trues:
        empty_return = 0.0, float('inf'), 0.0, 0.0, -1.0
        return empty_return + (None,) if return_details_for_vis else empty_return
    avg_loss = total_loss / len(np.concatenate(all_trues))
    all_preds, all_trues = np.concatenate(all_preds).flatten(), np.concatenate(all_trues).flatten()
    pcc, scc, rmse, r2 = 0.0, 0.0, float('inf'), -1.0
    try:
        if np.std(all_preds) > 1e-8 and np.std(all_trues) > 1e-8:
            pcc, _ = pearsonr(all_trues, all_preds)
            scc, _ = spearmanr(all_trues, all_preds)
        rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
        r2 = r2_score(all_trues, all_preds)
    except Exception as e:
        print(f"计算指标时出错: {e}")
    metrics = [avg_loss, rmse, pcc, scc, r2]
    if return_details_for_vis: return tuple(metrics) + ({'predictions': all_preds, 'trues': all_trues},)
    return tuple(metrics)


def visualize_results(history, test_details, model, output_dir, fold_num, model_name):
    # ... (此函数保持不变)
    pass


def visualize_cv_summary(all_folds_metrics_df, output_dir, model_name):
    # ... (此函数保持不变)
    pass


def main(cli_args):
    model_name = "HD_WFAN_FV_MSE"
    print(f"主程序: 开始执行RNA-DTA交叉验证 ({model_name} - 标准MSE版)...")
    print(f"主程序: 设备检测为: {device}")
    processed_data_dir = os.path.join(cli_args.data_dir, cli_args.input_file)
    try:
        complete_dataset = RNADataset(data_path=processed_data_dir)
        print(f"主程序: 预处理数据集加载成功，总样本数: {len(complete_dataset)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}");
        sys.exit(1)
    affinity_labels = np.array(complete_dataset.labels)
    binned_labels = KBinsDiscretizer(n_bins=cli_args.num_folds, encode='ordinal', strategy='uniform').fit_transform(
        affinity_labels.reshape(-1, 1))[:, 0]
    skf = RepeatedStratifiedKFold(n_splits=cli_args.num_folds, n_repeats=1, random_state=cli_args.seed)
    all_folds_metrics = []

    # --- 核心修改: 单折测试逻辑 ---
    all_splits = list(skf.split(np.zeros(len(complete_dataset)), binned_labels))

    if cli_args.target_fold is not None and 1 <= cli_args.target_fold <= cli_args.num_folds:
        target_fold_index = cli_args.target_fold - 1
        print(f"\n===== 检测到目标折数: 将只运行第 {cli_args.target_fold} 折 =====")
        splits_to_run = [(cli_args.target_fold, all_splits[target_fold_index])]
    else:
        print(f"\n===== 将运行全部 {cli_args.num_folds} 折交叉验证 =====")
        splits_to_run = list(enumerate(all_splits, 1))
    # --- 修改结束 ---

    for fold_num, (train_idx, test_idx) in splits_to_run:
        print(f"\n===== 开始处理第 {fold_num}/{cli_args.num_folds} 折 =====")
        train_loader = DataLoader(Subset(complete_dataset, train_idx), batch_size=cli_args.batch_size, shuffle=True,
                                  collate_fn=rna_dta_collate_fn, num_workers=cli_args.num_workers, pin_memory=True,
                                  drop_last=True, persistent_workers=cli_args.num_workers > 0)
        test_loader = DataLoader(Subset(complete_dataset, test_idx), batch_size=cli_args.batch_size, shuffle=False,
                                 collate_fn=rna_dta_collate_fn, num_workers=cli_args.num_workers, pin_memory=True,
                                 persistent_workers=cli_args.num_workers > 0)
        model = RNADTA_FM_CrossAttentionModel(**vars(cli_args)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
        loss_fn = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=cli_args.use_amp)
        history, best_pcc, epochs_no_improve, best_epoch = [], -1.0, 0, -1
        output_dir = os.path.dirname(cli_args.save_path)
        model_path = os.path.join(output_dir, f"{model_name}_fold_{fold_num}.pth")
        vis_dir = os.path.join(output_dir, f'visualizations/{model_name}_fold_{fold_num}')
        print(f"  开始训练... 最佳模型将保存至: {model_path}, 可视化图表将保存至: {vis_dir}")
        for epoch in range(1, cli_args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler)
            test_loss, rmse, pcc, scc, r2 = evaluate_epoch(model, test_loader, loss_fn, f"Epoch {epoch} Test")
            history.append(
                {'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss, 'pcc': pcc, 'rmse': rmse, 'r2': r2,
                 'scc': scc})
            epoch_log_summary = f"    Epoch {epoch}/{cli_args.epochs}: TrainLoss={train_loss:.4f}, TestLoss={test_loss:.4f} | TestMetrics: RMSE={rmse:.4f} PCC={pcc:.4f} R2={r2:.4f}"
            if pcc > best_pcc:
                best_pcc, best_epoch, epochs_no_improve = pcc, epoch, 0
                torch.save(model.state_dict(), model_path)
                epoch_log_summary += " * (新最佳PCC! 模型已保存)"
            else:
                epochs_no_improve += 1
                epoch_log_summary += f" (无改善 {epochs_no_improve}/{cli_args.early_stopping_patience})"
            print(epoch_log_summary)
            if epochs_no_improve >= cli_args.early_stopping_patience: print(f"\n    早停触发。"); break
        print(f"\n  当前折训练结束。最佳PCC出现在第 {best_epoch} 轮。")
        print(f"  加载最佳模型并进行最终评估和可视化...")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            final_loss, final_rmse, final_pcc, final_scc, final_r2, final_details = evaluate_epoch(model, test_loader,
                                                                                                   loss_fn,
                                                                                                   "Final Eval",
                                                                                                   return_details_for_vis=True)
            visualize_results(pd.DataFrame(history), final_details, model, vis_dir, fold_num, model_name)
            all_folds_metrics.append(
                {'fold': fold_num, 'best_test_epoch': best_epoch, 'pearson': final_pcc, 'rmse': final_rmse,
                 'spearman': final_scc, 'r2': final_r2})
        else:
            print("  警告：未找到已保存的最佳模型，跳过最终评估和可视化。")
        del model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()

    if cli_args.target_fold is None:
        results_df = pd.DataFrame(all_folds_metrics)
        csv_path = os.path.join(output_dir, f"{model_name}_all_folds_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n所有折的详细结果已保存到: {csv_path}")
        vis_summary_dir = os.path.join(os.path.dirname(cli_args.save_path), 'visualizations')
        visualize_cv_summary(results_df, vis_summary_dir, model_name)
        print(f"\n===== 交叉验证平均性能指标 ({model_name}版本) =====")
        print(results_df.drop(columns=['fold', 'best_test_epoch']).mean())
        print("==============================================")


if __name__ == "__main__":
    class ScriptArgs:
        # --- 核心修改: 增加目标折参数 ---
        target_fold = 4  # <-- 在这里设置您想跑的折数，例如 4。设为 None 或 0 则运行全部。
        # --------------------------------

        data_dir = 'RNA_DTA_data'
        input_file = 'processed_pt_files_simple_truncate'
        target_id_col, smiles_col, seq_col, affinity_col = 'Target_RNA_ID', 'SMILES', 'Target_RNA_sequence', 'pKd'
        st_file_dir = DEFAULT_ST_DIR
        pretrain_path = 'save/model_pre20.pth'
        save_path = 'save_rna_cv_results/rna_dta_model_final.pth'
        num_folds, seed, epochs, batch_size = 5, 42, 500, 32
        lr, weight_decay = 5e-5, 1e-5
        freeze_drug_encoder, use_amp, early_stopping_patience, num_workers = True, True, 100, 2
        deep_fusion_layers = 2
        cross_attention_embed_dim, cross_attention_num_heads, cross_attention_dropout = 128, 4, 0.1
        rna_fm_embed_dim = RNA_FM_DIM
        cnn_num_filters, cnn_kernel_sizes, rna_dropout = 128, [3, 5, 7], 0.3
        gnn_hidden_dim, gnn_layers, gnn_type, gnn_heads, gnn_pooling = 128, 3, 'GAT', 4, 'mean'
        rna_base_vocab_size, rna_base_embed_dim = RNA_BASE_VOCAB_SIZE, BASE_EMBED_DIM
        use_feature_gating, gating_mlp_hidden_dim = False, 32
        eskmer_dim, eskmer_projection_dim = ESKMER_DIM, 32
        mlp_pred_hidden_dims, mlp_pred_dropout = [512, 256], 0.3


    args_instance = ScriptArgs()
    # ... (打印参数和检查目录的代码) ...
    try:
        main(args_instance)
    except Exception as e_main:
        print(f"\n主程序执行过程中发生未捕获的严重错误: {e_main}")
        import traceback

        traceback.print_exc()
    print("\n交叉验证及可视化流程执行完毕。")

# # --- END OF FILE finetune_rna_dta_Regression_cv.py (Single Fold Execution Version) ---