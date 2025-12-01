# --- START OF FILE rna_dta_fm_cross_attention_model.py (Standard MSE Version) ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
import math

try:
    from downstream_until import CL_model_2d
    from rna_dta_dataset import RNA_BASE_VOCAB_SIZE, NUM_EDGE_FEATURES, RNA_FM_DIM, ESKMER_DIM, BASE_EMBED_DIM
    from process_dataset.pcqm4m import drug2emb_encoder
except ImportError as e:
    print(f"模型文件导入错误: {e}")
    import sys;

    sys.exit(1)


# =========================================================================================
# === 核心模块 (与您效果最好的HD-WFAN-FV版本相同) ===
# =========================================================================================
class CrossAttentionLayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(CrossAttentionLayerNorm, self).__init__()
        self.gamma, self.beta, self.variance_epsilon = nn.Parameter(torch.ones(hidden_size)), nn.Parameter(
            torch.zeros(hidden_size)), variance_epsilon

    def forward(self, x):
        u, s = x.mean(-1, keepdim=True), (x - x.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True)
        return self.gamma * ((x - u) / torch.sqrt(s + self.variance_epsilon)) + self.beta


class DisentangledCrossFusionFV(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(DisentangledCrossFusionFV, self).__init__()
        if hidden_size % num_attention_heads != 0: raise ValueError(
            f"Hidden size ({hidden_size}) must be divisible by num_heads ({num_attention_heads}).")
        self.num_attention_heads, self.attention_head_size = num_attention_heads, int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_drug_seq, self.key_drug_seq, self.value_drug_seq = [nn.Linear(hidden_size, self.all_head_size) for _
                                                                       in range(3)]
        self.query_drug_struct, self.key_drug_struct, self.value_drug_struct = [
            nn.Linear(hidden_size, self.all_head_size) for _ in range(3)]
        self.query_rna_seq, self.key_rna_seq, self.value_rna_seq = [nn.Linear(hidden_size, self.all_head_size) for _ in
                                                                    range(3)]
        self.query_rna_struct, self.key_rna_struct, self.value_rna_struct = [nn.Linear(hidden_size, self.all_head_size)
                                                                             for _ in range(3)]
        self.value_fusion_mlp = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size * 2), nn.ReLU(),
                                              nn.Linear(hidden_size * 2, hidden_size))
        self.fusion_weights_drug_on_rna, self.fusion_weights_rna_on_drug = nn.Parameter(torch.ones(4)), nn.Parameter(
            torch.ones(4))
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        return x.view(*x.size()[:-1], self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

    def forward(self, drug_seq_feat, drug_struct_feat, rna_seq_feat, rna_struct_feat):
        q_ds, k_ds = self.transpose_for_scores(self.query_drug_seq(drug_seq_feat)), self.transpose_for_scores(
            self.key_drug_seq(drug_seq_feat))
        q_dg, k_dg = self.transpose_for_scores(self.query_drug_struct(drug_struct_feat)), self.transpose_for_scores(
            self.key_drug_struct(drug_struct_feat))
        q_rs, k_rs = self.transpose_for_scores(self.query_rna_seq(rna_seq_feat)), self.transpose_for_scores(
            self.key_rna_seq(rna_seq_feat))
        q_rg, k_rg = self.transpose_for_scores(self.query_rna_struct(rna_struct_feat)), self.transpose_for_scores(
            self.key_rna_struct(rna_struct_feat))
        v_fused = self.value_fusion_mlp(
            torch.cat([f.squeeze(1) for f in [drug_seq_feat, drug_struct_feat, rna_seq_feat, rna_struct_feat]],
                      dim=-1)).unsqueeze(1)
        v_ds, v_dg = self.transpose_for_scores(self.value_drug_seq(v_fused)), self.transpose_for_scores(
            self.value_drug_struct(v_fused))
        v_rs, v_rg = self.transpose_for_scores(self.value_rna_seq(v_fused)), self.transpose_for_scores(
            self.value_rna_struct(v_fused))

        def compute_weighted_fused_attention(q_list, k_list, v_list, weights):
            normalized_weights = F.softmax(weights, dim=0)
            score11, score12 = torch.matmul(q_list[0], k_list[0].transpose(-1, -2)), torch.matmul(q_list[0],
                                                                                                  k_list[1].transpose(
                                                                                                      -1, -2))
            score21, score22 = torch.matmul(q_list[1], k_list[0].transpose(-1, -2)), torch.matmul(q_list[1],
                                                                                                  k_list[1].transpose(
                                                                                                      -1, -2))
            score_q1 = torch.cat([score11 * normalized_weights[0], score12 * normalized_weights[1]], dim=-1)
            score_q2 = torch.cat([score21 * normalized_weights[2], score22 * normalized_weights[3]], dim=-1)
            probs_q1, probs_q2 = F.softmax(score_q1 / math.sqrt(self.attention_head_size), dim=-1), F.softmax(
                score_q2 / math.sqrt(self.attention_head_size), dim=-1)
            v_combined = torch.cat(v_list, dim=2)
            return torch.matmul(self.dropout(probs_q1), v_combined), torch.matmul(self.dropout(probs_q2), v_combined)

        ctx_ds, ctx_dg = compute_weighted_fused_attention([q_ds, q_dg], [k_rs, k_rg], [v_rs, v_rg],
                                                          self.fusion_weights_drug_on_rna)
        ctx_rs, ctx_rg = compute_weighted_fused_attention([q_rs, q_rg], [k_ds, k_dg], [v_ds, v_dg],
                                                          self.fusion_weights_rna_on_drug)

        def reshape_context(context):
            context = context.permute(0, 2, 1, 3).contiguous()
            return context.view(*context.size()[:-2] + (self.all_head_size,))

        return reshape_context(ctx_ds), reshape_context(ctx_dg), reshape_context(ctx_rs), reshape_context(ctx_rg)


class DisentangledCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, attention_dropout, hidden_dropout):
        super(DisentangledCrossAttentionBlock, self).__init__()
        self.cross_attention = DisentangledCrossFusionFV(hidden_size, num_attention_heads, attention_dropout)
        self.dense_out_ds, self.dense_out_dg, self.dense_out_rs, self.dense_out_rg = (
        nn.Linear(hidden_size, hidden_size) for _ in range(4))
        self.norm_ds_attn, self.norm_dg_attn, self.norm_rs_attn, self.norm_rg_attn = (
        CrossAttentionLayerNorm(hidden_size) for _ in range(4))
        self.dropout_attn = nn.Dropout(hidden_dropout)
        self.ffn_ds, self.ffn_dg, self.ffn_rs, self.ffn_rg = (
        nn.Sequential(nn.Linear(hidden_size, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
        for _ in range(4))
        self.norm_ds_ffn, self.norm_dg_ffn, self.norm_rs_ffn, self.norm_rg_ffn = (CrossAttentionLayerNorm(hidden_size)
                                                                                  for _ in range(4))
        self.dropout_ffn = nn.Dropout(hidden_dropout)

    def forward(self, drug_seq_in, drug_struct_in, rna_seq_in, rna_struct_in):
        ctx_ds, ctx_dg, ctx_rs, ctx_rg = self.cross_attention(drug_seq_in, drug_struct_in, rna_seq_in, rna_struct_in)
        ds_after_attn = self.norm_ds_attn(drug_seq_in + self.dropout_attn(self.dense_out_ds(ctx_ds)))
        dg_after_attn = self.norm_dg_attn(drug_struct_in + self.dropout_attn(self.dense_out_dg(ctx_dg)))
        rs_after_attn = self.norm_rs_attn(rna_seq_in + self.dropout_attn(self.dense_out_rs(ctx_rs)))
        rg_after_attn = self.norm_rg_attn(rna_struct_in + self.dropout_attn(self.dense_out_rg(ctx_rg)))
        ds_out = self.norm_ds_ffn(ds_after_attn + self.dropout_ffn(self.ffn_ds(ds_after_attn)))
        dg_out = self.norm_dg_ffn(dg_after_attn + self.dropout_ffn(self.ffn_dg(dg_after_attn)))
        rs_out = self.norm_rs_ffn(rs_after_attn + self.dropout_ffn(self.ffn_rs(rs_after_attn)))
        rg_out = self.norm_rg_ffn(rg_after_attn + self.dropout_ffn(self.ffn_rg(rg_after_attn)))
        return ds_out, dg_out, rs_out, rg_out


class RNAEncoderFM_CNN_GNN_Disentangled(nn.Module):
    def __init__(self, rna_fm_embed_dim, cnn_num_filters=128, cnn_kernel_sizes=[3, 5, 7], cnn_dropout=0.25,
                 gnn_hidden_dim=128, gnn_layers=3, gnn_type='GAT', gnn_edge_dim_param=NUM_EDGE_FEATURES, gnn_heads=4,
                 gnn_dropout=0.2, gnn_pooling='mean', rna_base_vocab_size_param=RNA_BASE_VOCAB_SIZE,
                 rna_base_embed_dim_param=BASE_EMBED_DIM, eskmer_dim_param=ESKMER_DIM, eskmer_projection_dim=32,
                 use_feature_gating=True, gating_mlp_hidden_dim=32):
        super().__init__()
        self.use_feature_gating = use_feature_gating
        print(f"  RNA编码器初始化 (Standard MSE Version):")
        self.eskmer_projection = nn.Sequential(nn.Linear(eskmer_dim_param, eskmer_projection_dim), nn.ReLU(),
                                               nn.LayerNorm(eskmer_projection_dim))
        cnn_input_dim = rna_fm_embed_dim + eskmer_projection_dim
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=cnn_input_dim, out_channels=cnn_num_filters, kernel_size=k, padding=(k - 1) // 2) for
             k in cnn_kernel_sizes])
        self.cnn_dropout, self.cnn_output_dim = nn.Dropout(cnn_dropout), cnn_num_filters * len(cnn_kernel_sizes)
        self.gnn_type = gnn_type.upper()
        self.base_embedding = nn.Embedding(rna_base_vocab_size_param, rna_base_embed_dim_param)
        gnn_node_input_dim_after_concat = rna_fm_embed_dim + rna_base_embed_dim_param
        if self.use_feature_gating: self.feature_gate_mlp = nn.Sequential(
            nn.Linear(gnn_node_input_dim_after_concat, gating_mlp_hidden_dim), nn.ReLU(),
            nn.Linear(gating_mlp_hidden_dim, gnn_node_input_dim_after_concat), nn.Sigmoid())
        self.gnn_layers = nn.ModuleList()
        current_gnn_input_dim = gnn_node_input_dim_after_concat
        for i in range(gnn_layers):
            is_last_gat_layer, current_gat_heads = (self.gnn_type == 'GAT' and i == gnn_layers - 1), (
                1 if (self.gnn_type == 'GAT' and i == gnn_layers - 1) else gnn_heads)
            current_gnn_out_channels = gnn_hidden_dim if not (
                        self.gnn_type == 'GAT' and not is_last_gat_layer) else gnn_hidden_dim // gnn_heads
            if self.gnn_type == 'GCN':
                conv = GCNConv(current_gnn_input_dim, gnn_hidden_dim)
            elif self.gnn_type == 'GAT':
                conv = GATConv(current_gnn_input_dim, current_gnn_out_channels, heads=current_gat_heads,
                               dropout=gnn_dropout, add_self_loops=True, edge_dim=gnn_edge_dim_param,
                               concat=(not is_last_gat_layer))
            else:
                raise ValueError("GNN 类型必须是 'GCN' 或 'GAT'")
            self.gnn_layers.append(conv);
            current_gnn_input_dim = gnn_hidden_dim
        self.gnn_dropout = nn.Dropout(gnn_dropout)
        self.gnn_pool = global_mean_pool if gnn_pooling == 'mean' else global_add_pool
        self.gnn_output_dim = gnn_hidden_dim
        print("  RNA编码器: 将返回独立的CNN(序列)和GNN(结构)特征。\n")

    def forward(self, rna_fm_embeddings_cnn, rna_fm_mask_cnn, rna_local_eskmer_sequence, rna_graph_batch_gnn):
        projected_eskmer_seq = self.eskmer_projection(rna_local_eskmer_sequence)
        embedded_seq_for_cnn = torch.cat([rna_fm_embeddings_cnn, projected_eskmer_seq], dim=-1).permute(0, 2, 1)
        mask = rna_fm_mask_cnn.unsqueeze(1).repeat(1, embedded_seq_for_cnn.size(1), 1)
        embedded_seq_for_cnn = embedded_seq_for_cnn * mask.float()
        conv_outputs = [F.relu(conv(embedded_seq_for_cnn)) for conv in self.convs]
        pooled_outputs_cnn = [F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2) for conv_out in
                              conv_outputs]
        h_rna_seq = self.cnn_dropout(torch.cat(pooled_outputs_cnn, dim=1))
        node_features_fm_gnn = rna_graph_batch_gnn.x;
        base_indices_gnn = rna_graph_batch_gnn.base_indices
        node_features_base_emb_gnn = self.base_embedding(base_indices_gnn)
        current_node_features_gnn = torch.cat([node_features_fm_gnn, node_features_base_emb_gnn], dim=-1)
        if self.use_feature_gating: current_node_features_gnn = self.feature_gate_mlp(
            current_node_features_gnn) * current_node_features_gnn
        edge_index_gnn, gnn_batch_assignment = rna_graph_batch_gnn.edge_index, rna_graph_batch_gnn.batch
        edge_attr_gnn = rna_graph_batch_gnn.edge_attr if hasattr(rna_graph_batch_gnn, 'edge_attr') else None
        for i, layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'GAT':
                current_node_features_gnn = layer(current_node_features_gnn, edge_index_gnn, edge_attr=edge_attr_gnn)
            else:
                current_node_features_gnn = layer(current_node_features_gnn, edge_index_gnn)
            if i < len(self.gnn_layers) - 1: current_node_features_gnn = self.gnn_dropout(
                F.relu(current_node_features_gnn))
        h_rna_struct = self.gnn_pool(current_node_features_gnn, gnn_batch_assignment)
        return h_rna_seq, h_rna_struct


class RNADTA_FM_CrossAttentionModel(nn.Module):
    def __init__(self, pretrain_path = 'save/model_pre20.pth', device='cuda:0', deep_fusion_layers=2, cross_attention_embed_dim=128,
                 cross_attention_num_heads=4, cross_attention_dropout=0.1, rna_fm_embed_dim=RNA_FM_DIM,
                 cnn_num_filters=128, cnn_kernel_sizes=[3, 5, 7], rna_dropout=0.2, gnn_hidden_dim=128, gnn_layers=3,
                 gnn_type='GAT', gnn_heads=4, gnn_pooling='mean', rna_base_vocab_size=RNA_BASE_VOCAB_SIZE,
                 rna_base_embed_dim=BASE_EMBED_DIM, eskmer_dim=ESKMER_DIM, eskmer_projection_dim=32,
                 use_feature_gating=False, gating_mlp_hidden_dim=32, mlp_pred_hidden_dims=[512, 256],
                 mlp_pred_dropout=0.2, **kwargs):
        super().__init__()
        self.device = device
        self.drug_encoder_core = CL_model_2d().to(device)
        self._load_pretrained_drug_encoder(pretrain_path, device)
        self.drug_seq_dim, self.drug_struct_dim = 128, 128
        self.rna_encoder_module = RNAEncoderFM_CNN_GNN_Disentangled(rna_fm_embed_dim=rna_fm_embed_dim,
                                                                    cnn_num_filters=cnn_num_filters,
                                                                    cnn_kernel_sizes=cnn_kernel_sizes,
                                                                    cnn_dropout=rna_dropout,
                                                                    gnn_hidden_dim=gnn_hidden_dim,
                                                                    gnn_layers=gnn_layers, gnn_type=gnn_type,
                                                                    gnn_edge_dim_param=NUM_EDGE_FEATURES,
                                                                    gnn_heads=gnn_heads, gnn_dropout=rna_dropout,
                                                                    gnn_pooling=gnn_pooling,
                                                                    rna_base_vocab_size_param=rna_base_vocab_size,
                                                                    rna_base_embed_dim_param=rna_base_embed_dim,
                                                                    eskmer_dim_param=eskmer_dim,
                                                                    eskmer_projection_dim=eskmer_projection_dim,
                                                                    use_feature_gating=use_feature_gating,
                                                                    gating_mlp_hidden_dim=gating_mlp_hidden_dim)
        self.rna_seq_dim, self.rna_struct_dim = self.rna_encoder_module.cnn_output_dim, self.rna_encoder_module.gnn_output_dim
        self.proj_drug_seq, self.proj_drug_struct = nn.Linear(self.drug_seq_dim, cross_attention_embed_dim), nn.Linear(
            self.drug_struct_dim, cross_attention_embed_dim)
        self.proj_rna_seq, self.proj_rna_struct = nn.Linear(self.rna_seq_dim, cross_attention_embed_dim), nn.Linear(
            self.rna_struct_dim, cross_attention_embed_dim)
        self.deep_fusion_network = nn.ModuleList([DisentangledCrossAttentionBlock(hidden_size=cross_attention_embed_dim,
                                                                                  num_attention_heads=cross_attention_num_heads,
                                                                                  intermediate_size=cross_attention_embed_dim * 4,
                                                                                  attention_dropout=cross_attention_dropout,
                                                                                  hidden_dropout=0.1) for _ in
                                                  range(deep_fusion_layers)])

        # --- 核心修改: MLP预测头现在输出1个值 ---
        mlp_input_dim = cross_attention_embed_dim * 4
        self.mlp_prediction_head = self._create_mlp(mlp_input_dim, mlp_pred_hidden_dims, mlp_pred_dropout)

    def _create_mlp(self, input_dim, hidden_dims, dropout):
        layers = [];
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]);
            current_dim = hidden_dim
        # 最后一层输出1个值
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def _load_pretrained_drug_encoder(self, pretrain_path, device):
        try:
            state_dict = torch.load(pretrain_path, map_location=device)
            state_dict_to_load = state_dict.get('cl_model', state_dict)
            self.drug_encoder_core.load_state_dict({k.replace('module.', ''): v for k, v in state_dict_to_load.items()},
                                                   strict=False)
        except Exception as e:
            print(f"模型错误: 加载预训练药物权重 {pretrain_path} 失败: {e}");
            raise

    def forward(self, drug_graph_batch, rna_fm_embeddings_cnn_batch, rna_fm_mask_cnn_batch,
                rna_local_eskmer_sequence_batch, rna_graph_gnn_batch, original_smiles_list):
        drug_smiles_tokens_list, drug_smiles_masks_list = [], []
        for smi_str in original_smiles_list:
            tokens, mask = drug2emb_encoder(smi_str)
            drug_smiles_tokens_list.append(tokens);
            drug_smiles_masks_list.append(mask)
        drug_graph_batch.smiles, drug_graph_batch.mask = drug_smiles_tokens_list, drug_smiles_masks_list
        _, drug_struct_feat_raw, _, drug_seq_feat_raw = self.drug_encoder_core(drug_graph_batch)
        rna_seq_feat_raw, rna_struct_feat_raw = self.rna_encoder_module(rna_fm_embeddings_cnn_batch,
                                                                        rna_fm_mask_cnn_batch,
                                                                        rna_local_eskmer_sequence_batch,
                                                                        rna_graph_gnn_batch)
        drug_seq_feat = self.proj_drug_seq(drug_struct_feat_raw).unsqueeze(1)
        drug_struct_feat = self.proj_drug_struct(drug_struct_feat_raw).unsqueeze(1)
        rna_seq_feat = self.proj_rna_seq(rna_seq_feat_raw).unsqueeze(1)
        rna_struct_feat = self.proj_rna_struct(rna_struct_feat_raw).unsqueeze(1)
        current_ds, current_dg, current_rs, current_rg = drug_seq_feat, drug_struct_feat, rna_seq_feat, rna_struct_feat
        for fusion_block in self.deep_fusion_network:
            current_ds, current_dg, current_rs, current_rg = fusion_block(current_ds, current_dg, current_rs,
                                                                          current_rg)
        final_combined_for_mlp = torch.cat(
            [current_ds.squeeze(1), current_dg.squeeze(1), current_rs.squeeze(1), current_rg.squeeze(1)], dim=1)

        # --- 核心修改: 直接返回预测值 ---
        prediction = self.mlp_prediction_head(final_combined_for_mlp)

        return prediction.squeeze(-1)

# --- END OF FILE rna_dta_fm_cross_attention_model.py (Standard MSE Version) ---