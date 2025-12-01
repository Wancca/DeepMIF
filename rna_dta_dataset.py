# --- START OF FILE rna_dta_dataset.py (For Pre-processed .pt Files) ---

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import sys
import re
from rdkit import Chem
from torch_geometric.data import Data
from rdkit import RDLogger
from itertools import product

os.environ["PYTORCH_NO_WARNINGS"] = "1"
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

try:
    from process_dataset.DTA.utils.features import (
        atom_to_feature_vector,
        bond_to_feature_vector,
    )
except ImportError:
    print("错误: 无法导入 process_dataset 中的函数/对象。请确保路径设置正确。")
    sys.exit(1)

# --- 常量定义 ---
RNA_BASE_VOCAB = {'<pad>': 0, '<unk>': 1, 'A': 2, 'U': 3, 'G': 4, 'C': 5, 'N': 1}
RNA_BASE_VOCAB_SIZE = len(RNA_BASE_VOCAB)
RNA_FM_EMBEDDING_DIR = 'RNA_DTA_data/representations_cv'
RNA_FM_MAX_LEN = 512
RNA_FM_PAD_VALUE = 0.0
DRUG_MAX_ATOMS = 128

try:
    example_files = [f for f in os.listdir(RNA_FM_EMBEDDING_DIR) if f.endswith('.npy')]
    if example_files:
        example_path = os.path.join(RNA_FM_EMBEDDING_DIR, example_files[0])
        example_emb = np.load(example_path)
        RNA_FM_DIM = example_emb.shape[1]
    else:
        RNA_FM_DIM = 640
except Exception:
    RNA_FM_DIM = 640


BASE_EMBED_DIM=16
LOCAL_ESKMER_K = 3
LOCAL_ESKMER_WINDOW_SIZE = 31
ESKMER_BASES = ['A', 'U', 'G', 'C']
ESKMER_VOCAB_LIST = ["".join(p) for p in product(ESKMER_BASES, repeat=LOCAL_ESKMER_K)]
ESKMER_VOCAB_MAP = {kmer: i for i, kmer in enumerate(ESKMER_VOCAB_LIST)}
ESKMER_DIM = len(ESKMER_VOCAB_LIST)

BASE_EDGE_TYPES = ["WC_PAIR", "WOBBLE_PAIR", "STEM_BACKBONE", "LOOP_BACKBONE", "STEM_LOOP_JUNCTION"]
FRAGMENT_CONTEXT_FEATURES = ["IS_DEFINED_STEM_PAIR", "IS_DEFINED_STEM_BACKBONE", "IS_INTER_MODULE_BACKBONE"]
NUM_BASE_EDGE_FEATURES = len(BASE_EDGE_TYPES)
NUM_FRAGMENT_CONTEXT_FEATURES = len(FRAGMENT_CONTEXT_FEATURES)
NUM_EDGE_FEATURES = NUM_BASE_EDGE_FEATURES + NUM_FRAGMENT_CONTEXT_FEATURES
ST_FILE_DIR = 'RNA_DTA_data/st_outputs_final'
STRUCT_MODULE_CHARS = {'S', 'H', 'I', 'B', 'M', 'E'}
LOOP_MODULE_CHARS = {'H', 'I', 'B', 'M', 'E'}

# --- 辅助函数 (需要保留，因为预处理脚本会调用它们) ---
def _calculate_eskmer_for_subsequence(subsequence, k, vocab_map):
    L = len(subsequence)
    eskmer_counts = np.zeros(len(vocab_map), dtype=np.float32)
    if L < k: return eskmer_counts
    m1, m2 = (k + 1) // 2, k - ((k + 1) // 2)
    for kmer, kmer_idx in vocab_map.items():
        sub1, sub2 = kmer[:m1], kmer[m1:]
        count = 0
        for i in range(L - k + 1):
            for j in range(i + m1, L - m2 + 1):
                if subsequence[i:i + m1] == sub1 and subsequence[j:j + m2] == sub2: count += 1
        eskmer_counts[kmer_idx] = count
    return eskmer_counts

def seq_to_local_eskmer_sequence(sequence, k, window_size, vocab_map, max_len):
    L = len(sequence)
    local_eskmer_features = np.zeros((max_len, len(vocab_map)), dtype=np.float32)
    num_windows = L - window_size + 1
    if num_windows <= 0: return torch.from_numpy(local_eskmer_features), torch.zeros(max_len, dtype=torch.long)
    all_window_eskmers = np.array([_calculate_eskmer_for_subsequence(sequence[i:i + window_size], k, vocab_map) for i in range(num_windows)])
    norm_val = np.linalg.norm(all_window_eskmers);
    if norm_val > 0: all_window_eskmers /= norm_val
    effective_len = min(num_windows, max_len)
    local_eskmer_features[:effective_len, :] = all_window_eskmers[:effective_len, :]
    mask = torch.zeros(max_len, dtype=torch.long); mask[:effective_len] = 1
    return torch.from_numpy(local_eskmer_features), mask

def smiles_to_pyg_graph(smiles_string):
    mol = None; num_drug_bond_feature_categories = 3
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: mol = Chem.MolFromSmiles(smiles_string, sanitize=False); mol.UpdatePropertyCache(strict=False)
        if mol is None: return None
        atoms = mol.GetAtoms()
        if len(atoms) > DRUG_MAX_ATOMS:
            atom_indices_to_keep = list(range(DRUG_MAX_ATOMS))
            mol = Chem.PathToSubmol(mol, [bond.GetIdx() for bond in mol.GetBonds() if bond.GetBeginAtomIdx() in atom_indices_to_keep and bond.GetEndAtomIdx() in atom_indices_to_keep])
            if mol is None: return None
        atom_features_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
        if not atom_features_list: return None
        edges_list, edge_features_list = [], []
        if mol.GetNumBonds() > 0:
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)
                if not isinstance(edge_feature, (list, np.ndarray)) or len(edge_feature) != num_drug_bond_feature_categories: continue
                edges_list.extend([(i, j), (j, i)]); edge_features_list.extend([edge_feature, edge_feature])
        data = Data()
        data.x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
        data.num_nodes = len(atom_features_list)
        if edges_list:
            data.edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
            data.edge_attr = torch.from_numpy(np.array(edge_features_list, dtype=np.int64)).to(torch.long) if edge_features_list else torch.empty((0, num_drug_bond_feature_categories), dtype=torch.long)
        else:
            data.edge_index = torch.empty((2, 0), dtype=torch.long); data.edge_attr = torch.empty((0, num_drug_bond_feature_categories), dtype=torch.long)
        return data
    except Exception: return None

def get_rna_edge_attr_vector(base_type_idx=None, is_defined_stem_pair=False, is_defined_stem_backbone=False, is_inter_module_backbone=False):
    attr = torch.zeros(NUM_EDGE_FEATURES, dtype=torch.float);
    if base_type_idx is not None and 0 <= base_type_idx < NUM_BASE_EDGE_FEATURES: attr[base_type_idx] = 1.0
    if is_defined_stem_pair: attr[NUM_BASE_EDGE_FEATURES + 0] = 1.0
    if is_defined_stem_backbone: attr[NUM_BASE_EDGE_FEATURES + 1] = 1.0
    if is_inter_module_backbone: attr[NUM_BASE_EDGE_FEATURES + 2] = 1.0
    return attr

def get_rna_pair_type_and_context(base1, base2, is_in_defined_stem):
    pair = {base1.upper(), base2.upper()}; base_type_idx = None
    if pair == {'A', 'U'} or pair == {'G', 'C'}: base_type_idx = BASE_EDGE_TYPES.index("WC_PAIR")
    elif pair == {'G', 'U'}: base_type_idx = BASE_EDGE_TYPES.index("WOBBLE_PAIR")
    return get_rna_edge_attr_vector(base_type_idx=base_type_idx, is_defined_stem_pair=is_in_defined_stem if base_type_idx is not None else False)

def get_backbone_edge_attr_and_context(module_type_i, module_type_j, is_in_defined_stem):
    is_i_stem_generic, is_j_stem_generic = module_type_i == 'S', module_type_j == 'S'
    is_i_loop_generic, is_j_loop_generic = module_type_i in LOOP_MODULE_CHARS, module_type_j in LOOP_MODULE_CHARS
    base_type_idx = None
    if is_i_stem_generic and is_j_stem_generic: base_type_idx = BASE_EDGE_TYPES.index("STEM_BACKBONE")
    elif is_i_loop_generic and is_j_loop_generic: base_type_idx = BASE_EDGE_TYPES.index("LOOP_BACKBONE")
    else: base_type_idx = BASE_EDGE_TYPES.index("STEM_LOOP_JUNCTION")
    is_inter_module = (module_type_i != module_type_j)
    return get_rna_edge_attr_vector(base_type_idx=base_type_idx, is_defined_stem_backbone=is_in_defined_stem, is_inter_module_backbone=is_inter_module)

def rna_struct_to_graph_fragment_enhanced(sequence, structure_dotbracket, modules_sixth_line, defined_elements, node_features_fm=None, base_vocab=None):
    n = len(sequence)
    if n == 0 or len(structure_dotbracket) != n or len(modules_sixth_line) != n: return None
    if base_vocab is None or node_features_fm is None or node_features_fm.shape[0] != n: return None
    x_fm_for_gnn, unk_token_id = node_features_fm, base_vocab.get('<unk>', 1)
    base_indices_tensor = torch.tensor([base_vocab.get(base.upper(), unk_token_id) for base in sequence], dtype=torch.long)
    edge_list, edge_attributes_list = [], []
    is_in_defined_stem_array = [False] * n
    for el_id, el_type, ranges in defined_elements:
        if el_type == 'S':
            for r_start, r_end in ranges:
                for i in range(r_start - 1, r_end):
                    if 0 <= i < n: is_in_defined_stem_array[i] = True
    for i in range(n - 1):
        backbone_attr = get_backbone_edge_attr_and_context(modules_sixth_line[i], modules_sixth_line[i + 1], is_in_defined_stem_array[i] and is_in_defined_stem_array[i + 1])
        edge_list.extend([(i, i + 1), (i + 1, i)]); edge_attributes_list.extend([backbone_attr, backbone_attr])
    stack = []
    for i, char_dotbracket in enumerate(structure_dotbracket):
        if char_dotbracket == '(': stack.append(i)
        elif char_dotbracket == ')':
            if stack:
                j = stack.pop()
                pair_attr = get_rna_pair_type_and_context(sequence[i], sequence[j], is_in_defined_stem_array[i] and is_in_defined_stem_array[j])
                if pair_attr is not None and pair_attr[0:NUM_BASE_EDGE_FEATURES].sum() > 0:
                    edge_list.extend([(i, j), (j, i)]); edge_attributes_list.extend([pair_attr, pair_attr])
    if not edge_list: edge_index, edge_attr = torch.empty((2, 0), dtype=torch.long), torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float)
    else: edge_index, edge_attr = torch.tensor(edge_list, dtype=torch.long).t().contiguous(), torch.stack(edge_attributes_list, dim=0)
    return Data(x=x_fm_for_gnn, edge_index=edge_index, edge_attr=edge_attr, base_indices=base_indices_tensor, num_nodes=n)

def parse_st_file_enhanced(target_rna_id, st_file_directory):
    st_file_path = os.path.join(st_file_directory, f"{target_rna_id}.st")
    if not os.path.exists(st_file_path): return None, None, None, None
    try:
        with open(st_file_path, 'r') as f: lines = [line.strip() for line in f.readlines()]
        if len(lines) < 6: return None, None, None, None
        sequence_st, dot_bracket_st, modules_sixth_line_st = lines[3], lines[4], list(lines[5])
        defined_elements = []
        element_pattern = re.compile(r"([SHIBME])([\w.]+)\s+(\d+)\.\.(\d+)\s+\"([ACGUN]*)\"(?:\s+(\d+)\.\.(\d+)\s+\"([ACGUN]*)\")?(?:\s+\((\d+),(\d+)\)\s+([ACGUN]):([ACGUN]))?")
        for line in lines[7:]:
            if not line or line.startswith("#") or "segment" in line.lower(): continue
            match = element_pattern.match(line)
            if match:
                el_type_char, el_id_suffix = match.group(1), match.group(2)
                ranges = [(int(match.group(3)), int(match.group(4)))]
                if match.group(6) and match.group(7): ranges.append((int(match.group(6)), int(match.group(7))))
                defined_elements.append((f"{el_type_char}{el_id_suffix}", el_type_char, ranges))
        if not sequence_st or not dot_bracket_st or not modules_sixth_line_st: return None, None, None, None
        return sequence_st, dot_bracket_st, modules_sixth_line_st, defined_elements
    except Exception: return None, None, None, None

class RNADataset(Dataset):
    def __init__(self, data_path, **kwargs):
        super().__init__()
        self.processed_dir = data_path
        print(f"Dataset: 正在从预处理目录加载数据: {self.processed_dir}")
        if not os.path.isdir(self.processed_dir):
            raise FileNotFoundError(f"错误: 预处理数据目录未找到: '{self.processed_dir}'.\n请先运行 'preprocess_data_to_pt.py' 脚本。")
        self.file_paths = [os.path.join(self.processed_dir, f) for f in sorted(os.listdir(self.processed_dir)) if f.endswith('.pt')]
        if not self.file_paths:
            raise ValueError(f"错误: 在目录 '{self.processed_dir}' 中没有找到任何 .pt 文件。")
        print(f"Dataset: 成功找到 {len(self.file_paths)} 个预处理样本。")
        self.labels = [torch.load(p)['affinity'].item() for p in self.file_paths]
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        try:
            return torch.load(self.file_paths[idx])
        except Exception as e:
            print(f"警告: 加载文件 {self.file_paths[idx]} 失败: {e}. 跳过此样本。")
            return None

def rna_dta_collate_fn(batch_items_list):
    valid_samples = [item for item in batch_items_list if item is not None]
    if not valid_samples: return None
    collated_batch = {}
    from torch_geometric.data import Batch as PyGBatch
    all_keys = set().union(*(d.keys() for d in valid_samples))
    for key in all_keys:
        values = [sample.get(key) for sample in valid_samples]
        if any(v is None for v in values): continue
        if isinstance(values[0], Data):
            collated_batch[key] = PyGBatch.from_data_list(values)
        elif isinstance(values[0], torch.Tensor):
            collated_batch[key] = torch.stack(values, 0)
        else:
            collated_batch[key] = values
    if 'affinity' in collated_batch and collated_batch['affinity'].ndim > 1:
        collated_batch['affinity'] = collated_batch['affinity'].squeeze()
    return collated_batch

# --- END OF FILE rna_dta_dataset.py (For Pre-processed .pt Files) ---