import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import multiprocessing as mp
import os

# 🌟 全局缓存：预先获取所有的官能团提取函数
FG_FUNCS = {name: getattr(Fragments, name)
            for name in dir(Fragments) if name.startswith('fr_')}


def process_single_molecule(data_tuple):
    """
    处理单个分子的核心逻辑（在独立子进程中运行）。
    :param data_tuple: (index, smiles)
    :return: (nodes_list, edges_list)
    """
    idx, smi = data_tuple
    nodes = []
    edges = []

    # RDKit 核心计算
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol_node_id = f"MOL_{idx}"
    # 记录节点：(节点ID, 属性字典)
    nodes.append((mol_node_id, {'type': 'molecule', 'smiles': smi}))

    # 1. 提取骨架 (Scaffold)
    try:
        scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold_smi:
            scaffold_node_id = f"SCAFFOLD_{scaffold_smi}"
            nodes.append((scaffold_node_id, {'type': 'scaffold'}))
            # 记录连边：(起点ID, 终点ID, 属性字典)
            edges.append((mol_node_id, scaffold_node_id, {'relation': 'has_scaffold'}))
    except Exception:
        pass  # 忽略无标准骨架的极小分子

    # 2. 提取官能团 (Functional Groups)
    for fg_name, fg_func in FG_FUNCS.items():
        if fg_func(mol) > 0:
            clean_fg_name = fg_name.replace('fr_', '')
            fg_node_id = f"FG_{clean_fg_name}"
            nodes.append((fg_node_id, {'type': 'functional_group'}))
            edges.append((mol_node_id, fg_node_id, {'relation': 'has_group'}))

    return nodes, edges


def build_graph_multiprocess(smiles_list, num_workers=None):
    """
    多进程图谱构建调度器
    """
    # 默认留 1-2 个核心给系统，防止电脑卡死
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    print(f"\n🚀 启动多进程图谱构建，正在调用 {num_workers} 个 CPU 核心...")

    # 准备任务队列
    tasks = list(enumerate(smiles_list))

    # 1. 开启进程池，并行提取节点和连边
    with mp.Pool(num_workers) as pool:
        # imap_unordered + chunksize 配合 tqdm，是 Python 里最高效的多进程进度条写法
        chunk_size = max(1, len(tasks) // (num_workers * 4))
        results = list(tqdm(pool.imap_unordered(process_single_molecule, tasks, chunksize=chunk_size),
                            total=len(tasks), desc="⚙️ RDKit 拓扑计算"))

    # 2. 主进程合并结果入全局图谱
    print("\n🔄 正在将并行计算结果合并入全局图谱 (这一步很快)...")
    G = nx.Graph()
    for res in tqdm(results, desc="🔗 节点连边合并"):
        if res is not None:
            nodes, edges = res
            G.add_nodes_from(nodes)  # 批量添加节点
            G.add_edges_from(edges)  # 批量添加连边

    return G


if __name__ == "__main__":
    print("加载 20w SMILES 映射表...")
    df = pd.read_csv("../../data/my_200k_dataset.csv")
    my_smiles_list = df['canonical_smiles'].dropna().tolist()

    # 启动核心逻辑
    structural_graph = build_graph_multiprocess(my_smiles_list)

    # 战报输出
    print("\n✅ 图谱构建彻底完成！")
    print(f"📊 总节点数 (Nodes): {structural_graph.number_of_nodes()}")
    print(f"🔗 总连边数 (Edges): {structural_graph.number_of_edges()}")

    # 统计一下节点分布
    node_types = nx.get_node_attributes(structural_graph, 'type')
    from collections import Counter

    print(f"🧩 节点分布: {dict(Counter(node_types.values()))}")

    # 保存结果
    os.makedirs("../../data/processed", exist_ok=True)
    save_path = "../../data/processed/structural_graph.gml"

    print(f"\n💾 正在保存图谱至 {save_path} ...")
    nx.write_gml(structural_graph, save_path)
    print("🎉 保存成功！")