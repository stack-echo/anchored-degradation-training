import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import os

# 超参数设置
EMBED_DIM = 128  # 提取的图谱特征维度
POWER = 3  # 传播阶数 (相当于 GNN 的感受野，3阶足以覆盖分子到骨架的全局结构)
NORM_WEIGHT = 1.0  # 归一化权重


def compute_fastrp(G, dim=128, power=3):
    print("1. 正在提取节点与邻接矩阵...")
    nodes = list(G.nodes())
    # 建立 node_id 到 矩阵行索引 的映射
    node2idx = {node: i for i, node in enumerate(nodes)}

    # 获取极其节省内存的稀疏邻接矩阵 (Sparse Adjacency Matrix)
    A = nx.adjacency_matrix(G, nodelist=nodes)
    N = A.shape[0]

    print("2. 正在计算转移概率矩阵 (Degree Normalization)...")
    # 计算度矩阵的逆 D^{-1}
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0  # 防止除以0
    D_inv = sp.diags(np.power(degrees, -NORM_WEIGHT))

    # 转移矩阵 T = D^{-1} * A
    T = D_inv.dot(A)

    print(f"3. 正在生成高斯随机投影矩阵 (维度: {N} x {dim})...")
    # 生成基础的随机投影矩阵 R
    R = np.random.normal(0, 1.0 / np.sqrt(dim), size=(N, dim))

    print(f"4. 开始进行 {power} 阶 FastRP 矩阵传播 (极速乘法)...")
    embeddings = np.zeros((N, dim))
    current_matrix = R

    # 核心魔法：矩阵乘法代替了漫长的随机游走 (Random Walk)
    for p in tqdm(range(1, power + 1), desc="FastRP 迭代"):
        current_matrix = T.dot(current_matrix)
        # 将各阶的特征叠加 (相当于综合了 1跳、2跳、3跳 的邻居信息)
        embeddings += current_matrix

    # 最终的 L2 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embeddings = embeddings / norms

    print("5. 正在打包特征字典...")
    # 只提取 "molecule" 类型的节点特征 (我们不需要单独保存官能团的特征)
    mol_embeddings = {}
    node_types = nx.get_node_attributes(G, 'type')

    for node, idx in tqdm(node2idx.items(), desc="提取分子特征"):
        if node_types.get(node) == 'molecule':
            # 注意：你在建图时节点名叫 "MOL_0" 或 "MOL_CHEMBLxxxx"
            # 这里需要清洗一下名字，只保留 CHEMBL_ID 以便对齐
            clean_id = node.replace("MOL_", "")
            mol_embeddings[clean_id] = embeddings[idx].astype(np.float32)

    return mol_embeddings


if __name__ == "__main__":
    graph_path = "../../data/processed/structural_graph.gml"
    save_path = "../../data/processed/fastrp_embeddings.npz"

    print(f"📦 正在加载图谱: {graph_path} ...")
    G = nx.read_gml(graph_path)
    print(f"✅ 图谱加载完毕，节点数: {G.number_of_nodes()}")

    # 执行 FastRP 算法
    mol_embeddings_dict = compute_fastrp(G, dim=EMBED_DIM, power=POWER)

    print(f"\n💾 正在保存提取到的 {len(mol_embeddings_dict)} 个分子特征至 {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 使用 npz 格式保存字典，极致压缩
    np.savez_compressed(save_path, **mol_embeddings_dict)

    print("🎉 FastRP 图特征提取彻底完成！准备迎接你的模型训练吧！")