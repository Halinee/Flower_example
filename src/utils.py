from operator import itemgetter

import dgl
import numpy as np
import torch as th
from joblib import Parallel, cpu_count, delayed
from ogb.utils import smiles2graph
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from scipy import sparse
from torch_geometric.utils import tree_decomposition
from torch.utils.data.dataloader import default_collate


def pmap(fn, data, n_jobs=lambda cpus: cpus // 3, verbose=1, **kwargs):
    """
    Parallel map using joblib.
    :param fn: Pickleable fn to map over data.
    :param data: Input data (mappable)
    :param n_jobs: Either int or function that takes number of current cpus as input
      and returns the number of jobs to spawn.
    :param verbose: Whether to log output
    :param kwargs: Additional args for fn
    :return: Mapped output.
    """
    if not isinstance(n_jobs, int):
        n_jobs = n_jobs(cpu_count())
    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fn)(d, **kwargs) for d in data
    )


def pluck(d: dict, *keys):
    """
    Returns values of keys in dict in order.
      pluck({"a": 1, "b: 2}, "b", "a") = [2, 1]
    """
    return itemgetter(*keys)(d)


def exists(val):
    return val is not None


def dgl_collate(batch):
    """
    Collate special handling graph key in batch dict with dgl.batch
    """
    graphs = dgl.batch([x["graph"] for x in batch])
    default = {
        key: default_collate([d[key] for d in batch])
        for key in batch[0]
        if key != "graph"
    }
    return {"graph": graphs, **default}


# --------------------
# Graph coversion
# --------------------


def dgl_to_dict(g: dgl.DGLGraph) -> dict:
    """
    dgl.DGLGraph to serializable plain python dict format, special handling of
    "pos_enc" and "smiles" features.
    """
    d = {
        "node_feat": g.ndata["feat"].to(th.int8).numpy(),
        "edge_feat": g.edata["feat"].to(th.int8).numpy(),
        "num_nodes": g.number_of_nodes(),
        "edge_index": th.stack(g.edges()).to(th.int16).numpy(),
    }
    if "pos_enc" in g.ndata:
        d["pos_enc"] = g.ndata["pos_enc"].to(th.float16).numpy()
    if hasattr(g, "smiles"):
        d["smiles"] = g.smiles
    return d


def to_dgl(d: dict) -> dgl.DGLGraph:
    """
    Graph dict to dgl.DGLGraph, knowing how to handle "pos_enc" node features and
    "smiles" global features.
    :param d: input dict graph with numpy array data
    :return: DGLGraph with torch tensor data
    """
    if "edge_attr" in d:
        ek = "edge_attr"
    else:
        ek = "edge_feat"

    assert len(d[ek]) == len(d["edge_index"][0])
    assert len(d["node_feat"]) == d["num_nodes"]

    g = dgl.graph(
        (
            th.as_tensor(d["edge_index"][0], dtype=th.long),
            th.as_tensor(d["edge_index"][1], dtype=th.long),
        ),
        num_nodes=d["num_nodes"],
        idtype=th.int32,
    )
    g.ndata["feat"] = th.as_tensor(d["node_feat"], dtype=th.long)
    g.edata["feat"] = th.as_tensor(d[ek], dtype=th.long)
    if "pos_enc" in d:
        g.ndata["pos_enc"] = th.as_tensor(d["pos_enc"], dtype=th.float)
    if "smiles" in d:
        g.smiles = d["smiles"]
    return g


def mol_from_graph(graph: dict, atom_type_index=0, bond_type_index=0) -> Chem.Mol:
    """
    Given a molgraph featurized via OGB default smiles2graph, returns Mol object.
    FIXME: not recommended to use currently as it assumes atom and bond label indexes
    It is 25% faster than Chem.MolFromSmiles(smi), so if you already have the graph
    dict and featurization is standard, use this.
        > %timeit Chem.MolFromSmiles(smi)
        50.1 µs ± 308 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        > %timeit mol_from_graph(g)
        38.1 µs ± 299 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    """
    _BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]

    mol = Chem.RWMol()
    atoms = graph["node_feat"][:, atom_type_index].tolist()
    for a in atoms:
        mol.AddAtom(Chem.Atom(a + 1))
    row, col = graph["edge_index"]
    mask = row < col
    row, col = row[mask].tolist(), col[mask].tolist()

    bond_type = graph["edge_feat"][:, bond_type_index]
    bond_type = bond_type[mask].tolist()

    for i, j, bond in zip(row, col, bond_type):
        assert bond >= 0 and bond <= 3
        mol.AddBond(i, j, _BONDS[bond])

    return mol.GetMol()


# --------------------
# Featurization
# --------------------


def prepend_zero_features(arr: np.ndarray):
    """
    Given a 2D array prepends zeros to the feature dimension.
      (N, D) => (N, D+1)
    :param arr: input feature array
    :return: feature array with additional dimension
    """
    n, _ = arr.shape
    return np.concatenate([np.zeros((n, 1), dtype=int), arr], axis=1)


def with_position_encoding(g: dgl.DGLGraph, pos_enc_dim=8) -> dgl.DGLGraph:
    """
    Graph laplacian position encoding w/ Laplacian eigenvectors.
    :param g:
    :param pos_enc_dim:
    :return:
    """
    A = g.adjacency_matrix(False, scipy_fmt="csr").astype(float)
    N = sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sparse.eye(g.number_of_nodes()) - N * A * N

    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    pe = th.zeros((EigVec.shape[0], pos_enc_dim))
    dat = th.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float()
    pe[:, : dat.shape[-1]] = dat
    g.ndata["pos_enc"] = pe
    return g


def with_junction_tree(graph: dict, mol: Chem.Mol) -> dict:
    """
    Given graph dictionary and mol, finds cluster nodes and cluster connectivity and
    adds to graph dict. Similar to HIMP processing but returns a single graph instead
    of two graphs.
    https://github.com/rusty1s/himp-gnn/blob/master/transform.py
    """
    edge_index = graph["edge_index"]
    node_feat = graph["node_feat"]
    edge_feat = graph["edge_feat"]

    # noinspection PyTupleAssignmentBalance
    c2c, a2c, n_clusters, cluster_lbls = tree_decomposition(mol, return_vocab=True)
    n_atoms = graph["num_nodes"]
    c2c += n_atoms
    a2c[1] += n_atoms
    c2a = np.stack([a2c[1], a2c[0]])

    jt_edges = np.concatenate([a2c, c2a, c2c], 1)
    edge_index = np.concatenate([edge_index, jt_edges], 1)

    # prepend cluster feature dimension
    node_feat = prepend_zero_features(node_feat)
    edge_feat = prepend_zero_features(edge_feat)

    # append new features
    cluster_feat = np.zeros((n_clusters, node_feat.shape[1]), dtype=int)
    jt_edge_feat = np.zeros((jt_edges.shape[1], edge_feat.shape[1]), dtype=int)
    cluster_feat[:, 0] = cluster_lbls + 1
    jt_edge_feat[:, 0] = [1] * a2c.shape[1] + [2] * c2a.shape[1] + [3] * c2c.shape[1]
    node_feat = np.concatenate([node_feat, cluster_feat])
    edge_feat = np.concatenate([edge_feat, jt_edge_feat])

    graph["num_nodes"] = len(node_feat)
    graph["node_feat"] = node_feat
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_feat
    return graph


def with_virtual_nodes(graph: dict, n_virtual_nodes) -> dict:
    """
    Returns graph
    :param graph:
    :param n_virtual_nodes:
    :return:
    """
    edge_index = graph["edge_index"]
    node_feat = graph["node_feat"]
    edge_feat = graph["edge_feat"]

    n_real_nodes = len(node_feat)
    real_nodes = list(range(n_real_nodes))

    virtual_src = []
    virtual_dst = []
    for i in range(n_virtual_nodes):
        vn = n_real_nodes + i
        vn_copy = [vn] * n_real_nodes
        virtual_src.extend(real_nodes + vn_copy)
        virtual_dst.extend(vn_copy + real_nodes)
    virtual_edges = np.array([virtual_src, virtual_dst])
    edge_index = np.concatenate([edge_index, virtual_edges], axis=1)

    # prepend is_virtual feature dimension
    node_feat = prepend_zero_features(node_feat)
    edge_feat = prepend_zero_features(edge_feat)

    # append virtual features
    vn_feat = np.zeros((n_virtual_nodes, node_feat.shape[-1]), dtype=int)
    ve_feat = np.zeros((virtual_edges.shape[1], edge_feat.shape[1]), dtype=int)
    vn_feat[:, 0] = 1  # 0: not virtual, 1: is virtual
    ve_feat[:, 0] = 1
    node_feat = np.concatenate([node_feat, vn_feat])
    edge_feat = np.concatenate([edge_feat, ve_feat])

    graph["num_nodes"] = len(node_feat)
    graph["node_feat"] = node_feat
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_feat
    return graph


def smiles2graph_v2(
    smiles: str,
    smiles2graph=smiles2graph,
    add_junction_tree=True,
    n_virtual_nodes=1,
    add_pos_enc=True,
) -> dict:
    """
    :param smiles: input smiles
    :param smiles2graph: featurizer that converts smiles to molecular graph with keys
      ["num_nodes", "edge_index", "node_feat", "edge_feat"]
    :param add_junction_tree: whether to add junction tree nodes and connectivity
    :param n_virtual_nodes: whether to add fully connected virtual nodes
    :param add_pos_enc: whether to add laplacian positional encodings for nodes
    :return: graph dict
    """
    mol = Chem.MolFromSmiles(smiles)
    g = smiles2graph(smiles)

    # add cluster nodes and connectivity (junction tree)
    if add_junction_tree:
        g = with_junction_tree(g, mol)

    # add virtual nodes and connectivity
    if n_virtual_nodes > 0:
        g = with_virtual_nodes(g, n_virtual_nodes)

    # add "pos_enc" key, holding positional encoding of each node
    if add_pos_enc:
        g = to_dgl(g)
        g = with_position_encoding(g)
        g = dgl_to_dict(g)
    return g


#%%
if __name__ == "__main__":
    from ogb.utils import smiles2graph

    # check mol_from_graph round trip
    smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    mol = Chem.MolFromSmiles(smi)
    g = smiles2graph(smi)
    mol2 = mol_from_graph(g)
    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(mol2)
