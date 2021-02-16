import torch
from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
# import igraph
from features import *
from utils import dump_file, load_file
from transformers import BartTokenizer
from torch_geometric.data import Data, Batch
from sklearn import model_selection
from IPython import embed
import os
from torch_geometric import utils


# def mol2graph( mol ):
#     admatrix = rdmolops.GetAdjacencyMatrix( mol )
#     bondidxs = [ ( b.GetBeginAtomIdx(),b.GetEndAtomIdx() ) for b in mol.GetBonds() ]
#     adlist = np.ndarray.tolist( admatrix )
#     graph = igraph.Graph()
#     g = graph.Adjacency( adlist ).as_undirected()
#     for idx in g.vs.indices:
#         g.vs[ idx ][ "AtomicNum" ] = mol.GetAtomWithIdx( idx ).GetAtomicNum()
#         g.vs[ idx ][ "AtomicSymbole" ] = mol.GetAtomWithIdx( idx ).GetSymbol()
#     for bd in bondidxs:
#         btype = mol.GetBondBetweenAtoms( bd[0], bd[1] ).GetBondTypeAsDouble()
#         g.es[ g.get_eid(bd[0], bd[1]) ][ "BondType" ] = btype
#         print( bd, mol.GetBondBetweenAtoms( bd[0], bd[1] ).GetBondTypeAsDouble() )
#     return g
#     s = 'CN(C)[C@H]1[C@@H]2C[C@H]3C(=C(O)c4c(O)cccc4[C@@]3(C)O)C(=O)[C@]2(O)C(=O)\C(=C(/O)NCN5CCCC5)C1=O'
#     mol = Chem.MolFromSmiles(s)
#     am = Chem.GetAdjacencyMatrix(mol)
#     am
#     list(mol.GetBondBetweenAtoms())
#     mol.GetBondBetweenAtoms(0,1).GetBondTypeAsDouble()
#     [ ( b.GetBeginAtomIdx(),b.GetEndAtomIdx() ) for b in mol.GetBonds() ]
#
#     Chem.FragmentOnBonds(mol)
#     print(am)
#

def load_mole_data(args, filename, tokenizer):
    args.cache_filename = os.path.splitext(args.train_file)[0] + ".pkl"
    if args.use_cache and os.path.exists(args.cache_filename):
        print("Loading Cached Data...")
        data = load_file(args.cache_filename)
        train, val, test = data['train'], data['val'], data['test']
        args.in_dim = train[0]["graph_data"].x.shape[1]
        print("args.in_dim", args.in_dim)
        return train, val, test

    if os.path.exists(args.cache_filename):
        os.remove(args.cache_filename)

    df = pd.read_csv(filename)
    # samples=df.to_dict(orient='records')

    smiles = []
    suppl, targets = [], []

    for s, t in zip(list(df['smiles']), list(df[args.tgt_name])):
        try:
            c = Chem.MolFromSmiles(s)
        except Exception as e:
            print(e)
            continue
        if c:
            smiles.append(s)
            suppl.append(c)
            targets.append(t)

    # res, exclude
    all_atom_properties, dex = get_atom_properties(get_atom_symbols(suppl))
    args.in_dim = all_atom_properties[0].shape[1]
    print("args.in_dim")

    suppl = [suppl[d] for d in dex]
    targets = [targets[d] for d in dex]
    smiles = [smiles[d] for d in dex]

    # suppl = [Chem.MolFromSmiles(s) for s in smiles]

    # embed()
    # smiles = open("10_rndm_zinc_drugs_clean.smi").read().splitlines()
    print("adj_matrix")
    adj = list(get_adj_matrix_coo(suppl))
    number_of_bonds = list(get_num_bonds(suppl))

    print("bond_types")
    bond_types = list(get_bonds_info(suppl))

    # targets = list(get_targets(get_atom_symbols(suppl)))

    num_classes = max(targets) + 1
    print("num_classes", num_classes)
    print("edge_index")
    edge_index = [torch.tensor(target, dtype=torch.long) for target in adj]
    print("node_attr")
    node_attr = list(seperator(all_atom_properties))
    print("edge_attr")
    edge_attr = list(seperator(bond_types))
    print("Y_data")

    Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets

    print("Y_data", Y_data)
    instances = []
    print("start adding")
    for i, (s, node_a, edge_i, edge_a, tgt) in enumerate(zip(smiles, node_attr, edge_index, edge_attr, Y_data)):
        # print("print")
        # print(tgt)
        # print(node_a, edge_i, edge_a, tgt)
        print(i)
        print(s, tgt)
        instances.append({"smiles": s,
                          "id": i,
                          "graph_data": Data(x=node_a, edge_index=edge_i, edge_attr=edge_a, y=tgt),
                          "tokenizer": tokenizer
                          })

    # data_instance = list(map(list, zip(node_attr, edge_index, edge_attr, Y_data)))
    # instances=[{"smiles":smiles[instance_id],
    #             "edge_index":edge_index[instance_id],
    #             "node_attr":node_attr[instance_id],
    #             "edge_attr":edge_attr[instance_id],
    #             "Y_data":Y_data[instance_id],
    #             "instance_id":instance_id,
    #             "tokenizer":tokenizer} for instance_id in range(len(edge_index))]
    train, tmp = model_selection.train_test_split(instances, train_size=0.6, shuffle=True)
    val, test = model_selection.train_test_split(tmp, train_size=0.5, shuffle=True)  # This splits the tmp value

    if args.cache_filename:
        dump_file({"train": train, "val": val, "test": test}, args.cache_filename)

    return train, val, test

def load_data(args, filename, tokenizer):

    args.cache_filename = os.path.splitext(args.train_file)[0] + ".pkl"
    if args.use_cache and os.path.exists(args.cache_filename):

        print("Loading Cached Data...")
        data=load_file(args.cache_filename)
        train, val, test=data['train'],data['val'],data['test']
        args.in_dim = train[0]["graph_data"].x.shape[1]
        print("args.in_dim", args.in_dim)
        return train, val, test

    if os.path.exists(args.cache_filename):
        os.remove(args.cache_filename)

    df = pd.read_csv(filename)
    # samples=df.to_dict(orient='records')

    smiles = []
    suppl, targets = [], []

    for s, t in zip(list(df['smiles']), list(df[args.tgt_name])):
        try:
            c = Chem.MolFromSmiles(s)
        except Exception as e:
            print(e)
            continue
        if c:
            smiles.append(s)
            suppl.append(c)
            targets.append(t)

    # res, exclude
    all_atom_properties, dex = get_atom_properties(get_atom_symbols(suppl))
    args.in_dim = all_atom_properties[0].shape[1]
    print("args.in_dim")

    suppl = [suppl[d] for d in dex]
    targets = [targets[d] for d in dex]
    smiles = [smiles[d] for d in dex]

    # suppl = [Chem.MolFromSmiles(s) for s in smiles]

    # embed()
    # smiles = open("10_rndm_zinc_drugs_clean.smi").read().splitlines()
    print("adj_matrix")
    adj = list(get_adj_matrix_coo(suppl))
    number_of_bonds = list(get_num_bonds(suppl))

    print("bond_types")
    bond_types = list(get_bonds_info(suppl))

    # targets = list(get_targets(get_atom_symbols(suppl)))

    num_classes = max(targets) + 1
    print("num_classes", num_classes)
    print("edge_index")
    edge_index = [torch.tensor(target, dtype=torch.long) for target in adj]
    print("node_attr")
    node_attr = list(seperator(all_atom_properties))
    print("edge_attr")
    edge_attr = list(seperator(bond_types))
    print("Y_data")

    Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets

    print("Y_data", Y_data)
    instances = []
    print("start adding")
    for i, (s, node_a, edge_i, edge_a, tgt) in enumerate(zip(smiles, node_attr, edge_index, edge_attr, Y_data)):
        # print("print")
        # print(tgt)
        # print(node_a, edge_i, edge_a, tgt)
        print(i)
        print(s, tgt)
        instances.append({"smiles": s,
                          "id": i,
                          "graph_data": Data(x=node_a, edge_index=edge_i, edge_attr=edge_a, y=tgt),
                          "tokenizer": tokenizer
                          })

    # data_instance = list(map(list, zip(node_attr, edge_index, edge_attr, Y_data)))
    # instances=[{"smiles":smiles[instance_id],
    #             "edge_index":edge_index[instance_id],
    #             "node_attr":node_attr[instance_id],
    #             "edge_attr":edge_attr[instance_id],
    #             "Y_data":Y_data[instance_id],
    #             "instance_id":instance_id,
    #             "tokenizer":tokenizer} for instance_id in range(len(edge_index))]
    train, tmp = model_selection.train_test_split(instances, train_size=0.6, shuffle=True)
    val, test = model_selection.train_test_split(tmp, train_size=0.5, shuffle=True)  # This splits the tmp value

    if args.cache_filename:
        dump_file({"train": train, "val": val, "test": test}, args.cache_filename)

    return train, val, test
    # return instances


def collate_fn(batch):
    # max_len = max([len(f["input_ids"]) for f in batch])
    # input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    # input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    # input_ids = torch.tensor(input_ids, dtype=torch.long)
    # input_mask = torch.tensor(input_mask, dtype=torch.float)
    # index = torch.arange(start=0, end=len(batch))

    smiles = batch[0]["tokenizer"]([f["smiles"] for f in batch], return_tensors='pt', padding=True, )
    # edge_indices = [f["edge_index"] for f in batch]
    # node_attrs = [f["node_attr"] for f in batch]
    # edge_attrs = [f["edge_attr"] for f in batch]
    # Ys = [f["Y_data"] for f in batch]
    ids = [f["id"] for f in batch]

    # print("f[graph_data]", batch[0]["graph_data"])
    batch_graph_data = Batch.from_data_list([f["graph_data"] for f in batch])

    # Label Smoothing
    # output = (smiles, edge_indices, node_attrs,edge_attrs, Ys, ids)
    output = (smiles, batch_graph_data, ids)

    return output

# collate
