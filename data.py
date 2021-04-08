# import torch
# from rdkit import Chem
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import rdmolops
# import igraph
from pprint import pprint as pp
# from features import *
from utils import dump_file, load_file
# from transformers import BartTokenizer
from torch_geometric.data import Data, Batch
from sklearn import model_selection
from IPython import embed
import os
from torch_geometric import utils
import spacy
import json
# from collections import defaultdict
# import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from copy import deepcopy
import csv
from utils import *
import scipy

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

# def retrieve_atom_info(atoms):
#     res = []
#     dex = []
#
#     """new"""
#
#     # atom symbol to id mapping
#     atom_dict = {}
#     cur_atoms = [atom.GetSymbol() for atom in atoms]
#     tmp = []
#     # print("cur_atoms", cur_atoms)
#     for atom in cur_atoms:
#         if atom not in atom_dict:
#             atom_dict[atom] = len(atom_dict)
#         tmp.append([atom_dict[atom]])
#     res.append(tmp)
#     return res
#
#
# def retreive_graph_info(input_smile, args):
#     c = None
#     is_valid = False
#     if len(input_smile):
#         try:
#             c = Chem.MolFromSmiles(input_smile)
#         except Exception as e:
#             print(e)
#
#
# def get_graph_info(input_smiles, args):
#     smiles = []
#     suppl, targets = [], []
#     dex = []
#     for i, s in enumerate(list(input_smiles)):
#         if s == "[[NULL]]":
#             # print("isnull")
#             continue
#         try:
#             c = Chem.MolFromSmiles(s)
#         except Exception as e:
#             print(e)
#             continue
#         if c:
#             dex.append(i)
#             smiles.append(s)
#             suppl.append(c)
#
#     # res, exclude
#     all_atom_properties, tmp_dex = get_atom_properties(get_atom_symbols(suppl))
#
#     # updated valid dex
#     # dex is valid index in original input
#     dex = [dex[d] for d in tmp_dex]
#     assert len(tmp_dex) == len(dex), embed()
#
#     # args.in_dim = all_atom_properties[0].shape[1]
#     # print("args.in_dim")
#
#     suppl = [suppl[d] for d in tmp_dex]
#     # targets = [targets[d] for d in dex]
#     # smiles = [smiles[d] for d in dex]
#
#     # suppl = [Chem.MolFromSmiles(s) for s in smiles]
#
#     # embed()
#     # smiles = open("10_rndm_zinc_drugs_clean.smi").read().splitlines()
#     # print("adj_matrix")
#     adj = list(get_adj_matrix_coo(suppl))
#     # number_of_bonds = list(get_num_bonds(suppl))
#
#     # print("bond_types")
#     # bond_types = list(get_bonds_info(suppl))
#     bond_types = []
#
#     for k, mol in enumerate(suppl):
#         # tmp=[]
#         # for i, j in adj[k].t().numpy():
#         #     tmp.append(int(mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble()))
#
#         tmp = [int(mol.GetBondBetweenAtoms(int(i), int(j)).GetBondTypeAsDouble()) for i, j in adj[k].t().numpy()]
#         bond_types.append(tmp)
#
#     # targets = list(get_targets(get_atom_symbols(suppl)))
#
#     # num_classes = max(targets) + 1
#     # print("num_classes", num_classes)
#     print("edge_index")
#     edge_index = [torch.as_tensor(target, dtype=torch.long) for target in adj]
#     print(len(edge_index))
#     print(edge_index[0].shape)
#
#     print("node_attr")
#     node_attr = [torch.tensor(target, dtype=torch.long) for target in all_atom_properties]
#     print(len(node_attr))
#     print(node_attr[0].shape)
#
#     print("edge_attr")
#     edge_attr = [torch.tensor(target, dtype=torch.long) for target in bond_types]  # list(seperator(bond_types))
#     print(len(edge_attr))
#     print(edge_attr[0].shape)
#
#     # print("Y_data")
#     args.in_dim = args.g_dim
#     # Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets
#     in_dim = args.g_dim  # node_attr[0].shape[-1]
#
#     # dummy data
#     res = [Data(x=torch.rand(2, 1), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
#                 edge_attr=torch.tensor([1, 1], dtype=torch.long)) for _ in
#            range(len(input_smiles))]
#     res_mask = [0 for _ in range(len(input_smiles))]
#
#     # d is the actualy index in inputsmile
#     for i, d in enumerate(dex):
#         res_mask[d] = 1
#         res[d] = Data(x=node_attr[i], edge_index=edge_index[i], edge_attr=edge_attr[i])
#
#     return res_mask, res
#
#     return dex, node_attr, edge_index
#
#
# def load_mole_data2(args, filename, tokenizer):
#     args.cache_filename = os.path.splitext(args.train_file)[0] + ".pkl"
#     if args.use_cache and os.path.exists(args.cache_filename):
#         print("Loading Cached Data...")
#         data = load_file(args.cache_filename)
#         train, val, test = data['train'], data['val'], data['test']
#         # for f in [train, val, test]:
#         #     for d in f:
#         #         # d["graph_data"].x=torch.cat([d["graph_data"].x[:, :-13], d["graph_data"].x[:, 8:]])
#         #         d["graph_data"].x[:, 6] = torch.rand_like(d["graph_data"].x[:, 6])
#
#         # d["graph_data"].x = torch.rand_like(d["graph_data"].x)
#         # args.in_dim = train[0]["graph_data"].x.shape[1]
#         args.in_dim = args.g_dim
#
#         print("args.in_dim", args.in_dim)
#         return train, val, test
#
#     if os.path.exists(args.cache_filename):
#         os.remove(args.cache_filename)
#
#     df = pd.read_csv(filename)
#     # samples=df.to_dict(orient='records')
#
#     smiles = []
#     suppl, targets = [], []
#
#     for s, t in zip(list(df['smiles']), list(df[args.tgt_name])):
#         try:
#             c = Chem.MolFromSmiles(s)
#         except Exception as e:
#             print(e)
#             continue
#         if c:
#             smiles.append(s)
#             suppl.append(c)
#             targets.append(t)
#
#     # res, exclude
#     all_atom_properties, dex = get_atom_properties(get_atom_symbols(suppl))
#     args.in_dim = all_atom_properties[0].shape[1]
#
#     args.in_dim = args.g_dim
#     # all_atom_properties[0].shape[1]
#
#     suppl = [suppl[d] for d in dex]
#     targets = [targets[d] for d in dex]
#     smiles = [smiles[d] for d in dex]
#
#     adj = list(get_adj_matrix_coo(suppl))
#     number_of_bonds = list(get_num_bonds(suppl))
#     bond_types = []
#     # print("adj", adj)
#     # exit()
#     # for k, mol in enumerate(suppl):
#     #     # tmp=[]
#     #     # for i, j in adj[k].t().numpy():
#     #     #     tmp.append(int(mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble()))
#     #
#     #     tmp = [int(mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble()) for i,j in adj[k].t().numpy()]
#     #     bond_types.append(tmp)
#     #
#     # print(adj[0].shape)
#     # print(len(bond_types[0]))
#
#     # embed()
#     # targets = list(get_targets(get_atom_symbols(suppl)))
#
#     num_classes = max(targets) + 1
#     print("num_classes", num_classes)
#     print("edge_index")
#     edge_index = [torch.as_tensor(target, dtype=torch.long) for target in adj]
#     print("node_attr")
#     node_attr = [torch.tensor(target, dtype=torch.long) for target in all_atom_properties]
#     # list(seperator(all_atom_properties))
#     print("node_attr.shape", node_attr[0].shape)
#
#     print("edge_attr")
#     edge_attr = [torch.tensor(target, dtype=torch.long) for target in bond_types]
#     print("edge_attr.shape", edge_attr[0].shape)
#
#     print("Y_data")
#     Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets
#     print("Y_data", Y_data)
#
#     instances = []
#     print("start adding")
#     for i, (s, node_a, edge_i, edge_a, tgt) in enumerate(zip(smiles, node_attr, edge_index, edge_attr, Y_data)):
#         # print("print")
#         # print(tgt)
#         # print(node_a, edge_i, edge_a, tgt)
#         # print(i)
#         # print(s, tgt)
#         # print(Data(x=node_a, edge_index=edge_i, edge_attr=edge_a, y=tgt))
#         # exit()
#         instances.append({"smiles": s,
#                           "id": i,
#                           "graph_data": Data(x=node_a, edge_index=edge_i, edge_attr=edge_a, y=tgt),
#                           "tokenizer": tokenizer
#                           })
#
#     # data_instance = list(map(list, zip(node_attr, edge_index, edge_attr, Y_data)))
#     # instances=[{"smiles":smiles[instance_id],
#     #             "edge_index":edge_index[instance_id],
#     #             "node_attr":node_attr[instance_id],
#     #             "edge_attr":edge_attr[instance_id],
#     #             "Y_data":Y_data[instance_id],
#     #             "instance_id":instance_id,
#     #             "tokenizer":tokenizer} for instance_id in range(len(edge_index))]
#     train, tmp = model_selection.train_test_split(instances, train_size=0.4, shuffle=False)
#     val, test = model_selection.train_test_split(tmp, train_size=0.5, shuffle=False)  # This splits the tmp value
#
#     if args.cache_filename:
#         dump_file({"train": train, "val": val, "test": test}, args.cache_filename)
#
#     return train, val, test
#
#
# def load_mole_data(args, filename, tokenizer):
#     args.cache_filename = os.path.splitext(args.train_file)[0] + ".pkl"
#     if args.use_cache and os.path.exists(args.cache_filename):
#         print("Loading Cached Data...")
#         data = load_file(args.cache_filename)
#         train, val, test = data['train'], data['val'], data['test']
#         # for f in [train, val, test]:
#         #     for d in f:
#         #         # d["graph_data"].x=torch.cat([d["graph_data"].x[:, :-13], d["graph_data"].x[:, 8:]])
#         #         d["graph_data"].x[:, 6] = torch.rand_like(d["graph_data"].x[:, 6])
#
#         # d["graph_data"].x = torch.rand_like(d["graph_data"].x)
#         # args.in_dim = train[0]["graph_data"].x.shape[1]
#         args.in_dim = args.g_dim
#
#         print("args.in_dim", args.in_dim)
#         return train, val, test
#
#     if os.path.exists(args.cache_filename):
#         os.remove(args.cache_filename)
#
#     df = pd.read_csv(filename)
#     # samples=df.to_dict(orient='records')
#
#     smiles = []
#     suppl, targets = [], []
#
#     for s, t in zip(list(df['smiles']), list(df[args.tgt_name])):
#         try:
#             c = Chem.MolFromSmiles(s)
#         except Exception as e:
#             print(e)
#             continue
#         if c:
#             smiles.append(s)
#             suppl.append(c)
#             targets.append(t)
#
#     # res, exclude
#     all_atom_properties, dex = get_atom_properties(get_atom_symbols(suppl))
#     args.in_dim = args.g_dim
#     # all_atom_properties[0].shape[1]
#
#     suppl = [suppl[d] for d in dex]
#     targets = [targets[d] for d in dex]
#     smiles = [smiles[d] for d in dex]
#
#     adj = list(get_adj_matrix_coo(suppl))
#     number_of_bonds = list(get_num_bonds(suppl))
#     bond_types = []
#     # print("adj", adj)
#     # exit()
#     for k, mol in enumerate(suppl):
#         # tmp=[]
#         # for i, j in adj[k].t().numpy():
#         #     tmp.append(int(mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble()))
#
#         tmp = [int(mol.GetBondBetweenAtoms(int(i), int(j)).GetBondTypeAsDouble()) for i, j in adj[k].t().numpy()]
#         bond_types.append(tmp)
#
#     print(adj[0].shape)
#     print(len(bond_types[0]))
#
#     # embed()
#     # targets = list(get_targets(get_atom_symbols(suppl)))
#
#     num_classes = max(targets) + 1
#     print("num_classes", num_classes)
#     print("edge_index")
#     edge_index = [torch.as_tensor(target, dtype=torch.long) for target in adj]
#     print("node_attr")
#     node_attr = [torch.tensor(target, dtype=torch.long) for target in all_atom_properties]
#     # list(seperator(all_atom_properties))
#     print("node_attr.shape", node_attr[0].shape)
#
#     print("edge_attr")
#     edge_attr = [torch.tensor(target, dtype=torch.long) for target in bond_types]
#     print("edge_attr.shape", edge_attr[0].shape)
#
#     print("Y_data")
#     Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets
#     print("Y_data", Y_data)
#
#     instances = []
#     print("start adding")
#     for i, (s, node_a, edge_i, edge_a, tgt) in enumerate(zip(smiles, node_attr, edge_index, edge_attr, Y_data)):
#         # print("print")
#         # print(tgt)
#         # print(node_a, edge_i, edge_a, tgt)
#         # print(i)
#         # print(s, tgt)
#         # print(Data(x=node_a, edge_index=edge_i, edge_attr=edge_a, y=tgt))
#         # exit()
#         instances.append({"smiles": s,
#                           "id": i,
#                           "graph_data": Data(x=node_a, edge_index=edge_i, edge_attr=edge_a, y=tgt),
#                           "tokenizer": tokenizer
#                           })
#
#     # data_instance = list(map(list, zip(node_attr, edge_index, edge_attr, Y_data)))
#     # instances=[{"smiles":smiles[instance_id],
#     #             "edge_index":edge_index[instance_id],
#     #             "node_attr":node_attr[instance_id],
#     #             "edge_attr":edge_attr[instance_id],
#     #             "Y_data":Y_data[instance_id],
#     #             "instance_id":instance_id,
#     #             "tokenizer":tokenizer} for instance_id in range(len(edge_index))]
#     train, tmp = model_selection.train_test_split(instances, train_size=0.6, shuffle=False)
#     val, test = model_selection.train_test_split(tmp, train_size=0.5, shuffle=False)  # This splits the tmp value
#
#     if args.cache_filename:
#         dump_file({"train": train, "val": val, "test": test}, args.cache_filename)
#
#     return train, val, test


class ModalRetriever:
    def __init__(self):
        self.atom_dict = {}
        self.mention2cid, self.cmpd_info = load_file("data_online/ChemProt_Corpus/mention2ent.json"), \
                                           load_file("data_online/ChemProt_Corpus/cmpd_info.json")
        self.mention2protid, self.prot_info = load_file("data_online/ChemProt_Corpus/mention2protid.json"), \
                                              load_file("data_online/ChemProt_Corpus/prot_info.json")

    def get_prot(self, ent):
        # print("get_prot", ent)

        d_modal, d_modal_mask = ".", 0
        if ent in self.mention2protid:
            pid = self.mention2protid[ent]
            if pid is not None and pid in self.prot_info and "definition" in self.prot_info[pid]:
                d_modal, d_modal_mask = self.prot_info[pid]["definition"]['text'], 1

        return d_modal, d_modal_mask

    def get_mol(self, ent):
        # print("get_mol", ent)

        g_modal, g_modal_mask = "", 0
        d_modal, d_modal_mask = ".", 0
        if ent in self.mention2cid:
            cid = self.mention2cid[ent]
            # print("cid",cid)
            if cid is not None and str(cid) in self.cmpd_info:
                cid = str(cid)
                if "canonical_smiles" in self.cmpd_info[cid]:
                    g_modal, g_modal_mask = self.cmpd_info[cid]["canonical_smiles"], 1
                if "pubchem_description" in self.cmpd_info[cid] and 'descriptions' in self.cmpd_info[cid][
                    'pubchem_description'] and \
                        len(self.cmpd_info[cid]['pubchem_description']['descriptions']):
                    d_modal, d_modal_mask = self.cmpd_info[cid]['pubchem_description']['descriptions'][0][
                                                "description"], 1

        g_modal, g_modal_mask = self.get_graph(g_modal)
        return g_modal, g_modal_mask, d_modal, d_modal_mask

    def get_atom_info(self, mol):
        atoms = list(mol.GetAtoms())
        # atom symbol to id mapping
        cur_atoms = [atom.GetSymbol() for atom in atoms]
        tmp = []
        # print("len(self.atom_dict)", len(self.atom_dict))
        for atom in cur_atoms:
            if atom not in self.atom_dict:
                self.atom_dict[atom] = len(self.atom_dict)
            tmp.append([self.atom_dict[atom]])
        return tmp

    def get_edges(self, mol):
        adj = Chem.GetAdjacencyMatrix(mol)
        # print("am\n", am)
        for i in range(np.array(adj).shape[0]):
            for j in range(np.array(adj).shape[0]):
                if adj[i, j] >= 1:
                    adj[j, i] = adj[i, j]
        adj[adj > 1] = 1

        adj = scipy.sparse.csr_matrix(adj)
        return utils.from_scipy_sparse_matrix(adj)[0]

    def get_graph(self, input_smile):
        res = (Data(x=torch.rand(2, 1), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    edge_attr=torch.tensor([1, 1])), 0)

        if not len(input_smile): return res
        try:
            mol = Chem.MolFromSmiles(input_smile)
        except Exception as e:
            print(e)
            return res

        atom_properties = self.get_atom_info(mol)
        edge_index = self.get_edges(mol)
        bond_type = [int(mol.GetBondBetweenAtoms(int(i), int(j)).GetBondTypeAsDouble()) for i, j in
                     edge_index.t().numpy()]

        node_attr = torch.tensor(atom_properties, dtype=torch.long)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(bond_type, dtype=torch.long)

        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr), 1


def load_data_chemprot_re(args, filename, tokenizer=None):
    args.cache_filename = os.path.splitext(filename)[0] + ".pkl"

    if args.use_cache and os.path.exists(args.cache_filename):
        print("Loading Cached Data...", args.cache_filename)
        data = load_file(args.cache_filename)

        # print(sum([instance["ent1_g_mask"] for instance in data['instances']]) // len(data['instances']))
        # print(sum([instance["ent2_g_mask"] for instance in data['instances']]) // len(data['instances']))
        # print(sum([instance["ent1_d_mask"] for instance in data['instances']]) // len(data['instances']))
        # print(sum([instance["ent2_d_mask"] for instance in data['instances']]) // len(data['instances']))
        args.out_dim = len(data['rel2id'])
        # args.in_dim = data['instances'][0]["ent1_g"].x.shape[-1]
        args.in_dim = args.g_dim
        print("args.in_dim", args.in_dim)
        print("args.out_dim", args.out_dim)
        print(data['rel2id'])
        #
        # print(data['instances'][0]["modal_data"][0][0].x.dtype)
        # import torch

        return data['instances']

    instances = []
    # smiless1 = []
    # smiless2 = []
    # descriptions1 = []
    # descriptions2 = []

    mention2cid, cmpd_info = load_file("data_online/ChemProt_Corpus/mention2ent.json"), \
                             load_file("data_online/ChemProt_Corpus/cmpd_info.json")
    mention2protid, prot_info = load_file("data_online/ChemProt_Corpus/mention2protid.json"), \
                                load_file("data_online/ChemProt_Corpus/prot_info.json")

    rels = ['AGONIST-ACTIVATOR',
            'DOWNREGULATOR', 'SUBSTRATE_PRODUCT-OF',
            'AGONIST', 'INHIBITOR',
            'PRODUCT-OF', 'ANTAGONIST',
            'ACTIVATOR', 'INDIRECT-UPREGULATOR',
            'SUBSTRATE', 'INDIRECT-DOWNREGULATOR',
            'AGONIST-INHIBITOR', 'UPREGULATOR', ]
    rel2id = {rel: i for i, rel in enumerate(rels)}

    def fill_modal_data(ent, modal_feats, modal_feat_mask, is_prot=False):

        if not is_prot and ent in mention2cid:
            cid = mention2cid[ent]
            if cid is not None and str(cid) in cmpd_info:
                cid = str(cid)
                if "canonical_smiles" in cmpd_info[cid]:
                    # print("found canonical_smiles")
                    modal_feats[0].append(cmpd_info[cid]["canonical_smiles"])
                    modal_feat_mask[0].append(1)
                else:
                    modal_feats[0].append("[[NULL]]")  # @
                    modal_feat_mask[0].append(0)

                if "pubchem_description" in cmpd_info[cid] and 'descriptions' in cmpd_info[cid][
                    'pubchem_description'] and \
                        len(cmpd_info[cid]['pubchem_description']['descriptions']):
                    # print("dfound pubchem_description")
                    modal_feats[1].append(cmpd_info[cid]['pubchem_description']['descriptions'][0]["description"])
                    modal_feat_mask[1].append(1)
                else:
                    modal_feats[1].append("")  # [[NULL]]
                    modal_feat_mask[1].append(0)
                return
        else:
            if ent in mention2protid:
                pid = mention2protid[ent]
                if pid is not None and pid in prot_info:
                    modal_feats[0].append("")
                    modal_feat_mask[0].append(0)
                    if "definition" in prot_info[pid]:
                        # print("found1")
                        modal_feats[1].append(prot_info[pid]["definition"]['text'])
                        modal_feat_mask[1].append(1)
                    else:
                        modal_feats[1].append("")
                        modal_feat_mask[1].append(0)
                    return
        modal_feats[0].append("[[NULL]]")
        modal_feat_mask[0].append(0)
        modal_feats[1].append("")
        modal_feat_mask[1].append(0)

    texts, labels = [], []
    ent_pos = []

    with open(filename, mode="r", encoding="utf-8") as fin:
        modal_feats1 = [[], []]  # n row, each row is all for one modalities
        modal_feat_mask1 = [[], []]
        modal_feats2 = [[], []]  # n row, each row is all for one modalities
        modal_feat_mask2 = [[], []]
        # modal_feats={"smiles":[], "smiles":[], "smiles":[], "smiles":[]}
        for i, line in enumerate(fin):
            # if args.debug:
            #     # if i<100: continue
            #     if i>1000: break

            # print("\n", i, line)
            sample = json.loads(line.strip())

            # text label metadata
            assert not sample["metadata"]

            text = sample["text"]

            # exclusive
            ent1_spos, ent1_epos = text.find("<< ") + 3, text.find(" >>")
            ent2_spos, ent2_epos = text.find("[[ ") + 3, text.find(" ]]")

            ent1, ent2 = text[ent1_spos:ent1_epos], text[ent2_spos:ent2_epos]
            # lower case

            prior_tokens, mid_tokens, post_tokens = tokenizer.tokenize(text[:ent1_spos]), \
                                                    tokenizer.tokenize(text[ent1_epos + 3:ent2_spos]), \
                                                    tokenizer.tokenize(text[ent2_epos + 3:])
            ent1_tokens, ent2_tokens = ["*"] + tokenizer.tokenize(ent1) + \
                                       ["*"], ["*"] + tokenizer.tokenize(ent2) + ["*"]

            tokens = prior_tokens + ent1_tokens + mid_tokens + ent2_tokens + post_tokens
            s_pos = len(prior_tokens) + 1
            new_ent1_pos = (s_pos, s_pos + len(ent1_tokens))
            s_pos += len(ent1_tokens) + len(mid_tokens)
            new_ent2_pos = (s_pos, s_pos + len(ent2_tokens))

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            texts.append(input_ids)
            # texts.append(text.lower())
            labels.append(rel2id[sample["label"]])

            # print("ent1, ent2", ent1,"|",  ent2)
            # print("looking for ent1")

            if ent2 in mention2cid:
                ent1, ent2 = ent2, ent1
                new_ent1_pos, new_ent2_pos = new_ent2_pos, new_ent1_pos
            ent_pos.append([new_ent1_pos, new_ent2_pos])

            fill_modal_data(ent2, modal_feats2, modal_feat_mask2, is_prot=True)
            fill_modal_data(ent1, modal_feats1, modal_feat_mask1, is_prot=False)

            #
            # fill_modal_data(ent1, mention2cid, cmpd_info, modal_feats1, modal_feat_mask1)
            # # print("looking for ent2")
            # fill_modal_data(ent2, mention2cid, cmpd_info, modal_feats2, modal_feat_mask2)
            # print(modal_feats1)
            # print(modal_feats2)
            # print(len(modal_feats1[0]))
            # print(len(modal_feats1[1]))
            # print(len(modal_feat_mask1[0]))
            # print(len(modal_feat_mask1[1]))
            # print(len(modal_feats2[0]))
            # print(len(modal_feats2[1]))
            # print(len(modal_feat_mask2[0]))
            # print(len(modal_feat_mask2[1]))

    modal_feat_mask1[0], modal_feats1[0] = get_graph_info(modal_feats1[0], args)
    # modal_feat_mask2[0], modal_feats2[0] = get_graph_info(modal_feats2[0],args)
    # print(len(modal_feats1[0]))
    # print(len(modal_feats1[1]))
    # print(len(modal_feat_mask1[0]))
    # print(len(modal_feat_mask1[1]))
    # print(len(modal_feats2[0]))
    # print(len(modal_feats2[1]))
    # print(len(modal_feat_mask2[0]))
    # print(len(modal_feat_mask2[1]))

    # exit()

    # print(len(modal_feats1[0]))
    # print(len(modal_feat_mask1[0]))
    # print(len(modal_feats1[1]))
    # print(len(modal_feat_mask1[1]))
    # print(len(modal_feats2[0]))
    # print(len(modal_feat_mask2[0]))
    # print(len(modal_feats2[1]))
    # print(len(modal_feat_mask2[1]))
    #
    # print(modal_feats1[0][:5])
    # print(modal_feats1[1][:5])
    # print(modal_feats2[0][:5])
    # print(modal_feats2[1][:5])
    # print(modal_feat_mask1[0][:5])
    # print(modal_feat_mask1[1][:5])
    # print(modal_feat_mask2[0][:5])
    # print(modal_feat_mask2[1][:5])
    #
    # exit()
    assert len(modal_feat_mask1[0]) == len(modal_feats1[0]) == len(modal_feats2[0]) == len(modal_feats2[0]) == len(
        modal_feats2[1])

    valid_num = np.zeros((4))
    total_num = 0

    for i in range(len(texts)):
        total_num += 1
        valid_num += [modal_feat_mask1[0][i] == 1, modal_feat_mask1[1][i] == 1, modal_feat_mask2[1][i] == 1,
                      modal_feat_mask1[1][i] == 1 and modal_feat_mask2[1][i] == 1]

        # also add in original text for analyze purpose
        instances.append({"text": texts[i],
                          "ent_pos": ent_pos[i],
                          "id": i,
                          "label": labels[i],
                          "modal_data": [
                              [modal_feats1[0][i], modal_feats1[1][i], modal_feat_mask1[0][i],
                               modal_feat_mask1[1][i], ],
                              [modal_feats2[1][i], modal_feat_mask2[1][i]]
                          ],
                          "tokenizer": tokenizer
                          })
    print("instances", instances[0])

    # for i, (
    #         text, label, ent1_g, ent1_g_mask, ent1_d, ent1_d_mask, ent2_g, ent2_g_mask, ent2_d,
    #         ent2_d_mask) in enumerate(
    #     zip(texts, labels, modal_feats1[0], modal_feat_mask1[0], modal_feats1[1], modal_feat_mask1[1],
    #         modal_feats2[0], modal_feat_mask2[0], modal_feats2[1], modal_feat_mask2[1])):
    #
    #     total_num += 1
    #     if ent1_g_mask == 1: valid_num1g += 1
    #     if ent1_d_mask == 1: valid_num1d += 1
    #     if ent2_d_mask == 1: valid_num2d += 1
    #     # print("ent1_g_mask", ent1_g_mask)
    #     # print("ent1_d_mask", ent1_d_mask)
    #     # print("ent2_g_mask", ent2_g_mask)
    #     # print("ent2_g_mask", ent2_g_mask)
    #     instances.append({"text": text,
    #                       "id": i,
    #                       "label": label,
    #                       "ent1_g": ent1_g,
    #                       "ent1_g_mask": ent1_g_mask,
    #                       "ent1_d": ent1_d,
    #                       "ent1_d_mask": ent1_d_mask,
    #                       "ent2_g": ent2_g,
    #                       "ent2_g_mask": ent2_g_mask,
    #                       "ent2_d": ent2_d,
    #                       "ent2_d_mask": ent2_d_mask,
    #                       "tokenizer": tokenizer
    #                       })
    #
    # print("valid_num", valid_num)
    # print("total_num", total_num)
    # # print(instances)
    # # exit()
    print(total_num)
    print(valid_num)
    embed()
    args.out_dim = len(rel2id)
    # args.in_dim = instances[0]["ent1_g"].x.shape[-1]

    if args.cache_filename:
        dump_file({"instances": instances, "rel2id": rel2id}, args.cache_filename)

    # print(sum([instance["ent1_g_mask"] for instance in instances])//len(instances))
    # print(sum([instance["ent2_g_mask"] for instance in instances])//len(instances))
    # print(sum([instance["ent1_d_mask"] for instance in instances])//len(instances))
    # print(sum([instance["ent2_d_mask"] for instance in instances])//len(instances))
    return instances

    # for each sent

    if os.path.exists(args.cache_filename):
        os.remove(args.cache_filename)


def txt_and_entity_to_token(text, ent_pos_list, tokenizer):
    pass


def sent_with_entity_to_token_ids(sent, ent_pos_list):
    ent1, ent2 = ent_pos_list
    ent1_spos, ent1_epos = ent1
    ent2_spos, ent2_epos = ent2


class ChemProtDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, args, filename, tokenizer=None, modal_retriever=None):
        args.cache_filename = os.path.splitext(filename)[0] + ".pkl"
        if args.use_cache and os.path.exists(args.cache_filename):
            print("Loading Cached Data...", args.cache_filename)
            data = load_file(args.cache_filename)
            args.out_dim = len(data['rel2id'])
            print("args.in_dim", args.in_dim)
            print("args.out_dim", args.out_dim)
            print(data['rel2id'])
            self.instances = data['instances']
            # embed()
            return

        self.mention2cid, self.cmpd_info = load_file("data_online/ChemProt_Corpus/mention2ent.json"), \
                                           load_file("data_online/ChemProt_Corpus/cmpd_info.json")
        self.mention2protid, self.prot_info = load_file("data_online/ChemProt_Corpus/mention2protid.json"), \
                                              load_file("data_online/ChemProt_Corpus/prot_info.json")
        self.mention2concepts = load_file("data_online/ChemProt_Corpus/mention2concepts.json")

        # self.chem_mentions = load_file("data_online/ChemProt_Corpus/chem_mentions.json")
        self.get_mentions()

        self.modal_retriever = modal_retriever

        self.rels = ['AGONIST-ACTIVATOR',
                     'DOWNREGULATOR',
                     'SUBSTRATE_PRODUCT-OF',
                     'AGONIST',
                     'INHIBITOR',
                     'PRODUCT-OF',
                     'ANTAGONIST',
                     'ACTIVATOR',
                     'INDIRECT-UPREGULATOR',
                     'SUBSTRATE',
                     'INDIRECT-DOWNREGULATOR',
                     'AGONIST-INHIBITOR',
                     'UPREGULATOR', ]

        self.label_desc = {'AGONIST-ACTIVATOR': "Agonists bind to a receptor and increase its biological response.",
                           'DOWNREGULATOR': "Chemical down-regulates Gene/Protein",
                           'SUBSTRATE_PRODUCT-OF': "Chemicals that are both, substrate and products of enzymatic reactions.",
                           'AGONIST': "A Chemical binds to a receptor and alters the receptor state resulting in a biological response.",
                           'INHIBITOR': "Chemical binds to a Gene/Protein (typically a protein) and decreases its activity",
                           'PRODUCT-OF': "Chemical as the product of an enzymatic reaction or a transporter",
                           'ANTAGONIST': "Chemical reduces the action of another Chemical, generally an agonist.",
                           'ACTIVATOR': "Chemical binds to a Gene/Protein (typically a protein) and decreases its activity",
                           'INDIRECT-UPREGULATOR': "Chemicals that induce/stimulate/enhance the frequency, "
                                                   "rate or extent of gene expression or transcription, protein expression, "
                                                   "protein release/uptake or protein functions.",
                           'SUBSTRATE': "Chemical upon which a Gene/Protein (typically protein) acts.",
                           'INDIRECT-DOWNREGULATOR': "Chemicals that decrease gene expression or transcription, "
                                                     "protein expression, protein release /uptake or indirectly, "
                                                     "protein functions.",
                           'AGONIST-INHIBITOR': "Agonists bind to a receptor and decrease its biological response.",
                           'UPREGULATOR': "Chemical up-regulates Gene/Protein"}

        self.rel2id = {rel: i for i, rel in enumerate(self.rels)}
        print("self.rel2id", self.rel2id)

        args.out_dim = len(self.rel2id)

        with open(filename, mode="r", encoding="utf-8") as fin:
            self.original_data = [line.strip() for i, line in enumerate(fin)]

        self.tokenizer = tokenizer

        """loading"""
        self.instances = []
        label_text = [self.label_desc[lb] for lb in self.rels]
        # print('label_text', label_text)

        """get sentence segments and mention positions"""
        orig_data = {}
        for idx in range(len(self.original_data)):
            sample = json.loads(self.original_data[idx])
            tmp = text = sample["text"]
            for bracket in ["<< ", " >>", "[[ ", " ]]"]: tmp = tmp.replace(bracket, "")
            if tmp not in orig_data:
                orig_data[tmp] = set()
            orig_data[tmp].add((text.find("<< "), text.find(" >>") - 3))
            orig_data[tmp].add((text.find("[[ ") - 6, text.find(" ]]") - 9))

        for i, sent in enumerate(orig_data):
            tmp = {"range2segpos": {}, "segs": []}
            sorted_ranges = sorted(list(orig_data[sent]))

            prev_e = 0

            # context, mention1, context, mention2,...
            for j, rg in enumerate(sorted_ranges):
                s, e = rg

                tmp["segs"].append(sent[prev_e:s])
                tmp["segs"].append(sent[s:e])
                tmp["range2segpos"][rg] = j * 2 + 1
                prev_e = e
            tmp["segs"].append(sent[prev_e:])
            orig_data[sent] = tmp
            # print("orig_data",orig_data[sent])
        for idx in range(len(self.original_data)):
            # if torch.is_tensor(idx):
            #     idx = idx.tolist()

            sample = json.loads(self.original_data[idx])
            label = self.rel2id[sample["label"]]
            # print(label)

            assert not sample["metadata"]

            """convert to tokens"""
            text = sample["text"]
            tokenizer = self.tokenizer
            # print(text)

            # map all mentions with "*"
            tmp = text
            for bracket in ["<< ", " >>", "[[ ", " ]]"]: tmp = tmp.replace(bracket, "")
            segs = deepcopy(orig_data[tmp]["segs"])
            range2segpos = orig_data[tmp]["range2segpos"]
            # print(segs)

            # print(orig_data[tmp])
            # print((text.find("<< "), text.find(" >>") - 3))
            # print((text.find("[[ ") - 6, text.find(" ]]") - 9))
            # embed()

            # index in segs
            ind1 = range2segpos[(text.find("<< "), text.find(" >>") - 3)]
            ind2 = range2segpos[(text.find("[[ ") - 6, text.find(" ]]") - 9)]
            # tmp_segs[ind1] = "<< " + tmp_segs[ind1][1:-1] + " >>"
            # tmp_segs[ind2] = "[[ " + tmp_segs[ind2][1:-1] + " ]]"
            # text = "".join(tmp_segs)
            # # print(tmp_segs)

            # exclusive
            ent1_spos, ent1_epos = text.find("<< ") + 3, text.find(" >>")
            ent2_spos, ent2_epos = text.find("[[ ") + 3, text.find(" ]]")
            assert ent1_spos < ent2_spos, "ent1 after ent2"
            orig_ent1, orig_ent2 = text[ent1_spos:ent1_epos], text[ent2_spos:ent2_epos]
            # ent1, ent2 = orig_ent1, orig_ent2
            # print(orig_ent1, orig_ent2)
            # print(segs[ind1], segs[ind2])
            if args.add_concept:
                ent1_c, ent2_c = list(self.mention2concepts[orig_ent1].keys())[:5], \
                                 list(self.mention2concepts[orig_ent2].keys())[:5]
                ent1_c = " (e.g., " + ", ".join(ent1_c) + ")" if ent1_c else ""
                ent2_c = " (e.g., " + ", ".join(ent2_c) + ")" if ent2_c else ""
                segs[ind1] = segs[ind1] + ent1_c
                segs[ind2] = segs[ind2] + ent2_c
                # print(segs)

                # range1, range2, range3 = text[:ent1_epos], text[ent1_epos:ent2_epos], text[ent2_epos:]
                # ent1_c, ent2_c = list(self.mention2concepts[orig_ent1].keys())[:5], \
                #                  list(self.mention2concepts[orig_ent2].keys())[:5]
                # ent1_c = " (e.g., " + ", ".join(ent1_c) + ")" if ent1_c else ""
                # ent2_c = " (e.g., " + ", ".join(ent2_c) + ")" if ent2_c else ""
                # text = range1 + ent1_c + range2 + ent2_c + range3
                # print(text)
                #
                # ent1_spos, ent1_epos = text.find("<< ") + 3, text.find(" >>")
                # ent2_spos, ent2_epos = text.find("[[ ") + 3, text.find(" ]]")
                # ent1, ent2 = text[ent1_spos:ent1_epos], text[ent2_spos:ent2_epos]
            # lower case

            tokens = []
            cur_len = 0
            new_ent1_pos, new_ent2_pos = None, None
            for j, seg in enumerate(segs):
                seg_tks = tokenizer.tokenize(seg)

                # mention
                if j in range2segpos.values():
                    seg_tks = ["*"] + seg_tks + ["*"]
                if j == ind1:
                    new_ent1_pos = (1 + cur_len, 1 + cur_len + len(seg_tks))
                elif j == ind2:
                    new_ent2_pos = (1 + cur_len, 1 + cur_len + len(seg_tks))
                tokens += seg_tks
                cur_len += len(seg_tks)
            # print("tokens",tokens)
            # print("new_ent1_pos",new_ent1_pos)
            # print("new_ent2_pos",new_ent2_pos)
            # prior_tokens, mid_tokens, post_tokens = tokenizer.tokenize(text[:(ent1_spos - 3)]), \
            #                                         tokenizer.tokenize(text[(ent1_epos + 3):(ent2_spos - 3)]), \
            #                                         tokenizer.tokenize(text[(ent2_epos + 3):])
            # ent1_tokens, ent2_tokens = ["*"] + tokenizer.tokenize(ent1) + ["*"], \
            #                            ["*"] + tokenizer.tokenize(ent2) + ["*"]
            #
            # # print("ent1_tokens", ent1_tokens)
            # # print("ent2_tokens", ent2_tokens)
            # # print("prior_tokens", prior_tokens)
            # # print("mid_tokens", mid_tokens)
            # # print("post_tokens", post_tokens)
            # tokens = prior_tokens + ent1_tokens + mid_tokens + ent2_tokens + post_tokens
            #
            # # +1 for CLS
            # s_pos = len(prior_tokens) + 1
            # new_ent1_pos = (s_pos, s_pos + len(ent1_tokens))
            # s_pos += len(ent1_tokens) + len(mid_tokens)
            # new_ent2_pos = (s_pos, s_pos + len(ent2_tokens))

            # print(tokens)
            # print(new_ent1_pos)
            # print(new_ent2_pos)
            # embed()
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if orig_ent2 in self.chem_mentions:  # or ent1 in self.mention2protid
                # print("swapped")
                orig_ent1, orig_ent2 = orig_ent2, orig_ent1
                new_ent1_pos, new_ent2_pos = new_ent2_pos, new_ent1_pos
            # print(ent1)
            # print(ent2)
            # ent_pos = [new_ent1_pos, new_ent2_pos]

            ent_template = {
                "g": None,
                "g_mask": None,
                "t": None,
                "t_mask": None,
                "pos": None,
            }

            ent1_dict = deepcopy(ent_template)
            ent2_dict = deepcopy(ent_template)
            # print("empty1", ent1_dict)
            # print("empty2", ent2_dict)

            ent1_dict["pos"] = new_ent1_pos
            ent2_dict["pos"] = new_ent2_pos

            ent1_dict['g'], ent1_dict['g_mask'], ent1_dict['t'], ent1_dict['t_mask'] = self.modal_retriever.get_mol(
                orig_ent1)
            ent2_dict['t'], ent2_dict['t_mask'] = self.modal_retriever.get_prot(orig_ent2)
            # print(ent1_dict)

            # print(ent2_dict)
            # ent1_dict['t']=tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids( ent1_dict['t']))
            # ent2_dict['t'] = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(ent2_dict['t']))

            self.instances.append({"text": token_ids,
                                   "id": idx,
                                   "label": label,
                                   "ent": [ent1_dict, ent2_dict],
                                   "label_text": label_text,
                                   "tokenizer": tokenizer
                                   })

        if args.cache_filename:
            dump_file({"instances": self.instances, "rel2id": self.rel2id}, args.cache_filename)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def get_mentions(self):
        data_dir = 'data_online/ChemProt_Corpus/'
        tr = join(data_dir, "chemprot_training/chemprot_training_entities.tsv")
        dev = join(data_dir, "chemprot_development/chemprot_development_entities.tsv")
        test = join(data_dir, "chemprot_test_gs/chemprot_test_entities_gs.tsv")

        chem_mentions = {}
        prot_mentions = {}

        if os.path.exists(join(data_dir, "chem_mentions.json")):
            chem_mentions = load_file(join(data_dir, "chem_mentions.json"))
            prot_mentions = load_file(join(data_dir, "prot_mentions.json"))
        else:
            for file in [tr, dev, test]:
                with open(file, encoding='utf-8') as fd:
                    rd = list(csv.reader(fd, delimiter="\t", quotechar='"'))

                    for i, row in enumerate(rd):

                        if not row: continue
                        if row[2] == "CHEMICAL":
                            if row[-1] not in chem_mentions:
                                chem_mentions[row[-1]] = 0
                            chem_mentions[row[-1]] += 1
                        else:
                            if row[-1] not in prot_mentions:
                                prot_mentions[row[-1]] = 0
                            prot_mentions[row[-1]] += 1
            dump_file(chem_mentions, join(data_dir, "chem_mentions.json"))
            dump_file(prot_mentions, join(data_dir, "prot_mentions.json"))
        self.chem_mentions = chem_mentions
        self.prot_mentions = prot_mentions


def collate_fn(batch):
    # max_len = max([len(f["input_ids"]) for f in batch])
    # input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    # input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    # input_ids = torch.tensor(input_ids, dtype=torch.long)
    # input_mask = torch.tensor(input_mask, dtype=torch.float)
    # index = torch.arange(start=0, end=len(batch))

    smiles = batch[0]["tokenizer"]([f["smiles"] for f in batch], return_tensors='pt', padding=True, )
    # print("smiles", smiles)
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


class CustomBatch:
    def __init__(self, batch):
        max_len = max([len(f["text"]) for f in batch])
        input_ids = [f["text"] + [0] * (max_len - len(f["text"])) for f in batch]
        input_mask = [[1.0] * len(f["text"]) + [0.0] * (max_len - len(f["text"])) for f in batch]

        self.texts = torch.tensor(input_ids, dtype=torch.long)[:, :512]
        self.texts_attn_mask = torch.tensor(input_mask, dtype=torch.float)[:, :512]

        # print("self.texts ",self.texts )
        # print("self.texts_attn_mask ",self.texts_attn_mask )

        # self.texts = batch[0]["tokenizer"]([f["text"] for f in batch], return_tensors='pt', padding=True)
        self.ids = [f["id"] for f in batch]
        self.labels = torch.tensor([f["label"] for f in batch], dtype=torch.long)

        tokenizer = batch[0]["tokenizer"]
        self.label_text = tokenizer(batch[0]["label_text"], return_tensors='pt', padding=True)

        g_data = Batch.from_data_list([f["ent"][0]['g'] for f in batch])
        g_data.x = torch.as_tensor(g_data.x, dtype=torch.long)
        self.ent1_g = g_data
        self.ent1_g_mask = torch.tensor([f["ent"][0]['g_mask'] for f in batch]).unsqueeze(-1)

        self.ent1_d = tokenizer([f["ent"][0]['t'] for f in batch], return_tensors='pt', padding=True)
        self.ent1_d_mask = torch.tensor([f["ent"][0]['t_mask'] for f in batch]).unsqueeze(-1)
        # print("self.ent1_d_mask ",self.ent1_d_mask )

        self.ent2_d = tokenizer([f["ent"][1]['t'] for f in batch], return_tensors='pt', padding=True)
        self.ent2_d_mask = torch.tensor([f["ent"][1]['t_mask'] for f in batch]).unsqueeze(-1)
        self.concepts = tokenizer(["chemical compound", "gene/protein"], return_tensors='pt', padding=True)

        self.ent1_pos = torch.tensor([f["ent"][0]['pos'] for f in batch], dtype=torch.long)
        self.ent2_pos = torch.tensor([f["ent"][1]['pos'] for f in batch], dtype=torch.long)

        self.in_train = True
        # embed()

    def to(self, device):
        self.texts = self.texts.to(device)
        self.texts_attn_mask = self.texts_attn_mask.to(device)
        self.labels = self.labels.to(device)
        self.label_text = {key: self.label_text[key].to(device) for key in self.label_text}

        self.ent1_g = self.ent1_g.to(device)
        self.ent1_g_mask = self.ent1_g_mask.to(device)
        self.ent1_d = {key: self.ent1_d[key].to(device) for key in self.ent1_d}
        self.ent1_d_mask = self.ent1_d_mask.to(device)

        self.ent2_d = {key: self.ent2_d[key].to(device) for key in self.ent2_d}
        self.ent2_d_mask = self.ent2_d_mask.to(device)

        self.concepts = {key: self.concepts[key].to(device) for key in self.concepts}

        self.ent1_pos = self.ent1_pos.to(device)
        self.ent2_pos = self.ent2_pos.to(device)
        return self

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


def collate_wrapper(batch):
    return CustomBatch(batch)
