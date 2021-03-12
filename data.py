# import torch
# from rdkit import Chem
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import rdmolops
# import igraph
from pprint import pprint as pp
from features import *
from utils import dump_file, load_file
# from transformers import BartTokenizer
from torch_geometric.data import Data, Batch
from sklearn import model_selection
from IPython import embed
import os
# from torch_geometric import utils
import spacy
import json
# from collections import defaultdict
# import re
import pandas as pd
import torch

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

def get_graph_info(input_smiles, args):
    smiles = []
    suppl, targets = [], []
    dex = []
    for i, s in enumerate(list(input_smiles)):
        if s == "[[NULL]]":
            # print("isnull")
            continue
        try:
            c = Chem.MolFromSmiles(s)
        except Exception as e:
            print(e)
            continue
        if c:
            dex.append(i)
            smiles.append(s)
            suppl.append(c)

    # res, exclude
    all_atom_properties, tmp_dex = get_atom_properties(get_atom_symbols(suppl))

    # updated valid dex
    # dex is valid index in original input
    dex = [dex[d] for d in tmp_dex]
    assert len(tmp_dex) == len(dex), embed()

    # args.in_dim = all_atom_properties[0].shape[1]
    # print("args.in_dim")

    suppl = [suppl[d] for d in tmp_dex]
    # targets = [targets[d] for d in dex]
    # smiles = [smiles[d] for d in dex]

    # suppl = [Chem.MolFromSmiles(s) for s in smiles]

    # embed()
    # smiles = open("10_rndm_zinc_drugs_clean.smi").read().splitlines()
    # print("adj_matrix")
    adj = list(get_adj_matrix_coo(suppl))
    # number_of_bonds = list(get_num_bonds(suppl))

    # print("bond_types")
    # bond_types = list(get_bonds_info(suppl))
    bond_types = []

    for k, mol in enumerate(suppl):
        # tmp=[]
        # for i, j in adj[k].t().numpy():
        #     tmp.append(int(mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble()))

        tmp = [int(mol.GetBondBetweenAtoms(int(i), int(j)).GetBondTypeAsDouble()) for i, j in adj[k].t().numpy()]
        bond_types.append(tmp)

    # targets = list(get_targets(get_atom_symbols(suppl)))

    # num_classes = max(targets) + 1
    # print("num_classes", num_classes)
    print("edge_index")
    edge_index = [torch.as_tensor(target, dtype=torch.long) for target in adj]
    print(len(edge_index))
    print(edge_index[0].shape)

    print("node_attr")
    node_attr = [torch.tensor(target, dtype=torch.long) for target in all_atom_properties]
    print(len(node_attr))
    print(node_attr[0].shape)

    print("edge_attr")
    edge_attr = [torch.tensor(target, dtype=torch.long) for target in bond_types]  # list(seperator(bond_types))
    print(len(edge_attr))
    print(edge_attr[0].shape)

    # print("Y_data")
    args.in_dim = args.g_dim
    # Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets
    in_dim = args.g_dim  # node_attr[0].shape[-1]

    # dummy data
    res = [Data(x=torch.rand(2, 1), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                edge_attr=torch.tensor([1,1], dtype=torch.long)) for _ in
           range(len(input_smiles))]
    res_mask = [0 for _ in range(len(input_smiles))]

    # d is the actualy index in inputsmile
    for i, d in enumerate(dex):
        res_mask[d] = 1
        res[d] = Data(x=node_attr[i], edge_index=edge_index[i], edge_attr=edge_attr[i])

    return res_mask, res

    return dex, node_attr, edge_index


def load_mole_data2(args, filename, tokenizer):
    args.cache_filename = os.path.splitext(args.train_file)[0] + ".pkl"
    if args.use_cache and os.path.exists(args.cache_filename):
        print("Loading Cached Data...")
        data = load_file(args.cache_filename)
        train, val, test = data['train'], data['val'], data['test']
        # for f in [train, val, test]:
        #     for d in f:
        #         # d["graph_data"].x=torch.cat([d["graph_data"].x[:, :-13], d["graph_data"].x[:, 8:]])
        #         d["graph_data"].x[:, 6] = torch.rand_like(d["graph_data"].x[:, 6])

        # d["graph_data"].x = torch.rand_like(d["graph_data"].x)
        # args.in_dim = train[0]["graph_data"].x.shape[1]
        args.in_dim = args.g_dim

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

    args.in_dim = args.g_dim
    # all_atom_properties[0].shape[1]

    suppl = [suppl[d] for d in dex]
    targets = [targets[d] for d in dex]
    smiles = [smiles[d] for d in dex]

    adj = list(get_adj_matrix_coo(suppl))
    number_of_bonds = list(get_num_bonds(suppl))
    bond_types = []
    # print("adj", adj)
    # exit()
    # for k, mol in enumerate(suppl):
    #     # tmp=[]
    #     # for i, j in adj[k].t().numpy():
    #     #     tmp.append(int(mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble()))
    #
    #     tmp = [int(mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble()) for i,j in adj[k].t().numpy()]
    #     bond_types.append(tmp)
    #
    # print(adj[0].shape)
    # print(len(bond_types[0]))

    # embed()
    # targets = list(get_targets(get_atom_symbols(suppl)))

    num_classes = max(targets) + 1
    print("num_classes", num_classes)
    print("edge_index")
    edge_index = [torch.as_tensor(target, dtype=torch.long) for target in adj]
    print("node_attr")
    node_attr = [torch.tensor(target, dtype=torch.long) for target in all_atom_properties]
    # list(seperator(all_atom_properties))
    print("node_attr.shape", node_attr[0].shape)

    print("edge_attr")
    edge_attr = [torch.tensor(target, dtype=torch.long) for target in bond_types]
    print("edge_attr.shape", edge_attr[0].shape)

    print("Y_data")
    Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets
    print("Y_data", Y_data)

    instances = []
    print("start adding")
    for i, (s, node_a, edge_i, edge_a, tgt) in enumerate(zip(smiles, node_attr, edge_index, edge_attr, Y_data)):
        # print("print")
        # print(tgt)
        # print(node_a, edge_i, edge_a, tgt)
        # print(i)
        # print(s, tgt)
        # print(Data(x=node_a, edge_index=edge_i, edge_attr=edge_a, y=tgt))
        # exit()
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
    train, tmp = model_selection.train_test_split(instances, train_size=0.4, shuffle=False)
    val, test = model_selection.train_test_split(tmp, train_size=0.5, shuffle=False)  # This splits the tmp value

    if args.cache_filename:
        dump_file({"train": train, "val": val, "test": test}, args.cache_filename)

    return train, val, test


def load_mole_data(args, filename, tokenizer):
    args.cache_filename = os.path.splitext(args.train_file)[0] + ".pkl"
    if args.use_cache and os.path.exists(args.cache_filename):
        print("Loading Cached Data...")
        data = load_file(args.cache_filename)
        train, val, test = data['train'], data['val'], data['test']
        # for f in [train, val, test]:
        #     for d in f:
        #         # d["graph_data"].x=torch.cat([d["graph_data"].x[:, :-13], d["graph_data"].x[:, 8:]])
        #         d["graph_data"].x[:, 6] = torch.rand_like(d["graph_data"].x[:, 6])

        # d["graph_data"].x = torch.rand_like(d["graph_data"].x)
        # args.in_dim = train[0]["graph_data"].x.shape[1]
        args.in_dim = args.g_dim

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
    args.in_dim = args.g_dim
    # all_atom_properties[0].shape[1]

    suppl = [suppl[d] for d in dex]
    targets = [targets[d] for d in dex]
    smiles = [smiles[d] for d in dex]

    adj = list(get_adj_matrix_coo(suppl))
    number_of_bonds = list(get_num_bonds(suppl))
    bond_types = []
    # print("adj", adj)
    # exit()
    for k, mol in enumerate(suppl):
        # tmp=[]
        # for i, j in adj[k].t().numpy():
        #     tmp.append(int(mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble()))

        tmp = [int(mol.GetBondBetweenAtoms(int(i), int(j)).GetBondTypeAsDouble()) for i, j in adj[k].t().numpy()]
        bond_types.append(tmp)

    print(adj[0].shape)
    print(len(bond_types[0]))

    # embed()
    # targets = list(get_targets(get_atom_symbols(suppl)))

    num_classes = max(targets) + 1
    print("num_classes", num_classes)
    print("edge_index")
    edge_index = [torch.as_tensor(target, dtype=torch.long) for target in adj]
    print("node_attr")
    node_attr = [torch.tensor(target, dtype=torch.long) for target in all_atom_properties]
    # list(seperator(all_atom_properties))
    print("node_attr.shape", node_attr[0].shape)

    print("edge_attr")
    edge_attr = [torch.tensor(target, dtype=torch.long) for target in bond_types]
    print("edge_attr.shape", edge_attr[0].shape)

    print("Y_data")
    Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets
    print("Y_data", Y_data)

    instances = []
    print("start adding")
    for i, (s, node_a, edge_i, edge_a, tgt) in enumerate(zip(smiles, node_attr, edge_index, edge_attr, Y_data)):
        # print("print")
        # print(tgt)
        # print(node_a, edge_i, edge_a, tgt)
        # print(i)
        # print(s, tgt)
        # print(Data(x=node_a, edge_index=edge_i, edge_attr=edge_a, y=tgt))
        # exit()
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
    train, tmp = model_selection.train_test_split(instances, train_size=0.6, shuffle=False)
    val, test = model_selection.train_test_split(tmp, train_size=0.5, shuffle=False)  # This splits the tmp value

    if args.cache_filename:
        dump_file({"train": train, "val": val, "test": test}, args.cache_filename)

    return train, val, test


class ModalRetreiver:
    def __init__(self):
        pass


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
        args.in_dim=args.g_dim
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

    rels = ['AGONIST-ACTIVATOR', 'DOWNREGULATOR', 'SUBSTRATE_PRODUCT-OF',
            'AGONIST', 'INHIBITOR', 'PRODUCT-OF', 'ANTAGONIST', 'ACTIVATOR',
            'INDIRECT-UPREGULATOR', 'SUBSTRATE', 'INDIRECT-DOWNREGULATOR',
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
                    modal_feats[0].append("[[NULL]]")
                    modal_feat_mask[0].append(0)

                if "pubchem_description" in cmpd_info[cid] and 'descriptions' in cmpd_info[cid][
                    'pubchem_description'] and \
                        len(cmpd_info[cid]['pubchem_description']['descriptions']):
                    # print("dfound pubchem_description")
                    modal_feats[1].append(cmpd_info[cid]['pubchem_description']['descriptions'][0]["description"])
                    modal_feat_mask[1].append(1)
                else:
                    modal_feats[1].append("[[NULL]]")
                    modal_feat_mask[1].append(0)
                return
        else:
            if ent in mention2protid:
                pid = mention2protid[ent]
                if pid is not None and pid in prot_info:
                    modal_feats[0].append("[[NULL]]")
                    modal_feat_mask[0].append(0)
                    if "definition" in prot_info[pid]:
                        # print("found1")
                        modal_feats[1].append(prot_info[pid]["definition"]['text'])
                        modal_feat_mask[1].append(1)
                    else:
                        modal_feats[1].append("[[NULL]]")
                        modal_feat_mask[1].append(0)
                    return
        modal_feats[0].append("[[NULL]]")
        modal_feat_mask[0].append(0)
        modal_feats[1].append("[[NULL]]")
        modal_feat_mask[1].append(0)

    texts, labels = [], []

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

            ent1, ent2 = text[text.find("<< ") + 3:text.find(" >>")], text[text.find("[[ ") + 3:text.find(" ]]")]
            # lower case
            texts.append(text.lower())
            labels.append(rel2id[sample["label"]])

            # print("ent1, ent2", ent1,"|",  ent2)
            # print("looking for ent1")

            if ent2 in mention2cid:
                ent1, ent2 = ent2, ent1
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

    valid_num = np.zeros((3))
    total_num = 0

    for i in range(len(texts)):
        total_num += 1
        valid_num += [modal_feat_mask1[0][i] == 1, modal_feat_mask1[1][i] == 1, modal_feat_mask2[1][i] == 1]

        instances.append({"text": texts[i],
                          "id": i,
                          "label": labels[i],
                          "modal_data": [
                              [modal_feats1[0][i], modal_feats1[1][i], modal_feat_mask1[0][i],
                               modal_feat_mask1[1][i], ],
                              [modal_feats2[1][i], modal_feat_mask2[1][i]]
                          ],
                          "tokenizer": tokenizer
                          })
    print("instances",instances[0])
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


# def preprocess(file='D:\Research\MMLI\MMLI1\data\ChemProt_Corpus\cmpd_info.json'):
#     from torchtext.data import Field, BucketIterator, TabularDataset
#     from torchtext.vocab import GloVe
#
#     # json
#     original_data = load_file(file)
#     data = []
#     smiless = []
#     descriptions = []
#     for sample in original_data.values():
#         print(sample)
#         if 'canonical_smiles' in sample and "description" in sample:
#             smiless.append(sample['canonical_smiles'])
#             descriptions.append(sample["description"])
#             data.append([sample['canonical_smiles'], sample["description"]])
#     dex, node_attr, edge_index = get_graph_info(smiless)
#     smiless = smiless[dex]
#     descriptions = descriptions[dex]
#
#     dir = "data/ChemProt_Corpus/"
#     dump_file([{'description': d, 'smiles': s} for d, s in zip(descriptions, smiless)],
#               dir + "preprocessed_ae_samples.json")
#
#     with open(dir + "preprocessed_ae_samples_description.tsv", mode="w+") as f:
#         f.write('description\n')
#         for i, d in enumerate(descriptions):
#             if i > 0:  f.write('\n')
#             f.write(d)
#     # dump_file([{'description': d} for d  in descriptions], dir+"preprocessed_ae_samples_description.tsv")
#
#     spacy_en = spacy.load('en_core_web_sm')
#
#     def tokenize_en(text):
#         """
#         Tokenizes English text from a string into a list of strings
#         """
#         return [tok.text for tok in spacy_en.tokenizer(text)]
#
#     from torchtext.data import Example
#     SRC = Field(tokenize=tokenize_en,
#                 init_token='<sos>',
#                 eos_token='<eos>',
#                 lower=True, )
#     # desc = data.Field(lower=True, include_lengths=True, batch_first=True)
#     # LABEL = data.Field(sequential=False)
#     fields = [('description', SRC)]
#     tmp = TabularDataset(path=dir + "preprocessed_ae_samples_description.tsv", format="tsv", fields=fields)
#     SRC.build_vocab(tmp, min_freq=2, vectors=GloVe(name='6B', dim=300))  #
#
#     instances = []
#     print("start adding")
#     for i, (smiles, node_a, edge_i) in enumerate(zip(smiless, node_attr, edge_index)):
#         instances.append({"smiles": smiles,
#                           "id": i,
#                           "graph_data": Data(x=node_a, edge_index=edge_i, y=edge_i),
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
        self.texts = batch[0]["tokenizer"]([f["text"] for f in batch], return_tensors='pt', padding=True)
        self.ids = [f["id"] for f in batch]
        self.labels = torch.tensor([f["label"] for f in batch], dtype=torch.long)


        g_data=Batch.from_data_list([f["modal_data"][0][0] for f in batch])
        g_data.x=torch.as_tensor(g_data.x, dtype=torch.long)
        self.batch_modal_data = [[g_data,
                                  batch[0]["tokenizer"]([f["modal_data"][0][1] for f in batch], return_tensors='pt', padding=True),
                                  torch.tensor([f["modal_data"][0][2] for f in batch]).unsqueeze(-1),
                                  torch.tensor([f["modal_data"][0][3] for f in batch]).unsqueeze(-1)],
                                 [batch[0]["tokenizer"]([f["modal_data"][1][0] for f in batch], return_tensors='pt', padding=True),
                                  torch.tensor([f["modal_data"][1][1] for f in batch]).unsqueeze(-1)]]

        # print(self.batch_modal_data[0][0])
        # print(list(self.batch_modal_data[0][0]))
        # print(self.batch_modal_data[0][0][0].x)
        # print(self.batch_modal_data[0][0][0].x.dtype)
        #
        # embed()


        # self.batch_modal_data[0][0] = Batch.from_data_list([f["modal_data"][0][0] for f in batch])
        # self.batch_modal_data[0][1] = batch[0]["tokenizer"]([f["modal_data"][0][1] for f in batch], return_tensors='pt',
        #                                                     padding=True)
        # self.batch_modal_data[0][2] = torch.tensor([f["modal_data"][0][2] for f in batch]).unsqueeze(-1)
        # self.batch_modal_data[0][3] = torch.tensor([f["modal_data"][0][3] for f in batch]).unsqueeze(-1)
        #
        # self.batch_modal_data[1][0] = batch[0]["tokenizer"]([f["modal_data"][1][0] for f in batch], return_tensors='pt',
        #                                                     padding=True)
        # self.batch_modal_data[1][1] = torch.tensor([f["modal_data"][1][1] for f in batch]).unsqueeze(-1)

        self.in_train = True

    def to(self, device):




        inputs = {'texts': {key: self.texts[key].to(device) for key in self.texts},
                  "batch_ent1_d": {key: self.batch_modal_data[0][1][key].to(device) for key in self.batch_modal_data[0][1]},
                  "batch_ent1_d_mask": self.batch_modal_data[0][3].to(device),
                  "batch_ent2_d": {key: self.batch_modal_data[1][0][key].to(device) for key in self.batch_modal_data[1][0]},
                  "batch_ent2_d_mask": self.batch_modal_data[1][1].to(device),
                  "batch_ent1_g": self.batch_modal_data[0][0].to(device),
                  "batch_ent1_g_mask": self.batch_modal_data[0][2].to(device),
                  # "batch_ent2_g": batch[7].to(args.device),
                  # "batch_ent2_g_mask": batch[8].to(args.device),
                  "ids": self.ids,
                  "labels": self.labels.to(device),
                  'in_train': self.in_train,
                  }
        return inputs

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self
def collate_wrapper(batch):
    return CustomBatch(batch)


# def collate_fn_re(batch):
#     # max_len = max([len(f["input_ids"]) for f in batch])
#     # input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
#     # input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
#     # input_ids = torch.tensor(input_ids, dtype=torch.long)
#     # input_mask = torch.tensor(input_mask, dtype=torch.float)
#     # index = torch.arange(start=0, end=len(batch))
#     # {"text": text,
#     #  "id": i,
#     #  "label": label,
#     #  "ent1_g": ent1_g,
#     #  "ent1_g_mask": ent1_g_mask,
#     #  "ent1_d": ent1_d_mask,
#     #  "ent1_d_mask": ent1_d_mask,
#     #  "ent2_g": ent2_g_mask,
#     #  "ent2_g_mask": ent1_g_mask,
#     #  "ent2_d": ent1_d_mask,
#     #  "ent2_d_mask": ent1_d_mask,
#     #  "tokenizer": tokenizer
#     #  }
#     texts = batch[0]["tokenizer"]([f["text"] for f in batch], return_tensors='pt', padding=True)
#     ids = [f["id"] for f in batch]
#     labels = torch.tensor([f["label"] for f in batch], dtype=torch.long)
#     "modal_data": [
#         [modal_feats1[0][i], modal_feats1[1][i], modal_feat_mask1[0][i], modal_feat_mask1[0][i], ],
#         [modal_feats2[1][i], modal_feat_mask2[1][i]]
#     ],
#
#     batch_modal_data = [[[], [], [], []],
#                         [[], []]]
#
#     batch_modal_data[0][0] = Batch.from_data_list([f["modal_data"][0][0] for f in batch])
#     batch_modal_data[0][1] = batch[0]["tokenizer"]([f["modal_data"][0][1] for f in batch], return_tensors='pt',
#                                                    padding=True)
#     batch_modal_data[0][2] = torch.tensor([f["modal_data"][0][2] for f in batch]).unsqueeze(-1)
#     batch_modal_data[0][3] = torch.tensor([f["modal_data"][0][3] for f in batch]).unsqueeze(-1)
#
#     batch_modal_data[1][0] = batch[0]["tokenizer"]([f["modal_data"][1][0] for f in batch], return_tensors='pt',
#                                                    padding=True)
#     batch_modal_data[1][1] = torch.tensor([f["modal_data"][1][1] for f in batch]).unsqueeze(-1)
#
#     # batch_ent1_g = Batch.from_data_list([f["ent1_g"] for f in batch])
#     # batch_ent1_g_mask = torch.tensor([f["ent1_g_mask"] for f in batch]).unsqueeze(-1)
#     # batch_ent1_d = batch[0]["tokenizer"]([f["ent1_d"] for f in batch], return_tensors='pt', padding=True, )
#     # batch_ent1_d_mask = torch.tensor([f["ent1_d_mask"] for f in batch]).unsqueeze(-1)
#     # # print("batch_ent1_d")
#     # # pp(batch_ent1_d)
#     # # print("batch_ent1_g")
#     # # pp(batch_ent1_g)
#     # batch_ent2_g = Batch.from_data_list([f["ent2_g"] for f in batch])
#     # batch_ent2_g_mask = torch.tensor([f["ent2_g_mask"] for f in batch]).unsqueeze(-1)
#     # batch_ent2_d = batch[0]["tokenizer"]([f["ent2_d"] for f in batch], return_tensors='pt', padding=True, )
#     # batch_ent2_d_mask = torch.tensor([f["ent2_d_mask"] for f in batch]).unsqueeze(-1)
#     # # print("batch_ent2_d")
#     # # pp(batch_ent2_d)
#     # # print("batch_ent2_g")
#     # # pp(batch_ent2_g)
#     # # Label Smoothing
#     # # output = (smiles, edge_indices, node_attrs,edge_attrs, Ys, ids)
#
#     output = (texts,
#               batch_ent1_d,
#               batch_ent1_d_mask,
#               batch_ent2_d,
#               batch_ent2_d_mask,
#               batch_ent1_g,
#               batch_ent1_g_mask,
#               batch_ent2_g,
#               batch_ent2_g_mask,
#               ids,
#               labels,
#               )
#     return output
# collate
