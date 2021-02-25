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
# from IPython import embed
import os
# from torch_geometric import utils
import spacy
import json
# from collections import defaultdict
# import re
import pandas as pd


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

def get_graph_info(input_smiles):
    smiles = []
    suppl, targets = [], []
    dex = []
    for i, s in enumerate(list(input_smiles)):
        if s == "[[NULL]]":
            print("isnull")
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
    bond_types = list(get_bonds_info(suppl))

    # targets = list(get_targets(get_atom_symbols(suppl)))

    # num_classes = max(targets) + 1
    # print("num_classes", num_classes)
    # print("edge_index")
    edge_index = [torch.as_tensor(target, dtype=torch.long) for target in adj]
    # print("node_attr")
    node_attr = list(seperator(all_atom_properties))
    # print("edge_attr")
    edge_attr = list(seperator(bond_types))
    # print("Y_data")

    # Y_data = [torch.tensor(target, dtype=torch.float) for target in targets]  # targets
    in_dim = node_attr[0].shape[-1]

    # dummy data
    res = [Data(x=torch.rand(2, in_dim), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)) for _ in
           range(len(input_smiles))]
    res_mask = [0 for _ in range(len(input_smiles))]
    for i, d in enumerate(dex):
        res_mask[d] = 1
        res[d] = Data(x=node_attr[i], edge_index=edge_index[i])

    return res_mask, res

    return dex, node_attr, edge_index


def load_mole_data(args, filename, tokenizer):
    args.cache_filename = os.path.splitext(args.train_file)[0] + ".pkl"
    if args.use_cache and os.path.exists(args.cache_filename):
        print("Loading Cached Data...")
        data = load_file(args.cache_filename)
        train, val, test = data['train'], data['val'], data['test']
        for f in [train, val, test]:
            for d in f:
                # d["graph_data"].x=torch.cat([d["graph_data"].x[:, :-13], d["graph_data"].x[:, 8:]])
                d["graph_data"].x[:, 6] = torch.rand_like(d["graph_data"].x[:, 6])

                # d["graph_data"].x = torch.rand_like(d["graph_data"].x)
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
    train, tmp = model_selection.train_test_split(instances, train_size=0.6, shuffle=False)
    val, test = model_selection.train_test_split(tmp, train_size=0.5, shuffle=False)  # This splits the tmp value

    if args.cache_filename:
        dump_file({"train": train, "val": val, "test": test}, args.cache_filename)

    return train, val, test


def load_data_chemprot_re(args, filename, tokenizer=None):
    args.cache_filename = os.path.splitext(filename)[0] + ".pkl"


    if args.use_cache and os.path.exists(args.cache_filename):
        print("Loading Cached Data...", args.cache_filename)
        data = load_file(args.cache_filename)
        args.out_dim=len(data['rel2id'])
        args.in_dim=data['instances'][0]["ent1_g"].x.shape[-1]

        print("args.in_dim", args.in_dim)
        print("args.out_dim", args.out_dim)
        print(data['rel2id'])

        return data['instances']

    instances = []
    # smiless1 = []
    # smiless2 = []
    # descriptions1 = []
    # descriptions2 = []

    mention2cid, cmpd_info = load_file("data_online/ChemProt_Corpus/mention2ent.json"), \
                             load_file("data_online/ChemProt_Corpus/cmpd_info.json")

    rels = ['AGONIST-ACTIVATOR', 'DOWNREGULATOR', 'SUBSTRATE_PRODUCT-OF',
            'AGONIST', 'INHIBITOR', 'PRODUCT-OF', 'ANTAGONIST', 'ACTIVATOR',
            'INDIRECT-UPREGULATOR', 'SUBSTRATE', 'INDIRECT-DOWNREGULATOR',
            'AGONIST-INHIBITOR', 'UPREGULATOR', ]
    rel2id = {rel: i for i, rel in enumerate(rels)}

    def fill_modal_data(ent, mention2cid, cmpd_info, modal_feats, modal_feat_mask):
        if ent in mention2cid:
            cid = mention2cid[ent]
            if cid is not None and str(cid) in cmpd_info:
                cid=str(cid)
                if "canonical_smiles" in cmpd_info[cid]:
                    # print("found")
                    modal_feats[0].append(cmpd_info[cid]["canonical_smiles"])
                    modal_feat_mask[0].append(1)
                else:
                    modal_feats[0].append("[[NULL]]")
                    modal_feat_mask[0].append(0)

                if "pubchem_description" in cmpd_info[cid] and 'descriptions' in cmpd_info[cid]['pubchem_description'] and\
                    len(cmpd_info[cid]['pubchem_description']['descriptions']):
                    modal_feats[1].append(cmpd_info[cid]['pubchem_description']['descriptions'][0]["description"])
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
            if args.debug:
                # if i<100: continue
                if i>200: break

            # print("\n", i, line)
            sample = json.loads(line.strip())

            # text label metadata
            assert not sample["metadata"]

            text = sample["text"]
            texts.append(text)
            labels.append(rel2id[sample["label"]])

            ent1, ent2 = text[text.find("<< ") + 3:text.find(" >>")], text[text.find("[[ ") + 3:text.find(" ]]")]
            # print("ent1, ent2", ent1,"|",  ent2)
            # print("looking for ent1")

            fill_modal_data(ent1, mention2cid, cmpd_info, modal_feats1, modal_feat_mask1)
            # print("looking for ent2")
            fill_modal_data(ent2, mention2cid, cmpd_info, modal_feats2, modal_feat_mask2)
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



    modal_feat_mask1[0], modal_feats1[0] = get_graph_info(modal_feats1[0])
    modal_feat_mask2[0], modal_feats2[0] = get_graph_info(modal_feats2[0])
    # print(len(modal_feats1[0]))
    # print(len(modal_feats1[1]))
    # print(len(modal_feat_mask1[0]))
    # print(len(modal_feat_mask1[1]))
    # print(len(modal_feats2[0]))
    # print(len(modal_feats2[1]))
    # print(len(modal_feat_mask2[0]))
    # print(len(modal_feat_mask2[1]))

    # exit()


    assert len(modal_feat_mask1[0]) == len(modal_feats1[0]) == len(modal_feats2[0]) == len(modal_feats2[0]) == len(
        modal_feats2[1])

    for i, (
    text, label, ent1_g, ent1_g_mask, ent1_d, ent1_d_mask, ent2_g, ent2_g_mask, ent2_d, ent2_d_mask) in enumerate(
            zip(texts, labels, modal_feats1[0], modal_feat_mask1[0], modal_feats1[1], modal_feat_mask1[1],
                modal_feats2[0], modal_feat_mask2[0], modal_feats2[1], modal_feat_mask2[1])):


        instances.append({"text": text,
                          "id": i,
                          "label": label,
                          "ent1_g": ent1_g,
                          "ent1_g_mask": ent1_g_mask,
                          "ent1_d": ent1_d,
                          "ent1_d_mask": ent1_d_mask,
                          "ent2_g": ent2_g,
                          "ent2_g_mask": ent2_g_mask,
                          "ent2_d": ent2_d,
                          "ent2_d_mask": ent2_d_mask,
                          "tokenizer": tokenizer
                          })
    # print(instances)
    # exit()
    args.out_dim = len(rel2id)
    args.in_dim = instances[0]["ent1_g"].x.shape[-1]

    if args.cache_filename:
        dump_file({"instances": instances, "rel2id": rel2id}, args.cache_filename)

    return instances

    # for each sent

    if os.path.exists(args.cache_filename):
        os.remove(args.cache_filename)


def preprocess(file='D:\Research\MMLI\MMLI1\data\ChemProt_Corpus\cmpd_info.json'):
    from torchtext.data import Field, BucketIterator, TabularDataset
    from torchtext.vocab import GloVe

    # json
    original_data = load_file(file)
    data = []
    smiless = []
    descriptions = []
    for sample in original_data.values():
        print(sample)
        if 'canonical_smiles' in sample and "description" in sample:
            smiless.append(sample['canonical_smiles'])
            descriptions.append(sample["description"])
            data.append([sample['canonical_smiles'], sample["description"]])
    dex, node_attr, edge_index = get_graph_info(smiless)
    smiless = smiless[dex]
    descriptions = descriptions[dex]

    dir = "data/ChemProt_Corpus/"
    dump_file([{'description': d, 'smiles': s} for d, s in zip(descriptions, smiless)],
              dir + "preprocessed_ae_samples.json")

    with open(dir + "preprocessed_ae_samples_description.tsv", mode="w+") as f:
        f.write('description\n')
        for i, d in enumerate(descriptions):
            if i > 0:  f.write('\n')
            f.write(d)
    # dump_file([{'description': d} for d  in descriptions], dir+"preprocessed_ae_samples_description.tsv")

    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    from torchtext.data import Example
    SRC = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True, )
    # desc = data.Field(lower=True, include_lengths=True, batch_first=True)
    # LABEL = data.Field(sequential=False)
    fields = [('description', SRC)]
    tmp = TabularDataset(path=dir + "preprocessed_ae_samples_description.tsv", format="tsv", fields=fields)
    SRC.build_vocab(tmp, min_freq=2, vectors=GloVe(name='6B', dim=300))  #

    instances = []
    print("start adding")
    for i, (smiles, node_a, edge_i) in enumerate(zip(smiless, node_attr, edge_index)):
        instances.append({"smiles": smiles,
                          "id": i,
                          "graph_data": Data(x=node_a, edge_index=edge_i, y=edge_i),
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


class CustomBatch:
    def __init__(self, batch):
        # transposed_data = list(zip(*data))
        # self.inp = torch.stack(transposed_data[0], 0)
        # self.tgt = torch.stack(transposed_data[1], 0)
        self.texts = batch[0]["tokenizer"]([f["text"] for f in batch], return_tensors='pt', padding=True, )
        self.ids = [f["id"] for f in batch]
        self.labels = torch.tensor([f["label"] for f in batch], dtype=torch.long)
        self.batch_ent1_g = Batch.from_data_list([f["ent1_g"] for f in batch])
        self.batch_ent1_g_mask = [f["ent1_g_mask"] for f in batch]
        self.batch_ent1_d = batch[0]["tokenizer"]([f["ent1_d"] for f in batch], return_tensors='pt', padding=True, )
        self.batch_ent1_d_mask = [f["ent1_d_mask"] for f in batch]
        self.batch_ent2_g = Batch.from_data_list([f["ent2_g"] for f in batch])
        self.batch_ent2_g_mask = [f["ent2_g_mask"] for f in batch]
        self.batch_ent2_d = batch[0]["tokenizer"]([f["ent2_d"] for f in batch], return_tensors='pt', padding=True, )
        self.batch_ent2_d_mask = [f["ent2_d_mask"] for f in batch]
    # def to(self, device):
    #     texts={key: texts[key].to(args.device) for key in texts
    #      "batch_ent1_d": {key: batch_ent1_d[key].to(args.device) for key in batch_ent1_d},
    #      "batch_ent1_d_mask": batch[2].to(args.device),
    #      "batch_ent2_d": {key: batch_ent2_d[key].to(args.device) for key in batch_ent2_d},
    #      "batch_ent2_d_mask": batch[4].to(args.device),
    #      "batch_ent1_g": batch[5].to(args.device),
    #      "batch_ent1_g_mask": batch[6].to(args.device),
    #      "batch_ent2_g": batch[7].to(args.device),
    #      "batch_ent2_g_mask": batch[8].to(args.device),
    #      "ids": batch[9].to(args.device),
    #      "labels": batch[10].to(args.device),
    #      'in_train': True,
    #      }

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_fn_re(batch):
    # max_len = max([len(f["input_ids"]) for f in batch])
    # input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    # input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    # input_ids = torch.tensor(input_ids, dtype=torch.long)
    # input_mask = torch.tensor(input_mask, dtype=torch.float)
    # index = torch.arange(start=0, end=len(batch))
    # {"text": text,
    #  "id": i,
    #  "label": label,
    #  "ent1_g": ent1_g,
    #  "ent1_g_mask": ent1_g_mask,
    #  "ent1_d": ent1_d_mask,
    #  "ent1_d_mask": ent1_d_mask,
    #  "ent2_g": ent2_g_mask,
    #  "ent2_g_mask": ent1_g_mask,
    #  "ent2_d": ent1_d_mask,
    #  "ent2_d_mask": ent1_d_mask,
    #  "tokenizer": tokenizer
    #  }
    texts = batch[0]["tokenizer"]([f["text"] for f in batch], return_tensors='pt', padding=True, )
    ids = [f["id"] for f in batch]
    labels = torch.tensor([f["label"] for f in batch], dtype=torch.long)

    batch_ent1_g = Batch.from_data_list([f["ent1_g"] for f in batch])
    batch_ent1_g_mask = torch.tensor([f["ent1_g_mask"] for f in batch]).unsqueeze(-1)
    batch_ent1_d = batch[0]["tokenizer"]([f["ent1_d"] for f in batch], return_tensors='pt', padding=True,)
    batch_ent1_d_mask = torch.tensor([f["ent1_d_mask"] for f in batch]).unsqueeze(-1)
    # print("batch_ent1_d")
    # pp(batch_ent1_d)
    # print("batch_ent1_g")
    # pp(batch_ent1_g)
    batch_ent2_g = Batch.from_data_list([f["ent2_g"] for f in batch])
    batch_ent2_g_mask = torch.tensor([f["ent2_g_mask"] for f in batch]).unsqueeze(-1)
    batch_ent2_d = batch[0]["tokenizer"]([f["ent2_d"] for f in batch], return_tensors='pt', padding=True,)
    batch_ent2_d_mask = torch.tensor([f["ent2_d_mask"] for f in batch]).unsqueeze(-1)
    # print("batch_ent2_d")
    # pp(batch_ent2_d)
    # print("batch_ent2_g")
    # pp(batch_ent2_g)
    # Label Smoothing
    # output = (smiles, edge_indices, node_attrs,edge_attrs, Ys, ids)


    output = (texts,
              batch_ent1_d,
              batch_ent1_d_mask,
              batch_ent2_d,
              batch_ent2_d_mask,
              batch_ent1_g,
              batch_ent1_g_mask,
              batch_ent2_g,
              batch_ent2_g_mask,
              ids,
              labels,
              )
    return output
# collate
