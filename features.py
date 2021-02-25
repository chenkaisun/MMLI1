"""This file is comprised of basic UDFs that may be useful in Project Outcome.

There are a couple possible input types for each function:
1. molecules (the raw sdf file, opened to be iterated upon)
2. atom list (the atoms that each molecule of the sdf is comprised of, which can be represented by any unique ID)
"""

# TODO: get_atom_properties is generalized; adding a .specific_property to element(atom) would allow
#       access to different properties from the mendeleev package. Also note that this funciton is
#       dependant on the get_atom_symbols generator directly above it.

import scipy
import numpy as np
# import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler

# import tqdm
import torch

# import os.path as osp
import rdkit.Chem as Chem
# import networkx as nx

# from torch_geometric.data import Data, Dataset
from mendeleev import element
from torch_geometric import utils

# suppl = Chem.SDMolSupplier('10.sdf')
#
# smiles = open("10_rndm_zinc_drugs_clean.smi").read().splitlines()

# suppl = Chem.SDMolSupplier("qm9-100-sdf.sdf")

# targets = list(pd.read_csv("qm9-100.csv").mu)

# smiles = open("qm9-100-smiles.smi").read().splitlines()

# assert len(suppl) == len(targets) == len(smiles), "the datasets must be of the same length!"

# =================================================================================== #
"""                              GRAPH REPRESENTATION                               """


# =================================================================================== #

def get_adj_matrix_coo(molecules):
    for i, mol in enumerate(molecules):
        # print("i",i)
        # print("mol",mol)
        am = Chem.GetAdjacencyMatrix(mol)
        # print("am\n", am)
        for i in range(np.array(am).shape[0]):
            am[i, i] = 1

        # am[am>1]=1

        adj_mat = scipy.sparse.csr_matrix(am)

        # print("adj_mat", adj_mat.row)
        # print("adj_mat", adj_mat.col)

        # nx_graph = nx.from_scipy_sparse_matrix(adj_mat)
        # coo_matrix = nx.to_scipy_sparse_matrix(nx_graph, dtype=float, format="coo")

        # print("coo_matrix row", coo_matrix.row)
        # print("coo_matrix", coo_matrix.col)
        # print("utils.from_scipy_sparse_matrix(adj_mat)[0]", utils.from_scipy_sparse_matrix(adj_mat)[0])
        yield utils.from_scipy_sparse_matrix(adj_mat)[0]
        # yield coo_matrix.row, coo_matrix.col


# =================================================================================== #
"""                                GRAPH ATTRIBUTES                                 """


# =================================================================================== #

def get_num_bonds(molecules):
    for mol in molecules:
        number_of_bonds = mol.GetNumBonds()

        yield number_of_bonds


# =================================================================================== #
"""                                NODE ATTRIBUTES                                  """


# =================================================================================== #

def get_atom_symbols(molecules):
    for mol in molecules:
        atoms = mol.GetAtoms()
        # print("list(atoms)",list(atoms))
        yield list(atoms)

        # atom_symbols = [atom.GetSymbol() for atom in atoms]
        # yield atom_symbols


def get_prop(prop, atom, prop_dict):
    # print(prop_dict)
    # dynamic populating
    if not atom in prop_dict:
        prop_dict[atom] = {}
    if not prop in prop_dict[atom]:
        # print("atom", atom)
        # print("prop", prop)
        elt_atom = element(atom)
        if prop == "atomic_volume":
            prop_dict[atom][prop] = elt_atom.atomic_volume
        elif prop == "atomic_weight":
            prop_dict[atom][prop] = elt_atom.atomic_weight
        elif prop == "atomic_radius":
            prop_dict[atom][prop] = elt_atom.atomic_radius
        elif prop == "boiling_point":
            prop_dict[atom][prop] = elt_atom.boiling_point
        elif prop == "charge":
            prop_dict[atom][prop] = elt_atom.charge
        elif prop == "density":
            prop_dict[atom][prop] = elt_atom.density
    return prop_dict[atom][prop]

class AtomProp:
    def __init__(self):
        pass

def get_atom_properties(atom_list):
    res = []
    dex = []
    prop_dict = {}
    scaler = StandardScaler()
    for i, atoms in enumerate(atom_list):

        # if i>60: break
        print("i", i)
        # print("atoms",atoms)
        # for i, atom in enumerate(atoms):
        #     print(i, atom)
        #     element(atom).atomic_volume

        # atomic_number = [element(atom).atomic_number for atom in atoms]atomic_number,

        try:

            atomic_volume = [get_prop("atomic_volume", atom.GetSymbol(), prop_dict) for atom in atoms]
            atomic_weight = [get_prop("atomic_weight", atom.GetSymbol(), prop_dict) for atom in atoms]
            atomic_radius = [get_prop("atomic_radius", atom.GetSymbol(), prop_dict) for atom in atoms]
            boiling_point = [get_prop("boiling_point", atom.GetSymbol(), prop_dict) for atom in atoms]
            density = [get_prop("density", atom.GetSymbol(), prop_dict) for atom in atoms]

            total_valence = [atom.GetTotalValence() for atom in atoms]
            aromatic = [int(atom.GetIsAromatic()) for atom in atoms]
            fc = [atom.GetFormalCharge() for atom in atoms]

            # 8 types
            hybridization = np.zeros((len(atoms), 8))
            hybridization[np.arange(len(atoms)), [int(atom.GetHybridization()) for atom in atoms]] = 1
            # print("hybridization", hybridization)
            # hybridization = [int(atom.GetHybridization()) for atom in atoms]

            # 4 types
            chiral_tag = np.zeros((len(atoms), 4))
            chiral_tag[np.arange(len(atoms)), [int(atom.GetChiralTag()) for atom in atoms]] = 1
            # print("chiral_tag", chiral_tag)
            # chiral_tag = [int(atom.GetChiralTag()) for atom in atoms]
            # charge = [get_prop("charge", atom, prop_dict) for atom in atoms]

        except Exception as e:
            print(e)
            # exclude.append(i)
            continue

        all_atom_properties = list(zip(atomic_volume, atomic_weight, atomic_radius, boiling_point,
                                       density, total_valence, aromatic, fc))
        all_atom_properties = np.concatenate([all_atom_properties, hybridization, chiral_tag], axis=1)


        # print("all_atom_properties", scaler.fit_transform(all_atom_properties))
        # print("all_atom_properties", all_atom_properties.shape)
        # res.append((all_atom_properties-np.min(all_atom_properties, axis=0))/(1+np.max(all_atom_properties, axis=0)-np.min(all_atom_properties, axis=0)))

        res.append(scaler.fit_transform(all_atom_properties))
        print("res", res[-1].shape)
        #
        # res.append(scaler.fit_transform(all_atom_properties))

        dex.append(i)
    print("dex", dex)

    # normalization

    return res, dex
    # yield all_atom_properties


# for buh in all_atom_properties:
#         ah = buh[0]
#         print(type(ah))
#         for duh in buh:
#                 eh = duh[0]
#                 print(type(eh))

# =================================================================================== #
"""                                EDGE ATTRIBUTES                                  """


# =================================================================================== #

# def get_num_bonds(molecules):
#     for mol in suppl:
#         number_of_bonds = mol.GetNumBonds()
#
#         yield number_of_bonds

def get_bonds_info(molecules):
    for mol in molecules:
        number_of_bonds = mol.GetNumBonds()
        bond_types = [int(bond.GetBondTypeAsDouble()) for bond in mol.GetBonds()]

        yield bond_types


# for luh in bond_types:
#         ah = luh[0]
#         print(type(ah))
# =================================================================================== #
"""                                    TARGETS                                      """


# =================================================================================== #
def get_targets(atom_list):
    # Same as the get_atom_properties function from node attributes section
    for atoms in atom_list:
        boiling_points = list([element(atom).boiling_point for atom in atoms])

        yield boiling_points


def get_num_classes(atom_list):
    num_classes = []
    for atoms in atom_list:
        boiling_points = list([element(atom).boiling_point for atom in atoms])
        classes = len(sorted(list(set(boiling_points))))
        num_classes.append(classes)
    return max(num_classes)  # We return the max value, as this is the number of classes of the most diverse molecule


def normalize(numerical_dataset):
    raw = list(itertools.chain.from_iterable(numerical_dataset))

    maximum = max(raw)
    minimum = min(raw)

    for targets in numerical_dataset:
        norm_dataset = [(target - minimum) / (maximum - minimum) for target in targets]
        norm_dataset = torch.tensor(norm_dataset, dtype=torch.float)

        return norm_dataset


# norm_targets = normalize(targets)

# for buh in norm_targets:
#         print(type(ah))

# =================================================================================== #
"""                                 BUILD DATASETS                                  """


# =================================================================================== #

def seperator(datasets):
    for example in datasets:
        tensorized_example = torch.tensor(example, dtype=torch.float)

        yield tensorized_example

# edge_index = list(seperator(coo_adj_matrix))
# node_attr = list(seperator(all_atom_properties))
# edge_attr = list(seperator(bond_types))
# Y_data = list(seperator(targets))
#
# data_instance = list(map(list, zip(node_attr, edge_index, edge_attr, Y_data)))

# def return_data(zipped_data):
#     for instance in data_instance:
#         node_attr = instance[0]
#         edge_index = instance[1]
#         edge_attr = instance[2]
#         target = instance[3]
#
#         yield node_attr, edge_index, edge_attr, target
# def data_maker(datapoints):
#         for node_attr, edge_index, edge_attr, target in datapoints:
#                 data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=target)
#                 print(type(data))
#                 yield data
# print(list(data_maker(return_data(data_instance))))
