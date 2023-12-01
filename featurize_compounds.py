import torch
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from torch_geometric.data import Data


def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(bol_val) for bol_val in list(map(lambda s: x == s, permitted_list))
                       ]

    return binary_encoding

# Featurizing Atoms


def get_atom_features(atom):

    atoms_list = ['C', 'N', 'O', 'H', 'Li',
                  'Al', 'Si', 'S', 'Cl', 'F', 'P', 'I']

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), atoms_list)
    formal_charg_enc = one_hot_encoding(
        int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    n_neighbors_enc = one_hot_encoding(
        int(atom.GetDegree()), [[0, 1, 2, 3, 4, 5, "MoreThanFive"]])

    hybridization_types = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "other"]
    hybridization_enc = one_hot_encoding(
        int(atom.GetHybridization()), hybridization_types)

    ring_enc = [int(atom.IsInRing())]
    aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass = [float((atom.getMass()-10.812)/116.092)]

    vdw_radius = [
        float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())-1.5)/0.6)]
    covalent_radius = [
        float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())-0.64)/0.76)]

    n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                                                   18, 19, 20, "MoreThan20"])

    # Combine all features
    atom_feature_vector = atom_type_enc+formal_charg_enc+n_neighbors_enc+hybridization_enc+ring_enc +\
        aromatic_enc+atomic_mass+vdw_radius+covalent_radius+n_hydrogens_enc

    return np.array(atom_feature_vector)

# Featurize Chemical Bonds


def get_bond_features(bond):

    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.redchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    stereo_types = ['STEREOZ', 'STEREOE', 'STEREOANY', 'STEREONONE']

    bond_type_enc = one_hot_encoding(bond.GetBondType(), bond_types)
    bond_conj_enc = [int(bond.GetIsConjugated())]

    stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), stereo_types)

    bond_feature_vector = bond_type_enc+bond_conj_enc+stereo_type_enc

    return bond_feature_vector
