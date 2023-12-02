import torch
import pandas as pd
import numpy as np
from tdc.utils import get_label_map

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from featurize_compounds import get_atom_features, get_bond_features
from pair_graphs import PairData


def label_map_target(labels, num_classes=1317):  # Obtained from dataset analysis

    # Map labels
    label_map = get_label_map(
        name='TWOSIDES', task='DDI', name_column='Side Effect Name', path='../data/')

    # Inverse map to get numeric labels
    mapped_inverse = {v: k for k, v in label_map.items()}

    mapped_labels = [mapped_inverse.get(item, item) for item in labels]

    # Convert to vector of dimension (1,num_classes)
    target = torch.zeros(num_classes)

    for i in mapped_labels:
        target[i] = 1.0

    return target


def get_graphs(items):

    molecular_graphs = list()
    for item in items:
        smiles = [item[0], item[1]]
        graph_tensors = list()
        labels = item[2]

        for i in smiles:
            mol = Chem.MolFromSmiles(i)

            # featurize nodes and edges
            n_nodes = mol.GetNumAtoms()
            n_edges = 2*mol.GetNumBonds()

            # A reference for getting number of features
            unrelated_mol = Chem.MolFromSmiles("O=O")
            n_nodes_features = len(get_atom_features(
                unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(get_bond_features(
                unrelated_mol.GetBondBetweenAtoms(0, 1)))

            # feature matrix of Nodes
            X = np.zeros((n_nodes, n_nodes_features))

            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = get_atom_features(atom)

            X = torch.tensor(X, dtype=torch.float)

            # Construct edge array of shape (2,no of edges) as per PyG documentation
            rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)

            Edges = torch.stack([torch_rows, torch_cols], dim=0)

            # Construct the Edge Feature Array
            Edge_features = np.zeros((n_edges, n_edge_features))

            for (k, (z, j)) in enumerate(zip(rows, cols)):
                # Get bond Features for every
                Edge_features[k] = get_bond_features(
                    mol.GetBondBetweenAtoms(int(z), int(j)))

            Edge_features = torch.tensor(Edge_features, dtype=torch.float)
            graph_tensors.append(
                Data(x=X, edge_index=Edges, edge_attr=Edge_features))

        mol_graph1, mol_graph2 = graph_tensors[0], graph_tensors[1]

        y_tensor = label_map_target(labels)

        paired_graph = PairData(x_s=mol_graph1.x, edge_index_s=mol_graph1.edge_index,
                                edge_attr_s=mol_graph1.edge_attr,
                                x_t=mol_graph2.x, edge_index_t=mol_graph2.edge_index,
                                edge_attr_t=mol_graph2.edge_attr, target=y_tensor)
        molecular_graphs.append(paired_graph)
    return molecular_graphs
