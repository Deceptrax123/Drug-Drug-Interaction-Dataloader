# Use this to save the graphs to disk since the process function is not overridden.
# Once this is run, remove the process function and include the processed_paths property
# using the same format as raw_paths property

from typing import List, Tuple, Union
import torch
from torch_geometric.data import Dataset
import os
import os.path as osp
from dotenv import load_dotenv
import pickle
from molecule_dataloader import get_graphs


class MolecularGraphDataset(Dataset):
    def __init__(self, key, start, root, transform=None, pre_transform=None, pre_filter=None):
        self.key = key
        self.start = start
        self.root = root

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        load_dotenv('.env')
        rp = os.getenv(self.key)

        return [name for name in os.listdir(rp)[self.start:self.start+10000] if '_' not in name]

    @property
    def processed_file_names(self):
        load_dotenv('.env')
        l = 10000  # run in folds of 10,000 each

        proccessed_names = list()
        for n in range(self.start, self.start+l):
            proccessed_names.append('data_'+str(n)+'.pt')

        return proccessed_names

    @property
    def raw_paths(self):
        load_dotenv('.env')
        directory = os.getenv(self.key)
        return [os.path.join(directory, file) for file in os.listdir(directory)[self.start:self.start+10000] if '_' not in file]

    def process(self):
        idx = 0

        for raw_path in self.raw_paths:
            with open(raw_path, 'rb') as fp:
                bin = pickle.load(fp)

            data = get_graphs(bin)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(
                self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))

        return data
