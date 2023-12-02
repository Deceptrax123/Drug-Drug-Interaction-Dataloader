from typing import Any
from torch_geometric.data import Data


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs):
        if key == 'target':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
