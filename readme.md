# Torch Geometric Dataset for applications involving Chemical(SMILE Strings) Reactions

The repository contains the code for an effective way of making a Molecular Graph Dataset in Torch Geometric involving a pair of graphs.

The above implementation makes use of both node features(atoms) and edge features(bonds)

## Download Data
In order to download the Graphs, follow the following steps.

- Clone the Repository:```git clone git@github.com:Deceptrax123/GNN-Dataloader-For-Chemical-Reaction-Applications.git ```
- Run ```Scripts/download_pipeline.py``` and follow the steps to effectively save the .pt files.

## Train and Run
To train and evaluate GNN models, the following modifications need to be made to the dataset script.

- Do not override the ```process()``` function
- Use the  ```processed_paths``` property in the same way as mentioned in the <a href="https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset">docs</a> by giving the absolute path to each .pt file. This ensures the entire dataset isnt processed again.

## Featurizing nodes and edges

- Both node and edge level features have been included.
- For atoms we use features such as their hybridization, atomic mass, presence in an aromatic ring, formal charge etc
- For bonds we use features such as bond type(single,double, triple), stereochemical aspects and presence in a conjugation(Resonance)

## Batching

- Since 2 graphs need to be loaded, the Batching technique has been changed. More information on this can be found <a hred="https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html">here</a>.