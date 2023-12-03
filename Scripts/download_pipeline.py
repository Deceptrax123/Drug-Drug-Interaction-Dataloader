from tdc.multi_pred import DDI
from graph_dataset import MolecularGraphDataset
import pandas as pd
from dotenv import load_dotenv
import pickle
import os

# Download csvs and save the graphs as .pt files as required for graph dataloader classes


def download_data():
    # download original CSV from TDC
    data = DDI(name='TWOSIDES', path='../data')


def prepare_multi_label_csv():  # Reduce datasite size by combining reactions with same IDs and add labels to a list

    data = pd.read_csv("data/twosides.csv")

    # prepare column lists
    id1 = list()
    id2 = list()
    reactant1 = list()
    reactant2 = list()
    symptoms = list()

    original_id1s = data['ID1']
    original_ids2 = data['ID2']

    mapped_ids = list(zip(original_id1s, original_ids2))
    unique_elements = list(set(mapped_ids))

    for i in unique_elements:
        react_id1, react_id2 = i[0], i[1]

        # query based on ids and get reactants and interaction symptoms
        queried_data = data.query('ID1==@react_id1 and ID2==@react_id2')
        # All reactants are the same for same ids, query only 1
        reactant_1 = list(queried_data['X1'])[0]
        reactant_2 = list(queried_data['X2'])[0]

        side_effects = list(queried_data['Side Effect Name'])

        id1.append(react_id1)
        id2.append(react_id2)
        reactant1.append(reactant_1)
        reactant2.append(reactant_2)
        symptoms.append(side_effects)

    cleaned_dataframe = pd.DataFrame({
        'ID1': id1, 'ID2': id2, 'X1': reactant1, 'X2': reactant2, 'Side Effect Name': symptoms
    })

    # Convert to csv file
    cleaned_dataframe.to_csv("data/twosides_cleaned.csv")


def save_data_binaries():  # Save the tuple objects of (reactant1,reactant2,label) as binaries as required by Dataset clas
    load_dotenv(".env")
    data = pd.read_csv("data/twosides_cleaned.csv", converters={
        'Side Effect Name': pd.eval
    })

    x1s = data['X1']
    x2s = data['X2']
    labels = data['Side Effect Name']

    zipped = list(zip(x1s, x2s, labels))

    f_num = 0
    for i in zipped:
        file_name = os.getenv('tup_bins')+str(f_num)

        with open(file_name, 'wb') as fp:
            pickle.dump(i, fp)
        f_num += 1


def process_graphs():  # Save the Graphs as .pt files
    # process graphs and save them to disk.
    load_dotenv('.env')

    # Save the dataset to the path key points to
    dataset = MolecularGraphDataset(key='tup_bins')


# Run the download pipeline(only if u dont have the .pt files)
if __name__ == '__main__':

    print("click 1 to download data, 2 to download cleaned csvs, 3 to save data binaries and 4 to save graphs")
    print("If running for first time, proceed in order since all steps are needed to effectively download the data")
    opt = int(input())

    if opt == 1:
        download_data()
    elif opt == 2:
        prepare_multi_label_csv()
    elif opt == 3:
        save_data_binaries()
    elif opt == 4:
        process_graphs()
