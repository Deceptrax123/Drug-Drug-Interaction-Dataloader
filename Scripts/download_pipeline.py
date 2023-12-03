from tdc.multi_pred import DDI
from prepare_multi_label import prepare_multi_label_csv
from graph_dataset import MolecularGraphDataset
import pandas as pd
from dotenv import load_dotenv
import pickle
import os


def download_data():
    data = DDI(name='TWOSIDES', path='../data')


def save_data_binaries():
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


def process_graphs():
    # process graphs and save them to disk.
    load_dotenv('.env')

    # Save the dataset to the path key points to
    dataset = MolecularGraphDataset(key='tup_bins')


# Run the download pipeline(only if u dont have the .pt files)
if __name__ == '__main__':

    print("click 1 to download data, 2 to download cleaned csvs, 3 to save data binaries and 4 to save graphs")
    opt = int(input())

    if opt == 1:
        download_data()
    elif opt == 2:
        prepare_multi_label_csv()
    elif opt == 3:
        save_data_binaries()
    elif opt == 4:
        process_graphs()
