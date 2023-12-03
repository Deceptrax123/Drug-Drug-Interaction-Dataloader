from tdc.multi_pred import DDI
from prepare_multi_label import prepare_multi_label_csv
from molecule_dataloader import get_graphs
from sklearn.model_selection import train_test_split
import pandas as pd


def download_data():
    data = DDI(name='TWOSIDES', path='../data')


def download_graphs():
    data = pd.read_csv("/data/twosides_cleaned.csv", converters={
        'Side Effect Name': pd.eval
    })

    x1s = data['X1']
    x2s = data['X2']
    labels = data['Side Effect Name']

    zipped = list(zip(x1s, x2s, labels))

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    train, test = train_test_split(zipped, test_size=1-train_ratio)
    test, val = train_test_split(
        zipped, test_size=test_ratio/(test_ratio+validation_ratio), shuffle=False)

    split_dict = {
        'train': train,
        'test': test,
        'val': val
    }

    for i in split_dict.keys():
        get_graphs(items=split_dict[i], split=i)


# Run the pipeline

if __name__ == '__main__':

    print("click 1 to download data, 2 to download cleaned csvs and 3 to store graphs")
    opt = int(input())

    if opt == 1:
        download_data()
    elif opt == 2:
        prepare_multi_label_csv()
    elif opt == 3:
        download_graphs()
    else:
        print("Invalid Option")
