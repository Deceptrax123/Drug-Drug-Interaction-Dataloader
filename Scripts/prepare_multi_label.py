import pandas as pd
import numpy as np


def prepare_multi_label_csv():

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
