import os
import pandas as pd
from rdkit import Chem

# current file directory
root = os.path.dirname(os.path.abspath(__file__))


## read in: ./data/Table_S4.xlsx ############################################################
train_data = pd.read_excel(os.path.join(root, "data", "Table_S4.xlsx"), sheet_name=0)
test_data = pd.read_excel(os.path.join(root, "data", "Table_S4.xlsx"), sheet_name=1)
valid_data = pd.read_excel(os.path.join(root, "data", "Table_S4.xlsx"), sheet_name=2)

datasets = {"training_set":pd.DataFrame(train_data, columns=['No.', 'Smiles', 'activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100']), 
            "test_set":pd.DataFrame(test_data, columns=['No.', 'Smiles', 'activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100']), 
            "validation_set":pd.DataFrame(valid_data, columns=['No.', 'Smiles', 'activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100'])
            }

#drop smiles that cannot be processed by RDKIT
def drop_invalid_smiles(df):
    indices_to_drop = []
    inchikeys = []
    for i, s in enumerate(df["Smiles"]):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            indices_to_drop.append(i)
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except:
            inchikey = None
        inchikeys += [inchikey]
    df["inchikeys"] = inchikeys
    return df.drop(indices_to_drop)


for k,v in datasets.items():
    print(v.shape)
    df = drop_invalid_smiles(v)
    print(df.shape)
    df.to_csv(os.path.join(root, "data", f"{k}.csv"), index=False)