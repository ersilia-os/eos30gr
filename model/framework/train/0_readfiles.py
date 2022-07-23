from rdkit import Chem
from rdkit import RDConfig
import pandas as pd
import numpy as np

## read in: ./data/Table_S4.xlsx ############################################################
train_data = pd.read_excel('./data/Table_S4.xlsx', sheet_name=0)
test_data = pd.read_excel('./data/Table_S4.xlsx', sheet_name=1)
valid_data = pd.read_excel('./data/Table_S4.xlsx', sheet_name=2)

	# train_set len = 10422
	# test_set len = 1303 
	# valid_set len = 1302

train_set = pd.DataFrame(train_data, columns=['No.', 'Smiles', 'activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100'])
test_set = pd.DataFrame(test_data, columns=['No.', 'Smiles', 'activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100'])
valid_set = pd.DataFrame(valid_data, columns=['No.', 'Smiles', 'activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100'])

print(train_set.shape, test_set.shape, valid_set.shape)
print()

## convert to str instead of floats ##############################################################
train_set = train_set.astype({"activity10": str, "activity20": str, 'activity40': str, 'activity60':str, 'activity80':str, 'activity100':str}) 
test_set = test_set.astype({"activity10": str, "activity20": str, 'activity40': str, 'activity60':str, 'activity80':str, 'activity100':str}) 
valid_set = valid_set.astype({"activity10": str, "activity20": str, 'activity40': str, 'activity60':str, 'activity80':str, 'activity100':str}) 

props = ['No.', 'Smiles', 'activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100'];

## convert smiles to mols ############################################################
train_smiles = train_set["Smiles"].values.tolist()
test_smiles = test_set["Smiles"].values.tolist()
valid_smiles = valid_set["Smiles"].values.tolist()

train_mols, test_mols, valid_mols = [], [], []; 

for i in range(len(train_smiles)):
	mol = Chem.MolFromSmiles(train_smiles[i])
	
	if mol != None:
		for p in props:
			mol.SetProp(p, train_set[p][i])

		train_mols.append(mol);


for i in range(len(test_smiles)):
	mol = Chem.MolFromSmiles(test_smiles[i])
	
	if mol != None:
		for p in props:
			mol.SetProp(p, test_set[p][i])

		test_mols.append(mol);


for i in range(len(valid_smiles)):
	mol = Chem.MolFromSmiles(valid_smiles[i])
	
	if mol != None:
		for p in props:
			mol.SetProp(p, valid_set[p][i])

		valid_mols.append(mol);


# print(len(train_mols), len(test_mols), len(valid_mols)) ==> 10384, 1295, 1299

# mols => sdf:
train_writer = Chem.SDWriter('./input_file/trainingset.sdf')
test_writer = Chem.SDWriter('./input_file/testset.sdf')
valid_writer = Chem.SDWriter('./input_file/validationset.sdf')

for mol in train_mols:
	if mol is not None:
	   train_writer.write(mol)

for mol in test_mols:
	if mol is not None:
		test_writer.write(mol)

for mol in valid_mols:
	if mol is not None:
 		valid_writer.write(mol)
