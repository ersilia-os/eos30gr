import sys
import os
import csv
import tempfile
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem.ChemUtils import SDFToCSV
from standardiser import standardise
from lazyqsar.binary.morgan import MorganBinaryClassifier

root = os.path.abspath(os.path.dirname(__file__)) 

tmp_dir = tempfile.mkdtemp("eos-")

input_file = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])

smiles = []
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        smiles += [r[0]]

mols = []
for i, smi in enumerate(smiles):
    mol = Chem.MolFromSmiles(smi)
    mol = standardise.run(mol)
    if mol is not None:
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp("MoleculeIdentifier", "id-{0}".format(i))
    
    mols += [mol]

sdfile = os.path.join(tmp_dir, "input.sdf")
with open(sdfile, 'w') as file:
    writer = Chem.SDWriter(sdfile)
    for mol in mols:
        if mol is not None:
            writer.write(mol)
    writer.close()

# convert from SDF to CSV file
sdfile_in = Chem.SDMolSupplier(sdfile)
csvfile = open(os.path.join(tmp_dir, "input.csv"), 'w')
SDFToCSV.Convert(sdfile_in, csvfile, keyCol=None, stopAfter=- 1, includeChirality=False, smilesFrom='')
csvfile.close()

# load saved model
mdl_ckpt = os.path.join(root, "..", "..", "checkpoints", "model.joblib")
model = joblib.load(mdl_ckpt)

df = pd.read_csv(os.path.join(tmp_dir, "input.csv"))
X = list(df["SMILES"])
y = model.predict_proba(X)

y = y[:,1]

# write output in a .csv files
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["activity10"])
    for i in range(len(y)):
        writer.writerow([y[i]])
