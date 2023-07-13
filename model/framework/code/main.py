import sys
import os
import csv
import tempfile
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from mol2vec import features
from standardiser import standardise

input_file = sys.argv[1]
output_file = sys.argv[2]

root = os.path.abspath(os.path.dirname(__file__))

tmp_dir = tempfile.mkdtemp("eos-")

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
writer = Chem.SDWriter(sdfile)
for mol in mols:
    if mol is not None:
        writer.write(mol)

m2vfile = os.path.join(tmp_dir, "m2v.csv")
m2v_ckpt = os.path.abspath(os.path.join(root, "..", "..", "checkpoints", "model_300dim.pkl"))

features.featurize(sdfile, m2vfile, m2v_ckpt, 1, uncommon=None)

mdl_ckpt = os.path.join(root, "..", "..", "checkpoints", "model.joblib")
model = joblib.load(mdl_ckpt)

df = pd.read_csv(m2vfile)
cols = [col for col in list(df.columns) if col.startswith("mol2vec-")]
X = np.array(df[cols])

assert (X.shape[0] == len(smiles))

y = model.predict(X)

with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["activity80"])
    for i in range(len(y)):
        writer.writerow([y[i][0]])
