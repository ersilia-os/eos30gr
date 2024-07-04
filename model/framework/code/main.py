import sys
import os
import csv
import numpy as np
import deepchem as dc
import joblib
from ..train.train import model_builder


root = os.path.abspath(os.path.dirname(__file__)) 

input_file = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])

with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    X = [r[0] for r in reader]


# Restore the model from the checkpoint and run prediction on decoy threshold of 80 μM
mdl_ckpt = os.path.join(root, "..", "..", "checkpoints")
model = model_builder(mdl_ckpt)
model.restore(model_dir=mdl_ckpt, checkpoint=None)
y = model.predict(X)[:, 4, 1]


# write output in a .csv files
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["activity_80"])
    for i in range(len(y)):
        writer.writerow([y[i]])
