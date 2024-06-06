import sys
import os
import csv
import numpy as np
import joblib


root = os.path.abspath(os.path.dirname(__file__)) 

input_file = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])

with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    X = [r[0] for r in reader]


mdl_ckpt = os.path.join(root, "..", "..", "checkpoints", f"model_80.joblib")
model = joblib.load(mdl_ckpt)
y = model.predict_proba(X)[:,1]

# write output in a .csv files
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["activity_80"])
    for i in range(len(y)):
        writer.writerow([y[i]])
