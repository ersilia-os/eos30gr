import os
import csv
import sys
import tempfile
import numpy as np
from deepchem.models.optimizers import Adam
import deepchem as dc
import joblib

# current file directory
root = os.path.abspath(os.path.dirname(__file__))

# parse arguments
input_file = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])

# Extract SMILES strings from the input file
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    smiles = [r[1] for r in reader]

# Define the columns for the CSV data
columns = ['smiles', 'activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100']

# Create a temporary csv file
with tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='') as temp_file:
    writer = csv.writer(temp_file)
    writer.writerow(columns)  # Write header
    for sm in smiles:
        writer.writerow([sm] + [None] * (len(columns) - 1))

    temp_file_name = temp_file.name

# Define the featurizer used in training 
featurizer = dc.feat.CircularFingerprint(radius=2, size=2048)
loader = dc.data.CSVLoader(tasks=['activity10', 'activity20', 'activity40',
                                  'activity60', 'activity80', 'activity100'],
                                    feature_field="smiles", featurizer=featurizer)


# create dataset with from featurized input file 
dataset = loader.create_dataset(temp_file_name, shard_size=8192)

# The model
def model(model_dir, **model_params):
    n_layers = 3
    n_features = 2048
    tasks=['activity10', 'activity20', 'activity40', 'activity60', 'activity80', 'activity100']

    model = dc.models.MultitaskClassifier(
        n_tasks=len(tasks),
        n_features=n_features,
        layer_sizes=[200, 100, 50],
        dropouts=[.25] * n_layers,
        weight_init_stddevs=[.02] * n_layers,
        bias_init_consts=[1.] * n_layers,
        batch_size=256,
        optimizer=Adam(learning_rate=0.001),
        penalty_type="l2"
    )

    return model


# Restore the model from the checkpoint and run prediction on decoy threshold of 80 μM
mdl_ckpt = os.path.join(root, "..", "..", "checkpoints")
model = model(mdl_ckpt)
model.restore(model_dir=mdl_ckpt, checkpoint=None)
y = model.predict(dataset)[:, 4, 1]


# write output in a .csv files
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["activity_80"])
    for i in range(len(y)):
        writer.writerow([y[i]])
