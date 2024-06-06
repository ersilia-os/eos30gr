import lazyqsar as lq
import os
import pandas as pd
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(os.path.join(root, "data", "training_set.csv"))
test = pd.read_csv(os.path.join(root, "data", "test_set.csv"))
val = pd.read_csv(os.path.join(root, "data", "validation_set.csv"))
all = pd.concat([train, val, test], ignore_index=True)

c = 80 #as described in the publication

df = all[["No.","Smiles", "inchikeys", f"activity{c}"]]
print(df.shape)
df = df[~df[f"activity{c}"].isna()]
print(df.shape)
df = df.drop_duplicates(subset=["inchikeys"])
print(df.shape)
# Set X and Y
X_train = list(df["Smiles"])
y_train = list(df[f"activity{c}"])

model = lq.MorganBinaryClassifier(estimator_list=["rf"], time_budget_sec = 600)
model.fit(X_train, y_train)
model.save(os.path.join(root, "..", "..", "checkpoints", f"model_{c}.joblib"))