from tdc.single_pred import Tox
from sklearn.metrics import roc_curve, auc
import lazyqsar as lq
from rdkit import Chem
import pandas as pd
import numpy as np
import joblib
import shutil
from sklearn.metrics import roc_curve, auc

# Read in sets, filter out rows with null "activity10" vals
train = pd.read_csv('output_file/trainingset.csv')
train= train[~train["activity10"].isna()]
validate = pd.read_csv('output_file/validationset.csv')
validate = validate[~validate["activity10"].isna()]
test = pd.read_csv('output_file/testset.csv')
test = test[~test["activity10"].isna()]

df = pd.concat([train, validate, test], ignore_index=True)

# Set X and Y
X_train = list(df["SMILES"])
y_train = list(df["activity10"])

model = lq.MorganBinaryClassifier(estimator_list=["rf"], time_budget_sec = 2000)
model.fit(X_train, y_train)
model.save('../../checkpoints/model.joblib')