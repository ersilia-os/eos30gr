import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import lazyqsar as lq
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(os.path.join(root, "data", "training_set.csv"))
test = pd.read_csv(os.path.join(root, "data", "test_set.csv"))
val = pd.read_csv(os.path.join(root, "data", "validation_set.csv"))

#Actives are all molecules with Activity10
train_act = train[train["activity10"]==1]
test_act = test[train["activity10"]==1]
val_act = test[train["activity10"]==1]

#Inactives depend on the cutoff
cut_inact = [10,20,40,60,80,100]

for c in cut_inact:
    train_inact = train[train[f"activity{c}"]==0]
    test_inact = test[test[f"activity{c}"]==0]
    val_inact = val[val[f"activity{c}"]==0]
    X_train = train_act["Smiles"].tolist()+train_inact["Smiles"].tolist()
    y_train = [1 for i in range(len(train_act))]+[0 for i in range(len(train_inact))]
    print(len(train_act), len(train_inact), len(X_train), len(y_train))
    model = lq.ErsiliaBinaryClassifier(estimator_list=["rf"], time_budget_sec = 60)
    model.fit(X_train, y_train)
    X_test = test_act["Smiles"].tolist()+test_inact["Smiles"].tolist()
    y_test = [1 for i in range(len(test_act))]+[0 for i in range(len(test_inact))]
    test_proba1 = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, test_proba1)
    auroc = auc(fpr,tpr)
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auroc)
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(root, "figures", f"test_eosce_{c}_auroc.png"), dpi=300)
    X_val = val_act["Smiles"].tolist()+val_inact["Smiles"].tolist()
    y_val = [1 for i in range(len(val_act))]+[0 for i in range(len(val_inact))]
    val_proba1 = model.predict_proba(X_val)[:,1]
    fpr, tpr, _ = roc_curve(y_val, val_proba1)
    auroc = auc(fpr,tpr)
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auroc)
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(root, "figures", f"val_eosce_{c}_auroc.png"), dpi=300)