import random
import pandas as pd
import numpy as np
import deepchem as dc
import tensorflow as tf


from deepchem.models import MultitaskClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from deepchem.models.optimizers import Adam
from deepchem.utils.evaluate import Evaluator


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

featurizer = dc.feat.CircularFingerprint(radius=2, size=1024)
loader = dc.data.CSVLoader(tasks=['activity10', 'activity20', 'activity40',
                                  'activity60', 'activity80', 'activity100'],
                                    smiles_field="Smiles", featurizer=featurizer)

# Creating  deepchem datasets
dataset_train = loader.create_dataset('/data/training_set.csv')
dataset_test = loader.create_dataset('/data/test_set.csv')
dataset_validation = loader.create_dataset('/data/validation_set.csv')


# Utilizing the balancing transformer utility in deepchem to ensure dataset balance
def apply_transformer(dataset):
    print("About to transform data")
    transformer = dc.trans.BalancingTransformer(dataset=dataset)
    return transformer.transform(dataset), transformer

def transformer(dataset_train, dataset_test, dataset_validation):
    dataset_train, transformer_train = apply_transformer(dataset_train)
    dataset_test, transformer_test = apply_transformer(dataset_test)
    dataset_validation, transformer_validation = apply_transformer(dataset_validation)

    return (dataset_train, dataset_test, dataset_validation, 
            transformer_train, transformer_test, transformer_validation)
    
train_dataset, test_dataset, validation_dataset, transformer_train, transformer_test, transformer_validation = transformer(
    dataset_train, dataset_test, dataset_validation)
    
    
# Defining evaluation metrics

roc_auc_metrics = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
accuracy_metrics = dc.metrics.Metric(dc.metrics.accuracy_score, np.mean)
matthews_metrics = dc.metrics.Metric(dc.metrics.matthews_corrcoef, np.mean)


# Constructing a TensorFlow-based DeepChem MultitaskClassifier model

def model_builder(model_dir, **model_params):
    n_layers = 3
    n_features = train_dataset.X.shape[1]
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
 

# Train the model   
print("Starting Multi-Task-DNN")

shard_size = 2000
num_trials = 1
all_results = []


for trial in range(num_trials):
    print("Starting trial %d" % trial)
    optimizer = dc.hyper.GridHyperparamOpt(model_builder)
    params_dict = {
        "activation": ["relu"],
        "momentum": [.9],
        "init": ["glorot_uniform"],
        "learning_rate": [1e-3],
        "decay": [.0004],
        "nb_epoch": [20],
        "nesterov": [False],
        "nb_layers": [3],
        "batchnorm": [False],
        "penalty": [0.],
     }
    transformers = []
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
    params_dict, train_dataset, test_dataset, roc_auc_metrics, transformers)
    
    
# save model
best_model.save_checkpoint(model_dir='/content', max_checkpoints_to_keep=1)



# Evaluating model
def result(best_model, dataset_train, dataset_validation):
    print("Evaluating models")

    train_roc_auc_score, train_pertask_roc_auc_score = best_model.evaluate(
        dataset_train, [roc_auc_metrics], per_task_metrics=True)
    validation_roc_auc_score, validation_pertask_roc_auc_score = best_model.evaluate(
        dataset_validation, [roc_auc_metrics], per_task_metrics=True)

    train_accuracy_score, train_pertask_accuracy_score = best_model.evaluate(
        dataset_train, [accuracy_metrics], per_task_metrics=True)
    validation_accuracy_score, validation_pertask_accuracy_score = best_model.evaluate(
        dataset_validation, [accuracy_metrics], per_task_metrics=True)

    train_matthews_score, train_pertask_matthews_score = best_model.evaluate(
        dataset_train, [matthews_metrics], per_task_metrics=True)
    validation_matthews_score, validation_pertask_matthews_score = best_model.evaluate(
        dataset_validation, [matthews_metrics], per_task_metrics=True)

    table = PrettyTable()
    table.field_names = ["Metric", "Train Score", "Validation Score", "Train Per Task", "Validation Per Task"]

    table.add_row(["ROC AUC", train_roc_auc_score, validation_roc_auc_score,
                   train_pertask_roc_auc_score, validation_pertask_roc_auc_score])
    table.add_row(["Accuracy", train_accuracy_score, validation_accuracy_score,
                   train_pertask_accuracy_score, validation_pertask_accuracy_score])
    table.add_row(["Matthews", train_matthews_score, validation_matthews_score, 
                   train_pertask_matthews_score, validation_pertask_matthews_score])

    print(table)

    with open("/content/table.txt", "w") as file:
      file.write(table.get_string())

result(best_model, dataset_train, dataset_validation)



# Get predicted probabilities on the test datasets for different threshold of decoys (10 μM, 20 μM, 40 μM, 60 μM, 80 μM and 100 μM)

y_pred_proba = best_model.predict(dataset_test)
y_pred_proba_class = y_pred_proba[:, :, 1]

# Creating a DataFrame
y_pred = pd.DataFrame(y_pred_proba_class)
y_true = pd.DataFrame(dataset_test.y)
n_samples, n_classes = y_pred_proba_class.shape[0], y_pred_proba_class.shape[1]

custom_labels = ["10 μM", "20 μM", "40 μM", "60 μM", "80 μM", "100 μM"]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[i], y_pred[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class

plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='{0} (roc_auc_score = {1:0.3f})'.format(custom_labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for different threshold decoys on test sets')
plt.savefig('/content/all_classes.png')
plt.legend(loc="lower right")
plt.show()
