# Model pretrained parameters

This model has been retrained using the author's original training set and approach without mol2vec + MOE featurization. We have performed a validation of the results using the test and validation sets provided with the negative datasets defined at the following cut-offs: 10, 20, 40, 60, 80, 100 uM. 
The authors propose the following final model: We thus selected the multi-task DNN model with decoy threshold value of 80 μM as deephERG for further evaluation.

We are therefore providing a model trained on the 80uM cut-off data for negatives and 10uM cut-off data for positives.


| Metric | Train Score | Validation Score | Train Per Task | Validation Per Task |
|-----------------|-----------------|-----------------| -----------------|-----------------|                                                                                                                                                                  
| ROC AUC | mean-roc_auc_score: 0.88 | mean-roc_auc_score: 0.93 | 0.88, 0.88, 0.88, 0.88, 0.88, 0.88 | 0.93, 0.93, 0.93, 0.93, 0.93, 0.93 |
| Accuracy | mean-accuracy_score: 0.74 | mean-accuracy_score: 0.82 | 0.76, 0.74, 0.75, 0.74, 0.74, 0.73 | 0.83, 0.81, 0.82, 0.81, 0.8, 0.82 |
| Matthews | mean-matthews_corrcoef: 0.56| mean-matthews_corrcoef: 0.65 |0.57, 0.56, 0.57, 0.55, 0.55, 0.55 | 0.67, 0.64, 0.66, 0.64, 0.63, 0.66 |
