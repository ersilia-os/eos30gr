# Model pretrained parameters

This model has been retrained using the author's original training set. We have performed a validation of the results using the test and validation sets provided with the negative datasets defined at the following cut-offs: 10, 20, 40, 60, 80, 100 uM, replicating the author's work. 
The authors propose the following final model: We thus selected the multi-task DNN model building on MOE+Mol2vec descriptors with decoy threshold value of 80 μM as deephERG for further evaluation.

We are therefore providing a model trained on the 80uM cut-off data for negatives and 10uM cut-off data for positives.