# DeephERG prediction
## Model identifiers
- Slug: deepherg
- Ersilia ID: eos30gr
- Tags: hERG, toxicity, ML

# Model description
Tool for clasifying hERG blockers, using a re-trained version of the deephERG model. 
- Input: Compound
- Output: Probability of hERG blockade (80%) 
- Model type: Classification
- Training set: 12000
- Mode of training: Retrained.

# Source code
Cai C, Guo P, Zhou Y, et al. Deep Learning-Based Prediction of Drug-Induced Cardiotoxicity. J Chem Inf Model 59, 3(2019). https://doi.org/10.1021/acs.jcim.8b00769

- Code: https://github.com/ChengF-Lab/deephERG/blob/master/deephERG.py
- Checkpoints: N/A

# License
No license available. 

# History 
- Data was downloaded on 7/13/2022 from deephERG [GitHub] (https://github.com/ChengF-Lab/deephERG/blob/master/deephERG.py). 
- We have trained on a single task (blockade 80%) using KerasTuner. Original authors used DeepChem's hyperparameter optimizer.

# About us
The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission or [volunteer](https://www.ersilia.io/volunteer) with us!
