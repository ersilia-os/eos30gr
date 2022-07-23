from mol2vec import features

# input file format: *.sdf
# featurization
features.featurize('input_file/testset.sdf', 'output_file/testset_mol2vec.csv', 'input_file/model_300dim.pkl', 1, uncommon=None)
features.featurize('input_file/validationset.sdf', 'output_file/validationset_mol2vec.csv', 'input_file/model_300dim.pkl', 1, uncommon=None)
features.featurize('input_file/trainingset.sdf', 'output_file/trainingset_mol2vec.csv', 'input_file/model_300dim.pkl', 1, uncommon=None)