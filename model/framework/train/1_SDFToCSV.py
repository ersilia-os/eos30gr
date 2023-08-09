from rdkit import Chem
from rdkit.Chem.ChemUtils import SDFToCSV

# input file format: *.sdf
# output file format: *.csv

# test
test_out = open('output_file/testset.csv', 'w' )
test_in = Chem.SDMolSupplier('input_file/testset.sdf')
SDFToCSV.Convert(test_in, test_out, keyCol=None, stopAfter=- 1, includeChirality=False, smilesFrom='')
test_out.close()
# validate
vldt_out = open('output_file/validationset.csv', 'w' )
vldt_in = Chem.SDMolSupplier('input_file/validationset.sdf')
SDFToCSV.Convert(vldt_in, vldt_out, keyCol=None, stopAfter=- 1, includeChirality=False, smilesFrom='')
vldt_out.close()
# train
train_out = open('output_file/trainingset.csv', 'w')
train_in = Chem.SDMolSupplier('input_file/trainingset.sdf')
SDFToCSV.Convert(train_in, train_out, keyCol=None, stopAfter=- 1, includeChirality=False, smilesFrom='')
train_out.close()