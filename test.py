from pandas import RangeIndex
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from sklearn.linear_model import LinearRegression

import pandas

from qsardb import QsarDBPipeline

dataset = pandas.read_csv("esol.csv")
dataset.index = RangeIndex(start = 1, stop = len(dataset) + 1)

X = dataset[["SMILES", "Name"]]
y = dataset["logS"]

pipeline = QsarDBPipeline([
	("parser", SmilesToMolTransformer()),
	("descriptorizer", MolecularDescriptorTransformer(desc_list = ["MolLogP", "MolWt", "NumRotatableBonds", "NumAromaticRings"])),
	("regressor", LinearRegression())
])
pipeline.fit(X, y)

print("R2 = %.3f" % pipeline._estimator_steps().score(pipeline._encoded_descriptors(), y))

pipeline.export("ESOL.qdb")
