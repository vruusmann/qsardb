from pandas import RangeIndex
from rdkit import RDLogger
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import pandas

from qsardb import QDBPipeline

RDLogger.DisableLog("rdApp.*")

dataset = pandas.read_csv("esol.csv")
dataset.index = RangeIndex(start = 1, stop = len(dataset) + 1)

X = dataset[["SMILES", "Name"]]
y = dataset["logS"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 13)

pipeline = QDBPipeline([
	("parser", SmilesToMolTransformer()),
	("descriptorizer", MolecularDescriptorTransformer(desc_list = ["MolLogP", "MolWt", "NumRotatableBonds", "NumAromaticRings"])),
	("regressor", LinearRegression())
])
training = pipeline.fit(X_train, y_train)
validation = pipeline.validate(X_valid, y_valid)

print("Training R2 = %.3f" % r2_score(y_train, pipeline.predict(X_train)))
print("Validation R2 = %.3f" % r2_score(y_valid, validation))

pipeline.export("ESOL.qdb")
