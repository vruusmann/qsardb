from pandas import RangeIndex
from rdkit import RDLogger
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn2pmml.decoration import Alias
from sklearn2pmml.preprocessing import ExpressionTransformer

import pandas

from qsardb import QDBPipeline
from qsardb.rdkit import make_rdkit_pipeline

RDLogger.DisableLog("rdApp.*")

dataset = pandas.read_csv("esol.csv")
dataset.index = RangeIndex(start = 1, stop = len(dataset) + 1)

X = dataset[["SMILES", "Name"]]
y = dataset["logS"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 13)

rdkit_pipeline = make_rdkit_pipeline(["MolLogP", "MolWt", "NumRotatableBonds", "NumAromaticRings", "HeavyAtomCount"])

sklearn_pipeline = Pipeline([
	("featurizer", ColumnTransformer([
		("descriptors", "passthrough", [0, 1, 2]),
		("aromatic_proportion", Alias(ExpressionTransformer("6 * X[0] / X[1]"), name = "AromaticProportion"), [3, 4])
	])),
	("scaler", StandardScaler()),
	("regressor", LassoCV(random_state = 13))
])

pipeline = QDBPipeline([
	("rdkit", rdkit_pipeline),
	("sklearn", sklearn_pipeline)
])
training = pipeline.fit(X_train, y_train)
validation = pipeline.validate(X_valid, y_valid)

print("Training R2 = %.3f" % r2_score(y_train, pipeline.predict(X_train)))
print("Validation R2 = %.3f" % r2_score(y_valid, validation))

pipeline.to_qdb(name = "logS, RDKit descriptors").update(name = "ESOL aqueous solubility").store("ESOL.qdb.zip")
