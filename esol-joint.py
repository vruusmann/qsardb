from pandas import RangeIndex
from rdkit import RDLogger
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import collections
import pandas
import time

from qsardb import DescriptorPipeline, QDBPipeline
from qsardb.mordred import make_mordred_pipeline
from qsardb.rdkit import make_rdkit_pipeline

RDLogger.DisableLog("rdApp.*")

SAMPLE = 200

dataset = pandas.read_csv("esol.csv").sample(SAMPLE, random_state = 13)
dataset.index = RangeIndex(start = 1, stop = len(dataset) + 1)

X = dataset[["SMILES", "Name"]]
y = dataset["logS"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 13)

descriptors = DescriptorPipeline([
	("descriptorizer", ColumnTransformer([
		("rdkit", make_rdkit_pipeline(["MolLogP", "MolWt", "NumRotatableBonds", "NumAromaticRings", "HeavyAtomCount"]), [0]),
		("mordred", make_mordred_pipeline(), [0])
	], verbose_feature_names_out = False))
])

model = XGBRegressor(max_depth = 2, n_estimators = 12, learning_rate = 0.3, random_state = 13)

pipeline = QDBPipeline([
	("descriptors", descriptors),
	("model", model)
])

start = time.time()
pipeline.fit(X_train, y_train)
validation = pipeline.validate(X_valid, y_valid)
training = pipeline.datasets["training"]["predictions"]
print("fitted in %.0f s" % (time.time() - start))

print("Training R2 = %.3f" % r2_score(y_train, training))
print("Validation R2 = %.3f" % r2_score(y_valid, validation))

applications = pipeline._descriptor_steps().applications_out()
print()
print("descriptors by application: %s" % dict(collections.Counter(applications.values())))

booster = model.get_booster()
booster.feature_names = list(applications)
gain = booster.get_score(importance_type = "gain")
total = sum(gain.values())
print()
print("nodes: %d" % sum(len(tree.strip().splitlines()) for tree in booster.get_dump()))
print()
print("descriptors used by the model:")
for name, value in sorted(gain.items(), key = lambda item: -item[1]):
	print("\t%-24s %5.1f%%  %s" % (name, 100 * value / total, applications.get(name)))

pipeline.export("ESOL-joint.qdb.zip", name = "logS, RDKit and Mordred descriptors combined")
