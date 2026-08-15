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

RDKIT_NAMES = ["MolLogP", "MolWt", "NumRotatableBonds", "NumAromaticRings", "HeavyAtomCount"]

descriptors = DescriptorPipeline([
	("descriptorizer", ColumnTransformer([
		("rdkit", make_rdkit_pipeline(RDKIT_NAMES), [0]),
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
used = pipeline.used_descriptors()
print("fitted in %.0f s, %d of %d descriptors used" % (time.time() - start, len(used), pipeline.datasets["training"]["descriptors"].shape[1]))

descriptors = DescriptorPipeline([
	("descriptorizer", ColumnTransformer([
		("rdkit", make_rdkit_pipeline([name for name in RDKIT_NAMES if name in used]), [0]),
		("mordred", make_mordred_pipeline(names = sorted(used - set(RDKIT_NAMES))), [0])
	], verbose_feature_names_out = False))
])

pipeline = QDBPipeline([
	("descriptors", descriptors),
	("model", model)
])
pipeline.fit(X_train, y_train)
validation = pipeline.validate(X_valid, y_valid)
training = pipeline.datasets["training"]["predictions"]

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

pipeline.to_qdb(name = "logS, RDKit and Mordred descriptors combined").store("ESOL-joint.qdb.zip")
