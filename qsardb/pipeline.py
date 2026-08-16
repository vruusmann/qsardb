from pandas import DataFrame, Series
from rdkit import Chem
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

import importlib.metadata
import numpy
import os
import pandas
import pickle
import re
import shutil
import sklearn
import subprocess
import sys
import tempfile

from qsardb.core import QDB, format_values, parse_values

_PICKLE_PROTOCOL = 4

class DescriptorPipeline(Pipeline):

	def __init__(self, steps, memory = None, verbose = False):
		super().__init__(steps, memory = memory, verbose = verbose)
		self.set_output(transform = "pandas")

	def application_name(self):
		return None

	def descriptor_pipeline(self, names):
		return None

	def n_jobs(self):
		return getattr(self.steps[-1][1], "n_jobs", 1)

	def distill(self, names):
		distilled = self.descriptor_pipeline(names)
		if distilled is not None:
			return distilled
		return _distill(self.steps[-1][1], names)


	def get_feature_names_out(self, input_features = None):
		return numpy.asarray([str(name) for name in super().get_feature_names_out(input_features)], dtype = object)

	def applications_out(self):
		names = self.get_feature_names_out()
		application = self.application_name()
		if application is not None:
			return {name : application for name in names}
		applications = _applications_out(self.steps[-1][1])
		if len(applications) != len(names):
			raise ValueError("The pipeline reports %d applications for %d descriptors" % (len(applications), len(names)))
		return dict(zip(names, applications))

class QDBPipeline(Pipeline):

	def __init__(self, steps, memory = None, verbose = False):
		super().__init__(steps, memory = memory, verbose = verbose)
		self.datasets = []

	def fit(self, X, y = None, **fit_params):
		if self.datasets:
			raise ValueError("The pipeline has already been fitted, recording %s" % [dataset["type"] for dataset in self.datasets])
		self._check(X, y)
		descriptors = self._descriptor_steps().fit_transform(X[[X.columns[0]]])
		descriptors.index = X.index
		self.property_id = y.name
		self.datasets = []
		self._estimator_steps().fit(descriptors, y)
		self._record("training", X, y, descriptors)
		return self

	def validate(self, X, y = None, prediction_id = None):
		self._check(X, y)
		return self._record("validation", X, y, self._transform(X), prediction_id)

	def test(self, X, prediction_id = None):
		self._check(X, None)
		return self._record("testing", X, None, self._transform(X), prediction_id)

	def predict(self, X):
		descriptors = self._transform(X)
		return Series(self._estimator_steps().predict(descriptors), index = X.index, name = self.property_id)

	@classmethod
	def from_qdb(cls, qdb):
		models = {model["Id"] : model for model in qdb.containers["models"]}
		if len(models) != 1:
			raise ValueError("The archive holds %d models, select one of %s first" % (len(models), sorted(models)))
		model_id = list(models)[0]

		cargos = qdb.cargos["models"][model_id]
		if "pkl" not in cargos:
			raise ValueError("The model %s carries no pkl cargo, but %s" % (model_id, sorted(cargos)))

		pipeline = cls(pickle.loads(cargos["pkl"]).steps)
		pipeline.property_id = models[model_id]["PropertyId"]
		pipeline.datasets = pipeline._restore(qdb, model_id)
		return pipeline

	def _restore(self, qdb, model_id):
		structures = _restore_series(qdb.cargos["compounds"], "daylight-smiles")
		names = Series({container["Id"] : container.get("Name") for container in qdb.containers["compounds"]}, name = "Name")
		inchis = Series({container["Id"] : container.get("InChI") for container in qdb.containers["compounds"]}, name = "InChI")
		property = _restore_values(qdb.cargos["properties"][self.property_id]["values"])
		descriptors = DataFrame({container["Id"] : _restore_values(qdb.cargos["descriptors"][container["Id"]]["values"]) for container in qdb.containers["descriptors"]})

		datasets = []
		for container in qdb.containers["predictions"]:
			if container["ModelId"] != model_id:
				continue
			predictions = _restore_values(qdb.cargos["predictions"][container["Id"]]["values"])
			index = predictions.index
			datasets.append({
				"type" : container["Type"],
				"prediction_id" : container["Id"].split("-", 1)[-1],
				"structures" : structures[index],
				"names" : names[index] if names.notna().any() else None,
				"inchis" : inchis[index],
				"property" : property[index] if container["Type"] != "testing" else None,
				"descriptors" : descriptors.loc[index],
				"predictions" : predictions
			})
		return datasets

	def to_qdb(self, model_id = "1", name = None, description = None):
		if name is None:
			name = self.property_id

		qdb = QDB()
		qdb.files["requirements.txt"] = format_requirements(self)

		self._check_collisions()

		structures = self._merge("structures")
		inchis = self._merge("inchis")
		names = self._merge("names") if any(dataset["names"] is not None for dataset in self.datasets) else None
		for id, smiles in structures.items():
			qdb.add("compounds", {"Id" : str(id), "Name" : None if names is None else names.get(id), "InChI" : inchis[id]}, {"daylight-smiles" : smiles})

		property = self._merge("property")
		qdb.add("properties", {"Id" : self.property_id, "Name" : self.property_id}, {"values" : format_values(self.property_id, property)})

		descriptors = self._merge("descriptors")
		applications = self._descriptor_steps().applications_out()
		for id in descriptors.columns:
			cargos = {"values" : format_values(id, descriptors[id])}
			distilled = self._descriptor_steps().distill([id])
			if distilled is not None:
				cargos["pkl"] = pickle.dumps(distilled, protocol = _PICKLE_PROTOCOL)
			qdb.add("descriptors", {"Id" : id, "Name" : id, "Application" : applications.get(id)}, cargos)

		qdb.add("models", {"Id" : model_id, "Name" : name, "Description" : description, "PropertyId" : self.property_id}, {"pkl" : self._format_pickle(), "pmml" : self._format_pmml(descriptors.columns)})

		positions = {}
		for dataset in self.datasets:
			type = dataset["type"]
			positions[type] = positions.get(type, 0) + 1
			suffix = dataset["prediction_id"]
			if suffix is None:
				suffix = type if positions[type] == 1 else "%s-%d" % (type, positions[type])
			prediction_id = "%s-%s" % (model_id, suffix)
			qdb.add("predictions", {"Id" : prediction_id, "Name" : "%s, %s set" % (name, type), "ModelId" : model_id, "Type" : type, "Application" : sklearn_application()}, {"values" : format_values(prediction_id, dataset["predictions"])})

		return qdb

	def _check(self, X, y):
		if not isinstance(X, DataFrame) or len(X.columns) < 1:
			raise TypeError("X must be a DataFrame")
		if y is not None:
			if not isinstance(y, Series) or y.name is None:
				raise TypeError("y must be a named Series")
			if not X.index.equals(y.index):
				raise ValueError("X and y must share the same compound identifiers")

	def _record(self, type, X, y, descriptors, prediction_id = None):
		index = X.index.astype(str)
		X = X.set_axis(index)
		descriptors = descriptors.set_axis(index)
		if y is not None:
			y = y.set_axis(index)

		structures = X[X.columns[0]]
		inchis = format_inchis(structures)
		conflicting = sorted(inchis.groupby(level = 0).nunique().loc[lambda counts: counts > 1].index)
		if conflicting:
			raise ValueError("The %s set maps compound identifiers to more than one structure: %s" % (type, conflicting))
		predictions = Series(self._estimator_steps().predict(descriptors), index = X.index, name = self.property_id)
		self.datasets.append({
			"type" : type,
			"prediction_id" : prediction_id,
			"structures" : structures,
			"inchis" : inchis,
			"names" : self._select_names(X),
			"property" : y,
			"descriptors" : descriptors,
			"predictions" : predictions
		})
		return predictions

	def _check_collisions(self):
		inchis = pandas.concat([dataset["inchis"] for dataset in self.datasets])
		conflicting = sorted(inchis.groupby(level = 0).nunique().loc[lambda counts: counts > 1].index)
		if conflicting:
			raise ValueError("The datasets map compound identifiers to more than one structure: %s" % conflicting)

	def _transform(self, X):
		descriptors = self._descriptor_steps().transform(X[[X.columns[0]]])
		descriptors.index = X.index
		return descriptors

	def _select_names(self, X):
		for column in X.columns[1:]:
			if column.lower() == "name":
				return X[column]
		return None

	def _descriptor_steps(self):
		self._check_boundary()
		return self.steps[0][1]

	def _estimator_steps(self):
		self._check_boundary()
		return Pipeline(self.steps[1:])

	def _schema_step(self, steps):
		while isinstance(steps, Pipeline):
			steps = steps.steps[0][1]
		return steps

	def _check_boundary(self):
		if not isinstance(self.steps[0][1], DescriptorPipeline):
			raise ValueError("The first step must be a DescriptorPipeline")

	def training(self):
		for dataset in self.datasets:
			if dataset["type"] == "training":
				return dataset
		raise ValueError("The pipeline holds no training set")

	def _merge(self, key):
		parts = [dataset[key] for dataset in self.datasets if dataset[key] is not None]
		merged = pandas.concat(parts)
		return merged[~merged.index.duplicated(keep = "first")].sort_index()

	def used_descriptors(self):
		return _used_descriptors(self._format_pmml(self.training()["descriptors"].columns))

	def _format_pickle(self):
		distilled = self._descriptor_steps().distill(list(self.training()["descriptors"].columns))
		if distilled is None:
			distilled = self._descriptor_steps()
		else:
			distilled.fit(self.training()["structures"].to_frame())
		return pickle.dumps(Pipeline([("descriptors", distilled)] + list(self._estimator_steps().steps)), protocol = _PICKLE_PROTOCOL)

	def _format_pmml(self, descriptor_ids):
		active_fields = ["descriptors/" + id for id in descriptor_ids]
		estimator_steps = self._estimator_steps()
		schema_step = self._schema_step(estimator_steps)
		feature_names = getattr(schema_step, "feature_names_in_", None)

		if feature_names is not None:
			try:
				schema_step.feature_names_in_ = numpy.asarray(active_fields)
			except AttributeError:
				feature_names = None
		try:
			pmml_pipeline = make_pmml_pipeline(estimator_steps, active_fields = active_fields, target_fields = ["properties/" + self.property_id])
			directory = tempfile.mkdtemp()
			path = os.path.join(directory, "model.pmml")
			sklearn2pmml(pmml_pipeline, path)
			with open(path, "r", encoding = "UTF-8") as file:
				pmml = file.read()
			shutil.rmtree(directory)
		finally:
			if feature_names is not None:
				schema_step.feature_names_in_ = feature_names
		return pmml

def _distributions(names):
	mapping = importlib.metadata.packages_distributions()
	return {distribution.lower() for name in names for distribution in mapping.get(name, [])}

def _requires(distribution):
	names = set()
	for requirement in importlib.metadata.requires(distribution) or []:
		if "extra ==" not in requirement:
			names.add(requirement.split(";")[0].split()[0].split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower())
	return names

def _prune(distributions):
	required = set()
	for distribution in distributions:
		required.update(_requires(distribution))
	return distributions - required

_CAPTURE = "import pickle, sys; pickle.load(open(sys.argv[1], 'rb')); print(' '.join(sorted({name.split('.')[0] for name in sys.modules})))"

def _capture_modules(estimator):
	directory = tempfile.mkdtemp()
	path = os.path.join(directory, "estimator.pkl")
	with open(path, "wb") as file:
		pickle.dump(estimator, file)
	completed = subprocess.run([sys.executable, "-c", _CAPTURE, path], capture_output = True, text = True)
	shutil.rmtree(directory)
	if completed.returncode:
		raise ValueError("The fitted pipeline cannot be unpickled in a fresh interpreter, so the archive would not be executable. Every custom class it holds must be importable from a module rather than defined in a script. The loader reported: %s" % completed.stderr.strip().splitlines()[-1])
	return set(completed.stdout.split())

def format_requirements(estimator):
	distributions = _prune(_distributions(_capture_modules(estimator))) - {"qsardb", "pip", "setuptools"}
	return "\n".join("%s==%s" % (distribution, importlib.metadata.version(distribution)) for distribution in sorted(distributions)) + "\n"

def _restore_series(cargos, cargo_id):
	return Series({id : cargos[id][cargo_id].strip() for id in cargos})

def _restore_values(payload):
	values = {}
	for id, value in parse_values(payload).items():
		try:
			values[id] = float(value) if value is not None else float("nan")
		except ValueError:
			values[id] = value
	return Series(values)

def _used_descriptors(pmml):
	return {name[len("descriptors/"):] for name in re.findall(r"<DataField name=\"([^\"]+)\"", pmml) if name.startswith("descriptors/")}

def _distill(step, names):
	if isinstance(step, DescriptorPipeline):
		return step.distill(names)
	if isinstance(step, ColumnTransformer):
		branches = []
		for branch, transformer, columns in step.transformers_:
			if transformer in ("drop", "passthrough"):
				continue
			selected = [name for name in names if name in set(transformer.get_feature_names_out())]
			if selected:
				distilled = _distill(transformer, selected)
				if distilled is None:
					return None
				branches.append((branch, distilled, columns))
		if not branches:
			return None
		return DescriptorPipeline([("descriptorizer", ColumnTransformer(branches, verbose_feature_names_out = False))])
	if isinstance(step, Pipeline):
		return _distill(step.steps[-1][1], names)
	return None

def _applications_out(step):
	if isinstance(step, DescriptorPipeline):
		return list(step.applications_out().values())
	if isinstance(step, ColumnTransformer):
		applications = []
		for name, transformer, columns in step.transformers_:
			if transformer in ("drop", "passthrough"):
				continue
			applications.extend(_applications_out(transformer))
		return applications
	if isinstance(step, Pipeline):
		return _applications_out(step.steps[-1][1])
	return [None] * len(step.get_feature_names_out())

def sklearn_application():
	return "Scikit-Learn %s" % sklearn.__version__

def format_inchis(structures):
	return Series([Chem.MolToInchi(Chem.MolFromSmiles(smiles)) for smiles in structures], index = structures.index, name = "InChI")

