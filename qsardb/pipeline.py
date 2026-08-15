from pandas import DataFrame, Series
from rdkit import Chem
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

import numpy
import os
import pandas
import shutil
import sklearn
import tempfile

from qsardb.core import QDB, format_values

class DescriptorPipeline(Pipeline):

	def __init__(self, steps, memory = None, verbose = False):
		super().__init__(steps, memory = memory, verbose = verbose)
		self.set_output(transform = "pandas")

	def application_name(self):
		return None

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
		self.datasets = {}

	def fit(self, X, y = None, **fit_params):
		self._check(X, y)
		descriptors = self._descriptor_steps().fit_transform(X[[X.columns[0]]])
		descriptors.index = X.index
		self.property_id = y.name
		self.datasets = {}
		self._estimator_steps().fit(descriptors, y)
		self._record("training", X, y, descriptors)
		return self

	def validate(self, X, y = None):
		self._check(X, y)
		return self._record("validation", X, y, self._transform(X))

	def test(self, X):
		self._check(X, None)
		return self._record("testing", X, None, self._transform(X))

	def predict(self, X):
		descriptors = self._transform(X)
		return Series(self._estimator_steps().predict(descriptors), index = X.index, name = self.property_id)

	def export(self, path, name = None, description = None):
		if name is None:
			name = self.property_id

		qdb = QDB(name, description)

		self._check_collisions()

		structures = self._merge("structures")
		inchis = self._merge("inchis")
		names = self._merge("names") if any(dataset["names"] is not None for dataset in self.datasets.values()) else None
		for id, smiles in structures.items():
			qdb.add("compounds", {"Id" : str(id), "Name" : None if names is None else names.get(id), "InChI" : inchis[id]}, {"daylight-smiles" : smiles})

		property = self._merge("property")
		qdb.add("properties", {"Id" : self.property_id, "Name" : self.property_id}, {"values" : format_values(self.property_id, property)})

		descriptors = self._merge("descriptors")
		applications = self._descriptor_steps().applications_out()
		for id in descriptors.columns:
			qdb.add("descriptors", {"Id" : id, "Name" : id, "Application" : applications.get(id)}, {"values" : format_values(id, descriptors[id])})

		qdb.add("models", {"Id" : "1", "Name" : name, "PropertyId" : self.property_id}, {"pmml" : self._format_pmml(descriptors.columns)})

		for position, (type, dataset) in enumerate(self.datasets.items(), start = 1):
			qdb.add("predictions", {"Id" : str(position), "Name" : type.capitalize() + " set", "ModelId" : "1", "Type" : type, "Application" : sklearn_application()}, {"values" : format_values(type.capitalize() + " set", dataset["predictions"])})

		return qdb.store(path)

	def _check(self, X, y):
		if not isinstance(X, DataFrame) or len(X.columns) < 1:
			raise TypeError("X must be a DataFrame")
		if y is not None:
			if not isinstance(y, Series) or y.name is None:
				raise TypeError("y must be a named Series")
			if not X.index.equals(y.index):
				raise ValueError("X and y must share the same compound identifiers")

	def _record(self, type, X, y, descriptors):
		structures = X[X.columns[0]]
		inchis = format_inchis(structures)
		conflicting = sorted(inchis.groupby(level = 0).nunique().loc[lambda counts: counts > 1].index)
		if conflicting:
			raise ValueError("The %s set maps compound identifiers to more than one structure: %s" % (type, conflicting))
		predictions = Series(self._estimator_steps().predict(descriptors), index = X.index, name = self.property_id)
		self.datasets[type] = {
			"structures" : structures,
			"inchis" : inchis,
			"names" : self._select_names(X),
			"property" : y,
			"descriptors" : descriptors,
			"predictions" : predictions
		}
		return predictions

	def _check_collisions(self):
		inchis = pandas.concat([dataset["inchis"] for dataset in self.datasets.values()])
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

	def _merge(self, key):
		parts = [dataset[key] for dataset in self.datasets.values() if dataset[key] is not None]
		merged = pandas.concat(parts)
		return merged[~merged.index.duplicated(keep = "first")].sort_index()

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

