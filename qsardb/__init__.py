from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from xml.etree import ElementTree

import os
import pandas
import rdkit
import shutil
import sklearn
import tempfile
import zipfile

NAMESPACE = "http://www.qsardb.org/QDB"

ZIP_SUFFIXES = (".zip", ".qdb")

class QsarDBPipeline(Pipeline):

	def __init__(self, steps, memory = None, verbose = False):
		super().__init__(steps, memory = memory, verbose = verbose)
		self.set_output(transform = "pandas")
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

		if path.endswith(ZIP_SUFFIXES):
			directory = tempfile.mkdtemp()
			self._store(directory, name, description)
			self._store_zip(directory, path)
			shutil.rmtree(directory)
		else:
			self._store(path, name, description)
		return path

	def _check(self, X, y):
		if not isinstance(X, DataFrame) or len(X.columns) < 1:
			raise TypeError("X must be a DataFrame")
		if y is not None:
			if not isinstance(y, Series) or y.name is None:
				raise TypeError("y must be a named Series")
			if not X.index.equals(y.index):
				raise ValueError("X and y must share the same compound identifiers")

	def _record(self, type, X, y, descriptors):
		predictions = Series(self._estimator_steps().predict(descriptors), index = X.index, name = self.property_id)
		self.datasets[type] = {
			"structures" : X[X.columns[0]],
			"names" : self._select_names(X),
			"property" : y,
			"descriptors" : descriptors,
			"predictions" : predictions
		}
		return predictions

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
		return Pipeline(self.steps[:self._boundary()])

	def _estimator_steps(self):
		return Pipeline(self.steps[self._boundary():])

	def _boundary(self):
		positions = [position for position, (_, step) in enumerate(self.steps) if type(step).__module__.startswith("scikit_mol")]
		if not positions:
			raise ValueError("The pipeline must begin with scikit-mol steps")
		if positions != list(range(len(positions))):
			raise ValueError("The scikit-mol steps must precede all other steps")
		return len(positions)

	def _merge(self, key):
		parts = [dataset[key] for dataset in self.datasets.values() if dataset[key] is not None]
		merged = pandas.concat(parts)
		return merged[~merged.index.duplicated(keep = "last")].sort_index()

	def _store(self, directory, name, description):
		if os.path.exists(directory):
			shutil.rmtree(directory)
		os.makedirs(directory)

		self._store_xml(os.path.join(directory, "archive.xml"), "Archive", [{"Name" : name, "Description" : description}])

		structures = self._merge("structures")
		names = self._merge("names") if any(dataset["names"] is not None for dataset in self.datasets.values()) else None
		compounds = [{"Id" : str(id), "Name" : None if names is None else names.get(id), "Cargos" : "daylight-smiles"} for id in structures.index]
		self._store_registry(directory, "compounds", "CompoundRegistry", "Compound", compounds)
		for id, smiles in structures.items():
			self._store_cargo(directory, "compounds", str(id), "daylight-smiles", smiles)

		property = self._merge("property")
		self._store_registry(directory, "properties", "PropertyRegistry", "Property", [{"Id" : self.property_id, "Name" : self.property_id, "Cargos" : "values"}])
		self._store_cargo(directory, "properties", self.property_id, "values", format_values(self.property_id, property))

		descriptors = self._merge("descriptors")
		self._store_registry(directory, "descriptors", "DescriptorRegistry", "Descriptor", [{"Id" : id, "Name" : id, "Cargos" : "values", "Application" : rdkit_application()} for id in descriptors.columns])
		for id in descriptors.columns:
			self._store_cargo(directory, "descriptors", id, "values", format_values(id, descriptors[id]))

		self._store_registry(directory, "models", "ModelRegistry", "Model", [{"Id" : "1", "Name" : name, "Cargos" : "pmml", "PropertyId" : self.property_id}])
		self._store_cargo(directory, "models", "1", "pmml", self._format_pmml(descriptors.columns))

		predictions = []
		for position, (type, dataset) in enumerate(self.datasets.items(), start = 1):
			predictions.append({"Id" : str(position), "Name" : type.capitalize() + " set", "Cargos" : "values", "ModelId" : "1", "Type" : type, "Application" : sklearn_application()})
			self._store_cargo(directory, "predictions", str(position), "values", format_values(str(position), dataset["predictions"]))
		self._store_registry(directory, "predictions", "PredictionRegistry", "Prediction", predictions)

	def _store_registry(self, directory, type, registry_tag, container_tag, containers):
		self._store_xml(os.path.join(directory, type, type + ".xml"), registry_tag, containers, container_tag)

	def _store_xml(self, path, registry_tag, containers, container_tag = None):
		root = ElementTree.Element("{%s}%s" % (NAMESPACE, registry_tag))
		for attributes in containers:
			parent = root if container_tag is None else ElementTree.SubElement(root, "{%s}%s" % (NAMESPACE, container_tag))
			for tag, value in attributes.items():
				if value is not None:
					ElementTree.SubElement(parent, "{%s}%s" % (NAMESPACE, tag)).text = str(value)
		os.makedirs(os.path.dirname(path), exist_ok = True)
		tree = ElementTree.ElementTree(root)
		ElementTree.indent(tree)
		tree.write(path, encoding = "UTF-8", xml_declaration = True, default_namespace = NAMESPACE)

	def _store_cargo(self, directory, type, id, cargo_id, payload):
		os.makedirs(os.path.join(directory, type, id), exist_ok = True)
		with open(os.path.join(directory, type, id, cargo_id), "w", encoding = "UTF-8") as file:
			file.write(payload)

	def _store_zip(self, directory, path):
		with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
			for parent, _, names in os.walk(directory):
				for name in names:
					file_path = os.path.join(parent, name)
					archive.write(file_path, os.path.relpath(file_path, directory))

	def _format_pmml(self, descriptor_ids):
		pmml_pipeline = make_pmml_pipeline(self._estimator_steps(), active_fields = ["descriptors/" + id for id in descriptor_ids], target_fields = ["properties/" + self.property_id])
		directory = tempfile.mkdtemp()
		path = os.path.join(directory, "model.pmml")
		sklearn2pmml(pmml_pipeline, path)
		with open(path, "r", encoding = "UTF-8") as file:
			pmml = file.read()
		shutil.rmtree(directory)
		return pmml

def rdkit_application():
	return "RDKit %s" % rdkit.__version__

def sklearn_application():
	return "Scikit-Learn %s" % sklearn.__version__

def format_values(id, values):
	lines = ["Compound Id\t" + id]
	for compound_id, value in values.items():
		lines.append("%s\t%s" % (compound_id, round(float(value), 6)))
	return "\n".join(lines)
