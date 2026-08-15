from mordred import Calculator, descriptors
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import NoFitNeededMixin
from sklearn.base import BaseEstimator, TransformerMixin

import importlib.metadata
import numpy
import pandas

from qsardb.pipeline import DescriptorPipeline

class MordredPipeline(DescriptorPipeline):

	def application_name(self):
		return mordred_application()

	def descriptor_pipeline(self, names):
		return make_mordred_pipeline(names = list(names), n_jobs = self.n_jobs())

class MordredDescriptorTransformer(BaseEstimator, NoFitNeededMixin, TransformerMixin):

	def __init__(self, names = None, ignore_3D = True, n_jobs = 1):
		self.names = names
		self.ignore_3D = ignore_3D
		self.n_jobs = n_jobs

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		molecules = X.iloc[:, 0] if isinstance(X, pandas.DataFrame) else X
		values = self._calculator().pandas(list(molecules), quiet = True, nproc = self.n_jobs)
		values = values.fill_missing().astype("float64")
		values.index = getattr(X, "index", None)
		return values

	def get_feature_names_out(self, input_features = None):
		return numpy.asarray([str(descriptor) for descriptor in self._calculator().descriptors])

	def _calculator(self):
		available = Calculator(descriptors, ignore_3D = self.ignore_3D)
		if self.names is None:
			return available
		selected = {str(descriptor) : descriptor for descriptor in available.descriptors}
		return Calculator([selected[name] for name in self.names], ignore_3D = self.ignore_3D)

def make_mordred_pipeline(names = None, n_jobs = 1):
	return MordredPipeline([
		("parser", SmilesToMolTransformer()),
		("descriptorizer", MordredDescriptorTransformer(names = names, n_jobs = n_jobs))
	])

def mordred_application():
	return "Mordred %s" % importlib.metadata.version("mordredcommunity")
