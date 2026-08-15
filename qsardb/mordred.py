from mordred import Calculator, descriptors
from scikit_mol.conversions import SmilesToMolTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import importlib.metadata
import numpy
import pandas

from qsardb.rdkit import RDKitPipeline

class MordredPipeline(RDKitPipeline):
	pass

class MordredDescriptorTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, ignore_3D = True, n_jobs = 1):
		self.ignore_3D = ignore_3D
		self.n_jobs = n_jobs

	def fit(self, X, y = None):
		self.descriptor_names_ = [str(descriptor) for descriptor in self._calculator().descriptors]
		return self

	def transform(self, X):
		molecules = X.iloc[:, 0] if isinstance(X, pandas.DataFrame) else X
		values = self._calculator().pandas(list(molecules), quiet = True, nproc = self.n_jobs)
		values = values.fill_missing().astype("float64")
		values.index = getattr(X, "index", None)
		return values

	def get_feature_names_out(self, input_features = None):
		return numpy.asarray(self.descriptor_names_)

	def _calculator(self):
		return Calculator(descriptors, ignore_3D = self.ignore_3D)

def make_mordred_pipeline(n_jobs = 1):
	return MordredPipeline([
		("parser", SmilesToMolTransformer()),
		("descriptorizer", MordredDescriptorTransformer(n_jobs = n_jobs))
	])

def mordred_application():
	return "Mordred %s" % importlib.metadata.version("mordredcommunity")
