from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer

import rdkit

from qsardb.pipeline import DescriptorPipeline

class RDKitPipeline(DescriptorPipeline):

	def application_name(self):
		return rdkit_application()

	def descriptor_pipeline(self, names):
		return make_rdkit_pipeline(names = list(names), n_jobs = self.n_jobs())

def make_rdkit_pipeline(names = None, n_jobs = 1):
	return RDKitPipeline([
		("parser", SmilesToMolTransformer()),
		("descriptorizer", MolecularDescriptorTransformer(desc_list = names, n_jobs = n_jobs))
	])

def rdkit_application():
	return "RDKit %s" % rdkit.__version__
