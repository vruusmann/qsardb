from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer

import rdkit

from qsardb.pipeline import DescriptorPipeline

class RDKitPipeline(DescriptorPipeline):

	def application_name(self):
		return rdkit_application()

def make_rdkit_pipeline(desc_list):
	return RDKitPipeline([
		("parser", SmilesToMolTransformer()),
		("descriptorizer", MolecularDescriptorTransformer(desc_list = desc_list))
	])

def rdkit_application():
	return "RDKit %s" % rdkit.__version__
