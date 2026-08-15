from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer

from qsardb.pipeline import DescriptorPipeline

class RDKitPipeline(DescriptorPipeline):
	pass

def make_rdkit_pipeline(desc_list):
	return RDKitPipeline([
		("parser", SmilesToMolTransformer()),
		("descriptorizer", MolecularDescriptorTransformer(desc_list = desc_list))
	])
