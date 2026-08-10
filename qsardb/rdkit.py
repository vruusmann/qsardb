from pandas import Series
from rdkit import Chem
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from sklearn.pipeline import Pipeline

import rdkit

class RDKitPipeline(Pipeline):

	def __init__(self, steps, memory = None, verbose = False):
		super().__init__(steps, memory = memory, verbose = verbose)
		self.set_output(transform = "pandas")

def make_rdkit_pipeline(desc_list):
	return RDKitPipeline([
		("parser", SmilesToMolTransformer()),
		("descriptorizer", MolecularDescriptorTransformer(desc_list = desc_list))
	])

def rdkit_application():
	return "RDKit %s" % rdkit.__version__

def format_inchis(structures):
	return Series([Chem.MolToInchi(Chem.MolFromSmiles(smiles)) for smiles in structures], index = structures.index, name = "InChI")
