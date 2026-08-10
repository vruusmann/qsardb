from setuptools import setup

setup(
	name = "qsardb",
	version = "0.0.1",
	description = "QsarDB Python package",
	packages = ["qsardb"],
	install_requires = ["numpy", "pandas", "rdkit", "scikit-learn", "scikit-mol", "sklearn2pmml"]
)
