from setuptools import setup

setup(
	name = "qsardb",
	version = "0.0.1",
	description = "Creating QsarDB archives",
	packages = ["qsardb"],
	install_requires = ["pandas", "scikit-learn", "scikit-mol", "sklearn2pmml"]
)
