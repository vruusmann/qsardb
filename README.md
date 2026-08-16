# QsarDB Python package

A pure Python implementation of the QsarDB (QDB) archive format, together with a Scikit-Learn pipeline that trains a model and exports it as an archive.
Exported archives are executable: the model and the descriptors that feed it are stored as pickles alongside the PMML, so a loaded archive can score a new structure.

Descriptor calculation is supported out of the box for RDKit and Mordred, and the same pattern extends the package to any other Python descriptor library.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [QsarDB archives](#qsardb-archives)
	- [Classical archives](#classical-archives)
		- [Layout](#layout)
		- [Containers and cargos](#containers-and-cargos)
		- [Value cargos](#value-cargos)
	- [Python-enhanced archives](#python-enhanced-archives)
		- [Executable pickles](#executable-pickles)
		- [`requirements.txt`](#requirementstxt)
- [Python API](#python-api)
	- [`QDB`](#qdb)
		- [Loading and storing](#loading-and-storing)
		- [`select` and `merge`](#select-and-merge)
		- [`update`](#update)
		- [Normalisation](#normalisation)
	- [`QDBPipeline`](#qdbpipeline)
		- [Composition](#composition)
		- [Fitting and prediction sets](#fitting-and-prediction-sets)
		- [Interconversion with `QDB`](#interconversion-with-qdb)
	- [`DescriptorPipeline`](#descriptorpipeline)
		- [Built-in engines](#built-in-engines)
		- [Extending with another library](#extending-with-another-library)
- [Usage](#usage)
	- [Developing](#developing)
		- [A spot experiment](#a-spot-experiment)
		- [A branching experiment](#a-branching-experiment)
		- [Distilling a large descriptor set](#distilling-a-large-descriptor-set)
		- [Working across sessions](#working-across-sessions)
	- [Making predictions](#making-predictions)
		- [Through `QDBPipeline`](#through-qdbpipeline)
		- [Through the pickles](#through-the-pickles)
- [Examples](#examples)

## Install

```
pip install -e .[rdkit,mordred]
```

The base install reads and writes archives.
The `rdkit` and `mordred` extras add the corresponding descriptor pipelines.

## Quick start

A dataset of structures and measured values, four descriptors and a linear model:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import pandas

from qsardb import QDBPipeline
from qsardb.rdkit import make_rdkit_pipeline

dataset = pandas.read_csv("esol.csv", index_col = "Id")

X = dataset[["SMILES", "Name"]]
y = dataset["logS"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 13)

pipeline = QDBPipeline([
	("descriptors", make_rdkit_pipeline(["MolLogP", "MolWt", "NumRotatableBonds", "NumAromaticRings"])),
	("model", LinearRegression())
])
pipeline.fit(X_train, y_train)
validation = pipeline.validate(X_valid, y_valid)

print("validation R2 = %.3f" % r2_score(y_valid, validation))

(pipeline
	.to_qdb(model_id = "1", name = "logS from four RDKit descriptors")
	.store("logS.qdb.zip"))
```

```
validation R2 = 0.804
```

The index of the dataset becomes the compound identifiers, the first column of `X` holds the structures, and a column named `Name` becomes the compound names.
Nothing else needs saying: the descriptors that the model consumes, the values they took for every compound, the predictions for both sets and the software that computed them all follow from the pipeline itself.

What comes out is not a pickle of a fitted object but a QsarDB archive:

```
logS.qdb.zip
├── archive.xml
├── requirements.txt
├── compounds
│   ├── compounds.xml
│   ├── 1
│   │   └── daylight-smiles
│   └── ...
├── properties
│   ├── properties.xml
│   └── logS
│       └── values
├── descriptors
│   ├── descriptors.xml
│   ├── MolLogP
│   │   ├── values
│   │   └── pkl
│   └── ...
├── models
│   ├── models.xml
│   └── 1
│       ├── pmml
│       └── pkl
└── predictions
    ├── predictions.xml
    ├── 1-training
    │   └── values
    └── 1-validation
        └── values
```

The values are there to be checked, the PMML is there for anyone without Python, and the pickles are there so the archive can be run rather than only read.
Someone handed that file needs no code of yours:

```python
from qsardb import QDB, QDBPipeline

import pandas

pipeline = QDBPipeline.from_qdb(QDB.load("logS.qdb.zip"))

print(pipeline.predict(pandas.DataFrame({"SMILES" : ["CCO", "c1ccc2ccccc2c1"]}, index = ["ethanol", "naphthalene"])))
```

```
ethanol       -0.111868
naphthalene   -3.147085
Name: logS, dtype: float32
```

## QsarDB archives

### Classical archives

A classical archive describes a model.
The model container carries a `pmml` cargo, the descriptor containers carry the values that were fed to it, and the prediction containers carry what came out.
Everything needed to check the model is present, and a PMML evaluator can rerun it given descriptor values.

What such an archive cannot do is get from a structure to those descriptor values.
The `Application` attribute records that a descriptor came from, say, CDK 1.4.9, but not how to invoke it.

#### Layout

An archive is a directory tree, usually distributed as a ZIP file.
Each of the five containers is a directory holding a registry file and one subdirectory per entry, and each subdirectory holds that entry's cargos:

```
${archive}
├── archive.xml
├── compounds
│   ├── compounds.xml
│   ├── ${compound-id}
│   │   └── daylight-smiles
│   └── ...
├── properties
│   ├── properties.xml
│   ├── ${property-id}
│   │   └── values
│   └── ...
├── descriptors
│   ├── descriptors.xml
│   ├── ${descriptor-id}
│   │   └── values
│   └── ...
├── models
│   ├── models.xml
│   ├── ${model-id}
│   │   └── pmml
│   └── ...
└── predictions
    ├── predictions.xml
    ├── ${prediction-id}
    │   └── values
    └── ...
```

#### Containers and cargos

An archive holds five containers - compounds, properties, descriptors, models and predictions.
In Python a container is a dictionary of attributes and a cargo is its payload, a string or, for binary cargos such as `pkl` and `rds`, bytes.

```python
qdb.containers["models"]
qdb.cargos["models"][model_id]["pmml"]
qdb.files["requirements.txt"]
```

#### Value cargos

Property, descriptor and prediction containers carry a `values` cargo, a tab separated table keyed by compound identifier.
Values are written at the full precision of their own dtype, so a `float32` descriptor writes seven significant digits and a `float64` prediction writes seventeen.
A missing value is written as `N/A`, matching the reference implementation, which treats both null and `NaN` that way.

Descriptor values are stored as computed.
Anything derived from them - ratios, products, scaling - lives in the PMML as derived fields rather than as a descriptor of its own.
Field names in the PMML are namespaced as `descriptors/${descriptor-id}` and `properties/${property-id}`, while the pickles use the plain descriptor identifiers.

### Python-enhanced archives

An archive written by this package is a classical archive plus the Python-specific files marked below:

```
${archive}
├── archive.xml
├── requirements.txt                        <- Python-specific
├── compounds
│   ├── compounds.xml
│   ├── ${compound-id}
│   │   └── daylight-smiles
│   └── ...
├── properties
│   ├── properties.xml
│   ├── ${property-id}
│   │   └── values
│   └── ...
├── descriptors
│   ├── descriptors.xml
│   ├── ${descriptor-id}
│   │   ├── values
│   │   └── pkl                             <- Python-specific
│   └── ...
├── models
│   ├── models.xml
│   ├── ${model-id}
│   │   ├── pmml
│   │   └── pkl                             <- Python-specific
│   └── ...
└── predictions
    ├── predictions.xml
    ├── ${prediction-id}
    │   └── values
    └── ...
```

Everything else is where a classical archive puts it, so a reader that knows nothing about the additions still finds the PMML and the values where it expects them.

#### Executable pickles

The model container gains a `pkl` cargo beside its `pmml`: a pickled Scikit-Learn pipeline that takes structures, computes the descriptors it needs, and returns predictions.
Each descriptor container gains a `pkl` of its own, a pipeline that takes structures and returns that one descriptor.

The two model cargos differ in scope rather than duplicating each other.
`pmml` takes descriptor values and is readable by any PMML evaluator; `pkl` takes structures and needs Python.

#### `requirements.txt`

A file at the archive root pinning the packages needed to unpickle those cargos and call them.
It is derived by loading the pickled pipeline in a subprocess and recording what gets imported, then dropping anything already implied by another requirement.

An archive can therefore be executed anywhere those packages are installed.

## Python API

### `QDB`

`qsardb.QDB` is the archive.
It holds the five containers, it can hold any number of models, and it is the only class that touches the filesystem.

#### Loading and storing

```python
from qsardb import QDB

qdb = QDB.load("ONSMP010.qdb.zip")
qdb.store("ONSMP010")
```

A path ending in `.zip`, `.qdb` or `.qdb.zip` selects a ZIP file, anything else a directory.

#### `select` and `merge`

`select` returns one model as a standalone archive, `merge` combines standalone archives into one:

```python
qdb.select("packing")

(first
	.merge(second)
	.merge(third))
```

By default `select` prunes everything the model does not reference: the predictions of other models, the compounds those predictions covered, the other properties, and the descriptors absent from this model's PMML.
Pass `prune = False` to keep the archive whole.

`merge` is a union.
Compounds, properties and descriptors are merged by identifier, row by row for their values; a conflicting value raises.
Models and predictions raise on an identifier collision rather than being renumbered.

The two are inverses, so an archive can be taken apart and put back together without loss:

```python
(qdb
	.select("a")
	.merge(qdb.select("b")))
```

#### `update`

`merge` takes the name and description of its left operand.
`update` replaces either, and returns the archive so it can be chained:

```python
(qdb
	.update(name = "Three nested hypotheses", description = "Each branch was fitted in isolation.")
	.store("logS.qdb.zip"))
```

#### Normalisation

Reading and re-storing an archive normalises it: attributes are ordered as the schema declares, empty elements are dropped, `Cargos` is recomputed from the cargo files actually present, and anything outside the five containers and the root files is discarded.

### `QDBPipeline`

`qsardb.QDBPipeline` is a Scikit-Learn `Pipeline` that trains and applies exactly one model.
The single-model restriction is not ours: a Scikit-Learn pipeline ends in one estimator.

#### Composition

The first step must be a `DescriptorPipeline`, which turns structures into named descriptor columns.
Everything after it is the model.

```python
from sklearn.linear_model import LinearRegression

from qsardb import QDBPipeline
from qsardb.rdkit import make_rdkit_pipeline

pipeline = QDBPipeline([
	("descriptors", make_rdkit_pipeline(["MolLogP", "MolWt", "TPSA"])),
	("model", LinearRegression())
])
```

#### Fitting and prediction sets

```python
pipeline.fit(X_train, y_train)
pipeline.validate(X_tight, y_tight, prediction_id = "tight")
pipeline.validate(X_loose, y_loose, prediction_id = "loose")
pipeline.test(X_unknown)
```

`X` is a `DataFrame` whose first column holds structures; a column named `Name` is picked up as the compound name.
`y` is a named `Series` sharing the same index, and those index values become compound identifiers.
Compound identity is checked by InChI, so an identifier mapping to more than one structure is an error.

`fit` may be called once.
`validate` and `test` may be called any number of times, and each call appends a prediction container rather than replacing one.
Prediction identifiers are the model identifier and the set name joined by a hyphen, so the sets above become `1-tight`, `1-loose` and `1-testing`.
Naming them keeps them distinct when several models are merged into one archive.

#### Interconversion with `QDB`

```python
QDBPipeline.from_qdb(qdb)
pipeline.to_qdb(model_id = "1")
```

`from_qdb` requires a single-model archive, and restores the full state of the pipeline that wrote it.
`to_qdb` takes the model identifier, name and description, and writes them onto the model container; the archive name and description are set separately, through `QDB.update`.

Writing an archive is therefore two steps, and the second one decides the file name:

```python
(pipeline
	.to_qdb(model_id = "1", name = "logS from RDKit descriptors")
	.store("model.qdb.zip"))
```

### `DescriptorPipeline`

`qsardb.DescriptorPipeline` is the first step of a `QDBPipeline`: structures in, named descriptor columns out.

#### Built-in engines

`qsardb.rdkit.make_rdkit_pipeline(names, n_jobs)` and `qsardb.mordred.make_mordred_pipeline(names, n_jobs)` return one.
Passing no names computes every descriptor the library offers, 217 for RDKit and 1613 for Mordred.

Both can be combined in one pipeline, and each descriptor is attributed to the software that computed it:

```python
from sklearn.compose import ColumnTransformer

from qsardb import DescriptorPipeline
from qsardb.mordred import make_mordred_pipeline
from qsardb.rdkit import make_rdkit_pipeline

descriptors = DescriptorPipeline([
	("descriptorizer", ColumnTransformer([
		("rdkit", make_rdkit_pipeline(["MolLogP", "MolWt"]), [0]),
		("mordred", make_mordred_pipeline(), [0])
	], verbose_feature_names_out = False))
])
```

`applications_out()` reports the application per descriptor, and `to_qdb` writes it as the `Application` attribute.
A descriptor whose application cannot be determined is written without one.

#### Extending with another library

Neither RDKit nor Mordred offers a descriptor of molecular symmetry, although Carnelley's rule makes it a good predictor of melting point: benzene melts at 6 degrees and toluene, one methyl group less symmetric, at -95.
The example below adds two such descriptors, counting how many heavy atoms are distinguishable under the canonical ranking.

A new engine takes three pieces: a transformer that computes the descriptors, a `DescriptorPipeline` subclass that says who computed them and how to compute a subset of them alone, and a factory.

```python
from rdkit import Chem
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import NoFitNeededMixin
from sklearn.base import BaseEstimator, TransformerMixin

import numpy
import pandas

from qsardb import DescriptorPipeline

CALCULATORS = {
	"DistinctAtomFraction" : lambda molecule: len(set(Chem.CanonicalRankAtoms(molecule, breakTies = False))) / molecule.GetNumHeavyAtoms(),
	"EquivalentAtoms" : lambda molecule: molecule.GetNumHeavyAtoms() - len(set(Chem.CanonicalRankAtoms(molecule, breakTies = False)))
}

class SymmetryTransformer(BaseEstimator, NoFitNeededMixin, TransformerMixin):

	def __init__(self, names = None):
		self.names = names

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		names = self.get_feature_names_out()
		rows = [[CALCULATORS[name](molecule) for name in names] for molecule in X.iloc[:, 0]]
		return pandas.DataFrame(rows, columns = names, index = X.index)

	def get_feature_names_out(self, input_features = None):
		return numpy.asarray(list(CALCULATORS) if self.names is None else self.names)

class SymmetryPipeline(DescriptorPipeline):

	def application_name(self):
		return "Symmetry 1.0"

	def descriptor_pipeline(self, names):
		return make_symmetry_pipeline(names = list(names))

def make_symmetry_pipeline(names = None):
	return SymmetryPipeline([
		("parser", SmilesToMolTransformer()),
		("descriptorizer", SymmetryTransformer(names = names))
	])
```

The transformer should be stateless, so that a pipeline computing one descriptor can be built and pickled without being fitted; `NoFitNeededMixin` declares that.
It must also live in an importable module rather than in the script that uses it, because a class defined in `__main__` cannot be unpickled anywhere else, which would leave the archive inexecutable.
`to_qdb` checks this and refuses rather than writing such an archive.

`application_name` supplies the `Application` attribute of every descriptor the pipeline produces.
`descriptor_pipeline` builds a pipeline for a subset of names, which is what fills the per-descriptor `pkl` cargos and what distilling uses.
Both are optional: omitting the first leaves those descriptors without an attribution rather than guessing one, and omitting the second leaves them without a `pkl` cargo.

The new engine then combines with the built-in ones exactly as they combine with each other:

```python
descriptors = DescriptorPipeline([
	("descriptorizer", ColumnTransformer([
		("rdkit", make_rdkit_pipeline(["MolLogP", "MolWt"]), [0]),
		("symmetry", make_symmetry_pipeline(), [0])
	], verbose_feature_names_out = False))
])
```

```
MolLogP                RDKit 2026.03.5
MolWt                  RDKit 2026.03.5
DistinctAtomFraction   Symmetry 1.0
EquivalentAtoms        Symmetry 1.0
```

## Usage

### Developing

#### A spot experiment

One hypothesis, one model, one archive:

```python
pipeline = QDBPipeline([
	("descriptors", make_rdkit_pipeline(["MolLogP", "MolWt", "TPSA"])),
	("model", LinearRegression())
])
pipeline.fit(X_train, y_train)
pipeline.validate(X_valid, y_valid)

(pipeline
	.to_qdb(model_id = "1", name = "logS from RDKit descriptors")
	.store("logS.qdb.zip"))
```

#### A branching experiment

A QSAR study is usually several hypotheses tried against the same data.
Each hypothesis is a `QDBPipeline` of its own, and the archive is what collects them:

```python
BRANCHES = [
	("partition", "Partitioning alone", ["MolLogP"], "Dissolution is transfer into water, so logP alone should carry the signal."),
	("size", "Partitioning and size", ["MolLogP", "MolWt"], "Water must open a cavity around the solute, at a cost that scales with size."),
	("packing", "With crystal packing", ["MolLogP", "MolWt", "NumAromaticRings", "NumHDonors"], "A solid must first leave its lattice, so stacking and hydrogen bonding should matter.")
]

archives = []
for model_id, title, names, hypothesis in BRANCHES:
	pipeline = QDBPipeline([("descriptors", make_rdkit_pipeline(names)), ("model", LinearRegression())])
	pipeline.fit(X_train, y_train)
	pipeline.validate(X_valid, y_valid)
	verdict = r2_score(y_valid, pipeline.predict(X_valid))
	archives.append(pipeline.to_qdb(model_id = model_id, name = title, description = "HYPOTHESIS: %s VERDICT: validation R2 %.3f." % (hypothesis, verdict)))

deliverable = archives[0]
for other in archives[1:]:
	deliverable = deliverable.merge(other)

(deliverable
	.update(name = "Aqueous solubility, three nested hypotheses")
	.store("logS.qdb.zip"))
```

Every branch survives in the deliverable, the rejected ones included, and each can be taken out and used on its own.
Store what cannot be recomputed: the hypothesis and the verdict belong in the model description, while scores, residuals and applicability measures are derivable from the property and prediction values whenever they are wanted.

#### Distilling a large descriptor set

A model fitted on a large descriptor set typically uses a small part of it.
`used_descriptors()` reports the descriptors the fitted model actually references, so the pipeline can be rebuilt around them and refitted:

```python
pipeline.fit(X_train, y_train)
used = pipeline.used_descriptors()
```

Doing this before `to_qdb` keeps the archive to the descriptors that matter, and keeps the stored descriptor values aligned with what the model consumes.
[examples/esol-joint.py](examples/esol-joint.py) distills 1618 descriptors to 24 this way.

#### Working across sessions

An archive restores the full state of the pipeline that wrote it, so an experiment can be put down and picked up:

```python
pipeline = QDBPipeline.from_qdb(QDB.load("logS.qdb.zip").select("packing"))
pipeline.validate(X_new, y_new, prediction_id = "external")

(pipeline
	.to_qdb(model_id = "packing")
	.store("packing.qdb.zip"))
```

The restored pipeline carries its training, validation and testing sets, so a further `validate` appends to what is already there rather than starting over.
Nothing is recomputed on load; the stored state is taken as given.
`fit` refuses to run on a restored pipeline, because a second fit would leave the archive describing predictions the model no longer makes.

### Making predictions

#### Through `QDBPipeline`

```python
pipeline = QDBPipeline.from_qdb(QDB.load("model.qdb.zip"))
pipeline.predict(structures)
```

The result is a `Series` indexed by the identifiers of the structures passed in and named after the property.
A multi-model archive must be narrowed with `select` first.

#### Through the pickles

The model `pkl` is an ordinary Scikit-Learn pipeline, and can be used in either direction:

```python
model = pickle.loads(qdb.cargos["models"][model_id]["pkl"])

model.predict(structures)
model[1:].predict(descriptor_values)
```

Each descriptor `pkl` is a pipeline of its own, taking structures and returning that one descriptor:

```python
descriptor = pickle.loads(qdb.cargos["descriptors"][descriptor_id]["pkl"])

descriptor.transform(structures)
```

## Examples

[examples/esol.py](examples/esol.py) fits a linear model of aqueous solubility on RDKit descriptors, with a derived field computed from two of them, and exports an archive.

[examples/esol-joint.py](examples/esol-joint.py) combines RDKit and Mordred descriptors, fits a gradient boosted model, distills the descriptor set to the ones the model uses, and reports which software each came from.

Both read [examples/esol.csv](examples/esol.csv) and write their archive alongside it, so run them from within `examples`, with the package installed.
They need `xgboost` in addition to the package requirements.
