# QsarDB Python package

A pure Python implementation of the QsarDB (QDB) archive format, together with a Scikit-Learn pipeline that trains a model and exports it as an archive.
Exported archives are executable: the model and the descriptors that feed it are stored as pickles alongside the PMML, so a loaded archive can score a new structure.

## Install

```
pip install -e .[rdkit,mordred]
```

The base install reads and writes archives.
The `rdkit` and `mordred` extras add the corresponding descriptor pipelines.

## The two classes

`QDB` is the archive.
It holds compounds, properties, descriptors, models and predictions, it reads and writes files, and it can hold any number of models.

`QDBPipeline` is a Scikit-Learn `Pipeline` that trains and applies exactly one model.
The single-model restriction is not ours: a Scikit-Learn pipeline ends in one estimator.

The two convert into each other, and only `QDB` touches the filesystem:

```python
QDB.load(path)                 # file  -> QDB
qdb.store(path)                # QDB   -> file

QDBPipeline.from_qdb(qdb)      # QDB   -> QDBPipeline, requires a single-model archive
pipeline.to_qdb(model_id)      # QDBPipeline -> single-model QDB
```

So writing an archive is two steps, and the second one is where the file name is decided:

```python
pipeline.to_qdb(model_id = "1", name = "logS from RDKit descriptors").store("model.qdb.zip")
```

## Training a model

```python
from sklearn.linear_model import LinearRegression

from qsardb import QDBPipeline
from qsardb.rdkit import make_rdkit_pipeline

pipeline = QDBPipeline([
	("descriptors", make_rdkit_pipeline(["MolLogP", "MolWt", "TPSA"])),
	("model", LinearRegression())
])
pipeline.fit(X_train, y_train)
pipeline.validate(X_valid, y_valid)
```

The first step must be a `DescriptorPipeline`, which turns structures into named descriptor columns.
Everything after it is the model.

`X` is a `DataFrame` whose first column holds structures; a column named `Name` is picked up as the compound name.
`y` is a named `Series` sharing the same index, and those index values become compound identifiers.

`fit` may be called once.
`validate` and `test` may be called any number of times, and each call appends a prediction container rather than replacing one:

```python
pipeline.validate(X_tight, y_tight, prediction_id = "tight")
pipeline.validate(X_loose, y_loose, prediction_id = "loose")
pipeline.test(X_unknown)
```

Prediction identifiers are the model identifier and the set name joined by a hyphen, so the sets above become `1-tight`, `1-loose` and `1-testing`.
Naming them keeps them distinct when several models are merged into one archive.

Compound identity is checked by InChI, so an identifier mapping to more than one structure is an error.

## Single-model and multi-model archives

`to_qdb` always produces a single-model archive, and `from_qdb` always requires one.
Archives with several models are built and taken apart by two `QDB` methods:

```python
qdb = first.merge(second).merge(third)      # single-model -> multi-model
qdb.select("second")                        # multi-model  -> single-model
```

`merge` is a union.
Compounds, properties and descriptors are merged by identifier, row by row for their values; a conflicting value raises.
Models and predictions raise on an identifier collision rather than being renumbered, which is why `to_qdb` takes the model identifier.
The archive name and description are taken from the left operand and can be replaced afterwards:

```python
qdb.update(name = "Three nested hypotheses", description = "Each branch was fitted in isolation.")
```

`select` returns one model as a standalone archive.
By default it prunes everything the model does not reference: the predictions of other models, the compounds those predictions covered, the other properties, and the descriptors absent from this model's PMML.
Pass `prune = False` to keep the archive whole.

The two are inverses, so an archive can be taken apart and put back together without loss:

```python
qdb.select("a").merge(qdb.select("b"))
```

## A multi-stage experiment

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
deliverable.update(name = "Aqueous solubility, three nested hypotheses").store("logS.qdb.zip")
```

Every branch survives in the deliverable, the rejected ones included, and each can be taken out and used on its own.
Store what cannot be recomputed: the hypothesis and the verdict belong in the model description, while scores, residuals and applicability measures are derivable from the property and prediction values whenever they are wanted.

## Working across sessions

An archive restores the full state of the pipeline that wrote it, so an experiment can be put down and picked up:

```python
pipeline = QDBPipeline.from_qdb(QDB.load("logS.qdb.zip").select("packing"))
pipeline.validate(X_new, y_new, prediction_id = "external")
pipeline.to_qdb(model_id = "packing").store("packing.qdb.zip")
```

The restored pipeline carries its training, validation and testing sets, so a further `validate` appends to what is already there rather than starting over.
Nothing is recomputed on load; the stored state is taken as given.
`fit` refuses to run on a restored pipeline, because a second fit would leave the archive describing predictions the model no longer makes.

## Descriptors

`make_rdkit_pipeline(names, n_jobs)` and `make_mordred_pipeline(names, n_jobs)` return `DescriptorPipeline` instances.
Passing no names computes every descriptor the library offers.
Both can be combined, and each descriptor is attributed to the software that computed it:

```python
from sklearn.compose import ColumnTransformer

descriptors = DescriptorPipeline([
	("descriptorizer", ColumnTransformer([
		("rdkit", make_rdkit_pipeline(["MolLogP", "MolWt"]), [0]),
		("mordred", make_mordred_pipeline(), [0])
	], verbose_feature_names_out = False))
])
```

`applications_out()` reports the application per descriptor, and `to_qdb` writes it as the `Application` attribute.
A descriptor whose application cannot be determined is written without one.

A `DescriptorPipeline` can also be assembled by hand from any transformer that takes structures and returns named columns.

## Distilling

A model fitted on a large descriptor set typically uses a small part of it.
`used_descriptors()` reports the descriptors the fitted model actually references, so the pipeline can be rebuilt around them and refitted:

```python
pipeline.fit(X_train, y_train)
used = pipeline.used_descriptors()
```

Doing this before `to_qdb` keeps the archive to the descriptors that matter, and keeps the stored descriptor values aligned with what the model consumes.
`examples/esol-joint.py` distills 1618 descriptors to 24 this way.

## Executing an archive

The model container carries the fitted model twice.
`pmml` takes descriptor values and is readable by any PMML evaluator.
`pkl` is a pickled Scikit-Learn pipeline that takes structures, computes the descriptors it needs, and returns predictions:

```python
model = pickle.loads(qdb.cargos["models"]["1"]["pkl"])

model.predict(structures)                   # from structures
model[1:].predict(descriptor_values)        # from stored descriptor values
```

`QDBPipeline.from_qdb` wraps that pickle back into a `QDBPipeline`, so a loaded archive predicts through the same interface it was trained with:

```python
pipeline = QDBPipeline.from_qdb(QDB.load("model.qdb.zip"))
pipeline.predict(structures)
```

The result is a `Series` indexed by the identifiers of the structures passed in and named after the property.

Each descriptor container carries a `pkl` of its own, a pipeline that takes structures and returns that one descriptor.

`requirements.txt` at the archive root pins the packages needed to unpickle these and call them.
It is derived by loading the pickled pipeline in a subprocess and recording what gets imported, then dropping anything already implied by another requirement.

## Archive contents

Containers are dictionaries of attributes and cargos are payload strings, or bytes for binary cargos such as `pkl` and `rds`.
Files at the archive root, such as `requirements.txt`, are available as `QDB.files`.

Values are written at the full precision of their own dtype, so a float32 descriptor writes seven significant digits and a float64 prediction writes seventeen.
A missing value is written as `N/A`, matching the reference implementation, which treats both null and NaN that way.

Descriptor values are stored as computed; anything derived from them - ratios, products, scaling - lives in the PMML as derived fields.
Field names in the PMML are namespaced as `descriptors/{id}` and `properties/{id}`, while the pickles use the plain descriptor identifiers.

Reading and re-storing an archive normalises it: attributes are ordered as the schema declares, empty elements are dropped, `Cargos` is recomputed from the cargo files actually present, and anything outside the five containers and the root files is discarded.

## Examples

`examples/esol.py` fits a linear model of aqueous solubility on RDKit descriptors, with a derived field computed from two of them, and exports an archive.

`examples/esol-joint.py` combines RDKit and Mordred descriptors, fits a gradient boosted model, distills the descriptor set to the ones the model uses, and reports which software each came from.

Both read `esol.csv` and write their archive alongside it, so run them from within `examples`, with the package installed.
They need `xgboost` in addition to the package requirements.

## Not yet

An archive can be executed only where the packages named in its `requirements.txt` are installed; there is no implementation-independent descriptor specification, so a PMML evaluator alone cannot recompute descriptors from a structure.
Every compound occupies its own directory, which does not scale gracefully to large datasets.
