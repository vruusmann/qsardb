# QsarDB Python package

A pure Python implementation of the QsarDB (QDB) archive format, together with a Scikit-Learn pipeline that trains a model and exports it as an archive.
Exported archives are executable: the model and the descriptors that feed it are stored as pickles alongside the PMML, so a loaded archive can score a new structure.

## Install

```
pip install -e .[rdkit,mordred]
```

The base install reads and writes archives.
The `rdkit` and `mordred` extras add the corresponding descriptor pipelines.

## Archives

`QDB` reads and writes the archive itself, either as a directory or as a ZIP file.
A path ending in `.zip`, `.qdb` or `.qdb.zip` selects the latter.

```python
from qsardb import QDB

qdb = QDB.load("ONSMP010.qdb.zip")
qdb.store("ONSMP010")
```

Containers are dictionaries of attributes and cargos are payload strings, or bytes for binary cargos such as `pkl` and `rds`.
Files at the archive root, such as `requirements.txt`, are available as `QDB.files`.

Values are written at the full precision of their own dtype, so a float32 descriptor writes seven significant digits and a float64 prediction writes seventeen.
A missing value is written as `N/A`, matching the reference implementation, which treats both null and NaN that way.

Reading and re-storing an archive normalises it: attributes are ordered as the schema declares, empty elements are dropped, `Cargos` is recomputed from the cargo files actually present, and anything outside the five containers and the root files is discarded.

## Pipelines

`QDBPipeline` is a `Pipeline` whose first step must be a `DescriptorPipeline`.
That step turns structures into named descriptor columns; everything after it is the model.

```python
from qsardb import QDBPipeline
from qsardb.rdkit import make_rdkit_pipeline

pipeline = QDBPipeline([
	("descriptors", make_rdkit_pipeline(["MolLogP", "MolWt", "TPSA"])),
	("model", LinearRegression())
])
pipeline.fit(X, y)
pipeline.validate(X_valid, y_valid)
pipeline.export("model.qdb.zip")
```

`X` is a `DataFrame` whose first column holds structures; a column named `Name` is picked up as the compound name.
`y` is a named `Series` sharing the same index, and those index values become compound identifiers.
`fit`, `validate` and `test` record the three `Prediction` types.
Compound identity is checked by InChI, so an identifier mapping to more than one structure is an error.

`export` writes the compounds with their structures and InChI, the property, the descriptor values, the model, and one prediction container per recorded set.
Descriptor values are stored as computed; anything derived from them - ratios, products, scaling - lives in the PMML as derived fields.
Field names in the PMML are namespaced as `descriptors/{id}` and `properties/{id}`, while the pickles use the plain descriptor identifiers.

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

`applications_out()` reports the application per descriptor, and `export` writes it as the `Application` attribute.
A descriptor whose application cannot be determined is written without one.

A `DescriptorPipeline` can also be assembled by hand from any transformer that takes structures and returns named columns.

## Narrowing

A model fitted on a large descriptor set typically uses a small part of it.
`used_descriptors()` reports the descriptors the fitted model actually references, so the pipeline can be rebuilt around them and refitted:

```python
pipeline.fit(X_train, y_train)
used = pipeline.used_descriptors()
```

Doing this before `export` keeps the archive to the descriptors that matter, and keeps the stored descriptor values aligned with what the model consumes.
`examples/esol-joint.py` narrows 1618 descriptors to 24 this way.

## Executing an archive

The model container carries the fitted model twice.
`pmml` takes descriptor values and is readable by any PMML evaluator.
`pkl` is a pickled Scikit-Learn pipeline that takes structures, computes the descriptors it needs, and returns predictions:

```python
model = pickle.loads(qdb.cargos["models"]["1"]["pkl"])

model.predict(structures)                   # from structures
model[1:].predict(descriptor_values)        # from stored descriptor values
```

`QDBPipeline.load` wraps that pickle back into a `QDBPipeline`, so a loaded archive predicts through the same interface it was trained with.
It accepts an archive path or an already loaded `QDB`:

```python
from qsardb import QDBPipeline

pipeline = QDBPipeline.load("model.qdb.zip")
pipeline.predict(structures)
```

The result is a `Series` indexed by the identifiers of the structures passed in and named after the property.
An archive holding more than one model raises unless a `model_id` is given.

Each descriptor container carries a `pkl` of its own, a pipeline that takes structures and returns that one descriptor.

`requirements.txt` at the archive root pins the packages needed to unpickle these and call them.
It is derived by loading the pickled pipeline in a subprocess and recording what gets imported, then dropping anything already implied by another requirement.

## Examples

`examples/esol.py` fits a linear model of aqueous solubility on RDKit descriptors, with a derived field computed from two of them, and exports an archive.

`examples/esol-joint.py` combines RDKit and Mordred descriptors, fits a gradient boosted model, narrows the descriptor set to the ones the model uses, and reports which software each came from.

Both read `esol.csv` and write their archive alongside it, so run them from within `examples`, with the package installed.
They need `xgboost` in addition to the package requirements.

## Not yet

An archive can be executed only where the packages named in its `requirements.txt` are installed; there is no implementation-independent descriptor specification, so a PMML evaluator alone cannot recompute descriptors from a structure.
Every compound occupies its own directory, which does not scale gracefully to large datasets.
