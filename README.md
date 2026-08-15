# QsarDB Python package

A pure Python implementation of the QsarDB (QDB) archive format, together with a Scikit-Learn pipeline that trains a model and exports it as an archive.

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

Containers are dictionaries of attributes and cargos are payload strings, or bytes for binary cargos such as `rds`.
Reading and re-storing an archive normalises it: attributes are ordered as the schema declares, empty elements are dropped, `Cargos` is recomputed from the cargo files actually present, and anything outside the five containers is discarded.

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

`export` writes the compounds with their structures and InChI, the property, the descriptor values, the model as PMML, and one prediction container per recorded set.
Descriptor values are stored as computed; anything derived from them - ratios, products, scaling - lives in the PMML as derived fields.
Field names in the PMML are namespaced as `descriptors/{id}` and `properties/{id}`.

## Descriptors

`make_rdkit_pipeline(desc_list)` and `make_mordred_pipeline()` return `DescriptorPipeline` instances.
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

## Examples

`examples/esol.py` fits a linear model of aqueous solubility on RDKit descriptors, with a derived field computed from two of them, and exports an archive.

`examples/esol-joint.py` combines RDKit and Mordred descriptors, fits a gradient boosted model, and reports which software each selected descriptor came from.

Both read `esol.csv` and write their archive alongside it, so run them from within `examples`.
They need `xgboost` in addition to the package requirements.

## Not yet

Archives are not re-executable: an archive records which software computed a descriptor, but not enough to recompute it, so a loaded archive cannot score a new structure.
Descriptor and prediction values are stored rounded, which is sufficient for the values themselves but not always for reproducing a tree model's output from them.
Every compound occupies its own directory, which does not scale gracefully to large datasets.
