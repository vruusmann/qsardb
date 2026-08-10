from xml.etree import ElementTree

import os
import shutil
import tempfile
import zipfile

_CONTAINER_ATTRIBUTES = ("Id", "Name", "Description", "Labels", "Cargos")

_CONTAINERS = {
	"compounds" : ("CompoundRegistry", "Compound", _CONTAINER_ATTRIBUTES + ("Cas", "InChI")),
	"properties" : ("PropertyRegistry", "Property", _CONTAINER_ATTRIBUTES + ("Endpoint", "Species")),
	"descriptors" : ("DescriptorRegistry", "Descriptor", _CONTAINER_ATTRIBUTES + ("Application",)),
	"models" : ("ModelRegistry", "Model", _CONTAINER_ATTRIBUTES + ("PropertyId",)),
	"predictions" : ("PredictionRegistry", "Prediction", _CONTAINER_ATTRIBUTES + ("ModelId", "Type", "Application"))
}

_NAMESPACE = "http://www.qsardb.org/QDB"

_ZIP_SUFFIXES = (".zip", ".qdb")

class QDB(object):

	def __init__(self, name = None, description = None):
		self.name = name
		self.description = description
		self.containers = {type : [] for type in _CONTAINERS}
		self.cargos = {type : {} for type in _CONTAINERS}

	def add(self, type, attributes, cargos):
		attributes = dict(attributes)
		attributes["Cargos"] = " ".join(cargos.keys())
		self.containers[type].append(attributes)
		self.cargos[type][attributes["Id"]] = cargos

	def store(self, path):
		if path.endswith(_ZIP_SUFFIXES):
			directory = tempfile.mkdtemp()
			self._store(directory)
			self._store_zip(directory, path)
			shutil.rmtree(directory)
		else:
			self._store(path)
		return path

	def _store(self, directory):
		if os.path.exists(directory):
			shutil.rmtree(directory)
		os.makedirs(directory)

		self._store_xml(os.path.join(directory, "archive.xml"), "Archive", [{"Name" : self.name, "Description" : self.description}], ("Name", "Description"))

		for type, (registry_tag, container_tag, order) in _CONTAINERS.items():
			if not self.containers[type]:
				continue
			self._store_xml(os.path.join(directory, type, type + ".xml"), registry_tag, self.containers[type], order, container_tag)
			for id, cargos in self.cargos[type].items():
				for cargo_id, payload in cargos.items():
					self._store_cargo(directory, type, id, cargo_id, payload)

	def _store_xml(self, path, registry_tag, containers, order, container_tag = None):
		root = ElementTree.Element("{%s}%s" % (_NAMESPACE, registry_tag))
		for attributes in containers:
			parent = root if container_tag is None else ElementTree.SubElement(root, "{%s}%s" % (_NAMESPACE, container_tag))
			for tag in order:
				value = attributes.get(tag)
				if value is not None:
					ElementTree.SubElement(parent, "{%s}%s" % (_NAMESPACE, tag)).text = str(value)
		os.makedirs(os.path.dirname(path), exist_ok = True)
		tree = ElementTree.ElementTree(root)
		ElementTree.indent(tree)
		tree.write(path, encoding = "UTF-8", xml_declaration = True, default_namespace = _NAMESPACE)

	def _store_cargo(self, directory, type, id, cargo_id, payload):
		os.makedirs(os.path.join(directory, type, id), exist_ok = True)
		with open(os.path.join(directory, type, id, cargo_id), "w", encoding = "UTF-8") as file:
			file.write(payload)

	def _store_zip(self, directory, path):
		with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
			for parent, _, names in os.walk(directory):
				for name in names:
					file_path = os.path.join(parent, name)
					archive.write(file_path, os.path.relpath(file_path, directory))

def format_values(id, values):
	lines = ["Compound Id\t" + id]
	for compound_id, value in values.items():
		lines.append("%s\t%s" % (compound_id, round(float(value), 6)))
	return "\n".join(lines)
