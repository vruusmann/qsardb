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

	@classmethod
	def load(cls, path):
		if path.endswith(_ZIP_SUFFIXES):
			directory = tempfile.mkdtemp()
			with zipfile.ZipFile(path, "r") as archive:
				archive.extractall(directory)
			qdb = cls._load(directory)
			shutil.rmtree(directory)
		else:
			qdb = cls._load(path)
		return qdb

	@classmethod
	def _load(cls, directory):
		attributes = cls._load_xml(os.path.join(directory, "archive.xml"))[0]
		qdb = cls(attributes.get("Name"), attributes.get("Description"))

		for type in _CONTAINERS:
			path = os.path.join(directory, type, type + ".xml")
			if not os.path.exists(path):
				continue
			for attributes in cls._load_xml(path):
				cargos = {cargo_id : cls._load_cargo(directory, type, attributes["Id"], cargo_id) for cargo_id in attributes.get("Cargos", "").split()}
				qdb.add(type, attributes, cargos)
		return qdb

	@classmethod
	def _load_xml(cls, path):
		root = ElementTree.parse(path).getroot()
		containers = [root] if root.tag == "{%s}Archive" % _NAMESPACE else list(root)
		return [{element.tag.split("}")[-1] : element.text for element in container if element.text is not None and element.text.strip()} for container in containers]

	@classmethod
	def _load_cargo(cls, directory, type, id, cargo_id):
		with open(os.path.join(directory, type, id, cargo_id), "rb") as file:
			payload = file.read()
		try:
			return payload.decode("UTF-8")
		except UnicodeDecodeError:
			return payload

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
		path = os.path.join(directory, type, id, cargo_id)
		if isinstance(payload, bytes):
			with open(path, "wb") as file:
				file.write(payload)
		else:
			with open(path, "w", encoding = "UTF-8") as file:
				file.write(payload)

	def _store_zip(self, directory, path):
		with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
			for parent, _, names in os.walk(directory):
				for name in names:
					file_path = os.path.join(parent, name)
					archive.write(file_path, os.path.relpath(file_path, directory))

def format_values(header, values):
	lines = ["Compound Id\t" + header]
	for compound_id, value in values.items():
		lines.append("%s\t%s" % (compound_id, round(float(value), 6)))
	return "\n".join(lines)
