"""Exact graph contracts without exporter-specific names or serialization metadata."""

import hashlib
import json
import struct
from importlib.resources import files

import onnx
from onnx import numpy_helper

from .._io import object_hash
from .._protobuf import is_repeated


def _tensor(value):
    array = numpy_helper.to_array(value)
    if array.dtype.hasobject:
        payload = b"".join(
            len(item).to_bytes(8, "little") + item for item in array.flat
        )
    else:
        payload = array.tobytes()
    return [value.data_type, list(value.dims), hashlib.sha256(payload).hexdigest()]


class _Graph:
    def __init__(self, graph, outer=None, symbols=None):
        self.graph = graph
        self.outer = outer
        self.refs = {v.name: ["input", i] for i, v in enumerate(graph.input)}
        self.producers = {
            value: (i, port)
            for i, node in enumerate(graph.node)
            for port, value in enumerate(node.output)
            if value
        }
        self.weights = {v.name: v for v in graph.initializer}
        self.node_ids = {}
        self.nodes = []
        self.parameters = []
        self.symbols = {} if symbols is None else symbols
        self.active = set()
        if graph.sparse_initializer:
            raise ValueError("Sparse initializers are not in the export contract")

    def shape_type(self, value):
        def normalize(message):
            result = []
            for field, item in message.ListFields():
                if field.name == "denotation":
                    continue
                if field.name == "dim_param":
                    item = self.symbols.setdefault(item, len(self.symbols))
                elif field.type == field.TYPE_MESSAGE:
                    item = (
                        [normalize(v) for v in item]
                        if is_repeated(field)
                        else normalize(item)
                    )
                result.append([field.name, item])
            return result

        return normalize(value.type)

    def attribute(self, attr):
        kind = attr.type
        A = onnx.AttributeProto
        if attr.ref_attr_name:
            raise ValueError("Unresolved function attribute in export contract")
        if kind in (A.TENSOR, A.TENSORS):
            result = (
                _tensor(attr.t)
                if kind == A.TENSOR
                else [_tensor(t) for t in attr.tensors]
            )
        elif kind in (A.GRAPH, A.GRAPHS):
            graphs = [attr.g] if kind == A.GRAPH else attr.graphs
            result = [_Graph(g, self.resolve, self.symbols).build()[0] for g in graphs]
        elif kind in (A.FLOAT, A.FLOATS):
            values = [attr.f] if kind == A.FLOAT else attr.floats
            result = [struct.pack("<f", v).hex() for v in values]
        elif kind in (A.STRING, A.STRINGS):
            values = [attr.s] if kind == A.STRING else attr.strings
            result = [v.hex() for v in values]
        elif kind in (A.INT, A.INTS):
            result = attr.i if kind == A.INT else list(attr.ints)
        else:
            raise ValueError("Unsupported export attribute type: " + str(kind))
        return [attr.name, kind, result]

    def resolve(self, name):
        if not name:
            return ["absent"]
        if name in self.refs:
            return self.refs[name]
        if name in self.weights:
            reference = ["initializer", len(self.parameters)]
            self.parameters.append(_tensor(self.weights[name]))
        elif name in self.producers:
            index, port = self.producers[name]
            if index in self.active:
                raise ValueError("Cyclic graph in export contract")
            if index not in self.node_ids:
                ordinal = len(self.nodes)
                self.node_ids[index] = ordinal
                self.nodes.append(None)
                self.active.add(index)
                node = self.graph.node[index]
                inputs = [self.resolve(v) for v in node.input]
                attributes = [
                    self.attribute(a)
                    for a in sorted(node.attribute, key=lambda a: a.name)
                ]
                self.nodes[ordinal] = [
                    node.domain,
                    node.op_type,
                    inputs,
                    attributes,
                    [bool(v) for v in node.output],
                ]
                self.active.remove(index)
            reference = ["node", self.node_ids[index], port]
        elif self.outer:
            reference = ["outer", self.outer(name)]
        else:
            raise ValueError("Unresolved graph input: " + name)
        self.refs[name] = reference
        return reference

    def build(self):
        inputs = [self.shape_type(v) for v in self.graph.input]
        outputs = [
            [self.resolve(v.name), self.shape_type(v)] for v in self.graph.output
        ]
        if len(self.node_ids) != len(self.graph.node):
            # Export partitions are dependency-closed. Do not silently discard dead
            # branches, whose presence would otherwise escape an output-only hash.
            raise ValueError("Unreachable nodes in export graph contract")
        unused = sorted(
            (_tensor(v) for name, v in self.weights.items() if name not in self.refs),
            key=object_hash,
        )
        defaults = [
            [i, _tensor(self.weights[v.name])]
            for i, v in enumerate(self.graph.input)
            if v.name in self.weights
        ]
        result = dict(
            inputs=inputs,
            outputs=outputs,
            nodes=self.nodes,
            input_defaults=defaults,
            initializers=self.parameters,
            unused_initializers=unused,
        )
        names = dict(
            inputs=[v.name for v in self.graph.input],
            outputs=[v.name for v in self.graph.output],
            nodes=[
                self.graph.node[i].name
                for i in sorted(self.node_ids, key=self.node_ids.get)
            ],
        )
        return result, names


def graph_contract(model, *, network=False):
    if model.functions or model.training_info:
        raise ValueError("Function/training graphs are not in the export contract")
    graph = _Graph(model.graph)
    value, names = graph.build()
    value["opsets"] = sorted([o.domain, o.version] for o in model.opset_import)
    quantization = {}
    if network:
        from ..quant.engine import infer_model, layout

        inferred = infer_model(model)
        for scope in ("basic", "all"):
            plan = layout(inferred, scope)
            tensors = sorted(
                (graph.resolve(n) for n in plan["tensors"]), key=object_hash
            )
            edges = sorted(
                (
                    [graph.node_ids[i], port, graph.resolve(name)]
                    for (i, port), name in plan["edges"].items()
                ),
                key=object_hash,
            )
            quantization[scope] = dict(
                tensors=object_hash(tensors),
                edges=object_hash(edges),
                tensor_count=len(tensors),
                edge_count=len(edges),
            )
    return dict(structure=object_hash(value), quantization=quantization, names=names)


def validate_parts(parts, model_name):
    reference = json.loads(
        files("opennpu_quant.resources").joinpath("export-contracts.json").read_text()
    )
    if reference["id"] != object_hash(reference["models"]):
        raise ValueError("Release export reference checksum mismatch")
    expected = reference["models"].get(model_name)
    if expected is None or set(parts) != set(expected):
        raise ValueError("Release export graph roles differ: " + model_name)
    mappings = {}
    for role, model in parts.items():
        actual = graph_contract(model, network=role == "network")
        wanted = expected[role]
        if any(actual[key] != wanted[key] for key in ("structure", "quantization")):
            raise ValueError(
                "Release export structure mismatch: " + model_name + "/" + role
            )
        mappings[role] = {
            kind: list(zip(wanted["names"][kind], actual["names"][kind]))
            for kind in ("inputs", "outputs", "nodes")
        }
    if "postprocess" in mappings:
        outputs = dict(mappings["network"]["outputs"])
        if any(
            outputs.get(before) != after
            for before, after in mappings["postprocess"]["inputs"]
        ):
            raise ValueError("Release export connection mismatch: " + model_name)
    return dict(passed=True, reference=reference["id"], name_correspondence=mappings)
