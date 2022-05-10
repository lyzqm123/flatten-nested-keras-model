import tensorflow as tf
import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pprint
from typing import Optional, Union


class ModelConnectionSaver:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.model_connection = {}
        self.set_of_in_node_ids = set([])
        self.set_of_out_node_ids = set([])
        super().__init__()

    def _update_model_connection(self, layer_or_model: Union[tf.keras.Model, tf.keras.layers.Layer],
                                 in_node_ids: Optional[int], out_node_ids: Optional[int], depth: Optional[int]):
        is_model = isinstance(layer_or_model, tf.keras.Model)
        is_layer = isinstance(layer_or_model, tf.keras.layers.Layer)

        layer_or_model = layer_or_model
        name = layer_or_model.name
        if name not in self.model_connection.keys():
            self.model_connection[name] = {}
        self.model_connection[name].update(
            {"object": layer_or_model, "depth": depth})
        self.model_connection[name].update({"in_node_ids": in_node_ids})
        self.model_connection[name].update({"out_node_ids": out_node_ids})
        self.model_connection[name].update({"is_last": len(out_node_ids) == 0})
        self.model_connection[name].update({"is_layer": is_layer})
        self.model_connection[name].update({"is_model": is_model})

    def _update_graph(self, layer_or_model: Union[tf.keras.Model, tf.keras.layers.Layer],
                      in_node_ids: Optional[int], out_node_ids: Optional[int], depth: int):

        for in_node_id in in_node_ids:
            for out_node_id in out_node_ids:
                print("*{} [{}] {} => {} ".format(
                    '   '*depth, layer_or_model.name, in_node_id, out_node_id))
                self.graph.add_edge(in_node_id, out_node_id,
                                    name=layer_or_model.name)

    def update(self, layer_or_model: Union[tf.keras.Model, tf.keras.layers.Layer], in_node_ids: Optional[int], out_node_ids: Optional[int], depth: Optional[int]):
        is_model = isinstance(layer_or_model, tf.keras.Model)
        is_layer = isinstance(layer_or_model, tf.keras.layers.Layer)

        if not is_model and not is_layer:
            raise ValueError(
                "`layer_or_model: {}` should be one of the tf.keras.Model or tf.keras.layers.Layer.".format(layer_or_model))
        if not is_model:
            self._update_model_connection(
                layer_or_model, in_node_ids, out_node_ids, depth)
        self._update_graph(layer_or_model, in_node_ids, out_node_ids, depth)
        self.set_of_in_node_ids.update(in_node_ids)
        self.set_of_out_node_ids.update(out_node_ids)

    def get_layer(self, layer_name: str):
        if layer_name not in self.model_connection.keys():
            raise RuntimeError("")
        return self.model_connection[layer_name]["object"]

    def get_last_layer(self, depth):
        last_layer = None
        last_in_node_ids = []
        last_out_node_ids = []
        for _, info in self.model_connection.items():
            if info["is_last"] and info["depth"] == depth:
                last_layer = info["object"]
                in_node_ids = info["in_node_ids"]
                out_node_ids = info["out_node_ids"]
                last_in_node_ids.extend(in_node_ids)
                last_out_node_ids.extend(out_node_ids)
        return last_layer, last_in_node_ids, last_out_node_ids

    def save_topological_sorted_edges(self, topological_sorted_edges: set):
        set_of_topological_sorted_edges = set(
            [edge for edges in topological_sorted_edges for edge in edges])
        all_node_ids = self.set_of_in_node_ids | self.set_of_out_node_ids
        is_exist = len(set_of_topological_sorted_edges - all_node_ids) == 0
        if not is_exist:
            raise ValueError(
                "Some ids are not belong to `{}`.".format(all_node_ids))
        self.topological_sorted_edges = topological_sorted_edges

    def _remove_deprecated_input_tensor(self):
        to_remove_edges = []
        for edge in self.graph.edges.data():
            in_node = edge[0]
            out_node = edge[1]
            layer_name = edge[2]["name"]

            layer = self.connection_saver.get_layer(layer_name)

            # This layer should be remove if degree of outbound node is equal to zero.
            if isinstance(layer, tf.keras.layers.InputLayer) and \
                    self.graph.out_degree[in_node] == 1 and \
            self.graph.out_degree[out_node] == 0:
                to_remove_edges.append((in_node, out_node))

        self.graph.remove_edges_from(to_remove_edges)
        self.graph.remove_nodes_from(list(itertools.chain(*to_remove_edges)))

    def draw(self):
        if nx.is_directed_acyclic_graph(self.graph):
            topological_sorted_edges = list(
                nx.topological_sort(nx.line_graph(self.graph)))
            self.save_topological_sorted_edges(
                topological_sorted_edges)
        else:
            raise NotImplementedError("Non-DAG cases are not implemented.")

        edge_labels = nx.get_edge_attributes(self.graph, 'name')

        try:
            pos = nx.spring_layout(self.graph, scale=20,
                                   k=3/np.sqrt(self.graph.order()))
        except:
            raise ZeroDivisionError("Graph is empty...")

        nx.draw(self.graph, pos, node_color='r', with_labels=True)
        nx.draw_networkx_edge_labels(
            self.graph, pos=pos, alpha=0.6, edge_labels=edge_labels)

        graph_image_path = './flatten_model.png'
        try:
            plt.savefig("graph_image_path", format="PNG")
            print("The model graph image saved into `{}`.".format(graph_image_path))
        except:
            raise RuntimeError("Model graph saving error.")

    def print(self):
        pp = pprint.PrettyPrinter(indent=3)
        pp.pprint(self.model_connection)
