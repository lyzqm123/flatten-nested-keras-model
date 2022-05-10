import tensorflow as tf
import networkx as nx
from model_connection_saver import ModelConnectionSaver
from typing import Optional, Union


class IDTransformer:
    def __init__(self) -> None:
        self.decoder = {}
        self.counter = 0
        super().__init__()

    def __call__(self, number: Union[list, tuple, int, str, None]) -> Union[list, tuple, int]:
        is_list = isinstance(number, list)
        is_tuple = isinstance(number, tuple)
        ret = []

        if not is_list and not is_tuple:
            number = [number]

        for num in number:
            if num is None or num is not int:
                num = id(num)

            if num not in self.decoder.keys():
                self.decoder[num] = self.counter
                self.counter += 1

            ret.append(self.decoder[num])

        if is_list or is_tuple:
            return ret
        else:
            return ret[0]

    def possible(self, number: Union[list, tuple, int, str, None]) -> bool:
        if isinstance(number, list) or isinstance(number, tuple):
            return len(number) > 0
        return True

    def get_min_id(self, number: Union[list, tuple, int, str, None]) -> int:
        ret = self(number)
        return min(ret)

    def get_max_id(self, number: Union[list, tuple, int, str, None]) -> int:
        ret = self(number)
        return max(ret)


class ModelAnnotator:
    def __init__(self) -> None:
        self._build()
        super().__init__()

    def _build(self):
        self.id_transfomer = IDTransformer()
        self.connection_saver = ModelConnectionSaver()

    def _save_layer_or_model(self, layer_or_model: Union[tf.keras.layers.Layer, tf.keras.Model], in_node_ids: list, out_node_ids: list, depth: int):
        self.connection_saver.update(
            layer_or_model=layer_or_model, in_node_ids=in_node_ids, out_node_ids=out_node_ids, depth=depth)

    def _add_edge(self, layer_or_model: Union[tf.keras.layers.Layer, tf.keras.Model], depth: int) -> bool:
        if isinstance(layer_or_model, tf.keras.Model):
            model = layer_or_model
            if self.id_transfomer.possible(model.inbound_nodes) and \
                    self.id_transfomer.possible(model.outbound_nodes):
                in_node_ids = self.id_transfomer(model.inbound_nodes)
                out_node_ids = self.id_transfomer(model.outbound_nodes)

                self._save_layer_or_model(layer_or_model=model, in_node_ids=in_node_ids,
                                          out_node_ids=out_node_ids, depth=depth)
                last_layer, last_in_node_ids, _ = self.connection_saver.get_last_layer(
                    depth)

                if len(last_in_node_ids) != len(out_node_ids):
                    raise RuntimeError("Size of `{}` should be {} = {}".format(
                        last_in_node_ids, len(last_in_node_ids), len(out_node_ids)))

                self._save_layer_or_model(layer_or_model=last_layer, in_node_ids=last_in_node_ids,
                                          out_node_ids=out_node_ids, depth=depth)
            else:
                out_node_id = self.id_transfomer(None)
                last_layer, last_in_node_ids, _ = self.connection_saver.get_last_layer(
                    depth)
                self._save_layer_or_model(layer_or_model=last_layer, in_node_ids=last_in_node_ids,
                                          out_node_ids=[out_node_id], depth=depth)

        elif isinstance(layer_or_model, tf.keras.layers.Layer):
            ''' 
                This is the case of the normal keras layer such as `tf.layers.Conv2D`,  `tf.layers.BatchNormalization` ... 
                | `in_node_ids` |  => | keras layer | => | `out_node_ids` |
            '''
            layer = layer_or_model
            inbound_nodes = layer.inbound_nodes
            outbound_nodes = layer.outbound_nodes

            in_node_ids = self.id_transfomer(inbound_nodes)
            out_node_ids = self.id_transfomer(outbound_nodes)
            self._save_layer_or_model(layer_or_model=layer, in_node_ids=in_node_ids,
                                      out_node_ids=out_node_ids, depth=depth)
        return True

    def _model_loop(self, model: Union[tf.keras.Model, tf.keras.Sequential], depth=0):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                self._model_loop(layer, depth=depth + 1)
            elif isinstance(layer, tf.keras.layers.Layer):
                self._layer_loop(layer, depth)
            else:
                raise ValueError(
                    '`layer` can only be a `tf.keras.layers.Layer` instance. '
                    'You passed an instance of type: {input}.'.format(
                        input=layer.__class__.__name__))

        self._add_edge(model, depth)

    def _layer_loop(self, layer: Optional[tf.keras.layers.Layer], depth=0):
        if layer is None:
            raise ValueError('`layer` cannot be None.')
        if hasattr(layer, "_layers"):
            raise NotImplementedError("layer has a `_layers` attribute.")
        else:
            self._add_edge(layer, depth)

    def _annotate(self, model: Union[tf.keras.Model, tf.keras.Sequential]):
        self._model_loop(model)


    def annotate(self, model: Union[tf.keras.Model, tf.keras.Sequential]) -> ModelConnectionSaver:
        self._build()
        self._annotate(model)
        return self.connection_saver
