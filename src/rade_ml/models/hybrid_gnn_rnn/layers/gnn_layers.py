import copy
import logging
import tensorflow as tf
from typing import Dict, Any, Tuple, Union

# define logging at module level.
logger = logging.getLogger(__name__)


class GnnBlock(tf.keras.layers.Layer):
    """
    Graph neural network block with residual connections and multiple GNN layers: [GraphSAGE, MixedGraphSAGE]

    This block implements a stack of GNN layers with skip connections for better gradient flow and feature preservation.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        """
        Initialise the graphical neural network block.

        :param layer_config: dictionary containing general layer configuration & parameters.
        """
        # init call to super class
        super().__init__(**kwargs)

        # initiate required variables.
        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        # unpack layer configuration --> general.
        self.layers: int = 1
        self.layer_type: str | None = None
        self.dropout_rate: float = 0.0
        self.use_bias: bool | None = None
        self.use_residual: bool | None = None
        self.layer_norm: bool = False
        self._unpack_configuration(config=layer_config.get('general'))

        # unpack layer configuration --> parameters.
        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        # initiate variables to build.
        self.gnn_layers: list = []
        self.norm_layers: list = []
        self.input_projection: None = None
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_dropout') if (
                self.dropout_rate > 0.0) else None

    def build_gnn_layers(self) -> None:
        """
        Build dynamic GNN layers.

        :return:
        """
        # create gnn layers.
        for i in range(self.layers):
            # determine if this is the last layer for activation purposes.
            is_last_layer = i == self.layers - 1
            layer_activation = None if is_last_layer else self.activation # no activation on last layer for residual.

            # update gnn configuration with activation parameter for sub layers.
            layer_config = self.layer_config.copy()
            layer_config['parameters']['activation'] = None

            # create appropriate gnn sublayer.
            if self.layer_type.lower() == 'graph_sage':
                name = f'{self.name}_{self.layer_type.lower()}_{i}'
                layer = GraphSage(layer_config=layer_config, name=name)
            elif self.layer_type.lower() == 'mixed_graph_sage':
                name = f'{self.name}_{self.layer_type.lower()}_{i}'
                layer = MixedGraphSage(layer_config=layer_config, name=name)
            else:
                raise ValueError(f"Undefined layer type, got {self.layer_type.lower()}")

            # append gnn layers to list.
            self.__setattr__(name, layer)
            self.gnn_layers.append(layer)

            # create batch normalisation layer, if needed.
            if self.layer_norm:
                name = f'{self.name}_ln_{i}'
                layer = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=name)

                # append batch norm layer
                self.__setattr__(name, layer)
                self.norm_layers.append(layer)

        # initialise gnn layer and batch norm layer containers.
        self.__setattr__("gnn_layers", self.gnn_layers)
        self.__setattr__("norm_layers", self.norm_layers)

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        """
        Build gnn block with given input shape.

        :param input_shape: tuple of input shapes:
            - trade features [num_trades, features]
            - adjacency matrix [num_trades, num_trades]
        :return:
        """
        # extract input shapes.
        _, _ = input_shape

        # build dynamic gnn layers
        self.build_gnn_layers()

        # build input projection.
        if self.use_residual and self.input_projection is None:
            self.input_projection = tf.keras.layers.Dense(
                units=self.units, kernel_initializer=self.kernel_initializer, use_bias=False,
                name=f'{self.name}_projection'
            )
            self.__setattr__('input_projection', self.input_projection)

        # update layer build flag.
        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor]], training: bool = False) -> tf.Tensor:
        """
        Forward pass of GnnBlock

        :param inputs: tuple of inputs:
            - trade features [num_trades, features]
            - adjacency matrix [num_trades, num_trades]
        :param training: whether in training mode.
        :return:
        """
        # extract inputs
        features, adjacency = inputs

        # store input for residual connections.
        residual = features

        # project input if needed for residual connections.
        if self.use_residual:
            residual = self.input_projection(residual)

        # apply input to gnn sub layers.
        x = features
        for i, gnn_layer in enumerate(self.gnn_layers):
            # apply gnn layer.
            x = gnn_layer((x, adjacency), training=training)

            # apply batch normalisation, if needed.
            if self.layer_norm and i < len(self.norm_layers):
                x = self.norm_layers[i](x, training=training)

            # apply dropout, if needed.
            if i < self.layers - 1:
                x = self._activation(x)
                if hasattr(self, 'dropout') and self.dropout is not None:
                    x = self.dropout(x, training=training)

        # add residual connection.
        if self.use_residual:
            x += residual

        # block level linearity.
        x = self._activation(x)

        # return gnn layer output --> [num_trades, gnn_units]
        return x

    def compute_output_shape(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> tf.TensorShape:
        """
        Compute output shape of the layer.

        :param input_shape: tuple of input shapes:
            - trade features [num_trades, features]
            - adjacency matrix [num_trades, num_trades]
        :return:
        """
        features_shape, _ = input_shape
        return tf.TensorShape([features_shape[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for serializing the layer.

        :return:
        """
        config = super(GnnBlock, self).get_config()
        config.update({
            'layer_config': self.layer_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GnnBlock":
        """
        Instantiates the GnnBlock from its config.

        :param config:
        :return:
        """
        return cls(**config)

    def _activation(self, output: tf.Tensor):
        """
        Activation function helper
        :param output: tf.tensor to apply activation function.
        :return:
        """
        if self.activation == 'leaky_relu':
            return tf.nn.leaky_relu(output, alpha=0.2)
        elif self.activation:
            return tf.keras.activations.get(self.activation)(output)
        return output

    def _unpack_configuration(
            self, config: Dict[str, Any]
    ) -> None:
        """
        Unpack configuration elements into separate variables.
        :param config: dictionary configuration.
        :return:
        """
        for k, v in config.items():
            setattr(self, k, v)


class GraphSage(tf.keras.layers.Layer):
    """
    Inductive GraphSAGE layer with mean, sum or max aggregator.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        """
        Initialise graph sage layer.

        :param layer_config: dictionary containing general layer configuration & parameters.
        """
        # init call to super class
        super().__init__(**kwargs)

        # initiate required variables.
        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        # unpack layer configuration --> general.
        self.layers: int = 1
        self.layer_type: str | None = None
        self.dropout_rate: float = 0.0
        self.use_bias: bool | None = None
        self.aggregation_op: str = 'mean'
        self._unpack_configuration(config=layer_config.get('general'))

        # unpack layer configuration --> parameters.
        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        # initiate variables to build.
        self.dense_self = None
        self.dense_neigh = None
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_dropout') if (
                self.dropout_rate > 0.0) else None


    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        """
        Build the layer with input shape.

        :param input_shape: tuple of input shapes:
            - trade features [num_trades, features]
            - adjacency matrix [num_trades, num_trades]
        :return:
        """
        # extract input shapes.
        _, _ = input_shape

        # self and neighbours transformations.
        self.dense_self = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias,
            name=f'{self.name}_weights_self'
        )
        self.dense_neigh = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias,
            name=f'{self.name}_weights_neigh'
        )

        # update layer build flag.
        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor]], training: bool = False) -> tf.Tensor:
        """
        Forward pass of the GraphSage layer.

        :param inputs: tuple of inputs:
            - trade features [num_trades, features]
            - adjacency matrix [num_trades, num_trades]
        :param training: whether in training mode.
        :return:
        """
        # extract inputs
        x, a = inputs
        num_trades = tf.shape(x)[0]

        # create flag for sparse tensor adjacency matrix.
        is_sparse = isinstance(a, tf.SparseTensor)

        # optional input dropout.
        if getattr(self, "dropout", None) is not None:
            x = self.dropout(x, training=training)

        # choose aggregator.
        if self.aggregation_op.lower() == 'mean':
            # mean aggregator. -> since adj is row normalised sum and mean are the same.
            if is_sparse:
                neigh_summary = tf.sparse.sparse_dense_matmul(a, x)
            else:
                neigh_summary = tf.matmul(a, x)
        elif self.aggregation_op.lower() == 'max':
            if is_sparse:
                # sparse max via segment max
                rows = a.indices[:, 0]
                cols = a.indices[:, 1]
                gathered = tf.gather(x, cols)
                neigh_summary = tf.math.unsorted_segment_max(data=gathered, segment_ids=rows, num_segments=num_trades)
                neigh_summary = tf.where(tf.math.is_finite(neigh_summary), neigh_summary, tf.zeros_like(neigh_summary))
            else:
                # dense max via boolean mask.
                mask = a > 0
                idx = tf.where(mask)
                rows = idx[:, 0]
                cols = idx[:, 1]
                gathered = tf.gather(x, cols)
                neigh_summary = tf.math.unsorted_segment_max(gathered, rows, num_segments=num_trades)
                neigh_summary = tf.where(tf.math.is_finite(neigh_summary), neigh_summary, tf.zeros_like(neigh_summary))
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregation_op.lower()}...")

        # linear transformation and combine.
        h_self = self.dense_self(x)                             # [n, gnn_units]
        h_neigh = self.dense_neigh(neigh_summary)               # [n, gnn_units]
        out = h_self + h_neigh                                  # [n, gnn_units]
        return tf.keras.activations.get(self.activation)(out)   # [n, gnn_units]

    def compute_output_shape(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> tf.TensorShape:
        """
        Compute output shape of the layer.
        :param input_shape:
        :return:
        """
        features_shape, _ = input_shape
        return tf.TensorShape([features_shape[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for serializing the layer.

        :return:
        """
        config = super(GraphSage, self).get_config()
        config.update({
            'layer_config': self.layer_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphSage":
        """
        Instantiates the GraphSAGE from its config.

        :param config:
        :return:
        """
        return cls(**config)

    def _unpack_configuration(
            self, config: Dict[str, Any]
    ) -> None:
        """
        Unpack configuration elements into separate variables.
        :param config: dictionary configuration.
        :return:
        """
        for k, v in config.items():
            setattr(self, k, v)


class MixedGraphSage(tf.keras.layers.Layer):
    """
    Inductive mixed aggregation GraphSage layer; concatenates mean, sum or max neighbours features.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        """
        Initialise mixed graph sage layer.

        :param layer_config: dictionary containing general layer configuration & parameters.
        """
        # init call to super class
        super().__init__(**kwargs)

        # initiate required variables.
        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        # unpack layer configuration --> general.
        self.layers: int = 1
        self.layer_type: str | None = None
        self.dropout_rate: float = 0.0
        self.use_bias: bool | None = None
        self._unpack_configuration(config=layer_config.get('general'))

        # unpack layer configuration --> parameters.
        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        # initiate variables to build.
        self.fusion_dense = None
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_dropout') if (
                self.dropout_rate > 0.0) else None

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        """
        Build the layer with input shape.

        :param input_shape: tuple of input shapes:
            - trade features [num_trades, features]
            - adjacency matrix [num_trades, num_trades]
        :return:
        """
        # extract input shapes.
        _, _ = input_shape

        # self and neighbours transformations.
        self.fusion_dense = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias,
            name=f'{self.name}_fusion_dense'
        )

        # update layer build flag.
        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor]], training: bool = False) -> tf.Tensor:
        """
        Forward pass of MixedGraphSage layer.

        :param inputs: tuple of inputs:
            - trade features [num_trades, features]
            - adjacency matrix [num_trades, num_trades]
        :param training: whether in training mode.
        :return:
        """
        # extract inputs
        x, a = inputs
        num_trades = tf.shape(x)[0]

        # create flag for sparse tensor adjacency matrix.
        is_sparse = isinstance(a, tf.SparseTensor)

        # optional input dropout.
        if getattr(self, "dropout", None) is not None:
            x = self.dropout(x, training=training)

        # mean aggregator. -> since adj is row normalised sum and mean are the same.
        if is_sparse:
            neigh_mean = tf.sparse.sparse_dense_matmul(a, x)
        else:
            neigh_mean = tf.matmul(a, x)

        # max aggregation
        if is_sparse:
            # sparse max via segment max
            rows = a.indices[:, 0]
            cols = a.indices[:, 1]
            gathered = tf.gather(x, cols)
            neigh_max = tf.math.unsorted_segment_max(data=gathered, segment_ids=rows, num_segments=num_trades)
            neigh_max = tf.where(tf.math.is_finite(neigh_max), neigh_max, tf.zeros_like(neigh_max))
        else:
            # dense max via boolean mask.
            mask = a > 0
            idx = tf.where(mask)
            rows = idx[:, 0]
            cols = idx[:, 1]
            gathered = tf.gather(x, cols)
            neigh_max = tf.math.unsorted_segment_max(gathered, rows, num_segments=num_trades)
            neigh_max = tf.where(tf.math.is_finite(neigh_max), neigh_max, tf.zeros_like(neigh_max))

        # concatenate self, mean, sum, max
        concat_feats = tf.concat([x, neigh_mean, neigh_max], axis=1)     # [n, 4f]
        out = self.fusion_dense(concat_feats)                                       # [n, gnn_units]
        return tf.keras.activations.get(self.activation)(out)

    def compute_output_shape(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> tf.TensorShape:
        """
        Compute output shape of the layer.
        :param input_shape:
        :return:
        """
        features_shape, _ = input_shape
        return tf.TensorShape([features_shape[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for serializing the layer.

        :return:
        """
        config = super(MixedGraphSage, self).get_config()
        config.update({
            'layer_config': self.layer_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MixedGraphSage":
        """
        Instantiates the GraphSAGE from its config.

        :param config:
        :return:
        """
        return cls(**config)

    def _unpack_configuration(
            self, config: Dict[str, Any]
    ) -> None:
        """
        Unpack configuration elements into separate variables.
        :param config: dictionary configuration.
        :return:
        """
        for k, v in config.items():
            setattr(self, k, v)
