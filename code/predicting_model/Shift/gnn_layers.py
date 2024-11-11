import tensorflow as tf
import tensorflow_gnn as tfgnn

class hadamard_and_pooling(tfgnn.keras.layers.AnyToAnyConvolutionBase):
    
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self._message_fn = tf.keras.layers.Sequential(
            [tf.keras.layers.Dense(units, "softplus"),
             tf.keras.layers.Dense(units)]
        )
        
    def get_config(self):
        return dict(units=self._message_fn.units, **super().get_config())
    
    def convolve(self, *, sender_node_input, sender_edge_input, receiver_input,
                 broadcast_from_sender_node, broadcast_from_receiver, pool_to_receiver,
                 training):
        inputs = []
        inputs.append(tfgnn.pool_edges_to_node(sender_edge_input, edge_set_name="bond", node_tag=tfgnn.SOURCE, reduce_type="prod"))

        edge_message = tfgnn.combine_values([edge_embedding.edge_sets["bond"].__getitem__(tfgnn.HIDDEN_STATE), 
                                            edge_message.edge_sets["bond"].__getitem__(tfgnn.HIDDEN_STATE)], combine_type="sum")

        graph_message = tfgnn.pool_edges_to_node(edge_message, edge_set_name="bond", node_tag=tfgnn.SOURCE, reduce_type="prod")

        graph_message = tfgnn.pool_neighbors_to_node(graph_message, edge_set_name="bond", to_tag=tfgnn.SOURCE, reduce_type="sum")

        graph_message = self._message_fn(graph_message)

        return tfgnn.combine_values([node_embedding.node_sets["atom"].__getitem__(tfgnn.HIDDEN_STATE), 
                                    graph_message.node_sets["atom"].__getitem__(tfgnn.HIDDEN_STATE)], combine_type="sum")