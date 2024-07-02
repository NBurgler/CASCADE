import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models import mt_albis
import pandas as pd
import pickle
import sys
import functools
import matplotlib.pyplot as plt
import datetime

import wandb
from wandb.keras import WandbCallback

'''wandb.init(
        project="nmr-prediction",
        dir="/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/H/DFTNN",
        config={
        "learning_rate": 5E-4,
        "dataset": "cascade",
        "epochs": 10,
        "setting": "tfgnn",
    }
)'''

sys.path.append('code/predicting_model')

class GradientLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(GradientLogger, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for layer in self.model.layers:
                if hasattr(layer, 'trainable_weights'):
                    for weight in layer.trainable_weights:
                        gradients = self.model.optimizer.get_gradients(self.model.total_loss, weight)
                        for grad, var in zip(gradients, self.model.trainable_weights):
                            tf.summary.histogram(var.name + '/gradients', grad, step=epoch)
            self.writer.flush()
    
def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(tf.expand_dims(distances, 1) - (-mu + delta * k))**2 / delta
    return tf.math.exp(logits)

def set_initial_node_state(node_set, *, node_set_name):
    # one-hot encoded features are embedded immediately
    atom_num_embedding = tf.keras.layers.Dense(64, name="atom_num_embedding")(node_set["atom_num"])
    chiral_tag_embedding = tf.keras.layers.Dense(64, name="chiral_tag_embedding")(node_set["chiral_tag"])
    hybridization_embedding = tf.keras.layers.Dense(64, name="hybridization_embedding")(node_set["hybridization"])

    # other numerical features are first reshaped...
    '''degree = tf.keras.layers.Reshape((-1, 1))(node_set["degree"])
    explicit_valence = tf.keras.layers.Reshape((-1, 1))(node_set["explicit_valence"])
    formal_charge = tf.keras.layers.Reshape((-1, 1))(node_set["formal_charge"])
    implicit_valence = tf.keras.layers.Reshape((-1, 1))(node_set["implicit_valence"])
    is_aromatic = tf.keras.layers.Reshape((-1, 1))(node_set["is_aromatic"])
    no_implicit = tf.keras.layers.Reshape((-1, 1))(node_set["no_implicit"])
    num_explicit_Hs = tf.keras.layers.Reshape((-1, 1))(node_set["num_explicit_Hs"])
    num_implicit_Hs = tf.keras.layers.Reshape((-1, 1))(node_set["num_implicit_Hs"])
    num_radical_electrons = tf.keras.layers.Reshape((-1, 1))(node_set["num_radical_electrons"])
    total_degree = tf.keras.layers.Reshape((-1, 1))(node_set["total_degree"])
    total_num_Hs = tf.keras.layers.Reshape((-1, 1))(node_set["total_num_Hs"])
    total_valence = tf.keras.layers.Reshape((-1, 1))(node_set["total_valence"])'''

    degree = tf.keras.layers.Reshape((-1,))(node_set["degree"])
    explicit_valence = tf.keras.layers.Reshape((-1,))(node_set["explicit_valence"])
    formal_charge = tf.keras.layers.Reshape((-1,))(node_set["formal_charge"])
    implicit_valence = tf.keras.layers.Reshape((-1,))(node_set["implicit_valence"])
    is_aromatic = tf.keras.layers.Reshape((-1,))(node_set["is_aromatic"])
    no_implicit = tf.keras.layers.Reshape((-1,))(node_set["no_implicit"])
    num_explicit_Hs = tf.keras.layers.Reshape((-1,))(node_set["num_explicit_Hs"])
    num_implicit_Hs = tf.keras.layers.Reshape((-1,))(node_set["num_implicit_Hs"])
    num_radical_electrons = tf.keras.layers.Reshape((-1,))(node_set["num_radical_electrons"])
    total_degree = tf.keras.layers.Reshape((-1,))(node_set["total_degree"])
    total_num_Hs = tf.keras.layers.Reshape((-1,))(node_set["total_num_Hs"])
    total_valence = tf.keras.layers.Reshape((-1,))(node_set["total_valence"])

    # ... and then concatenated so that they can be embedded as well
    numerical_features = tf.keras.layers.Concatenate(axis=-1)([degree, explicit_valence, formal_charge, implicit_valence, is_aromatic, no_implicit,
                                                              num_explicit_Hs, num_implicit_Hs, num_radical_electrons, total_degree, total_num_Hs, 
                                                              total_valence])
    #numerical_features = tf.keras.backend.print_tensor(numerical_features, summarize=-1)
    numerical_embedding = tf.keras.layers.Dense(64, name="numerical_embedding")(numerical_features)
    #numerical_embedding = tf.keras.backend.print_tensor(numerical_embedding, summarize=-1)
    
    # the one-hot and numerical embeddings are concatenated and fed to another dense layer
    concatenated_embedding = tf.keras.layers.Concatenate()([atom_num_embedding, chiral_tag_embedding,
                                                            hybridization_embedding, numerical_embedding])
    #concatenated_embedding = tf.keras.backend.print_tensor(concatenated_embedding)
    return tf.keras.layers.Dense(256, name="node_init")(concatenated_embedding)

def set_initial_edge_state(edge_set, *, edge_set_name):
    distance = tf.keras.layers.Reshape((-1,))(edge_set["distance"])
    #distance = tf.keras.backend.print_tensor(distance, summarize=-1)
    rbf_distance = rbf_expansion(edge_set["distance"])
    rbf_distance = tf.keras.layers.Reshape((-1,))(rbf_distance)

    #rbf_distance = tf.keras.backend.print_tensor(rbf_distance, summarize=-1)
    # TODO: add other features
    edge_embedding = tf.keras.layers.Dense(256, name="edge_init")(rbf_distance)
    #edge_embedding = tf.keras.backend.print_tensor(edge_embedding, summarize=-1)
    return rbf_distance
    
def dense(units, activation=None):
    """A Dense layer with regularization (L2 and Dropout)."""
    l2_regularization = 5e-4
    dropout_rate = 0.5
    regularizer = tf.keras.regularizers.l2(l2_regularization)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            bias_initializer=tf.keras.initializers.Zeros()),
        tf.keras.layers.Dropout(dropout_rate)
    ])

def edge_updating():
    return tf.keras.Sequential([
        dense(512, activation="relu"),
        dense(256),
        dense(256, activation="relu"),
        dense(256, activation="relu")])


def node_updating():
    return tf.keras.Sequential([
        dense(256, activation="relu"),
        dense(256)])

def readout_layers():
    return tf.keras.Sequential([
        dense(256, activation="relu"),
        dense(256, activation="relu"),
        dense(128, activation="relu"),
        dense(1)])


def model_fn(graph_tensor_spec: tfgnn.GraphTensorSpec):
    #preprocessing layers
    graph = inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(graph)
    #labels = tf.keras.backend.print_tensor(graph.node_sets["_readout"].__getitem__("shift"), summarize=-1)
    # Message passing layers
    # First, the edges are updated according to the layers in edge_updating
    # Then, the updated edge states are pooled to the node states (hadamard product + element-wise sum)
    # Finally, the pooled edge states are updated according to the layers in node_updating

    for _ in range(3):
        graph = mt_albis.MtAlbisGraphUpdate(
            units=256,
            message_dim=128,
            receiver_tag=tfgnn.TARGET,
            simple_conv_reduce_type="mean|sum",
            state_dropout_rate=0.2,
            l2_regularization=1e-5,
            normalization_type="layer",
            next_state_type="residual",
            # More hyperparameters like edge_dropout_rate can be added here.
        )(graph)
        '''graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
                {"bond": tfgnn.keras.layers.SimpleConv(
                    message_fn = dense(512, activation="relu"), 
                    reduce_type="sum", 
                    receiver_tag=tfgnn.TARGET)},
                next_state=tfgnn.keras.layers.ResidualNextState(dense(256, activation="relu"))
            )}
        )(graph)'''
        '''},
            edge_sets={"bond": tfgnn.keras.layers.EdgeSetUpdate(
                next_state=tfgnn.keras.layers.ResidualNextState(edge_updating())
            )}
        )(graph)'''
    '''graph = tfgnn.keras.layers.GraphUpdate(
            edge_sets={"bond": tfgnn.keras.layers.EdgeSetUpdate(
                    next_state=tfgnn.keras.layers.ResidualNextState(
                        residual_block=edge_updating()
                )
            )},
            node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
                {"bond": tfgnn.keras.layers.Pool(tag=tfgnn.TARGET, reduce_type="prod|sum")},
                next_state=tfgnn.keras.layers.ResidualNextState(
                    residual_block=node_updating()
                )
            )}
        )(graph)'''

    #readout layers
    '''graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
            {"bond": tfgnn.keras.layers.Pool(tag=tfgnn.TARGET, reduce_type="sum")},
            next_state=tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(128))
        )}
    )(graph)

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
            {"bond": tfgnn.keras.layers.Pool(tag=tfgnn.TARGET, reduce_type="sum")},
            next_state=tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(64))
        )}
    )(graph)

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
            {"bond": tfgnn.keras.layers.Pool(tag=tfgnn.TARGET, reduce_type="sum")},
            next_state=tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(1))
        )}
    )(graph)'''
    #output = graph

    '''readout_features = tfgnn.keras.layers.StructuredReadout("shift")(graph)
    logits = tf.keras.layers.Dense(256, activation="relu")(readout_features)
    logits = tf.keras.layers.Dense(256, activation="relu")(logits)
    logits = tf.keras.layers.Dense(128, activation="relu")(logits)
    output = tf.keras.layers.Dense(1)(logits)'''
    
    return tf.keras.Model(inputs=[inputs], outputs=[graph])

if __name__ == "__main__":
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"
    batch_size = 32
    initial_learning_rate = 5E-4
    epochs = 10
    epoch_divisor = 1

    train_ds_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_train.tfrecords"])
    test_ds_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_test.tfrecords"])
    valid_ds_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_valid.tfrecords"])

    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    train_size = 63565
    valid_size = 13622
    test_size = 13622

    task = runner.NodeMeanAbsoluteError("shift", label_feature_name="shift")
    steps_per_epoch = train_size // batch_size // epoch_divisor
    validation_steps = valid_size // batch_size // epoch_divisor
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    optimizer_fn = functools.partial(tf.keras.optimizers.Adam, learning_rate=learning_rate)

    filepath = path + "gnn/models/"
    log_dir = path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, period=1, verbose=1)
    gradient_logger = GradientLogger(log_dir=log_dir)

    trainer = runner.KerasTrainer(
        strategy=tf.distribute.MirroredStrategy(),
        model_dir=path + "tmp/gnn_model/",
        callbacks=[tensorboard_callback, checkpoint],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        restore_best_weights=False,
        checkpoint_every_n_steps="never",
        summarize_every_n_steps="never",
        backup_and_restore=False,
    )

    model_exporter = runner.KerasModelExporter(output_names="shifts")

    runner.run(
        gtspec=graph_tensor_spec,
        train_ds_provider=train_ds_provider,
        valid_ds_provider=valid_ds_provider,
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        trainer=trainer,
        task=task,
        global_batch_size=batch_size,
        epochs=epochs,
        model_exporters=[model_exporter]
    )