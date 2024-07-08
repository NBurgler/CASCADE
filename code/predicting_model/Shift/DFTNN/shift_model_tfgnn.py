import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import matplotlib.pyplot as plt

os.chdir("C:/Users/niels/Documents/repo/CASCADE")

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
    return concatenated_embedding

def set_initial_edge_state(edge_set, *, edge_set_name):
    distance = tf.keras.layers.Reshape((-1,))(edge_set["distance"])
    #distance = tf.keras.backend.print_tensor(distance, summarize=-1)
    rbf_distance = rbf_expansion(edge_set["distance"])
    rbf_distance = tf.keras.layers.Reshape((-1,))(rbf_distance)

    #rbf_distance = tf.keras.backend.print_tensor(rbf_distance, summarize=-1)
    # TODO: add other features
    #distance = tf.keras.backend.print_tensor(distance, summarize=-1)
    edge_embedding = tf.keras.layers.Dense(256, name="edge_init")(rbf_distance)
    #edge_embedding = tf.keras.backend.print_tensor(edge_embedding, summarize=-1)
    return edge_embedding

def _build_model(graph_tensor_spec):
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    #graph = input_graph.merge_batch_to_components()

    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(input_graph)

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

    for _ in range(3):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
                {"bond": tfgnn.keras.layers.SimpleConv(
                    message_fn = dense(512, activation="relu"), 
                    reduce_type="sum", 
                    receiver_tag=tfgnn.TARGET)},
                next_state=tfgnn.keras.layers.ResidualNextState(dense(256, activation="relu"))
            )}
        )(graph)

    readout_features = tfgnn.keras.layers.StructuredReadout("shift")(graph)
    logits = tf.keras.layers.Dense(1)(readout_features)

    return tf.keras.Model(inputs=[input_graph], outputs=[logits])


if __name__ == "__main__":
    batch_size = 32
    initial_learning_rate = 5E-4
    epochs = 5
    epoch_divisor = 1

    train_path = os.path.join(os.getcwd(), 'data\own_data\shift_train.tfrecords')
    val_path = os.path.join(os.getcwd(), 'data\own_data\shift_valid.tfrecords')

    train_size = 63565
    valid_size = 13622
    test_size = 13622

    steps_per_epoch = train_size // batch_size // epoch_divisor
    validation_steps = valid_size // batch_size // epoch_divisor
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    train_ds = tf.data.TFRecordDataset([train_path])
    val_ds = tf.data.TFRecordDataset([val_path])
    train_ds = train_ds.batch(batch_size=batch_size).repeat()
    val_ds = val_ds.batch(batch_size=batch_size)

    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    train_ds = train_ds.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
    val_ds = val_ds.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
    preproc_input_spec = train_ds.element_spec

    # preprocessing
    preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)
    graph = preproc_input
    graph = graph.merge_batch_to_components()
    labels = tfgnn.keras.layers.Readout(node_set_name="_readout",
                                    feature_name="shift")(graph)
    graph = graph.remove_features(node_sets={"_readout": ["shift"]})
    preproc_model = tf.keras.Model(preproc_input, (graph, labels))
    train_ds = train_ds.map(preproc_model)
    val_ds = val_ds.map(preproc_model)

    # model
    model_input_spec, _ = train_ds.element_spec
    model = _build_model(model_input_spec)

    loss = tf.keras.losses.MeanAbsoluteError()
    metrics = [tf.keras.losses.MeanAbsoluteError()]

    model.compile(tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)
    model.summary()

    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs, validation_data=val_ds)
    #history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs, validation_data=val_ds)

    for k, hist in history.history.items():
        plt.plot(hist)
        plt.title(k)
        plt.show()