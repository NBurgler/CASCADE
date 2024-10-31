import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import matplotlib.pyplot as plt
import datetime

def node_preprocessing(node_set, *, node_set_name):
    features = node_set.get_features_dict()

    features["degree"] = tf.keras.layers.Embedding(8, 4)
    features["explicit_valence"] = tf.keras.layers.Embedding(8, 4)
    features["formal_charge"] = tf.keras.layers.Embedding()
    features["implicit_valence"]
    features["num_explicit_hs"]
    features["num_implicit_hs"]

    return features


def edge_preprocessing(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    '''norm = tf.keras.layers.Normalization(axis=None)
    norm.adapt(features["distance"])
    norm(features["distance"])'''
    print(features["distance"])

    #features["distance"] = tf.keras.layers.BatchNormalization(axis=-1)(features["distance"])

    return features

def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(tf.expand_dims(distances, 1) - (-mu + delta * k))**2 / delta
    return tf.math.exp(logits)

def set_initial_node_state(node_set, *, node_set_name):
    # embed the different input features
    atom_sym_embedding = tf.keras.layers.Dense(1, name="atom_sym_embedding")(node_set["atom_sym"])
    chiral_tag_embedding = tf.keras.layers.Dense(2, name="chiral_tag_embedding")(node_set["chiral_tag"])
    hybridization_embedding = tf.keras.layers.Dense(2, name="hybridization_embedding")(node_set["hybridization"])
    degree_embedding = tf.keras.layers.Embedding(5, 2, name="degree_embedding")(node_set["degree"])
    formal_charge_embedding = tf.keras.layers.Embedding(3, 1, name="formal_charge_embedding")(node_set["formal_charge"])
    is_aromatic_embedding = tf.keras.layers.Embedding(2, 1, name="is_aromatic_embedding")(node_set["is_aromatic"])
    no_implicit_embedding = tf.keras.layers.Embedding(2, 1, name="no_implicit_embedding")(node_set["no_implicit"])
    num_Hs_embedding = tf.keras.layers.Embedding(5, 2, name="num_Hs_embedding")(node_set["num_Hs"])
    valence_embedding = tf.keras.layers.Embedding(5, 2, name="valence_embedding")(node_set["valence"])

    # concatenate the embeddings
    concatenated_embedding = tf.keras.layers.Concatenate()([atom_sym_embedding, chiral_tag_embedding,
                                                            hybridization_embedding, degree_embedding,
                                                            formal_charge_embedding, is_aromatic_embedding,
                                                            no_implicit_embedding, num_Hs_embedding,
                                                            valence_embedding])
    #concatenated_embedding = tf.keras.backend.print_tensor(concatenated_embedding)
    return tf.keras.layers.Dense(256, name="node_embedding")(concatenated_embedding)

def set_initial_edge_state(edge_set, *, edge_set_name):
    if edge_set_name == "bond":
        normalized_distance = tf.keras.layers.Reshape((-1,))(edge_set["distance"])
        #normalized_distance = tf.keras.backend.print_tensor(normalized_distance, summarize=-1)
        bond_type_embedding = tf.keras.layers.Dense(2, name="bond_type_embedding")(edge_set["bond_type"])
        is_conjugated_embedding = tf.keras.layers.Embedding(2, 1, name="is_conjugated_embedding")(edge_set["is_conjugated"])
        stereo_embedding = tf.keras.layers.Dense(2, name="stereo_embedding")(edge_set["stereo"])
        #distance = tf.keras.backend.print_tensor(distance, summarize=-1)
        #rbf_distance = rbf_expansion(edge_set["distance"])
        #rbf_distance = tf.keras.layers.Reshape((-1,))(rbf_distance)

        #rbf_distance = tf.keras.backend.print_tensor(rbf_distance, summarize=-1)
        # TODO: add other features
        #distance = tf.keras.backend.print_tensor(distance, summarize=-1)
        edge_embedding = tf.keras.layers.Concatenate()([normalized_distance, bond_type_embedding, is_conjugated_embedding, stereo_embedding])
    elif edge_set_name == "interatomic_distance":
        return rbf_expansion(edge_set["distance"])
    #edge_embedding = tf.keras.backend.print_tensor(edge_embedding, summarize=-1)
    return tf.keras.layers.Dense(256, name="edge_embedding")(edge_embedding)

def dense(units, activation=None):
        """A Dense layer with regularization (L2 and Dropout)."""
        l2_regularization = 1e-5
        #dropout_rate = 0.1
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer=tf.keras.initializers.Zeros()),
            #tf.keras.layers.Dropout(dropout_rate)
        ])

def edge_updating():
    return tf.keras.Sequential([
        dense(512, activation="relu"),
        dense(256),
        dense(256, activation="relu"),
        dense(256, activation="relu")])


def node_updating():
    return tf.keras.Sequential([
        dense(256),
        dense(256, activation="relu"),
        dense(256)])

def readout_layers():
    return tf.keras.Sequential([
        dense(256, activation="relu"),
        dense(256, activation="relu")])


def _build_model(graph_tensor_spec):
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    #graph = input_graph.merge_batch_to_components()

    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(input_graph)

    for _ in range(3):
        graph = tfgnn.keras.layers.GraphUpdate(
            edge_sets={"interatomic_distance": tfgnn.keras.layers.EdgeSetUpdate(
                next_state=tfgnn.keras.layers.ResidualNextState(node_updating())
            )},
            node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
                {"interatomic_distance": tfgnn.keras.layers.Pool(
                    reduce_type="mean|sum", 
                    tag=tfgnn.TARGET)},
                next_state=tfgnn.keras.layers.ResidualNextState(edge_updating())
            )}   
        )(graph)

    readout_features = tfgnn.keras.layers.StructuredReadout("shape")(graph)
    logits = readout_layers()(readout_features)

    logits = tf.expand_dims(logits, axis=1)
    logits = tf.concat([logits,logits,logits,logits,logits,logits], axis=1)

    lstm = tf.keras.layers.LSTM(8, activation=tf.keras.activations.softmax, return_sequences=True)(logits)

    return tf.keras.Model(inputs=[input_graph], outputs=[lstm])


if __name__ == "__main__":
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "C:/Users/niels/Documents/repo/CASCADE/"
    batch_size = 32
    initial_learning_rate = 5E-4
    epochs = 5
    epoch_divisor = 1

    train_path = path + "data/own_data/shape/own_train.tfrecords"
    val_path = path + "data/own_data/shape/own_valid.tfrecords"

    train_size = 63324
    valid_size = 13569
    test_size = 13571

    steps_per_epoch = train_size // batch_size // epoch_divisor
    validation_steps = valid_size // batch_size // epoch_divisor
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    train_ds = tf.data.TFRecordDataset([train_path])
    val_ds = tf.data.TFRecordDataset([val_path])
    train_ds = train_ds.batch(batch_size=batch_size).repeat()
    val_ds = val_ds.batch(batch_size=batch_size)

    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchemaMult.pbtxt")
    example_input_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    train_ds = train_ds.map(tfgnn.keras.layers.ParseExample(example_input_spec))
    val_ds = val_ds.map(tfgnn.keras.layers.ParseExample(example_input_spec))
    preproc_input_spec = train_ds.element_spec

    # preprocessing
    preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)
    #graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=node_preprocessing, edge_sets_fn=edge_preprocessing)(preproc_input)
    graph = preproc_input
    graph = graph.merge_batch_to_components()
    labels = tfgnn.keras.layers.Readout(node_set_name="_readout",
                                    feature_name="shape")(graph)
    graph = graph.remove_features(node_sets={"_readout": ["shape"]})
    preproc_model = tf.keras.Model(preproc_input, (graph, labels))
    train_ds = train_ds.map(preproc_model)
    val_ds = val_ds.map(preproc_model)

    # model
    model_input_spec, _ = train_ds.element_spec
    model = _build_model(model_input_spec)

    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.losses.CategoricalCrossentropy()]

    model.compile(tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)
    model.summary()

    code_path = path + "code/predicting_model/Multiplicity/"
    filepath = code_path + "gnn/models/mult_model/checkpoint.weights.h5"
    log_dir = code_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, save_freq="epoch", verbose=1, monitor="val_categorical_crossentropy", save_weights_only=True)
    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback, checkpoint]) 

    #load best weights before saving
    model.load_weights(filepath)

    serving_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name="examples")
    preproc_input = tfgnn.keras.layers.ParseExample(example_input_spec)(serving_input)
    serving_model_input, _ = preproc_model(preproc_input)
    serving_logits = model(serving_model_input)
    serving_output = {"shape": serving_logits}
    exported_model = tf.keras.Model(serving_input, serving_output)
    exported_model.export(code_path + "gnn/models/mult_model")
   
    #for layer in model.layers: print(layer.get_config(), layer.get_weights())