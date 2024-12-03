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
    one_hot_inputs = tf.keras.layers.Concatenate()([node_set["atom_sym"], node_set["chiral_tag"], node_set["hybridization"]])

    
    integer_inputs = tf.cast(tf.keras.layers.Concatenate()([node_set["degree"], node_set["formal_charge"], node_set["is_aromatic"], 
                                                   node_set["no_implicit"], node_set["num_Hs"], node_set["valence"]]), tf.float32)
    

    # concatenate the embeddings
    concatenated_inputs = tf.keras.layers.Concatenate()([one_hot_inputs, integer_inputs])
    #concatenated_embedding = tf.keras.backend.print_tensor(concatenated_embedding)
    return tf.keras.layers.Dense(256, name="node_embedding")(concatenated_inputs)

def set_initial_edge_state(edge_set, *, edge_set_name):
    if edge_set_name == "bond":
        one_hot_inputs = tf.keras.layers.Concatenate()([edge_set["bond_type"], edge_set["stereo"]])
        integer_inputs = tf.cast(edge_set["is_conjugated"], tf.float32)
        float_inputs = edge_set["normalized_distance"]
                                
        edge_inputs = tf.keras.layers.Concatenate()([one_hot_inputs, integer_inputs, float_inputs])

    elif edge_set_name == "interatomic_distance":
        return rbf_expansion(edge_set["distance"])
    #edge_embedding = tf.keras.backend.print_tensor(edge_embedding, summarize=-1)
    return tf.keras.layers.Dense(256, name="edge_embedding")(edge_inputs)

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
        dense(256, activation="relu"),
        dense(128, activation="relu"),
        tf.keras.layers.Dense(1)])


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

    readout_features = tfgnn.keras.layers.StructuredReadout("shift")(graph)
    logits = readout_layers()(readout_features)

    return tf.keras.Model(inputs=[input_graph], outputs=[logits])


if __name__ == "__main__":
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    
    batch_size = 32
    initial_learning_rate = 5E-4
    epochs = 10
    epoch_divisor = 1

    train_path = path + "data/own_data/shift/DFT_train.tfrecords"
    val_path = path + "data/own_data/shift/DFT_valid.tfrecords"


    #train_size = 8
    #valid_size = 2
    #test_size = 2

    #train_size = 63324
    #valid_size = 13569
    #test_size = 13571

    #train_size = 175324
    #valid_size = 37570
    #test_size =  37570

    train_size = 5217
    valid_size = 1118
    test_size = 1119

    steps_per_epoch = train_size // batch_size // epoch_divisor
    validation_steps = valid_size // batch_size // epoch_divisor
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    train_ds = tf.data.TFRecordDataset([train_path])
    val_ds = tf.data.TFRecordDataset([val_path])
    train_ds = train_ds.batch(batch_size=batch_size).repeat()
    val_ds = val_ds.batch(batch_size=batch_size)

    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchema.pbtxt")
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

    model.compile(tf.keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics)
    model.summary()

    code_path = path + "code/predicting_model/Shift/DFTNN/"
    filepath = code_path + "gnn/models/DFT_model/checkpoint.weights.h5"
    log_dir = code_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, save_freq="epoch", verbose=1, monitor="val_mean_absolute_error", save_weights_only=True)
    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback, checkpoint]) 

    #load best weights before saving
    model.load_weights(filepath)

    serving_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name="examples")
    preproc_input = tfgnn.keras.layers.ParseExample(example_input_spec)(serving_input)
    serving_model_input, _ = preproc_model(preproc_input)
    serving_logits = model(serving_model_input)
    serving_output = {"shifts": serving_logits}
    exported_model = tf.keras.Model(serving_input, serving_output)
    exported_model.export(code_path + "gnn/models/DFT_model")
   
    #for layer in model.layers: print(layer.get_config(), layer.get_weights())