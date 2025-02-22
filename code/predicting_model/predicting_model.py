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
        tf.keras.layers.Dense(1, activation="relu")], name="shift")

def mask_couplings(inputs):
    shapes, couplings = inputs
    shapes_decoded = tf.argmax(tf.stop_gradient(shapes), axis=-1)

    # mask is 0 when shape is s or m (argmax == (0 V 1))
    mask = tf.cast(tf.logical_not(tf.reduce_any(tf.stack([shapes_decoded == 0, shapes_decoded == 1]), axis=0)), tf.float32)

    mask = tf.expand_dims(mask, axis=-1)

    masked_couplings = couplings * mask

    return masked_couplings

def update_graph(graph, new_feature, new_feature_name):
    # Access the existing features of the "atom" node set
    existing_features = graph.node_sets["atom"].features

    # Ensure that new_feature is a tf.Tensor
    if not isinstance(new_feature, tf.Tensor):
        raise ValueError("new_feature must be a tf.Tensor")

    # Create a dictionary for the updated features by combining existing features with the new feature
    updated_features = {
        **existing_features,  # Keep the existing features
        new_feature_name: new_feature  # Add the new feature (with the correct name)
    }

    # Replace the features in the graph
    updated_graph = graph.replace_features(node_sets={"atom": updated_features})

    return updated_graph

class GraphUpdateLayer(tf.keras.layers.Layer):
    def __init__(self, new_feature_name, **kwargs):
        super().__init__(**kwargs)
        self.new_feature_name = new_feature_name

    def call(self, inputs):
        graph, new_feature = inputs

        return update_graph(graph, new_feature, self.new_feature_name)
    
class GNNLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_update = tfgnn.keras.layers.GraphUpdate(
                edge_sets={"interatomic_distance": tfgnn.keras.layers.EdgeSetUpdate(
                    next_state=tfgnn.keras.layers.ResidualNextState(node_updating())
                )},
                node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
                    {"interatomic_distance": tfgnn.keras.layers.Pool(
                        reduce_type="mean|sum", 
                        tag=tfgnn.TARGET)},
                    next_state=tfgnn.keras.layers.ResidualNextState(edge_updating())
                )} 
        )

    def call(self, graph):
        for _ in range(3):
            graph = self.graph_update(graph)

        return graph


def _build_model(graph_tensor_spec):
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)

    base_graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(input_graph)

    gnn_layer = GNNLayer()

    ### SHIFT PREDICTION ###
    graph = gnn_layer(base_graph)
    readout_features = tfgnn.keras.layers.StructuredReadout("hydrogen")(graph)
    shift_output = readout_layers()(readout_features)
    
    graph_with_shift = GraphUpdateLayer(new_feature_name="shift")([base_graph, shift_output])

    ### SHAPE PREDICTION ###
    graph = gnn_layer(graph_with_shift)
    graph_output = tfgnn.keras.layers.StructuredReadout("hydrogen")(graph)
    shape_branch = tf.keras.layers.Dense(128)(graph_output)
    shape_branch = tf.keras.layers.RepeatVector(4)(shape_branch)
    shape_branch = tf.keras.layers.GRU(256, return_sequences=True)(shape_branch)
    shape_branch = tf.keras.layers.GRU(64, return_sequences=True)(shape_branch)
    shape_branch = tf.keras.layers.GRU(64, return_sequences=True)(shape_branch)
    shape_branch = tf.keras.layers.Dense(8)(shape_branch)
    shape_output = tf.keras.layers.Softmax(name="shape")(shape_branch)

    graph_with_shape = GraphUpdateLayer(new_feature_name="shape")([base_graph, shape_output])

    ### COUPLING CONSTANT PREDICTION ###
    graph = gnn_layer(graph_with_shape)
    graph_output = tfgnn.keras.layers.StructuredReadout("hydrogen")(graph)
    coupling_branch = tf.keras.layers.Dense(128)(graph_output)
    coupling_branch = tf.keras.layers.RepeatVector(4)(coupling_branch)
    coupling_branch = tf.keras.layers.GRU(256, return_sequences=True)(coupling_branch)
    coupling_branch = tf.keras.layers.GRU(128, return_sequences=True)(coupling_branch)
    coupling_output = tf.keras.layers.Dense(1, activation="relu", name="coupling_output")(coupling_branch)
    coupling_output = tf.keras.layers.Lambda(mask_couplings, name="coupling")([shape_output, coupling_output])

    combined_output = tf.keras.layers.concatenate([shape_output, coupling_output], name="combined")

    return tf.keras.Model(inputs=[input_graph], outputs={"shift": shift_output, "shape": shape_output, "coupling": coupling_output, "combined": combined_output})


class PreprocessingModel(tf.keras.Model):
    def call(self, inputs):
        graph_tensor = inputs.merge_batch_to_components()

        shift_labels = tfgnn.keras.layers.Readout(node_set_name="_readout",
                                        feature_name="shift")(graph_tensor)
        shape_labels = tfgnn.keras.layers.Readout(node_set_name="_readout",
                                        feature_name="shape")(graph_tensor)
        coupling_labels = tfgnn.keras.layers.Readout(node_set_name="_readout",
                                        feature_name="coupling")(graph_tensor)
        
        dummy_labels = tf.fill([tf.shape(coupling_labels)[0]], tf.constant(float('nan'), dtype=coupling_labels.dtype))

        graph_tensor = graph_tensor.remove_features(node_sets={"_readout": ["shift", "shape", "coupling"]})

        labels = {
            "shift": shift_labels,
            "shape": shape_labels,
            "coupling": coupling_labels,
            "combined": dummy_labels
        }

        return graph_tensor, labels

def weighted_cce(class_weights):
    def loss(y_true, y_pred):
        cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_loss = cce_loss * class_weights
        return tf.reduce_mean(weighted_loss)
    
    return loss

def count_invalid_shapes(y_true, y_pred):    # A shape is invalid if the any token except the first is "m" or if any token after an "s" is not "s"
    invalid_count = 0
    shape_decoded = tf.argmax(y_pred, axis=-1)
    for shape in shape_decoded:
        m_indices = tf.where(tf.equal(shape_decoded, 0))
        m_invalid_0 = False
        m_invalid_1 = False
        if tf.size(m_indices) > 0:
            first_m = tf.reduce_min(m_indices)
            m_invalid_0 = tf.reduce_any(tf.not_equal(shape[first_m + 1:], 1)) # Something after an "m" is not an "s"
            m_invalid_1 = tf.reduce_any(tf.equal(shape[1:], 0)) # "m" is present, but not at the first position

        s_indices = tf.where(tf.equal(shape_decoded, 0))
        s_invalid = False
        if tf.size(s_indices) > 0:
            first_s = tf.reduce_min(s_indices)
            s_invalid = tf.reduce_any(tf.not_equal(shape[first_s + 1:], 1)) # Something after an "s" is not an "s"

        if tf.reduce_any([m_invalid_0, m_invalid_1, s_invalid]):  # if any one is invalid, the invalid count goes up
            invalid_count += 1

    return invalid_count


def count_invalid_couplings(y_true, y_pred):    # A coupling is invalid if a token with an s or m as shape has a value for the coupling constant
    invalid_count = 0
    shape_pred, coupling_pred = tf.split(y_pred, [8, 1], axis=-1)

    shape_decoded = tf.argmax(shape_pred, axis=-1)
    for coupling in coupling_pred:
        coupling = tf.squeeze(coupling, axis=-1)
        i = 0
        invalid_coupling = False
        zero_indices = tf.where(tf.equal(coupling, 0.0))
        zero_indices = tf.squeeze(zero_indices, axis=-1)

        shape_m_or_s = tf.where(tf.logical_or(tf.equal(shape_decoded[i], 0), tf.equal(shape_decoded[i], 1)))
        shape_m_or_s = tf.squeeze(shape_m_or_s, axis=-1)

        if tf.equal(tf.shape(shape_m_or_s), tf.shape(zero_indices)):
            invalid_coupling = tf.reduce_any(tf.not_equal(shape_m_or_s, zero_indices))   # Check if zero values are (only) given for m or s shapes
        else:
            invalid_coupling = True   # If there is a different number of zero's and m/s, it's always invalid
            
        if invalid_coupling:
            invalid_count += 1
        
        i += 1

    return invalid_count

if __name__ == "__main__":
    #path = "/home1/s3665828/code/CASCADE/"
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    
    batch_size = 32
    initial_learning_rate = 5E-4
    epochs = 500
    epoch_divisor = 1

    train_path = path + "data/own_data/All/own_train.tfrecords.gzip"
    val_path = path + "data/own_data/All/own_valid.tfrecords.gzip"

    train_size = 63324
    valid_size = 13569
    test_size = 13571

    steps_per_epoch = train_size // batch_size // epoch_divisor
    validation_steps = valid_size // batch_size // epoch_divisor
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    train_ds = tf.data.TFRecordDataset([train_path], compression_type="GZIP")
    val_ds = tf.data.TFRecordDataset([val_path], compression_type="GZIP")
    train_ds = train_ds.batch(batch_size=batch_size).repeat()
    val_ds = val_ds.batch(batch_size=batch_size)

    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchemaComplete.pbtxt")
    example_input_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    train_ds = train_ds.map(tfgnn.keras.layers.ParseExample(example_input_spec))
    val_ds = val_ds.map(tfgnn.keras.layers.ParseExample(example_input_spec))
    preproc_input_spec = train_ds.element_spec

    # preprocessing
    preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)
    preproc_model = PreprocessingModel()
    train_ds = train_ds.map(preproc_model)
    val_ds = val_ds.map(preproc_model)

    # model
    model_input_spec, _ = train_ds.element_spec
    model = _build_model(model_input_spec)

    loss = {"shift": tf.keras.losses.MeanAbsoluteError(),
            "shape": weighted_cce([1.0,0.4,0.15,0.05]),
            "coupling": tf.keras.losses.MeanAbsoluteError()}
    metrics = {"shape": count_invalid_shapes,
              "combined": count_invalid_couplings}
    loss_weights = {"shift": 1.0,
                    "shape": 0.0,
                    "coupling": 0.0}

    model.compile(tf.keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics, loss_weights=loss_weights)
    model.summary()

    code_path = path + "code/predicting_model/"
    filepath = code_path + "gnn/models/test_model/checkpoint.weights.h5"
    log_dir = code_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, save_freq="epoch", verbose=1, monitor="val_loss", save_weights_only=True)
    history1 = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=25, validation_data=val_ds, callbacks=[tensorboard_callback, checkpoint])

    loss_weights = {"shift": 1.0,
                    "shape": 0.5,
                    "coupling": 0.5}
    model.compile(tf.keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics, loss_weights=loss_weights)
    history2 = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=25, validation_data=val_ds, callbacks=[tensorboard_callback, checkpoint])

    loss_weights = {"shift": 1.0,
                    "shape": 0.0,
                    "coupling": 0.0}
    model.compile(tf.keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics, loss_weights=loss_weights)
    history3 = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs-50, validation_data=val_ds, callbacks=[tensorboard_callback, checkpoint])

    history = {
    key: history1.history[key] + history2.history[key] + history3.history[key]
    for key in history1.history
    }

    #load best weights before saving
    model.load_weights(filepath)

    serving_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name="examples")
    preproc_input = tfgnn.keras.layers.ParseExample(example_input_spec)(serving_input)
    serving_model_input, _ = preproc_model(preproc_input)
    serving_logits = model(serving_model_input)
    serving_output = {"shifts": serving_logits["shift"], "shape": serving_logits["shape"], "coupling_constants": serving_logits["coupling"]}
    exported_model = tf.keras.Model(serving_input, serving_output)
    exported_model.export(code_path + "gnn/models/test_model")
   
    #for layer in model.layers: print(layer.get_config(), layer.get_weights())