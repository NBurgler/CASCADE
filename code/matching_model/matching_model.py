import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn

def predict_shift(path, input_dict):
    shift_model = tf.saved_model.load(path + "code/predicting_model/Shift/DFTNN/gnn/models/DFT_model")
    signature_fn = shift_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    output_dict = signature_fn(**input_dict)
    logits = output_dict["shifts"]
    return logits

def predict_shape(path, input_graph):
    shape_model = tf.saved_model.load(path + "code/predicting_model/Shape/gnn/models/own_model")
    signature_fn = shape_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

def predict_couplings(path, input_graph):
    coupling_model = tf.saved_model.load(path + "code/predicting_model/Coupling/gnn/models/own_model")
    signature_fn = coupling_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

def make_predictions(path):
    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchema.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    dataset = tf.data.TFRecordDataset([path + "data/own_data/Shift/own_train.tfrecords.gzip"], compression_type="GZIP")
    examples = next(iter(dataset.batch(1)))
    example = tf.reshape(examples[0], (1,))
    input_graph = tfgnn.parse_example(graph_spec, example)
    input_dict = {"examples": example}
    shifts = predict_shift(path, input_dict)
    print(input_graph.node_sets["atom"].__getitem__("shift"))
    print(shifts)


def parse_example(serialized_example):
    feature_description = {
        "predicted_shift": tf.io.FixedLenFeature([1], tf.float32),
        "predicted_shape": tf.io.FixedLenFeature([32], tf.float32),  # 4x8 flattened
        "predicted_coupling": tf.io.FixedLenFeature([4], tf.float32),
        "observed_shift": tf.io.FixedLenFeature([1], tf.float32),
        "observed_shape": tf.io.FixedLenFeature([32], tf.float32),  # 4x8 flattened
        "observed_coupling": tf.io.FixedLenFeature([4], tf.float32),
        "match": tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(serialized_example, feature_description)

    # Reshape shape features
    features["predicted_shape"] = tf.reshape(features["predicted_shape"], (4, 8))
    features["observed_shape"] = tf.reshape(features["observed_shape"], (4, 8))
    
    # Separate inputs and label
    inputs = {
        "predicted_shift": features["predicted_shift"],
        "predicted_shape": features["predicted_shape"],
        "predicted_coupling": features["predicted_coupling"],
        "observed_shift": features["observed_shift"],
        "observed_shape": features["observed_shape"],
        "observed_coupling": features["observed_coupling"],
    }
    label = tf.cast(features["match"], tf.float32)
    
    return inputs, label


def _build_embedding_model():
    # Shift input
    shift_input = tf.keras.layers.Input(shape=(1,), name="shift")
    shift_dense = tf.keras.layers.Dense(16, activation="relu")(shift_input)

    # Shape input (4 one-hot vectors)
    shape_input = tf.keras.layers.Input(shape=(4, 8), name="shape")
    shape_flat = tf.keras.layers.Flatten()(shape_input)
    shape_dense = tf.keras.layers.Dense(32, activation="relu")(shape_flat)

    # Coupling constants input
    coupling_input = tf.keras.layers.Input(shape=(4,), name="coupling")
    coupling_dense = tf.keras.layers.Dense(16, activation="relu")(coupling_input)

    # Combine all features
    combined = tf.keras.layers.Concatenate()([shift_dense, shape_dense, coupling_dense])
    embedding = tf.keras.layers.Dense(64, activation="relu", name="embedding")(combined)

    return tf.keras.Model(inputs=[shift_input, shape_input, coupling_input], outputs=embedding, name="embedding_model")



def _build_model():
    predicted_shift = tf.keras.layers.Input(shape=(1,), name="predicted_shift")
    predicted_shape = tf.keras.layers.Input(shape=(4, 8), name="predicted_shape")
    predicted_coupling = tf.keras.layers.Input(shape=(4,), name="predicted_coupling")

    observed_shift = tf.keras.layers.Input(shape=(1,), name="observed_shift")
    observed_shape = tf.keras.layers.Input(shape=(4, 8), name="observed_shape")
    observed_coupling = tf.keras.layers.Input(shape=(4,), name="observed_coupling")

    embedding_model = _build_embedding_model()
    predicted_embedding = embedding_model(
        {"shift": predicted_shift, "shape": predicted_shape, "coupling": predicted_coupling}
    )
    observed_embedding = embedding_model(
        {"shift": observed_shift, "shape": observed_shape, "coupling": observed_coupling}
    )

    distance = tf.keras.layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=-1))([predicted_embedding, observed_embedding])

    distance = tf.keras.layers.Flatten()(distance)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(distance)

    return tf.keras.Model(
        inputs=[predicted_shift, predicted_shape, predicted_coupling, observed_shift, observed_shape, observed_coupling],
        outputs=output,
        name="siamese_network"
    )


if __name__ == "__main__":
    #path = "/home1/s3665828/code/CASCADE/"
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"

    train_path = path + "data/own_data/matching_model/own_train.tfrecords.gzip"
    val_path = path + "data/own_data/matching_model/own_valid.tfrecords.gzip"

    train_ds = tf.data.TFRecordDataset([train_path], compression_type="GZIP")
    val_ds = tf.data.TFRecordDataset([val_path], compression_type="GZIP")

    train_ds = train_ds.map(parse_example).shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(parse_example).shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    '''
    print(train_ds)
    iterator = iter(train_ds)
    print(iterator.get_next())
    input = iterator.get_next()
    '''

    model = _build_model()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    model.fit(train_ds, epochs= 1000)