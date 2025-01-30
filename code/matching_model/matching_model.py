import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
import datetime

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

    distance = tf.keras.layers.Flatten(name="distance")(distance)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(distance)

    return tf.keras.Model(
        inputs=[predicted_shift, predicted_shape, predicted_coupling, observed_shift, observed_shape, observed_coupling],
        outputs=output,
        name="siamese_network"
    )


if __name__ == "__main__":
    #path = "/home1/s3665828/code/CASCADE/"
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "C:/Users/niels/Documents/repo/CASCADE/"

    train_path = path + "data/own_data/matching_model/own_train.tfrecords.gzip"
    val_path = path + "data/own_data/matching_model/own_valid.tfrecords.gzip"

    train_ds = tf.data.TFRecordDataset([train_path], compression_type="GZIP")
    val_ds = tf.data.TFRecordDataset([val_path], compression_type="GZIP")

    train_size = 63324
    valid_size = 13569
    test_size = 13571

    batch_size = 32
    epoch_divisor = 1

    steps_per_epoch = train_size // batch_size // epoch_divisor
    validation_steps = valid_size // batch_size // epoch_divisor

    train_ds = train_ds.map(parse_example).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(parse_example).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    full_model = _build_model()
    full_model.compile(loss='binary_crossentropy', optimizer='adam')
    full_model.summary()

    code_path = path + "code/matching_model/"
    filepath = code_path + "gnn/models/matching_model/checkpoint.weights.h5"
    log_dir = code_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, save_freq="epoch", verbose=1, monitor="val_loss", save_weights_only=True)

    history = full_model.fit(train_ds, epochs=1000, validation_data=val_ds, callbacks=[tensorboard_callback, checkpoint])

    full_model.load_weights(filepath)

    distance_model = tf.keras.Model(inputs=full_model.input, outputs=full_model.get_layer('distance').output)

    full_model.save(code_path + "gnn/models/matching_model")
    distance_model.save(code_path + "gnn/models/distance_model")