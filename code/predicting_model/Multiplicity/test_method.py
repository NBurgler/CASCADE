import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner

def _build_model(graph_tensor_spec):
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    #graph = input_graph.merge_batch_to_components()

    graph = tfgnn.keras.layers.StructuredReadoutIntoFeature(key="shift", 
                                                  feature_name="shift",
                                                  new_feature_name="new_shift")(input_graph)
    logits = graph.node_sets['readout'].__getitem__("new_shift")
    print(logits)
    print(graph.node_sets['readout'].__getitem__("new_shift"))
    print(graph.node_sets['readout'].__getitem__("shift"))

    return tf.keras.Model(inputs=[input_graph], outputs=[logits])


if __name__ == "__main__":
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    train_path = path + "data/own_data/own_data_train.tfrecords"
    val_path = path + "data/own_data/own_data_valid.tfrecords"

    batch_size = 64
    initial_learning_rate = 5E-3
    epochs = 10
    epoch_divisor = 1

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

    model_input_spec, _ = train_ds.element_spec
    model = _build_model(model_input_spec)

    loss = tf.keras.losses.MeanAbsoluteError()
    metrics = [tf.keras.losses.MeanAbsoluteError()]

    model.compile(tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)
    model.summary()

    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs, validation_data=val_ds) 