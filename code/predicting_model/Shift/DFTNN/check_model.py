import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import itertools
from tensorflow_gnn import runner

path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"

graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
model = tf.saved_model.load(path + "tmp/gnn_model/export")
signature_fn = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

num_examples = 2
#dataset = tf.data.TFRecordDataset(filenames=["data/own_data/shift_train.tfrecords"])
dataset_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_train.tfrecords"])
dataset = dataset_provider.get_dataset(tf.distribute.InputContext())

example = next(iter(dataset.batch(1)))
input_graph = tfgnn.parse_example(graph_spec, example)
print(input_graph.context.__getitem__("smiles"))
print(input_graph.node_sets["_readout"].__getitem__("shift"))
print(input_graph.node_sets["atom"].__getitem__("atom_num"))
degree = input_graph.node_sets["atom"].__getitem__("degree")
charge = input_graph.node_sets["atom"].__getitem__("formal_charge")
print(degree)
print(charge)
degree = tf.keras.layers.Reshape((-1, 1))(degree)
charge = tf.keras.layers.Reshape((-1, 1))(charge)
print(degree)
print(charge)
print("-----")
check = tf.keras.layers.Concatenate(axis=2)([degree, charge])
#check = tf.concat([degree, charge], axis=1)
print(check)
check = tf.squeeze(check)
print(check)
#print(tf.expand_dims(check, axis=-1))

input_dict = {"examples": example}
output_dict = signature_fn(**input_dict)
logits = output_dict["shifts"]

#print(logits)