import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import itertools
from tensorflow_gnn import runner

path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"

train_ds_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_train.tfrecords"])
dataset = tf.data.TFRecordDataset(filenames=["data/own_data/shift_train.tfrecords"])
graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


def _clean_example_for_serving(graph_tensor):
  graph_tensor = graph_tensor.remove_features(node_sets={"_readout": ["shift"]})
  serialized_example = tfgnn.write_example(graph_tensor)
  return serialized_example.SerializeToString()

num_examples = 2
dataset = train_ds_provider.get_dataset(tf.distribute.InputContext())
serialized_examples = [_clean_example_for_serving(gt)
                        for gt in itertools.islice(dataset, num_examples)]

dataset = tf.data.Dataset.from_tensor_slices(serialized_examples)

model = tf.saved_model.load(path + "gnn/models/")
signature_fn = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
input_dict = {"examples": next(iter(dataset.batch(2)))}

output_dict = signature_fn(**input_dict)
logits = output_dict["shift"]
#print(logits)