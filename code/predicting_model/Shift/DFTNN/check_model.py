import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import itertools
from tensorflow_gnn import runner

path = "C:/Users/niels/Documents/repo/CASCADE/code/predicting_model/Shift/DFTNN/"

model = tf.saved_model.load(path + "tmp/gnn_model/export")
signature_fn = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

def _clean_example_for_serving(graph_tensor):
  #graph_tensor = graph_tensor.remove_features(node_sets={"paper": ["label"]})
  print(graph_tensor)
  serialized_example = tfgnn.write_example(graph_tensor)
  return serialized_example.SerializeToString()

num_examples = 2
#dataset = tf.data.TFRecordDataset(filenames=["data/own_data/shift_train.tfrecords"])
dataset_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_train.tfrecords"])
dataset = dataset_provider.get_dataset(tf.distribute.InputContext())
print(dataset)
serialized_examples = [_clean_example_for_serving(gt)
                       for gt in itertools.islice(dataset, num_examples)]

ds = tf.data.Dataset.from_tensor_slices(serialized_examples)

input_dict = {"examples": next(iter(ds.batch(2)))}

output_dict = signature_fn(**input_dict)
logits = output_dict["shifts"]
#print(logits)