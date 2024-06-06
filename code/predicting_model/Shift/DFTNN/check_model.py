import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import itertools
from tensorflow_gnn import runner

path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"

model = tf.saved_model.load(path + "tmp/gnn_model/export")
signature_fn = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

num_examples = 2
#dataset = tf.data.TFRecordDataset(filenames=["data/own_data/shift_train.tfrecords"])
dataset_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_train.tfrecords"])
dataset = dataset_provider.get_dataset(tf.distribute.InputContext())

sample = next(iter(dataset.batch(1)))

input_dict = {"examples": sample}
output_dict = signature_fn(**input_dict)
logits = output_dict["shifts"]

print(logits)