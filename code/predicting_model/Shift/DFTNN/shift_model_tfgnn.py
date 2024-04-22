import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd
import pickle
import sys

sys.path.append('code/predicting_model')

path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"

dataset = tf.data.TFRecordDataset(filenames=["data/own_data/shift_graph.tfrecords"])
graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
print(graph_tensor_spec)
dataset = dataset.map(
    lambda serialized: tfgnn.parse_single_example(graph_tensor_spec, serialized))

for i, graph in enumerate(dataset.take(3)):
    print(f"Input {i}: {graph}")