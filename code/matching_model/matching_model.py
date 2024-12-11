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

    


if __name__ == "__main__":
    #path = "/home1/s3665828/code/CASCADE/"
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    make_predictions(path)
    labels = pd.read_csv(path + "code/matching_model/matching_data.csv.gz", index_col=0)