import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import math

def one_hot_encode_shape(shape):            # The output will be a matrix of four one-hots, where each one-hot encodes for a shape
    one_hot_shapes = np.empty((0,4,8), dtype=int)   # i.e. 'dtp' will be encoded as matrix of four one-hots where the first one encodes                     
    indices = np.ones(4, dtype=int)         # for 'd', the second for 't', the third for 'p', and the rest for 's'
    if len(shape) > 4:
        indices = [0, 1, 1, 1]  # msss
    else:
        for i in range(len(shape)):
            if shape[i] == '-': 
                indices *= -1  # only for non-H atoms
                break   
            elif shape[i] == 'm': indices[i] = 0
            elif shape[i] == 's': indices[i] = 1
            elif shape[i] == 'd': indices[i] = 2
            elif shape[i] == 't': indices[i] = 3
            elif shape[i] == 'q': indices[i] = 4
            elif shape[i] == 'p': indices[i] = 5
            elif shape[i] == 'h': indices[i] = 6
            elif shape[i] == 'v': indices[i] = 7
        
    one_hot_shapes = np.append(one_hot_shapes, np.expand_dims(tf.one_hot(tf.convert_to_tensor(indices), 8), axis=0), axis=0)
    return one_hot_shapes


def convert_coupling_constants(coupling_string):    # Convert the couplings string into an array of length 4
    coupling_constants = [0.0, 0.0, 0.0, 0.0]                # Any entries without a value will be 0.0
    for i, value in enumerate(coupling_string.split(";")):
        if value != "-" and i < 4:
            coupling_constants[i] = float(value)

    return coupling_constants


def serialize_example(predicted, observed, match):
    """
    Creates a tf.train.Example from one sample.
    """
    feature = {
        "predicted_shift": tf.train.Feature(float_list=tf.train.FloatList(value=[predicted["shift"]])),
        "predicted_shape": tf.train.Feature(float_list=tf.train.FloatList(value=predicted["shape"].flatten())),
        "predicted_coupling": tf.train.Feature(float_list=tf.train.FloatList(value=predicted["coupling"])),
        "observed_shift": tf.train.Feature(float_list=tf.train.FloatList(value=[observed["shift"]])),
        "observed_shape": tf.train.Feature(float_list=tf.train.FloatList(value=observed["shape"].flatten())),
        "observed_coupling": tf.train.Feature(float_list=tf.train.FloatList(value=observed["coupling"])),
        "match": tf.train.Feature(int64_list=tf.train.Int64List(value=[match])),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

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


def create_tensors(path):
    observations = pd.read_csv(path + "code/matching_model/matching_data.csv.gz", index_col=0)
    shift_data = pd.read_csv(path + "data/own_data/all_shift_results.csv.gz", index_col=0)
    predictions = pd.read_csv(path + "data/own_data/shape_and_coupling_results.csv.gz", index_col=0)
    #predictions['Coupling'] = predictions['predicted_coupling'].str.strip('[]').str.split().apply(lambda x: np.array(x, dtype=float))

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    train_data = tf.io.TFRecordWriter(path + "data/own_data/matching_model/own_train.tfrecords.gzip", options=options)
    test_data = tf.io.TFRecordWriter(path + "data/own_data/matching_model/own_test.tfrecords.gzip", options=options)
    valid_data = tf.io.TFRecordWriter(path + "data/own_data/matching_model/own_valid.tfrecords.gzip", options=options)
    all_data = tf.io.TFRecordWriter(path + "data/own_data/matching_model/own_data.tfrecords.gzip", options=options)

    total = len(observations)
    total = 1000
    n_train = math.floor(total*0.7)
    n_valid = math.floor(total*0.15)
    n_test = total - n_train - n_valid
    print("Total dataset size: " + str(total))
    print("Training set size: " + str(n_train))
    print("Validation set size: " + str(n_valid))
    print("Testing set size: " + str(n_test))
    
    for idx, sample in tqdm(observations.iterrows()):
        if idx == 1000: break
        match = np.random.choice([0, 1])
        if (match):
            prediction = predictions.loc[(predictions["mol_id"] == sample["mol_id"]) & (predictions["index"] == sample["atom_idx"])].iloc[0]
            shift_sample = shift_data.loc[(predictions["mol_id"] == sample["mol_id"]) & (predictions["index"] == sample["atom_idx"])].iloc[0]
        else:
            shift_sample = shift_data.sample().iloc[0]
            while(abs(shift_sample["predicted_shift"] - sample["Shift"]) <= 1.0):   # Condition for finding non-matches
                shift_sample = shift_data.sample().iloc[0]

            prediction = predictions[(predictions["mol_id"] == shift_sample["mol_id"]) & (predictions["index"] == shift_sample["index"])].iloc[0]
        
        predicted_couplings = prediction['predicted_coupling'].strip('[]').split()
        predicted_couplings = [float(i) for i in predicted_couplings]

        observed = {"shift": sample["Shift"], 
                    "shape": one_hot_encode_shape(sample["Shape"]),
                    "coupling": convert_coupling_constants(sample["Coupling"])}
        
        predicted = {"shift": shift_sample["predicted_shift"], 
                    "shape": one_hot_encode_shape(prediction["predicted_shape"]),
                    "coupling": predicted_couplings}

        example = serialize_example(predicted, observed, match)
        all_data.write(example)
        if idx < n_train:
            train_data.write(example)
        elif idx < (n_train+n_valid):
            valid_data.write(example)
        else:
            test_data.write(example)
        


if __name__ == "__main__":
    #path = "/home1/s3665828/code/CASCADE/"
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"

    atom_df = pd.read_csv(path + "code/predicting_model/Coupling/own_atom.csv.gz", index_col=0)
    dataset = atom_df.loc[atom_df["atom_symbol"] == "H"]
    dataset = dataset.drop(columns=["atom_symbol", "chiral_tag", "degree", "formal_charge", "hybridization", 
                            "is_aromatic", "no_implicit", "num_Hs", "valence"])
    dataset.to_csv(path + "code/matching_model/matching_data.csv.gz", compression='gzip')
    create_tensors(path)
