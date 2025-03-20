import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
import pandas as pd
from tqdm import tqdm
import pulp
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
import ast

import sys
from pathlib import Path

# Add the 'main_project' folder to sys.path
parent_dir = Path(__file__).resolve().parent.parent  # This gets the project root (main_project)
sys.path.append(str(parent_dir))

from predicting_model import create_graph_tensor


def one_hot_encode_shape(shape):        # The output will be a matrix of four one-hots, where each one-hot encodes for a shape
    indices = np.ones(4, dtype=int)     # i.e. 'dtp' will be encoded as matrix of four one-hots where the first one encodes
    if len(shape) > 4:                  # for 'd', the second for 't', the third for 'p', and the rest for 's'
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
        
    one_hot_shape = tf.one_hot(tf.convert_to_tensor(indices), 8)
    return one_hot_shape

def convert_coupling_constants(coupling_string):    # Convert the couplings string into an array of length 4
    coupling_constants = [0.0, 0.0, 0.0, 0.0]                # Any entries without a value will be 0.0
    for i, value in enumerate(coupling_string.split(";")):
        if value != "-" and i < 4:
            coupling_constants[i] = float(value)

    return coupling_constants

def predict_shift(path, input_dict):
    shift_model = tf.saved_model.load(path + "code/predicting_model/Shift/gnn/models/shift_model_best")
    signature_fn = shift_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    output_dict = signature_fn(**input_dict)
    logits = output_dict["shifts"]
    return logits


def predict_shape_and_coupling(path, input_dict):
    shape_model = tf.saved_model.load(path + "code/predicting_model/Shape_And_Coupling/gnn/models/shape_and_coupling_model_0")
    signature_fn = shape_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    output_dict = signature_fn(**input_dict)
    shape_logits = output_dict["shape"]
    coupling_logits = np.round(output_dict["coupling_constants"], 2)
    
    return shape_logits, coupling_logits

def predict_all(path, input_dict):
    shape_model = tf.saved_model.load(path + "code/predicting_model/gnn/models/predicting_model_6_3")
    signature_fn = shape_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    output_dict = signature_fn(**input_dict)
    shift_logits = output_dict["shift"]
    shape_logits = output_dict["shape"]
    coupling_logits = np.round(output_dict["coupling_constants"], 2)
    
    return shift_logits, shape_logits, coupling_logits


# From a smiles, create the necessary features and predict shift, shape, and coupling constants. Returns all predicted peaks
def predict_peaks(smiles="C#CCC1CCOCO1"):
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "/home1/s3665828/code/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/" ## TODO: CHANGE PATH TO BE RELATIVE

    mol_df, atom_df, bond_df, distance_df = create_graph_tensor.create_dictionary(2, path, type=type, smiles=smiles, name=smiles)
    if mol_df is None:
        return None, None

    mol_df.insert(5, "n_distance", len(distance_df["distance"]))
    example = create_graph_tensor.create_single_tensor(mol_df, atom_df, bond_df, distance_df)   # Create tensor for shift prediction
    example = tf.reshape(example, (1,))
    input_dict = {"examples": example}

    shifts, shapes, couplings = predict_all(path, input_dict)
    shifts = np.squeeze(shifts.numpy())
    shapes = shapes.numpy()

    predicted_peaks = []

    for idx in range(len(shifts)):
        predicted_peak = {"shift": shifts[idx], 
                        "shape": shapes[idx],
                        "coupling": couplings[idx].flatten()}
        predicted_peaks.append(predicted_peak)

    atom_idx = atom_df.loc[atom_df["atom_symbol"] == "H"].index.to_numpy()
    return predicted_peaks, atom_idx


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
    
    return inputs

def minimize_distance(matrix, amount):
    m, n = len(matrix), len(matrix[0])  # Rows and Columns
    prob = LpProblem("MinimizeDistance", LpMinimize)

    # Define binary variables x_ij (1 if cell [i][j] is selected, 0 otherwise)
    x = [[LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n)] for i in range(m)]

    # Objective function: Minimize total distance
    prob += lpSum(matrix[i][j] * x[i][j] for i in range(m) for j in range(n))

    # Constraint: Each row must have exactly one selected value
    for i in range(m):
        prob += lpSum(x[i][j] for j in range(n)) == 1

    # Constraint: Each column must have at least one selected value
    for j in range(n):
        if amount[j] != (None or '-'):
            prob += lpSum(x[i][j] for i in range(m)) == int(amount[j])
        else:
            prob += lpSum(x[i][j] for i in range(m)) >= 1

    prob.solve(PULP_CBC_CMD(msg=False))

    # Extract the selected values and total cost
    selected = [(i, j) for i in range(m) for j in range(n) if x[i][j].value() == 1]
    total_cost = sum(matrix[i][j] for i, j in selected)
    return selected, total_cost

def create_distance_matrix(predicted_peaks, observed_peaks):
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "/home1/s3665828/code/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    
    distance_matrix = np.empty(shape=(len(predicted_peaks), len(observed_peaks)))

    model = tf.keras.models.load_model(path + "code/matching_model/gnn/models/distance_model")
    signature_fn = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    i = 0
    for pred_peak in predicted_peaks:
        j = 0
        for obs_peak in observed_peaks:
            pred_shift = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(pred_peak["shift"]), axis=0), axis=0)
            pred_shape = tf.expand_dims(tf.convert_to_tensor(pred_peak["shape"]), axis=0)
            pred_coupling = tf.expand_dims(tf.convert_to_tensor(pred_peak["coupling"]), axis=0)
            obs_shift = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(obs_peak["shift"]), axis=0), axis=0)
            obs_shape = tf.expand_dims(tf.convert_to_tensor(one_hot_encode_shape(obs_peak["shape"])), axis=0)
            obs_coupling = tf.expand_dims(tf.convert_to_tensor(convert_coupling_constants(obs_peak["coupling"])), axis=0)

            sample = [pred_shift, pred_shape, pred_coupling, obs_shift, obs_shape, obs_coupling]
            distance = model.predict(sample, verbose=0)
            distance_matrix[i][j] = distance[0][0]
            j += 1
        i += 1

    return distance_matrix

def dataframe_to_list_of_dicts(peak_dataframe):
    peak_list = []
    for i, peak in peak_dataframe.iterrows():
        shift = float(peak["shift"])
        shape = peak["shape"]
        coupling = (";").join(peak["coupling"].split(", "))
        peak_list.append({"shift": shift, "shape": shape, "coupling": coupling})

    return peak_list

def evaluate_matching_model(model, dataset):
    model = tf.keras.models.load_model(path + "code/matching_model/gnn/models/distance_model")
    signature_fn = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    mol_df = pd.read_csv(path + "code/matching_model/mol_data.csv.gz", compression="gzip", index_col=0)

    observed_peaks_df = pd.read_csv(path + "code/matching_model/peak_data.csv.gz", compression="gzip", index_col=0)

    correct_peaks = 0
    total_peaks = 0
    correct_mols = 0
    incorrect_mols = []
    for i, mol in mol_df.iterrows():
        mol_id = mol["mol_id"]
        observed_peaks = observed_peaks_df.loc[observed_peaks_df["mol_id"] == mol_id]
        observed_atom_idx = observed_peaks["atom_idx"].reset_index()
        amount = observed_peaks["amount"].to_list()

        observed_peaks = dataframe_to_list_of_dicts(observed_peaks)

        predicted_peaks, predicted_atom_idx = predict_peaks(mol["smiles"])
        if predicted_peaks is None:
            continue
        
        distance_matrix = create_distance_matrix(predicted_peaks, observed_peaks)

        selected, total_cost = minimize_distance(distance_matrix, amount)
        incorrect_mol = None
        for predicted_idx, observed_idx in selected:
            if str(predicted_atom_idx[predicted_idx]) in observed_atom_idx["atom_idx"].iloc[observed_idx]:
                correct_peaks += 1
            else:
                incorrect_mol = mol_df["smiles"].loc[mol_df["mol_id"] == mol_id].values[0]
            total_peaks += 1

        if incorrect_mol == None:
            correct_mols += 1
        else:
            incorrect_mols.append(mol_df["smiles"].loc[mol_df["mol_id"] == mol_id].values[0])

    with open(path + "code/matching_model/incorrect_predictions.txt", 'w') as f:
     f.write("Mols Correctly Predicted: ")
     f.write(str(correct_mols) + "/" + str(len(mol_df)))
     f.write("\n")
     f.write("Peaks Correctly Predicted: ")
     f.write(str(correct_peaks) + "/" + str(total_peaks))
     f.write("\n")
     f.write("Molecules with errors:")
     f.write("\n")
     for smiles in incorrect_mols:
         f.write(smiles)
         f.write("\n")

def convert_shape_one_hot(one_hot_matrix):
    shape = ''
    for one_hot in one_hot_matrix:
        if np.argmax(one_hot) == 0: shape += "m"
        elif np.argmax(one_hot) == 1: shape += "s"
        elif np.argmax(one_hot) == 2: shape += "d"
        elif np.argmax(one_hot) == 3: shape += "t"
        elif np.argmax(one_hot) == 4: shape += "q"
        elif np.argmax(one_hot) == 5: shape += "p"
        elif np.argmax(one_hot) == 6: shape += "h"
        elif np.argmax(one_hot) == 7: shape += "v"

    return shape

def create_predicting_model_results(model, dataset, num_samples, name):
    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchemaComplete.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    output = {"molecule":[], "mol_id":[], "index":[], "predicted_shift":[], "target_shift":[], "target_coupling":[], "predicted_coupling":[],
              "coupling_pred_1":[], "coupling_pred_2":[], "coupling_pred_3":[], "coupling_pred_4":[],
              "coupling_target_1":[], "coupling_target_2":[], "coupling_target_3":[], "coupling_target_4":[],
              "target_shape":[], "predicted_shape":[], "converted_shape":[],
              "shape_pred_1":[], "shape_pred_2":[], "shape_pred_3":[], "shape_pred_4":[], "shape_target_1":[], 
              "shape_target_2":[], "shape_target_3":[], "shape_target_4":[]}
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    examples = next(iter(dataset.batch(num_samples)))
    for i in tqdm(range(num_samples)):
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        input_dict = {"examples": example}
        output_dict = signature_fn(**input_dict)
        shift_logits = np.round(output_dict["shifts"], 2)
        coupling_logits = np.round(output_dict["coupling_constants"], 2)
        shape_logits = output_dict["shape"]
        shift_labels = input_graph.node_sets["_readout"].__getitem__("shift").to_tensor()[0]
        coupling_labels = input_graph.node_sets["_readout"].__getitem__("coupling").to_tensor()[0]
        shape_labels = input_graph.node_sets["_readout"].__getitem__("shape").to_tensor()[0]
        smiles = input_graph.context.__getitem__("smiles")
        smiles = tf.get_static_value(smiles).astype(str)[0][0][0]
        mol_id = input_graph.context.__getitem__("_mol_id")
        mol_id = tf.get_static_value(mol_id).astype(str)[0][0][0]
        index = input_graph.edge_sets["_readout/hydrogen"].adjacency.source[0]
        if coupling_logits.ndim == 1:
            coupling_logits = [coupling_logits]
            shape_logits = [shape_logits]

        for j, predicted_couplings in enumerate(coupling_logits):
            predicted_shift = tf.get_static_value(shift_logits[j])[0]
            target_shift = tf.get_static_value(shift_labels[j])[0]

            target_couplings = coupling_labels[j]
            target_couplings = tf.get_static_value(target_couplings).flatten()
            predicted_couplings = tf.get_static_value(predicted_couplings).flatten()

            output["molecule"].append(smiles)
            output["mol_id"].append(mol_id)
            output["index"].append(tf.get_static_value(index[j]))

            output["target_shift"].append(target_shift)
            output["predicted_shift"].append(predicted_shift)

            output["target_coupling"].append(target_couplings)
            output["predicted_coupling"].append(predicted_couplings)

            output["coupling_pred_1"].append(tf.get_static_value(predicted_couplings[0]))
            output["coupling_pred_2"].append(tf.get_static_value(predicted_couplings[1]))
            output["coupling_pred_3"].append(tf.get_static_value(predicted_couplings[2]))
            output["coupling_pred_4"].append(tf.get_static_value(predicted_couplings[3]))

            output["coupling_target_1"].append(tf.get_static_value(target_couplings[0]))
            output["coupling_target_2"].append(tf.get_static_value(target_couplings[1]))
            output["coupling_target_3"].append(tf.get_static_value(target_couplings[2]))
            output["coupling_target_4"].append(tf.get_static_value(target_couplings[3]))

            predicted_shape = shape_logits[j]
            converted_labels = convert_shape_one_hot(shape_labels[j])
            converted_predictions = convert_shape_one_hot(predicted_shape)

            output["target_shape"].append(tf.get_static_value(converted_labels))
            output["predicted_shape"].append(tf.get_static_value(predicted_shape))
            output["converted_shape"].append(tf.get_static_value(converted_predictions))

            output["shape_pred_1"].append(converted_predictions[0])
            output["shape_pred_2"].append(converted_predictions[1])
            output["shape_pred_3"].append(converted_predictions[2])
            output["shape_pred_4"].append(converted_predictions[3])

            output["shape_target_1"].append(converted_labels[0])
            output["shape_target_2"].append(converted_labels[1])
            output["shape_target_3"].append(converted_labels[2])
            output["shape_target_4"].append(converted_labels[3])

    
    lengths = {key: (len(key), len(value)) for key, value in output.items()}

    output_df = pd.DataFrame.from_dict(output)
    print(output_df[["molecule", "index", "target_shift", "target_shape", "target_coupling", "predicted_shift", "converted_shape", "predicted_coupling"]])
    total = len(output_df)  

    output_df.to_csv(path + "data/own_data/" + str(name) + ".csv.gz", compression='gzip')

def convert_predicted_shapes(s):
    # Remove all unwanted characters (brackets and unnecessary spaces)
    s = s.replace('[', '').replace(']', '').strip()
    
    # Split the string by spaces and convert them to floats
    numbers = list(map(float, s.split()))
    
    # Reshape the list of numbers into a 4x8 matrix
    return np.array(numbers).reshape(4, 8)

def convert_couplings(s):
    # Remove the square brackets and split by space
    s = s.strip('[]')
    # Convert the space-separated string elements into a list of floats
    float_list = list(map(float, s.split()))
    # Convert the list to a numpy array
    return np.array(float_list)

def compute_cce(predicted, true):
    # Flatten both predicted and true arrays
    predicted_flat = predicted.flatten()
    true_flat = true.flatten()
    
    # Use np.clip to avoid log(0), which is undefined
    epsilon = 1e-15  # Small epsilon value to prevent log(0)
    predicted_flat = np.clip(predicted_flat, epsilon, 1 - epsilon)
    
    # Calculate Cross-Entropy
    cce = -np.sum(true_flat * np.log(predicted_flat))
    return cce

def count_invalid_shapes(predicted_shape):
    wrong_m_loc = 0
    non_s_after_s = 0
    non_s_after_m = 0
    if "m" in predicted_shape[1:]:
        wrong_m_loc = 1
    for i in range(3):
        if predicted_shape[i] == 's' and predicted_shape[i + 1] != 's':
            non_s_after_s = 1
        if predicted_shape[i] == 'm' and predicted_shape[i + 1] != 's':
            non_s_after_m = 1

    return [wrong_m_loc, non_s_after_s, non_s_after_m]

def count_invalid_couplings(predicted_coupling, predicted_shape):
    number_of_zeros_needed = predicted_shape.count("m") + predicted_shape.count("s")
    number_of_zeros = sum(1 for x in predicted_coupling if x == 0.0)
    if number_of_zeros < number_of_zeros_needed:  # Too many coupling constants
        return [0, 1]
    elif number_of_zeros > number_of_zeros_needed: # Too few coupling constants
        return [1, 0]
    
    return [0, 0]
    


def evaluate_predicting_model(results):
    total = len(results)
    results["shift_mae"] = np.abs(results["target_shift"] - results["predicted_shift"])

    results["target_shape_one_hot"] = results["target_shape"].apply(one_hot_encode_shape)
    results["target_shape_one_hot"] = results["target_shape_one_hot"].apply(np.array)
    results["predicted_shape"] = results["predicted_shape"].apply(convert_predicted_shapes)
    results["shape_cce"] = results.apply(lambda row: compute_cce(row["predicted_shape"], row["target_shape_one_hot"]), axis=1)
    results["invalid_shapes"] = results["converted_shape"].apply(count_invalid_shapes)

    results["predicted_coupling"] = results["predicted_coupling"].apply(convert_couplings)
    results["target_coupling"] = results["target_coupling"].apply(convert_couplings)
    results["invalid_coupling"] = results.apply(lambda row: count_invalid_couplings(row['predicted_coupling'], row['converted_shape']), axis=1)

    results["coupling_mae"] = np.abs(results["target_coupling"] - results["predicted_coupling"])

    print("Shift MAE: " + str(np.round(np.mean(results["shift_mae"]), 2)))
    print("Shape CCE (With M): " + str(np.round(np.mean(results["shape_cce"]), 2)))
    print("Shape CCE (Without M): " + str(np.round(np.mean(results.loc[results["target_shape"] != "msss"]["shift_mae"]), 2)))
    print("Coupling MAE (With M): " + str(np.round(np.mean(np.mean(results["coupling_mae"])), 2)))
    print("Coupling MAE (Without M): " + str(np.round(np.mean(np.mean(results.loc[results["target_shape"] != "msss"]["coupling_mae"])), 2)))
    print()
    print("INVALID SHAPES")
    wrong_m_loc, non_s_after_s, non_s_after_m = results["invalid_shapes"].apply(pd.Series).sum(axis=0)
    print("Wrong M Location: " + str(wrong_m_loc))
    print("Non-s after s: " + str(non_s_after_s))
    print("Non-s after m: " + str(non_s_after_m))
    print()
    print("INVALID COUPLING CONSTANTS")
    not_enough, too_many = results["invalid_coupling"].apply(pd.Series).sum(axis=0)
    print("Not enough coupling constants: " + str(not_enough))
    print("Too many coupling constants: " + str(too_many))
    

if __name__ == '__main__':
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "/home1/s3665828/code/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"

    train_size = 63324
    valid_size = 13569
    test_size = 13571

    print("TRAIN RESULTS")
    evaluate_predicting_model(pd.read_csv(path + "data/own_data/separate_model_train.csv.gz"))
    print()
    print("VALIDATION RESULTS")
    evaluate_predicting_model(pd.read_csv(path + "data/own_data/separate_model_valid.csv.gz"))
    print()
    print("TEST RESULTS")
    evaluate_predicting_model(pd.read_csv(path + "data/own_data/separate_model_test.csv.gz"))
    '''dataset = tf.data.TFRecordDataset([path + "data/own_data/All/own_train.tfrecords.gzip"], compression_type="GZIP")
    model = tf.saved_model.load(path + "code/predicting_model/gnn/models/separate_model")
    create_predicting_model_results(model, dataset, num_samples=train_size, name="separate_model_train")

    dataset = tf.data.TFRecordDataset([path + "data/own_data/All/own_valid.tfrecords.gzip"], compression_type="GZIP")
    create_predicting_model_results(model, dataset, num_samples=valid_size, name="separate_model_valid")

    dataset = tf.data.TFRecordDataset([path + "data/own_data/All/own_test.tfrecords.gzip"], compression_type="GZIP")
    create_predicting_model_results(model, dataset, num_samples=test_size, name="separate_model_test")'''
    