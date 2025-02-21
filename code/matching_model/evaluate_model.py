import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

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


# From a smiles, create the necessarry features and predict shift, shape, and coupling constants. Returns all predicted peaks
def predict_peaks(smiles="C#CCC1CCOCO1"):
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "/home1/s3665828/code/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/" ## TODO: CHANGE PATH TO BE RELATIVE

    mol_df, atom_df, bond_df, distance_df = create_graph_tensor.create_dictionary(2, path, type=type, smiles=smiles, name=smiles)

    example = create_graph_tensor.create_single_tensor(mol_df, atom_df, bond_df, distance_df)   # Create tensor for shift prediction
    example = tf.reshape(example, (1,))
    input_dict = {"examples": example}
    shifts = np.squeeze(predict_shift(path, input_dict).numpy())

    shift_df = atom_df.copy()
    shift_df.loc[shift_df["atom_symbol"] == "H", "Shift"] = shifts

    example = create_graph_tensor.create_single_tensor(mol_df, atom_df, bond_df, distance_df, shift_df) # Create tensor for shape/coupling prediction
    example = tf.reshape(example, (1,))
    input_dict = {"examples": example}
    shapes, couplings = predict_shape_and_coupling(path, input_dict)
    shapes = shapes.numpy()

    predicted_peaks = []

    for idx in range(len(shifts)):
        predicted_peak = {"shift": shifts[idx], 
                        "shape": shapes[idx],
                        "coupling": couplings[idx].flatten()}
        predicted_peaks.append(predicted_peak)

    atom_idx = shift_df.loc[shift_df["atom_symbol"] == "H"].index.to_numpy()
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


if __name__ == '__main__':
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "/home1/s3665828/code/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"


    model = tf.keras.models.load_model(path + "code/matching_model/gnn/models/distance_model")
    signature_fn = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    mol_df = pd.read_csv(path + "code/matching_model/mol_data.csv.gz", compression="gzip", index_col=0)

    #predicted_peaks, atom_indices = predict_peaks("C#CCC1CCOCO1")

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