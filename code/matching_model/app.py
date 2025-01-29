import streamlit as st
import numpy as np
import pandas as pd
from evaluate_model import predict_peaks, create_distance_matrix, minimize_distance
import matplotlib.pyplot as plt
import matplotlib
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from io import BytesIO
from PIL import Image
import math

import sys
from pathlib import Path


# Add the sibling directory to sys.path
sibling_dir = Path(__file__).resolve().parent.parent / "predicting_model"
print(sibling_dir)
sys.path.append(str(sibling_dir))

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

    return shape[0] + shape[1:].replace("s", "")

def convert_coupling_constants(coupling_constants_array):
    coupling_constants = ''
    for coupling_constant in coupling_constants_array:
        if coupling_constant != 0:
            if coupling_constants != '':
                coupling_constants += ", "
            coupling_constants += str(coupling_constant)

    if coupling_constants == '':
        coupling_constants = "-"

    return coupling_constants


def draw_mol_image(smiles, color_per_idx):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        if atom.GetSymbol() == "H":
            atom.SetProp('atomNote', str(index))

    def hex_to_rgb(hex_color):
        return matplotlib.colors.hex2color(hex_color)
    
    atom_colors = {int(index): hex_to_rgb(color) for index, color in color_per_idx}

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors)
    d.FinishDrawing()
    png_data = d.GetDrawingText()

    mol_image = Image.open(BytesIO(png_data))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mol_image)  # Show the RDKit image
    ax.axis("off")
    return fig

def highlight_cells(row, highlight_indices):
    print(highlight_indices)
    # Create a list of styles for the row, initially empty
    styles = ['' for _ in range(len(row))]
    
    # Iterate over the columns (skip the first column, which is index 0)
    for col_index in range(1, len(row)):  # Start from column index 1
        # Check if the (row index, column index) is in highlight_indices
        if (row.name, col_index-1) in highlight_indices:
            styles[col_index] = 'background-color: yellow'  # Highlight the specific cell
    
    return styles

def highlight_row(row):
    color = row['color']
    return ['background-color: {}'.format(color)] * len(row)

def assign_random_colors(n):
    # Generate a list of 'n' unique colors from the 'tab20' colormap
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / n) for i in range(n)]  # Generate 'n' unique colors
    # Convert RGBA to hex
    hex_colors = [matplotlib.colors.rgb2hex(color[:3]) for color in colors]
    return hex_colors

st.set_page_config(layout="wide")

# Title of the app
st.title("NMR Peak Matching")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.write("### Please enter the following properties of the observed peaks:")
    st.write("""
    - **Shift**: The chemical shift of the peak
    - **Shape**: The shape/multiplicity of the peak. You can enter a combination of up to 4 of the following tokens (e.g dtd):
        - m     (multiplet)
        - s     (singlet)
        - d     (doublet)
        - t     (triplet)
        - q     (quadruplet)
        - p     (quintet/pentuplet)
        - h     (sextet)
        - v     (septet)
    - **Coupling Constants**: Enter the coupling constants in descending order seperated by a comma and a space (e.g. 10.24, 5.42, 3.54). If there are no coupling constants, enter -.
    """)

    ##### Observed Peaks #####
    if "observed_peaks_df" not in st.session_state:
        st.session_state.observed_peaks_df = pd.DataFrame([{"shift":None, "shape":None, "coupling":None}])

    if "observed_peaks" not in st.session_state:
        st.session_state.observed_peaks = []

    df = pd.DataFrame([{"shift":None, "shape":None, "coupling":None}])
    st.write("Observed Peaks:")
    st.session_state.observed_peaks_df = st.data_editor(df, 
                                                        num_rows="dynamic",
                                                        column_order=("shift", "shape", "coupling"),
                                                        column_config={
                                                        "shift": "Shift",
                                                        "shape": "Shape",
                                                        "coupling": "Coupling Constants"
                                                        },
                                                        hide_index=True)

    if not st.session_state.observed_peaks_df.isna().any().any(): 
        observed_peak_dict = st.session_state.observed_peaks_df.to_dict(orient="records")
        observed_peaks = []
        for i, peak in enumerate(observed_peak_dict):
            shift = float(peak["shift"])
            shape = peak["shape"]
            coupling = (";").join(peak["coupling"].split(", "))
            observed_peaks.append({"shift": shift, "shape": shape, "coupling": coupling})

        st.session_state.observed_peaks = observed_peaks

with col2:
    ##### Molecule #####
    if "selected" not in st.session_state:
        st.session_state.selected = []

    if "colors_in_order" not in st.session_state:
        st.session_state.colors_in_order = []

    if "atom_idx" not in st.session_state:
        st.session_state.atom_idx = []

    smiles = st.text_input("Enter a smiles:", value="C#CCC1CCOCO1")

    if "molecule_image" not in st.session_state:
        st.session_state.molecule_image = None

    if st.session_state.selected != []:
        colored_df = st.session_state.observed_peaks_df.copy()
        
        colored_df["color"] = assign_random_colors(len(colored_df))
        st.session_state.colors_in_order = colored_df["color"].values

        styled_df = colored_df.style.apply(highlight_row, axis=1)

        color_per_idx = [(st.session_state.atom_idx[pred_idx], st.session_state.colors_in_order[obs_idx]) for (pred_idx, obs_idx) in st.session_state.selected]
        st.session_state.molecule_image = draw_mol_image(smiles, color_per_idx)

    if st.button("Submit Smiles"):
        color_per_idx = [(st.session_state.atom_idx[pred_idx], st.session_state.colors_in_order[obs_idx]) for (pred_idx, obs_idx) in st.session_state.selected]
        st.session_state.molecule_image = draw_mol_image(smiles, color_per_idx)

    if st.session_state.molecule_image:
        st.write("Molecule:")
        st.pyplot(st.session_state.molecule_image)

    ##### Matching Table #####
    if st.session_state.selected != []:
        st.dataframe(styled_df,
                     column_order=("shift", "shape", "coupling"),
                     column_config={
                        "shift": "Shift",
                        "shape": "Shape",
                        "coupling": "Coupling Constants"
                        },
                     hide_index=True)
        

    with st.expander("See Peak Match Table"):
        if st.session_state.selected != []:
            st.session_state.atom_idx = np.array([peak[0] for peak in st.session_state.predicted_peak_table])
            table = []
            for i in range(len(st.session_state.atom_idx)):
                match = {}
                match = st.session_state.observed_peaks[st.session_state.selected[i][1]].copy()
                match["atom_idx"] = st.session_state.atom_idx[i]
                table.append(match)

            st.dataframe(table,
                        column_config={"atom_idx": "Atom Index", 
                                        "shift": "Shift", 
                                        "shape": "Shape", 
                                        "coupling": "Coupling Constants"},
                        column_order=("atom_idx", "shift", "shape", "coupling"))
        


with col3:
    ##### Predicted Peaks #####
    if "predicted_peaks" not in st.session_state:
        st.session_state.predicted_peaks = None

    if "predicted_peak_table" not in st.session_state:
        st.session_state.predicted_peak_table = None

    if st.button("Predict peaks"):
        st.session_state.predicted_peak_table = []
        predicted_peaks, atom_indices = predict_peaks(smiles)
        st.session_state.predicted_peaks = predicted_peaks
        for i, peak in enumerate(predicted_peaks):
            index = atom_indices[i]
            shift = float(round(peak["shift"], 4))
            shape = convert_shape_one_hot(peak["shape"])
            coupling = convert_coupling_constants(peak["coupling"])
            st.session_state.predicted_peak_table.append([index, shift, shape, coupling])

    if st.session_state.predicted_peak_table:
        st.write("Predicted Peaks:")
        predicted_peak_df = pd.DataFrame(st.session_state.predicted_peak_table, columns=["Atom Index", "Shift", "Shape", "Coupling Constants"])
        st.dataframe(predicted_peak_df,
                    column_config={"Shift": st.column_config.NumberColumn(format="%.2f")},
                    hide_index=True)
        
    ##### Distance Matrix #####
    if "distance_matrix" not in st.session_state:
        st.session_state.distance_matrix = np.empty((0))

    if "distance_matrix_df" not in st.session_state:
        st.session_state.distance_matrix_df = pd.DataFrame()

    if "total_cost" not in st.session_state:
        st.session_state.total_cost = None

    if st.button("Create Distance Matrix"):
        distance_matrix = create_distance_matrix(st.session_state.predicted_peaks, st.session_state.observed_peaks)
        st.session_state.distance_matrix = distance_matrix

        st.session_state.atom_idx = np.array([peak[0] for peak in st.session_state.predicted_peak_table])
        distance_matrix_with_idx = np.hstack((st.session_state.atom_idx.reshape(-1, 1), distance_matrix))

        num_peaks = distance_matrix.shape[1]
        column_names = ["Atom Index"] + [f"Peak {i}" for i in range(1, num_peaks+1)]

        st.session_state.distance_matrix_df = pd.DataFrame(distance_matrix_with_idx, columns=column_names)
        st.session_state.selected = []

    if st.session_state.distance_matrix.size != 0:
        st.write("Distance Matrix:")

    if not st.session_state.distance_matrix_df.empty:
        df_styled = st.session_state.distance_matrix_df.style.apply(highlight_cells, highlight_indices=st.session_state.selected, axis=1)
        st.dataframe(df_styled,
                    column_config={"Atom Index": st.column_config.NumberColumn(format="%d")},
                    hide_index=True)

    if st.session_state.total_cost != None:
        st.write("Total Distance: " + str(round(st.session_state.total_cost,2)))

    if st.button("Minimize Distance"):
        st.session_state.selected, st.session_state.total_cost = minimize_distance(st.session_state.distance_matrix)
        st.rerun()
