import streamlit as st
import numpy as np
import pandas as pd
from evaluate_model import predict_peaks, create_distance_matrix, minimize_distance
import matplotlib.pyplot as plt
import matplotlib
import random
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from io import BytesIO
from PIL import Image
import math
import base64

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
    
    if color_per_idx != None:
        atom_colors = {int(index): hex_to_rgb(color) for index, color in color_per_idx}
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors)
    else:
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol)

    d.FinishDrawing()
    png_data = d.GetDrawingText()

    mol_image = Image.open(BytesIO(png_data))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mol_image)  # Show the RDKit image
    ax.axis("off")
    return fig

def highlight_cells(row, highlight_indices):
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

def fig_to_base64(fig):
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)  # Reset buffer
    
    # Convert to Base64 string
    img_base64 = base64.b64encode(img_bytes.read()).decode()
    
    return f"data:image/png;base64,{img_base64}"

############################################################
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
        st.session_state.observed_peaks_df = pd.DataFrame([{"shift":None, "shape":None, "coupling":None, "amount":None}])

    if "observed_peaks" not in st.session_state:
        st.session_state.observed_peaks = []

    if "observed_peak_amount" not in st.session_state:
        st.session_state.observed_peak_amount = []

    if "observed_peak_text" not in st.session_state:
        st.session_state.observed_peak_text = ''

    st.write("You can enter the peaks in the publication style format (e.g. 1H NMR (400 MHz, CDCl3) δ 8.32 (d, J = 5.6 Hz, 1H, CHAr), 8.22 (s, 1H, CHAr), etc.)), or you can enter the properties in the table directly.")

    st.session_state.observed_peak_text = st.text_area("Enter the peaks as text")

    #df = pd.DataFrame([{"shift":None, "shape":None, "coupling":None, "amount":None}])
    df = pd.DataFrame(columns=["shift", "shape", "coupling", "amount"])
    if st.session_state.observed_peak_text != '':
        text = st.session_state.observed_peak_text
        text = text[:-1].split('δ', 1)[-1]  # Remove everything up to the δ

        while text[-1] != ")": # Remove any trailing characters after the last bracket
            text = text[:-1]

        shifts = re.sub(r'\([^)]*\)', '', text).strip().split(" , ") # Only the shifts are not in brackets
        shapes = []
        couplings = []
        amounts = []
        
        text_in_brackets = re.findall(r'\(([^)]*)\)', text) # Extract text inside brackets

        for peak_properties in text_in_brackets:
            properties = peak_properties.split(', ')
            shapes.append(properties[0]) # Shape is always first
            if (shapes[-1] == 'm') or (shapes[-1] == "s"): # Check if there are coupling constants
                couplings.append('-')
                amounts.append(properties[1][:-1])
            else:
                coupling = ''
                for i in range(len(shapes[-1])):  # the number of coupling constants matches the number of shapes
                    coupling += properties[1+i]
                    if i != len(shapes[-1])-1:
                        coupling += ", "

                couplings.append(coupling[4:-3])
                amounts.append(properties[1 + len(shapes[-1])][:-1]) # Remove the H to get the amount

        for i, shift in enumerate(shifts):
            if "-" in shift or "–" in shift:
                start, end = map(float, re.findall(r'\d+\.\d+', shift)) # Gather both shifts
                avg = f"{(start + end) / 2:.2f}"  # Format to 2 decimal places
                shifts[i] = avg

        df["shift"] = shifts
        df["shape"] = shapes
        df["coupling"] = couplings
        df["amount"] = amounts

    st.write("Observed Peaks:")
    st.session_state.observed_peaks_df = st.data_editor(df, 
                                                        num_rows="dynamic",
                                                        column_order=("shift", "shape", "coupling", "amount"),
                                                        column_config={
                                                        "shift": "Shift",
                                                        "shape": "Shape",
                                                        "coupling": "Coupling Constants",
                                                        "amount": "Amount"
                                                        },
                                                        hide_index=True)

    if not st.session_state.observed_peaks_df.isna().any().any(): 
        observed_peak_dict = st.session_state.observed_peaks_df.to_dict(orient="records")
        observed_peaks = []
        observed_peak_amount = []
        for i, peak in enumerate(observed_peak_dict):
            shift = float(peak["shift"])
            shape = peak["shape"]
            coupling = (";").join(peak["coupling"].split(", "))
            observed_peaks.append({"shift": shift, "shape": shape, "coupling": coupling})
            observed_peak_amount.append(peak["amount"])

        st.session_state.observed_peaks = observed_peaks
        st.session_state.observed_peak_amount = observed_peak_amount

with col2:
    ##### Molecule #####
    if "selected" not in st.session_state:
        st.session_state.selected = []

    if "colors_in_order" not in st.session_state:
        st.session_state.colors_in_order = []

    if "atom_idx" not in st.session_state:
        st.session_state.atom_idx = []

    if "memory_df" not in st.session_state:
        st.session_state.memory_df = pd.DataFrame()

    smiles = st.text_input("Enter a smiles:")

    if "molecule_image" not in st.session_state:
        st.session_state.molecule_image = None

    if "img_base64" not in st.session_state:
        st.session_state.img_base64 = None

    if st.session_state.selected != []:
        colored_df = st.session_state.observed_peaks_df.copy()
        
        colored_df["color"] = assign_random_colors(len(colored_df))
        st.session_state.colors_in_order = colored_df["color"].values

        st.session_state.matching_table = colored_df.style.apply(highlight_row, axis=1)

        color_per_idx = [(st.session_state.atom_idx[pred_idx], st.session_state.colors_in_order[obs_idx]) for (pred_idx, obs_idx) in st.session_state.selected]
        st.session_state.molecule_image = draw_mol_image(smiles, color_per_idx)

    subcol_1, subcol_2 = st.columns(2)

    with subcol_1:
        if st.button("Submit Smiles"):
            color_per_idx = None
            st.session_state.molecule_image = draw_mol_image(smiles, color_per_idx)
    
    with subcol_2:
        if st.button("Reset Predictions"):
            st.session_state.predicted_peaks = None
            st.session_state.predicted_peak_table = None
            st.session_state.distance_matrix = np.empty((0))
            st.session_state.distance_matrix_df = pd.DataFrame()
            st.session_state.total_cost = None
            st.session_state.selected = []
            st.session_state.colors_in_order = []
            color_per_idx = None
            st.session_state.molecule_image = draw_mol_image(smiles, color_per_idx)
            


    if st.session_state.molecule_image:
        st.write("Molecule:")
        st.pyplot(st.session_state.molecule_image)
        st.session_state.img_base64 = fig_to_base64(st.session_state.molecule_image)
        plt.close(st.session_state.molecule_image)

    ##### Matching Table #####
    if "matching_table" not in st.session_state:
        st.session_state.matching_table = None

    if st.session_state.matching_table != None:
        st.dataframe(st.session_state.matching_table,
                     column_order=("shift", "shape", "coupling", "amount"),
                     column_config={
                        "shift": "Shift",
                        "shape": "Shape",
                        "coupling": "Coupling Constants",
                        "amount": "Amount"
                        },
                     hide_index=True)

    ##### Memory Table #####
    if not st.session_state.memory_df.empty:
        st.dataframe(st.session_state.memory_df,
                     column_config={
                         "Image": st.column_config.ImageColumn("Molecule Image")
                         },
                         hide_index=True)


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
        st.session_state.selected, st.session_state.total_cost = minimize_distance(st.session_state.distance_matrix, st.session_state.observed_peak_amount)
        data_row = pd.DataFrame({"Image": st.session_state.img_base64,
                                     "Molecule": smiles,
                                     "Distance": st.session_state.total_cost
                                     }, index = [0])
            
        st.session_state.memory_df = pd.concat([st.session_state.memory_df, data_row], ignore_index=True)
        st.rerun()
