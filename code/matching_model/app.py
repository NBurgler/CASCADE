import streamlit as st
import numpy as np
import pandas as pd
from evaluate_model import predict_and_match
import matplotlib.pyplot as plt
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


def draw_mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        atom.SetProp('atomNote', str(index))
    
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    png_data = d.GetDrawingText()

    mol_image = Image.open(BytesIO(png_data))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mol_image)  # Show the RDKit image
    ax.axis("off")
    return fig

# Function to highlight specific cells in a DataFrame
def highlight_cells(dataframe, highlight_coords):
    def highlight(val, row, col):
        # Check if the current cell's coordinates are in the highlight list
        if (row, col) in highlight_coords:
            return "background-color: yellow; font-weight: bold;"
        return ""

    # Apply styles for each cell
    styled_df = dataframe.style.apply(
        lambda row: [highlight(val, row.name, col) for col, val in enumerate(row)],
        axis=1,
    )
    return styled_df

# Title of the app
st.title("NMR Peak Matching")

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

if "observed_peaks" not in st.session_state:
    st.session_state.observed_peaks = []

# Initialize session state to store the table data
if "observed_peak_table" not in st.session_state:
    st.session_state.observed_peak_table = []  # Start with an empty list

# Create input boxes for the three properties (on the same line)
col1, col2, col3 = st.columns(3)
with col1:
    shift = st.text_input("Shift", key="shift")
with col2:
    shape = st.text_input("Shape", key="shape")
with col3:
    coupling = st.text_input("Coupling Constants", key="coupling")

# Add a submit button
if st.button("Submit Peak"):
    observed_peak = {"shift": None, "shape": None, "coupling": None}
    # Validate inputs (ensure none of the fields are empty)
    if shift and shape and coupling:
        # Append the new row to the session state table_data
        st.session_state.observed_peak_table.append([shift, shape, coupling])
        observed_peak["shift"] = float(shift)
        observed_peak["shape"] = shape
        observed_peak["coupling"] = (";").join(coupling.split(", "))
        st.session_state.observed_peaks.append(observed_peak)
        # Clear input fields
        st.session_state.prop1 = ""
        st.session_state.prop2 = ""
        st.session_state.prop3 = ""
    else:
        st.warning("Please fill in all three properties before submitting.")

# Display the table
if st.session_state.observed_peak_table:
    st.write("Peak Table:")
    df = pd.DataFrame(st.session_state.observed_peak_table, columns=["Shift", "Shape", "Coupling Constants"])
    st.table(df)


# Smiles input
smiles = st.text_input("Enter a smiles:", value="C#CCC1CCOCO1")

if "molecule_image" not in st.session_state:
    st.session_state.molecule_image = None

if st.button("Submit Smiles"):
    st.session_state.molecule_image = draw_mol_image(smiles)

if st.session_state.molecule_image:
    st.pyplot(st.session_state.molecule_image)

if "predicted_peak_table" not in st.session_state:
    st.session_state.predicted_peak_table = []

if "predicted_peaks" not in st.session_state:
    st.session_state.predicted_peaks = None

if st.button("Predict peaks"):
    print(st.session_state.observed_peaks)
    predicted_peaks, observed_peaks, distance_matrix, selected, total_cost, atom_indices = predict_and_match(smiles, st.session_state.observed_peaks)
    st.session_state.predicted_peaks = predicted_peaks
    print(st.session_state.predicted_peaks)
    for peak in predicted_peaks:
        shift = float(round(peak["shift"], 4))
        shape = convert_shape_one_hot(peak["shape"])
        coupling = convert_coupling_constants(peak["coupling"])
        st.session_state.predicted_peak_table.append([shift, shape, coupling])

    st.session_state.distance_matrix = distance_matrix

if st.session_state.predicted_peak_table:
    st.write("Predicted Peak Table:")
    df = pd.DataFrame(st.session_state.predicted_peak_table, columns=["Shift", "Shape", "Coupling Constants"])
    pd.to_numeric(df["Shift"], downcast='float')
    st.table(df)