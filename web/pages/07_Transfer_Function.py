
import streamlit as st
import os
import json
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Transfer Function")

# --- Schematic ---
# Path logic similar to Exploded Architecture to find the image
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent # web/pages -> web -> THESIS
scheme_path = root_dir / "docs" / "img" / "scheme.png"

if scheme_path.exists():
    st.image(str(scheme_path), caption="Schematic of the 4x7 Super-Kernel-Nuller component.", use_container_width=True)

st.markdown("""
This page illustrates the complexity of the analytical approach for modeling the complete system.
The schematic above shows the physical architecture of the component. Each section corresponds to a transfer matrix.
Below, we detail each matricial step.
""")

st.header("Matrix Composition")
# Equation with renamed variables: I->X, S->R, Split->Y
st.latex(r"M_{global} = R_{layer} \cdot X_{23,45} \cdot P_{9-14} \cdot Y_{splitter} \cdot N_{layer} \cdot X_{2-3} \cdot P_{5-8} \cdot N_{layer} \cdot P_{1-4}")

# --- Load Matrices ---
matrices_file = current_dir.parent / "assets" / "transfer_function" / "matrices.json"
matrices_data = {}
if matrices_file.exists():
    with open(matrices_file, 'r', encoding='utf-8') as f:
        matrices_data = json.load(f)
        
# Load Global Matrix
global_matrix_file = current_dir.parent / "assets" / "transfer_function" / "transfer_matrix.txt"
global_matrix_content = ""
if global_matrix_file.exists():
    with open(global_matrix_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Cleanup
            content = content.strip()
            if content.startswith("$\\displaystyle"):
                content = content.replace("$\\displaystyle", "")
            if content.endswith("$"):
                content = content[:-1]
            global_matrix_content = content.strip()

# --- Tabs ---
# Order: Flow from Input (Right in equation, Left in diagram)
# P1-4 -> Nlayer -> P5-8 -> X2-3 -> Nlayer -> Ysplitter -> P9-14 -> X23-45 -> Rlayer

tab_names = [
    "Phase P1-4", 
    "Nuller 1", 
    "Phase P5-8", 
    "Crossover X2-3", 
    "Nuller 2", 
    "Splitter Y", 
    "Phase P9-14", 
    "Crossover X23-45", 
    "Recombiner R"
]

tabs = st.tabs(tab_names)

# Helper to display matrix
def show_matrix(tab, key, title, notes=""):
    with tab:
        st.subheader(title)
        if notes:
            st.info(notes)
        
        if key in matrices_data:
            st.latex(matrices_data[key])
        else:
            st.warning(f"Matrix '{key}' not found (incomplete extraction).")

# 1. Phase P1-4 (p_1_4)
show_matrix(tabs[0], "p_1_4", "Phase Shifters Layer 1", "Phase injection on the 4 inputs.")

# 2. Nuller 1 (Nlayer)
show_matrix(tabs[1], "Nlayer", "Nulling Stage 1", "First stage of Nulling couplers.")

# 3. Phase P5-8 (p_5_8)
show_matrix(tabs[2], "p_5_8", "Phase Shifters Layer 2", "Intermediate phase correction.")

# 4. Crossover X2-3 (invert_2_3)
show_matrix(tabs[3], "invert_2_3", "Crossover Channels 2-3", "Crossing of waveguides 2 and 3.")

# 5. Nuller 2 (Nlayer again) - Reusing Nlayer data as it is the same physical component block usually
show_matrix(tabs[4], "Nlayer", "Nulling Stage 2", "Second stage of Nulling couplers.")

# 6. Splitter Y (splitters)
show_matrix(tabs[5], "splitters", "Y-Junction Splitters", "Power splitting, transition from 4 to 7 waveguides.")

# 7. Phase P9-14 (p_9_14)
show_matrix(tabs[6], "p_9_14", "Phase Shifters Layer 3", "Final stage of phase shifters on the 7 channels.")

# 8. Crossover X23-45 (invert_23_45)
show_matrix(tabs[7], "invert_23_45", "Crossover Channels 2-3 & 4-5", "Final waveguide rearrangement.")

# 9. Recombiner R (Slayer)
show_matrix(tabs[8], "Slayer", "Recombination Layer", "Final mixing stage to obtain the final outputs.")

# 10. Global M
st.subheader("Analytical Global Matrix")
st.write("The product of all these matrices yields the following expression:")
if global_matrix_content:
    st.latex(global_matrix_content)
else:
    st.error("Global matrix file missing.")

st.warning("As you can see, the expression explodes in complexity, making pure symbolic analysis very cumbersome.")
