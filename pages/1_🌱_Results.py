import streamlit as st

import os
import sys

from PIL import Image
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

base_path = sys.argv[1]

file_names = []
dirs = []

for root, dirs, files in os.walk(base_path + "results"):
    for file in files:
            filename=os.path.join(root, file)
            file_names.append(filename)

df = pd.DataFrame({'File Name' : file_names})

gd = GridOptionsBuilder.from_dataframe(df)
gd.configure_pagination(enabled=True)
gd.configure_selection(selection_mode="single", use_checkbox=True)
gd.configure_column("File Name", headerCheckboxSelection = True)

file_table = AgGrid(df, fit_columns_on_grid_load=True, gridOptions=gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)

visualize = st.button('Visualize')

if visualize:

    selected_result = file_table["selected_rows"]
    file_name = selected_result[0]['File Name']

    image = Image.open(file_name)
    # leaf_count = len(pred_boxes)
    # # kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{leaf_count}</h1>", unsafe_allow_html=True)
    st.subheader('Output Image')
    st.image(image)

    # st.markdown("**Detected Leaves**")
    # kpi1_text = st.markdown("0")
    # st.markdown('---')