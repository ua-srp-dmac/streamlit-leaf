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

for root, dirs, files in os.walk(base_path + 'results'):
    for file in files:
        filename=os.path.join(root, file)
        file_names.append(filename)

df = pd.DataFrame({'File Name' : file_names})

gd = GridOptionsBuilder.from_dataframe(df)
gd.configure_pagination(enabled=True)
gd.configure_selection(selection_mode='single', use_checkbox=True)
gd.configure_column('File Name', headerCheckboxSelection = True)

st.title('Leaf Segmentation')
st.header('Results')

file_table = AgGrid(df, fit_columns_on_grid_load=True, gridOptions=gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)

visualize = st.button('Visualize')

if visualize:

    selected_result = file_table['selected_rows']
    file_path = selected_result[0]['File Name']
    
    # get file_name without extension
    file_name = file_path.split('/')[-1]

    image = Image.open(file_path)

    st.subheader('Output Image')
    st.text(file_name)
    st.image(image)
