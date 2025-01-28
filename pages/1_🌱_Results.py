import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

# Get the results path
results_path = sys.argv[3]
full_results_path = Path("/data-store") / results_path
full_results_path = str(full_results_path.resolve())

# Collect file names
file_names = []
dirs = []

for root, dirs, files in os.walk(full_results_path):
    for file in files:
        filename = os.path.join(root, file)
        file_names.append(filename)

df = pd.DataFrame({'File Name': file_names})

# Configure AgGrid
gd = GridOptionsBuilder.from_dataframe(df)
gd.configure_pagination(enabled=True)
gd.configure_selection(selection_mode='single', use_checkbox=True)
gd.configure_column('File Name', headerCheckboxSelection=True)

st.title('Leaf Segmentation')
st.header('Results')

# Display AgGrid
file_table = AgGrid(df, fit_columns_on_grid_load=True, gridOptions=gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)

visualize = st.button('Visualize')

if visualize:
    selected_result = file_table['selected_rows']
    
    if selected_result:
        file_path = selected_result[0]['File Name']
        # Get file name
        file_name = os.path.basename(file_path)

        # Check file extension
        file_extension = os.path.splitext(file_path)[1].lower()

        st.subheader('Output')
        st.text(file_name)

        if file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            # Display image
            image = Image.open(file_path)
            st.image(image, caption=file_name)
        elif file_extension == '.csv':
            # Display CSV
            csv_data = pd.read_csv(file_path)
            st.dataframe(csv_data)
        else:
            st.warning(f"Unsupported file type: {file_extension}")

        # Add a download button
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            st.download_button(
                label="Download File",
                data=file_bytes,
                file_name=file_name,
                mime="application/octet-stream"
            )
    else:
        st.warning("Please select a file to visualize.")
