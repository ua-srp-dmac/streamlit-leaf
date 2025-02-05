import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# detectron2 imports
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data.catalog import Metadata
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode

import os
import datetime
import time
import sys
import cv2
import base64
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import csv
import re
import gc

from PIL import Image
from PIL.ExifTags import TAGS

# pyzbar is used for decoding QR codes
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol

# streamlit imports
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

from matplotlib import pyplot as plt

from qreader import QReader

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(png_file):
    """ Reads png image so it can be used as a background-image in css. 
    """
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(
    png_file,
    background_position="50% 10%",
    margin_top="10%",
    image_width="70%",
    image_height="",
):
    """ Builds makrrkup and css for logo.
    """

    binary_string = get_base64_of_bin_file(png_file)

    return (
        """
            <style>
                [data-testid="stSidebarNav"] {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    background-position: %s;
                    margin-top: %s;
                    background-size: %s %s;
                }
            </style>
        """ % (
            binary_string,
            background_position,
            margin_top,
            image_width,
            image_height,
        )
    )


def add_logo(png_file):
    """ Streamlit does not easily support adding a logo to the top
        of the sidebar due to how the multi-page menu is rendered.

        This is a workaround for placing the logo at the top of
        the sidebar.
    """

    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )

def prepend_data_store(path):
    """Ensure path is prefixed with /data-store correctly."""
    if path.startswith('/'):
        return os.path.join('/data-store', path.lstrip('/'))
    return os.path.join('/data-store', path)

def setup():
    """ App setup that needs to run once at initialization.
    """
    # get base data path from user input
    data_path = sys.argv[1]
    model_file =  sys.argv[2]
    results_path = sys.argv[3]
    run_on_cyverse = sys.argv[4] if len(sys.argv) > 4 else None

    if run_on_cyverse == 'True':
        data_path = prepend_data_store(data_path)
        model_file = prepend_data_store(model_file)
        results_path = prepend_data_store(results_path)

    return data_path, model_file, results_path


@st.cache()
def setup_model(model_file):
    """ Setup model and metadata from trained weights.
    """

    leaf_cfg = get_cfg()

    leaf_cfg.MODEL.DEVICE='cpu'
    
    leaf_cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    ))

    # set to number of classes (qr, leaf, red-square)
    leaf_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    
    # path to trained weights
    leaf_cfg.MODEL.WEIGHTS = model_file 
    
    # set a custom testing threshold
    leaf_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  

    leaf_predictor = DefaultPredictor(leaf_cfg)

    # set up metadata
    leaf_metadata = Metadata()
    leaf_metadata.set(thing_classes = ['leaf', 'qr', 'red-square'])

    return (leaf_predictor, leaf_metadata)

def write_results_to_csv(csv_file_path, batch_results):
    """ Write batch results to a CSV file. """
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file, 
            fieldnames=["Image Name", "Leaf Pixel Area", "Leaf Count", "Red Square Pixel Area", "QR Code Pixel Area", "Leaf Area cm2", "Decoded QR"]
        )

        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write the batch results
        writer.writerows(batch_results)

def run_inference(batch, batch_idx, batch_size, total_images):
    """Run prediction on a batch of images with optimized memory usage."""

    batch_results = []

    for index, item in enumerate(batch):
        file_path = item['file_path']
        file_name = item['file_name']
        date = item['date']

        progress_bar_text.write(
            f"Processing image {((batch_size * batch_idx) + index + 1)} of {total_images} ({file_name})"
        )

        # Load image only when needed
        with Image.open(file_path) as img:
            image_np = np.array(img)

        # Run inference
        outputs = leaf_predictor(image_np)

        # Convert tensors to NumPy early
        pred_boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        class_labels = outputs['instances'].pred_classes.cpu().numpy()
        pred_masks = outputs['instances'].pred_masks.cpu().numpy()

        # Initialize counts
        leaf_pixel_area = 0
        red_square_pixel_area = 0
        qr_code_pixel_area = 0
        leaf_count = 0
        leaf_area = None
        qr_indices = []

        # Process predicted masks
        for i, label in enumerate(class_labels):
            mask = pred_masks[i]
            pixel_count = mask.sum()

            if label == 0:  # Leaf
                leaf_pixel_area += pixel_count
                leaf_count += 1
            elif label == 1:  # QR code
                qr_code_pixel_area += pixel_count
                qr_indices.append(i)
            elif label == 2:  # Red square
                red_square_pixel_area += pixel_count

        # Calculate leaf area
        if red_square_pixel_area:
            leaf_area = (4 * leaf_pixel_area) / red_square_pixel_area
        elif qr_code_pixel_area:
            leaf_area = (1.44 * leaf_pixel_area) / qr_code_pixel_area

        # Decode QR code
        qr_result_decoded = None
        if qr_indices:
            bbox = pred_boxes[qr_indices[0]]
            x0, y0, x1, y1 = map(round, [bbox[0] - 500, bbox[1] - 500, bbox[2] + 500, bbox[3] + 500])

            # Crop image for QR decoding
            with Image.open(file_path) as img:
                crop_img = img.crop((x0, y0, x1, y1))

            # Resize image
            crop_img_resized = crop_img.resize(
                (int(crop_img.width * 0.4), int(crop_img.height * 0.4)), Image.ANTIALIAS
            )

            crop_img_resized.save("qr_crop.jpg")

            # Run external QR decoder
            cmd = subprocess.run(
                ["python", "/app/decode_qr.py"],
                capture_output=True,
                check=False
            )

            stdout = cmd.stdout.decode()
            qr_result_decoded = stdout if stdout else None

            if not qr_result_decoded:
                qreader = QReader()
                decoded_text = qreader.detect_and_decode(image=image_np)
                qr_result_decoded = decoded_text[0] if decoded_text else "Failed to decode"

        # Collect results
        batch_results.append({
            "Image Name": file_name,
            "Leaf Pixel Area": leaf_pixel_area,
            "Leaf Count": leaf_count,
            "Red Square Pixel Area": red_square_pixel_area,
            "QR Code Pixel Area": qr_code_pixel_area,
            "Leaf Area cm2": leaf_area,
            "Decoded QR": qr_result_decoded or "Failed to decode"
        })

        # Rename original file if needed
        image_dir = '/'.join(file_path.split('/')[:-1])
        old_file_name = file_path.split('/')[-1]

        new_file_name = f"{date}_{qr_result_decoded}" if qr_result_decoded else f"{date}_{file_name}"

        if rename_files_option:
            new_file_path = f"{image_dir}/{new_file_name}.JPG"
            if not old_file_name.startswith(date):
                os.rename(file_path, new_file_path)

        # Save visualization results
        if save_masks_to_img:
            v = Visualizer(image_np[:, :, ::-1], metadata=leaf_metadata, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
            out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            result_image = Image.fromarray(out.get_image()[:, :, ::-1])

            save_path = Path(results_path) / analysis_name / f"{new_file_name}-result.JPG"
            result_image.save(save_path)

        # Free memory for each iteration
        del image_np
        del outputs
        del pred_boxes, class_labels, pred_masks
        gc.collect()

        progress_bar.progress(((batch_size * batch_idx) + index + 1) / total_images)

    # Save batch results
    write_results_to_csv(Path(results_path) / analysis_name / 'results.csv', batch_results)

    # Final cleanup
    del batch_results
    gc.collect()
    
    
def get_files(data_path):
    """ Walk through file directory and gather necessary information to
        display file list.
    """

    file_names = []
    modified_dates = []
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            filename=os.path.join(root, file)
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                stats=os.stat(filename)
                file_names.append(filename)
                modified_dates.append(
                    datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %-I:%M %p')
                )    

    return (file_names, modified_dates)
        

#--------------------- STREAMLIT INTERFACE ----------------------#

# show logo at top of sidebar
add_logo('/app/images/srp-logo.png')

# setup
data_path, model_file, results_path = setup()
leaf_predictor, leaf_metadata = setup_model(model_file)

st.header('Leaf Segmentation App')

# options
st.subheader('1. Configure Options')
st.markdown('If you\'d like to rename file according to the naming convention, select **Rename Files**.')


# Get current date in YYYY-MM-DD format
current_date = datetime.datetime.now().strftime("%Y-%m-%d_analysis")

# Create a text input field with current date as the default
analysis_name = st.text_input("Analysis Name", value=current_date)

# Validate the input
is_valid = True
if " " in analysis_name:
    st.error("Analysis Name cannot contain spaces.")
    is_valid = False
elif not re.match(r"^[a-zA-Z0-9_-]+$", analysis_name):
    st.error("Only letters, numbers, underscores, and dashes are allowed.")
    is_valid = False
elif not analysis_name.strip():
    st.error("Analysis Name is required.")
    is_valid = False

button_disabled = not is_valid

rename_files_option = st.checkbox('Rename files', value=False)
save_masks_to_img = st.checkbox('Save segmentation results to image', value=False)

st.subheader('2. Select Files')
st.markdown('Select the files you\'d like to analyze.')

progress_bar_text = st.empty()

# walk through directory to display files in table
file_names, modified_dates = get_files(data_path)

# set up AgGrid
df = pd.DataFrame({'File Name' : file_names, 'Last Updated': modified_dates})
gd = GridOptionsBuilder.from_dataframe(df)
gd.configure_pagination(enabled=False)
gd.configure_selection(selection_mode='multiple', use_checkbox=True)
gd.configure_column('File Name', headerCheckboxSelection = True)

# display AgGrid
file_table = AgGrid(
    df,
    height=500, 
    fit_columns_on_grid_load=True,
    gridOptions=gd.build(),
    update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED | GridUpdateMode.MODEL_CHANGED )

run = st.button('Run', disabled=button_disabled)


if run:
    with st.spinner('Running inference...'):

        # Ensure directories exist
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(os.path.join(results_path, analysis_name), exist_ok=True)

        progress_bar = st.progress(0)
        progress_bar_text = st.empty()

        selected_rows = file_table['selected_rows']
        batch_size = 10
        total_batches = (len(selected_rows) + batch_size - 1) // batch_size  # Efficient batch count

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(selected_rows))
            batch = []

            progress_bar_text.write(f"Preparing batch {batch_idx + 1} of {total_batches}")

            for index in range(start_idx, end_idx):
                row = selected_rows[index]
                file_path = row['File Name']
                file_name = os.path.splitext(os.path.basename(file_path))[0]  # More efficient parsing

                # Read DateTime EXIF metadata (only the needed tag)
                with Image.open(file_path) as image:
                    exifdata = image.getexif()
                    img_date = exifdata.get(306, '')  # 306 = DateTime tag
                
                date = img_date.split(' ')[0].replace(':', '-') if img_date else ''

                batch.append({
                    'file_path': file_path,  # Store path, not full image
                    'file_name': file_name,
                    'date': date
                })

            run_inference(batch, batch_idx, batch_size, total_images=len(selected_rows))

        time.sleep(1)
        progress_bar.empty()
        progress_bar_text.write('Processing complete 🎉')


    


 
            