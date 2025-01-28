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

@st.cache()
def setup():
    """ App setup that needs to run once at initialization.
    """
    # get base data path from user input
    data_path = sys.argv[1]
    model_file = sys.argv[2]
    results_path = sys.argv[3]

    # Prefix all paths with /data-store/
    full_data_path = Path("/data-store") / data_path
    full_model_file = Path("/data-store") / model_file
    full_results_path = Path("/data-store") / results_path

    print('FULL results:', full_results_path)

    # if results path doesn't exist, create it
    if not os.path.isdir(full_results_path):
        print('In if statement', full_results_path)
        os.mkdir(full_results_path) 

    return full_data_path, str(full_model_file), full_results_path


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
    """ Run prediction on batch of images.
    """

    batch_results = []

    for index, image in enumerate(batch):

        progress_bar_text.write(f"Processing image {((batch_size * batch_idx) + index + 1)} of {total_images} ({image['file_name']})")

        # run interence on selected image
        outputs = leaf_predictor(image['image'])

        # get bboxes and class labels
        pred_boxes = outputs['instances'].pred_boxes.tensor.numpy()
        class_labels = outputs['instances'].pred_classes.numpy()
        pred_masks = outputs['instances'].pred_masks.numpy()

        # Initialize counts for the current image
        leaf_pixel_area = 0
        red_square_pixel_area = 0
        qr_code_pixel_area = 0
        leaf_count = 0
        leaf_area = None

        qr_indices = []
        
        # get indices of leaves, qr codes, and red reference square
        for i, label in enumerate(class_labels):
            mask = pred_masks[i]
            pixel_count = mask.sum()  # Count the number of True pixels

            if label == 0:  # Leaf
                leaf_pixel_area += pixel_count
                leaf_count += 1
            elif label == 1:  # QR code
                qr_code_pixel_area += pixel_count
                qr_indices.append(i)
            elif label == 2:  # Red square
                red_square_pixel_area += pixel_count
        
        # get leaf area in cm2 using red square as reference, or QR code if red square not detected
        if (red_square_pixel_area):
            leaf_area = leaf_area = (4 * leaf_pixel_area) / red_square_pixel_area
        elif (qr_code_pixel_area):
            leaf_area = leaf_area = (1.44 * leaf_pixel_area) / qr_code_pixel_area


        # if qr code was detected, decode
        crop_img = None
        
        if len(qr_indices):

            # get first qr code
            bbox = pred_boxes[qr_indices[0]]

            # (x0, y0, x1, y1)
            x0 = round(bbox[0].item()-500)
            y0 = round(bbox[1].item()-500)
            x1 = round(bbox[2].item()+500)
            y1 = round(bbox[3].item()+500)

            # crop to bounding box for QR decoding
            crop_img = image['image'][ y0:y1, x0:x1]
            
            scale_percent = 40 # percent of original size
            width = int(crop_img.shape[1] * scale_percent / 100)
            height = int(crop_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            crop_img_resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

            image_to_save = Image.fromarray(crop_img_resized)
            image_to_save.save("qr_crop.jpg")
       
        qr_result_decoded = None

        cmd = subprocess.run(
            ["python", "/app/decode_qr.py"],
            capture_output=True,
            check=False
        )
        
        stdout = cmd.stdout.decode()

        if stdout:
            qr_result_decoded = stdout
        else:
            qreader = QReader()

            decoded_text = qreader.detect_and_decode(image=image['image'])

            if len(decoded_text):
                qr_result_decoded = decoded_text[0]
        

        # Collect results for the current image
        batch_results.append({
            "Image Name": image['file_name'],
            "Leaf Pixel Area": leaf_pixel_area,
            "Leaf Count": leaf_count,
            "Red Square Pixel Area": red_square_pixel_area,
            "QR Code Pixel Area": qr_code_pixel_area,
            "Leaf Area cm2": leaf_area,
            "Decoded QR": qr_result_decoded if qr_result_decoded is not None else "Failed to decode"
        })

        # get directory current image is in
        image_dir = image['file_path'].split('/')[:-1]
        image_dir = '/'.join(image_dir)

        old_file_name = image['file_path'].split('/')[-1]
        new_file_name = None

        # if QR was decoded, name results file w/ plant ID
        if qr_result_decoded:
            new_file_name = image['date'] + '_' + qr_result_decoded
        else:
            new_file_name = image['date'] + '_' + image['file_name']


        # rename original file
        if rename_files_option:
 
            new_file_path = image_dir + '/' + new_file_name + '.JPG'
            
            if not old_file_name.startswith(image['date']):
                os.rename(image['file_path'], new_file_path)

        # save leaf results to file
        if save_masks_to_img: 
            # set up results visualizer
            v = Visualizer(image['image'][:, :, ::-1],
                metadata=leaf_metadata, 
                scale=0.5, 
                instance_mode=ColorMode.SEGMENTATION
            )

            out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            result_arr = out.get_image()[:, :, ::-1]
            result_image = Image.fromarray(result_arr)

            if not old_file_name.startswith(image['date']):
                save_path = Path(results_path) / f"{new_file_name}-result.JPG"
            else:
                save_path = Path(results_path) / f"{old_file_name.split('.')[0]}-result.JPG"


            result_image.save(save_path)
        
        progress_bar.progress(((batch_size * batch_idx) + index + 1) / (total_images))
    
    write_results_to_csv(Path(results_path) / 'results.csv', batch_results)
    
    del batch  # Remove batch from memory
    del outputs  # Clear the outputs
    
    
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

run = st.button('Run')


if run:
    with st.spinner('Running inference...'):

        progress_bar = st.progress(0)
        progress_bar_text = st.empty()

        # set up batch
        selected_rows = file_table['selected_rows']
        batch = []
        batch_size = 10

        total_batches = len(selected_rows) // batch_size + (1 if len(selected_rows) % batch_size != 0 else 0)
    
        for batch_idx in range(total_batches):
            # Get the current batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(selected_rows))
            batch = []
            
            progress_bar_text.write(f"Preparing batch {batch_idx + 1} of {total_batches}")
        
            for index in range(start_idx, end_idx):
                row = selected_rows[index]
                file_path = row['File Name']
                image = Image.open(file_path)

                # get file_name without extension
                file_name = file_path.split('/')[-1].split('.')[0]
                
                # get DateTime from exif data
                exifdata = image.getexif()
                img_date = ''

                for tag_id in exifdata:
                    
                    # get the tag name, instead of human unreadable tag id
                    tag = TAGS.get(tag_id, tag_id)
                    data = exifdata.get(tag_id)

                    if tag == 'DateTime': 
                        img_date = data
                
                date = img_date.split(' ')[0].replace(':', '-')

                batch.append({
                    'image': np.array(image),
                    'image_cv2': cv2.imread(file_path),
                    'file_name': file_name,
                    'file_path': file_path,
                    'date': date
                })
            
            run_inference(batch, batch_idx, batch_size, total_images=len(selected_rows))

        time.sleep(1)
        progress_bar.empty()
        progress_bar_text.write('Processing complete 🎉')


    


 
            