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
    base_path = sys.argv[1]

    # if results path doesn't exist, create it
    if not os.path.isdir(base_path + 'results'):
        os.mkdir(base_path + 'results') 

    return base_path


@st.cache()
def setup_model(base_path):
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
    leaf_cfg.MODEL.WEIGHTS = base_path + 'models/leaf_qr_model.pth' 
    
    # set a custom testing threshold
    leaf_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  

    leaf_predictor = DefaultPredictor(leaf_cfg)

    # set up metadata
    leaf_metadata = Metadata()
    leaf_metadata.set(thing_classes = ['leaf', 'qr', 'red-square'])

    return (leaf_predictor, leaf_metadata)


def run_inference(batch):
    """ Run prediction on batch of images.
    """

    for index, image in enumerate(batch):

        # run interence on selected image
        outputs = leaf_predictor(image['image'])

        # get bboxes and class labels
        pred_boxes = outputs['instances'].pred_boxes.tensor.numpy()
        class_labels = outputs['instances'].pred_classes.numpy()

        qr_indices = []
        leaf_indices = []
        
        # get indices of leaves and qr codes
        for i, label in enumerate(class_labels):
            if label == 0: # leaf
                leaf_indices.append(i)
            elif label == 1: # qr
                qr_indices.append(i)

        qr_result_decoded = None
        
        # if qr code was detected, decode
        if len(qr_indices):

            # get first qr code
            bbox = pred_boxes[qr_indices[0]]

            # (x0, y0, x1, y1)
            x0 = round(bbox[0].item())
            y0 = round(bbox[1].item())
            x1 = round(bbox[2].item())
            y1 = round(bbox[3].item())

            crop_mask = cv2.inRange(crop_img,(0,0,0),(120,120,120))
            thresholded = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
            inverted = 255-thresholded # black-in-white

            scale_percent = 20 # percent of original size
            width = int(crop_img.shape[1] * scale_percent / 100)
            height = int(crop_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            
            resize image
            crop_img_resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

            st.image(crop_img_resized)

            image_scaled = crop_img.resize((int(round(x*.2)), int(round(y*.2))))

            st.image(crop_img)
            st.image(inverted)

        # crop to bounding box for QR decoding
        crop_img = image['image_cv2']
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        st.image(crop_img)


        qreader = QReader()
        decoded_text = qreader.detect_and_decode(image=crop_img)
        print('decoded_text', decoded_text)
            
        qr_result = decode(crop_img, symbols=[ZBarSymbol.QRCODE])
        print('QR Result:', qr_result)

        if len(qr_result):
            qr_result_decoded = qr_result[0].data.decode('utf-8') 

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
        if run_model_option: 
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
                save_path = Path(base_path + 'results/' + new_file_name + '-result.JPG')
            else:
                save_path = Path(base_path + 'results/' + old_file_name.split('.')[0] + '-result.JPG')


            result_image.save(save_path)
        
        progress_bar.progress((index + 2) / (len(batch) + 1))
    
    
def get_files(base_path):
    """ Walk through file directory and gather necessary information to
        display file list.
    """

    file_names = []
    modified_dates = []
    
    for root, dirs, files in os.walk(base_path + 'data'):
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
base_path = setup()
leaf_predictor, leaf_metadata = setup_model(base_path)

st.header('Leaf Segmentation App')

# options
st.subheader('1. Configure Options')
st.markdown('If you\'d like to rename file according to the naming convention, select **Rename Files**.')

rename_files_option = st.checkbox('Rename files', value=True)
run_model_option = st.checkbox('Save segmentation results to image', value=True)

st.subheader('2. Select Files')
st.markdown('Select the files you\'d like to analyze.')

# walk through directory to display files in table
file_names, modified_dates = get_files(base_path)

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

        # set up batch
        selected_rows = file_table['selected_rows']
        batch = []
        
        for index, row in enumerate(selected_rows):
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
        
        progress_bar.progress(1 / (len(batch) + 1))
        
        run_inference(batch)

        time.sleep(1)
        progress_bar.empty()


    


 
            