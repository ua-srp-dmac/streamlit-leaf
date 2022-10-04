import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data.catalog import Metadata
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode

import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
from torchvision import transforms

import streamlit as st

from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder



@st.cache()
def setup():
    """ Setup model from trained weights.
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
    leaf_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    leaf_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    leaf_cfg.MODEL.WEIGHTS = base_path + "models/leaf_qr_model.pth" # path to trained weights
    leaf_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    leaf_predictor = DefaultPredictor(leaf_cfg)

    # set up metadata
    leaf_metadata = Metadata()
    leaf_metadata.set(thing_classes = ['leaf', 'qr', 'red-square'])

    return (leaf_predictor, leaf_metadata)


@st.cache()
def run_inference(batch):
    """ Run prediction on batch of images.
    """

    for index, image in enumerate(batch):

        # run interence on selected image
        outputs = leaf_predictor(image["image"])

        # get bboxes and class labels
        pred_boxes = outputs["instances"].pred_boxes.tensor.numpy()
        class_labels = outputs["instances"].pred_classes.numpy()

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

            # crop to bounding box for QR decoding
            crop_img = image["image"][ y0:y1, x0:x1]

            # decode QR code
            qr_result = decode(crop_img, symbols=[ZBarSymbol.QRCODE])

            if len(qr_result):
                qr_result_decoded = qr_result[0].data.decode("utf-8") 

        new_file_name = None

        # if QR was decoded, name results file w/ plant ID
        if qr_result_decoded:
            new_file_name = image['date'] + '_' + qr_result_decoded
        else:
            new_file_name = image['date'] + '_' + image['file_name']

        # rename original file
        if rename_files_option:

            print('rename files option')
            image_dir = image['file_path'].split('/')[:-1]
            new_file_path = '/'.join(image_dir)
            new_file_path = new_file_path + '/' + new_file_name + '.jpg'

            os.rename(image['file_path'], new_file_path)

        # save leaf results to file
        if run_model_option: 
            # set up results visualizer
            v = Visualizer(image["image"][:, :, ::-1],
                metadata=leaf_metadata, 
                scale=0.5, 
                instance_mode=ColorMode.SEGMENTATION
            )

            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_arr = out.get_image()[:, :, ::-1]
            result_image = Image.fromarray(result_arr)

            save_path = Path(base_path + "results/" + new_file_name + "-result.jpg")

            result_image.save(save_path)

#--------------------- STREAMLIT INTERFACE ----------------------#

base_path = setup()
leaf_predictor, leaf_metadata = setup_model(base_path)

st.title('Leaf Segmentation')

st.header('Options')

rename_files_option = st.checkbox('Rename files')
run_model_option = st.checkbox('Run Leaf Segmentation Model')

st.header('Files')

# walk through directory to display files in table
file_names = []
dirs = []

for root, dirs, files in os.walk(base_path + "data"):
    for file in files:
        filename=os.path.join(root, file)
        file_names.append(filename)

# set up AgGrid
df = pd.DataFrame({'File Name' : file_names})
gd = GridOptionsBuilder.from_dataframe(df)
gd.configure_pagination(enabled=True)
gd.configure_selection(selection_mode="multiple", use_checkbox=True)
gd.configure_column("File Name", headerCheckboxSelection = True)

# display AgGrid
file_table = AgGrid(df, fit_columns_on_grid_load=True, gridOptions=gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)

run = st.button('Run')

if run:

    # set up batch
    selected_rows = file_table['selected_rows']
    batch = []
    
    for row in selected_rows:
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
            'file_name': file_name,
            'file_path': file_path,
            'date': date
        })
    
    print(batch)
    
    run_inference(batch)

    


 
            