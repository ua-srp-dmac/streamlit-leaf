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

import sys

st.title('Leaf Segmentation')
st.header('Files')

# get base data path from user input
base_path = sys.argv[1]

leaf_cfg = get_cfg()
leaf_cfg.MODEL.DEVICE='cpu'
leaf_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
leaf_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
leaf_cfg.MODEL.WEIGHTS = base_path + "models/leaf_qr_model.pth" # path to the model we just trained
leaf_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

leaf_predictor = DefaultPredictor(leaf_cfg)

leaf_metadata = Metadata()
leaf_metadata.set(thing_classes = ['leaf', 'qr', 'red-square'])

if not os.path.isdir(base_path + 'results'):
    os.mkdir(base_path + 'results') 


@st.cache()
def run_inference(batch):

    for index, image in enumerate(batch):

        # run interence on selected image
        outputs = leaf_predictor(image["image"])

        # get bboxes and class labels
        pred_boxes = outputs["instances"].pred_boxes.tensor.numpy()
        class_labels = outputs["instances"].pred_classes.numpy()

        qr_indices = []
        leaf_indices = []

        print(class_labels)
        
        # get indices of leaves and qr codes
        for i, label in enumerate(class_labels):
            print(label)
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
            print(qr_result[0].data)
            qr_result_decoded = qr_result[0].data.decode("utf-8") 

        # set up results visualizer
        v = Visualizer(image["image"][:, :, ::-1],
            metadata=leaf_metadata, 
            scale=0.5, 
            instance_mode=ColorMode.SEGMENTATION
        )

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_arr = out.get_image()[:, :, ::-1]
        result_image = Image.fromarray(result_arr)

        # if QR was decoded, name results file w/ plant ID
        if qr_result_decoded:
            save_path = Path(base_path + "results/" + image['date'] + '_' + qr_result_decoded + "-result.jpg")
        else:
            save_path = Path(base_path + "results/" + image['date'] + '_' + image['file_name'] + "-result.jpg")

        result_image.save(save_path)
    

file_names = []
dirs = []

for root, dirs, files in os.walk(base_path + "data"):
    for file in files:
            filename=os.path.join(root, file)
            file_names.append(filename)

df = pd.DataFrame({'File Name' : file_names})

gd = GridOptionsBuilder.from_dataframe(df)
gd.configure_pagination(enabled=True)
gd.configure_selection(selection_mode="single", use_checkbox=True)
gd.configure_column("File Name", headerCheckboxSelection = True)

file_table = AgGrid(df, fit_columns_on_grid_load=True, gridOptions=gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)

run = st.button('Run')

if run:

    # set up batch
    selected_rows = file_table["selected_rows"]
    batch = []
    
    for row in selected_rows:
        file_path = row["File Name"]
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

        batch.append({"image": np.array(image), "file_name": file_name, "date": date })
    
    run_inference(batch)

    


 
            