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


# get base data path from user input
base_path = sys.argv[1]

# if results path doesn't exist, create it
if not os.path.isdir(base_path + 'results'):
    os.mkdir(base_path + 'results') 


# --------------- SETUP MODEL WITH TRAINED WEIGHTS ----------------#
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

# --------------- START RUN INFERENCE FUNCTION ----------------#
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

#--------------------- STREAMLIT INTERFACE ----------------------#

st.title('Leaf Segmentation')
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
gd.configure_selection(selection_mode="single", use_checkbox=True)
gd.configure_column("File Name", headerCheckboxSelection = True)

# display AgGrid
file_table = AgGrid(df, fit_columns_on_grid_load=True, gridOptions=gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)
st.header('Models')

models = []
for root, dirs, files in os.walk(base_path + "models"):
    for model in files:
        model = os.path.join(root, file)
        models.append(model)

df2 = pd.DataFrame({'Models' : models})
gdm = GridOptionsBuilder.from_dataframe(df2)
gdm.configure_pagination(enabled=True)
gdm.configure_selection(selection_mode="single", use_checkbox=True)
gdm.configure_column("Models", headerCheckboxSelection = True)
model_table = AgGrid(df2,fit_columns_on_grid_load=True, gridOptions=gdm.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)



path = "/cyverse/data"
dir = []
file_name = []

os.walk(path)
file_name = [x[0] for x in os.walk(path)]

option = st.selectbox('Select Directory', file_name, index = len(file_name) - 1)

st.write('Files in: ', option)

file_names = []

file_names = (file for file in os.listdir(option) 
         if os.path.isfile(os.path.join(option, file)))

df3 = pd.DataFrame({'File_Name' : file_names})
gds = GridOptionsBuilder.from_dataframe(df3)
gds.configure_selection(selection_mode="single", use_checkbox=True)
AgGrid(df3, fit_columns_on_grid_load=True, gridOptions = gds.build(), update_mode = GridUpdateMode.SELECTION_CHANGED)

  
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




 
            