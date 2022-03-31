import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import Metadata
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from pathlib import Path

import requests
import argparse

import os, json, random

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_option_menu import option_menu

from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol

from torchvision import transforms


DEMO_IMAGE = '250-8.JPG'

st.title('Leaf Segmentation')

main_menu = option_menu(None, ["Home", "Results"], 
    icons=['house', 'cloud-upload'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

# sidebar
# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 350px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 350px;
#         margin-left: -350px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.sidebar.title('Leaf Segmentation')
# st.sidebar.subheader('Parameters')

def get_dirs_inside_dir(folder):
    return [my_dir for my_dir in list(map(lambda x:os.path.basename(x), sorted(Path(folder).iterdir(), key=os.path.getmtime, reverse=True))) if os.path.isdir(os.path.join(folder, my_dir))
            and my_dir != '__pycache__' and my_dir != '.ipynb_checkpoints' and my_dir != 'API']

def list_folders_in_folder(folder):
    return [file for file in os.listdir(folder) if os.path.isdir(os.path.join(folder, file))]

def show_dir_tree(folder):
    with st.expander(f"Show {os.path.basename(folder)} folder tree"):
        for line in tree(Path.home() / folder):
            st.write(line)

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

@st.cache()
def run_inference(batch):

    print(batch)

    leaf_cfg = get_cfg()
    leaf_cfg.MODEL.DEVICE='cpu'
    leaf_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    leaf_cfg.DATASETS.TEST = ()
    leaf_cfg.DATALOADER.NUM_WORKERS = 2
    leaf_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    leaf_cfg.SOLVER.IMS_PER_BATCH = 2
    leaf_cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    leaf_cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    leaf_cfg.SOLVER.STEPS = []        # do not decay learning rate
    leaf_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (qrcode). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    leaf_cfg.MODEL.WEIGHTS = "leaf_model.pth"  # path to the model we just trained
    leaf_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

    os.makedirs(leaf_cfg.OUTPUT_DIR, exist_ok=True)

    leaf_predictor = DefaultPredictor(leaf_cfg)

    qr_cfg = get_cfg()
    qr_cfg.MODEL.DEVICE='cpu'
    qr_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    qr_cfg.DATASETS.TEST = ()
    qr_cfg.DATALOADER.NUM_WORKERS = 2
    qr_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    qr_cfg.SOLVER.IMS_PER_BATCH = 2
    qr_cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    qr_cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    qr_cfg.SOLVER.STEPS = []        # do not decay learning rate
    qr_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (qrcode). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    qr_cfg.MODEL.WEIGHTS = "qr_model.pth"  # path to the model we just trained
    qr_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

    os.makedirs(qr_cfg.OUTPUT_DIR, exist_ok=True)

    qr_predictor = DefaultPredictor(qr_cfg)

    leaf_metadata = Metadata()
    leaf_metadata.set(thing_classes = ['leaf'])

    if not os.path.isdir('/results'):
        os.mkdir('/results') 

    for index, image in enumerate(batch):

        leaf_outputs = leaf_predictor(image["image"])
        qr_outputs = qr_predictor(image["image"])
        # pred_boxes = leaf_outputs["instances"].pred_boxes
        # print("Found " + str(len(pred_boxes)) + " leaves")

        pred_boxes = qr_outputs["instances"].pred_boxes

        qr_result_decoded = None

        qr_bbox = pred_boxes.tensor.numpy()

        print(qr_bbox)

        if len(qr_bbox):

            bbox = qr_bbox[0]

            # (x0, y0, x1, y1)

            x0 = round(bbox[0].item())
            y0 = round(bbox[1].item())
            x1 = round(bbox[2].item())
            y1 = round(bbox[3].item())

            crop_img = image["image"][ y0:y1, x0:x1]

            # zbar
            qr_result = decode(crop_img, symbols=[ZBarSymbol.QRCODE])
            print(qr_result[0].data )
            qr_result_decoded = qr_result[0].data.decode("utf-8") 

        v = Visualizer(image["image"][:, :, ::-1],
            metadata=leaf_metadata, 
            scale=0.5, 
            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        out = v.draw_instance_predictions(leaf_outputs["instances"].to("cpu"))
        result_arr = out.get_image()[:, :, ::-1]
        result_image = Image.fromarray(result_arr)

        if qr_result_decoded:
            save_path = Path("/results/" + qr_result_decoded + "-result.jpg")
        else:
            save_path = Path("/results/result.jpg")

        result_image.save(save_path)


if main_menu =='About':
    st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')

elif main_menu =="Results":

    file_names = []
    dirs = []

    for root, dirs, files in os.walk("/results"):
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
   

elif main_menu == 'Upload':
    img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

elif main_menu =='Home':

    file_names = []
    dirs = []

    for root, dirs, files in os.walk("/iplant/home/michellito"):
        for file in files:
                filename=os.path.join(root, file)
                file_names.append(filename)

    df = pd.DataFrame({'File Name' : file_names})

    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_pagination(enabled=True)
    gd.configure_selection(selection_mode="single", use_checkbox=True)
    gd.configure_column("File Name", headerCheckboxSelection = True)

    file_table = AgGrid(df, fit_columns_on_grid_load=True, gridOptions=gd.build(), update_mode=GridUpdateMode.SELECTION_CHANGED)

    # st.sidebar.text('Original Image')
    # st.sidebar.image(image)
    run = st.button('Run')
    
    if run:

        # set up batch
        selected_rows = file_table["selected_rows"]
        print(selected_rows)
        batch = []

        convert_tensor = transforms.ToTensor()
        
        for row in selected_rows:
            image = Image.open(row["File Name"])
            batch.append({"image": np.array(image)})
        
        run_inference(batch)

        # main_menu = "Results"

        # leaf_count = 0

        # if img_file_buffer is not None:
        #     image = np.array(Image.open(img_file_buffer))

        # else:
        #     demo_image = DEMO_IMAGE
        #     image = np.array(Image.open(demo_image))
    


 
            