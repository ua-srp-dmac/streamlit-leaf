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

from torchvision import transforms


DEMO_IMAGE = '250-8.JPG'

st.title('Leaf Segmentation')

main_menu = option_menu(None, ["Home", "Results", "Upload", 'About'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
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

    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (qrcode). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.WEIGHTS = "leaf_model.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)
    outputs = []

    for image in batch:
        output = predictor(image["image"])
        print(output["instances"])
        outputs.append(output)


    # test image
    # outputs = predictor(batch)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    # print(outputs["instances"].pred_boxes)
    # pred_boxes = outputs["instances"].pred_boxes

    # print("Found " + str(len(pred_boxes)) + " leaves")

    # leaf_count = len(pred_boxes)
    # kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{leaf_count}</h1>", unsafe_allow_html=True)

    # leaf_metadata = Metadata()
    # leaf_metadata.set(thing_classes = ['leaf'])

    # v = Visualizer(image[:, :, ::-1],
    #     metadata=leaf_metadata, 
    #     scale=0.5, 
    #     instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    # )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # st.subheader('Output Image')
    # st.image(out.get_image()[:, :, ::-1], use_column_width= True)


if main_menu =='About':
    st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')

elif main_menu =="Results":
    st.markdown("**Detected Leaves**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

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
    gd.configure_selection(selection_mode="multiple", use_checkbox=True)
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

        # leaf_count = 0

        # if img_file_buffer is not None:
        #     image = np.array(Image.open(img_file_buffer))

        # else:
        #     demo_image = DEMO_IMAGE
        #     image = np.array(Image.open(demo_image))
    


 
            