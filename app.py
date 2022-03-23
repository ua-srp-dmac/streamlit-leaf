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

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = '250-8.JPG'

st.title('Leaf Segmentation')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Leaf Segmentation')
st.sidebar.subheader('Parameters')

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

# App modes
app_mode = st.sidebar.selectbox(
    'Choose the App mode', [
        'About App',
        'Leaf Segmentation',
        'QR Code'
    ]
)

if app_mode =='About App':
    st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('''
          # About Me \n 
            Hey this is ** Ritesh Kanjee ** from **Augmented Startups**. \n
           
            If you are interested in building more Computer Vision apps like this one then visit the **Vision Store** at
            www.augmentedstartups.info/visionstore \n
            
            Also check us out on Social Media
            - [YouTube](https://augmentedstartups.info/YouTube)
            - [LinkedIn](https://augmentedstartups.info/LinkedIn)
            - [Facebook](https://augmentedstartups.info/Facebook)
            - [Discord](https://augmentedstartups.info/Discord)
        
            If you are feeling generous you can buy me a **cup of  coffee ** from [HERE](https://augmentedstartups.info/ByMeACoffee)
             
            ''')


elif app_mode =='Leaf Segmentation':

    
    st.sidebar.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Detected Leaves**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    # max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    # st.sidebar.markdown('---')
    # detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    # st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    # st.sidebar.text('Original Image')
    # st.sidebar.image(image)
    run = st.sidebar.button('Run')
    

    if run:

        leaf_count = 0

        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))

        else:
            demo_image = DEMO_IMAGE
            image = np.array(Image.open(demo_image))
    
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

        # test image
        outputs = predictor(image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        pred_boxes = outputs["instances"].pred_boxes

        print("Found " + str(len(pred_boxes)) + " leaves")

        leaf_count = len(pred_boxes)
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{leaf_count}</h1>", unsafe_allow_html=True)

        leaf_metadata = Metadata()
        leaf_metadata.set(thing_classes = ['leaf'])

        v = Visualizer(image[:, :, ::-1],
            metadata=leaf_metadata, 
            scale=0.5, 
            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.subheader('Output Image')
        st.image(out.get_image()[:, :, ::-1], use_column_width= True)


elif app_mode =='QR Code':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Detected Leaves**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    leaf_count = 0
    
    
    # Dashboard
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            leaf_count += 1

            #print('face_landmarks:', face_landmarks)

            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{leaf_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image,use_column_width= True)
# Watch Tutorial at www.augmentedstartups.info/YouTube