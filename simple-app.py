import streamlit as st
from streamlit_option_menu import option_menu

st.title('Leaf Segmentation')

main_menu = option_menu(None, ["Home", "Results"], 
    icons=['house', 'cloud-upload'], 
    menu_icon="cast", default_index=0, orientation="horizontal")


if main_menu =='Home':
    st.markdown('In this application we are using **Detectron2** for leaf segmentation. **StreamLit** is to create the Web Graphical User Interface (GUI) ')

elif main_menu == 'Results':
    st.markdown('Results page')

    



 
            