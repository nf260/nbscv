import streamlit as st
import numpy as np
import cv2
from joblib import load

from functions import (
    bs_detect,
    spot_metrics,
    calc_multispot_prob,
    draw_bs_contours,
    draw_bounding_box
)

st.title("DBS Vision - Single image analysis")

# === Parameter Section 1: ROI ===
st.header("1. Parameters")
st.write("Leave default values, unless you know what you are doing")
st.subheader("Co-ordinates of outer rectange")
st.write("The entire DBS must fall within this region")
col1, col2 = st.columns(2)
with col1:
    x_min = st.number_input("x_min", value=1)
    x_max = st.number_input("x_max", value=459)
with col2:
    y_min = st.number_input("y_min", value=50)
    y_max = st.number_input("y_max", value=299)

# === Parameter Section 2: Search Circle ===
st.subheader("Search Area (Circle)")
st.write("The circular punching region of the puncher. At least part of the DBS must fall within this region")
col3, col4, col5 = st.columns(3)
with col3:
    center_x = st.number_input("Center X", value=209)
with col4:
    center_y = st.number_input("Center Y", value=139)
with col5:
    radius = st.number_input("Radius", value=68)

center = (center_x, center_y)

# === Parameter Section 3: Scaling ===
st.subheader("Image Scaling")
st.write("Conversion between pixels and mm")
mm_per_pixel = st.number_input("mm per pixel", value=0.1161)

# === Upload Image Section ===
st.header("2. Upload Image")
st.write("This should be an original image from the Panthera puncher, with image size 752 x 480 pixels")
uploaded_file = st.file_uploader("Upload a .jpg/.jpeg/.png image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = img[0:300, 150:610]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load models
    scaler = load('log_model_scaler_220828.joblib')
    log_model = load('log_model_final_220828.joblib')

    # Define columns
    cols = ['contour_index','area','perimeter_mm','roundness','equiv_diam_mm', 
            'long_mm','short_mm', 'elongation', 'circular_extent',
            'hull_area', 'solidity', 'hull_perimeter', 'convexity', 
            'number_punches', 'average_punch_area']
    ml_cols = ['roundness','elongation','circular_extent','solidity','convexity']

    # Run processing pipeline
    contours, hierarchy = bs_detect(img, x_min, x_max, y_min, y_max, select_punched=True)
    spot_met = spot_metrics(contours, hierarchy, mm_per_pixel, center, radius, select_punched=True)
    prob_ms_list = calc_multispot_prob(spot_met, cols, ml_cols, scaler, log_model)
    contour_img = draw_bs_contours(img_rgb.copy(), contours, hierarchy, center, radius, select_punched=True)
    bounding_box_img = draw_bounding_box(
        contour_img.copy(),
        contours,
        hierarchy,
        center,
        radius,
        spot_met,
        multispot_prob_list=prob_ms_list,
        select_punched=True,
        diam_range=(8, 16),
        prob_multi_limit=0.50
    )

    st.header("Bounding box image")
    st.image(bounding_box_img, channels="RGB", use_container_width=True)

else:
    st.info("Awaiting image upload...")
