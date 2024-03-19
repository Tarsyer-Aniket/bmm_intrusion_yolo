import streamlit as st
from PIL import Image
import cv2
from streamlit_cropper import st_cropper
import numpy as np
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import time
import pickle
import math
from tarsyer import helper_function_ui as hfu
import os
st.set_option('deprecation.showfileUploaderEncoding', False)


SAVE_DIR = 'ROI/'
os.system(f'mkdir -p {SAVE_DIR}')
CONFIG_PATH = 'tarsyer-config.pkl'

# Upload an image and set some options for demo purposes
st.header("Retail - ROI Generator")

img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
camera_detail = st.sidebar.text_input('camera_detail', value='camera_detail')
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)

# print(img_file)

drawing_mode = st.sidebar.selectbox("Drawing tool:", ("line", "point",)) # "circle", "transform"))

if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 12)

box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker(label="Stroke color hex: ", value='#FF0012')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]



points = None
json_data = None
model_width, model_height = 300, 300

if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    st.write(img.width, img.height)

    cropped_coordinates = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio, return_type="box")

    roi_x1, roi_y1, roi_x2, roi_y2 = cropped_coordinates["left"], cropped_coordinates["top"], cropped_coordinates["left"]+cropped_coordinates["width"], cropped_coordinates["top"]+cropped_coordinates["height"]
    st.write(cropped_coordinates)
    if abs(cropped_coordinates["width"]-cropped_coordinates["height"])> 3:
        st.error("ROI not square")
    elif roi_x2 > img.width or roi_y2 > img.height:
        st.error("ROI out of image")
    else:
        img_numpy = np.array(img.convert('RGB'))
        cropped_img_numpy = img_numpy[roi_y1:roi_y2, roi_x1:roi_x2]
        # cv2.imwrite(f'{SAVE_DIR}test.jpg', cv2.cvtColor(cropped_img_numpy, cv2.COLOR_RGB2BGR))
        

        cropped_img = Image.fromarray(cropped_img_numpy)
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=cropped_img,
            update_streamlit=realtime_update,
            height=model_height,
            width=model_width,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )

        # Do something interesting with the image data and paths
        # if canvas_result.image_data is not None:
        #     st.image(canvas_result.image_data)
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
            
            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype("str")
            json_data = canvas_result.json_data["objects"]

            
            st.dataframe(objects)

        if json_data is not None:
            if len(json_data) == 4: #len(json_data) == 4:
                # st.info('4 entry found')
                ### validations
                if json_data[0]['type'] == 'line' and json_data[1]['type'] == 'circle' and json_data[2]['type'] == 'circle' and json_data[3]['type'] == 'circle':
                    # st.info('pass 1')
                    if json_data[1]['width'] == 24 and json_data[2]['width'] == 24 and json_data[1]['height'] == 24 and json_data[2]['height'] == 24 and json_data[3]['width'] == json_data[3]['height']:
                        # st.info('pass 2')
                        points = []
                        for dict_ in json_data:
                            if dict_['type'] == 'circle' and dict_['originX'] == 'left' and dict_['originY'] == 'center':
                                x1_ = int(dict_['left']+dict_['width']/2)
                                y1_ = dict_['top']
                                # x1_ = dict_['left']
                                # y1_ = dict_['top']

                                # angle_in_degree = 90 - dict_['angle']
                                # if angle_in_degree < 0:
                                #     angle_in_degree = 360 - angle_in_degree
                                # radius = dict_['width']
                                # cx1, cy1 = find_radius_endpoint(x1_, y1_, radius, angle_in_degree)

                                # points.append((cx1, cy1))
                                points.append((x1_, y1_))
                    else:
                        st.error('2 point not found')
                else:
                    st.error('1 line 2 point 1 circle not found')

        if points is not None:
            if len(points) == 3:
                # st.info(f'ROI points = {points}')
                roi_dict = {}
                roi_dict['filename'] = img_file.name
                roi_dict['camera_detail'] = camera_detail
                roi_dict['crop_coord'] = (roi_x1, roi_y1, roi_x2, roi_y2)
                roi_dict['model_dim'] = (model_height, model_width)
                roi_dict['image_size'] = (img.width, img.height)
                roi_dict['line'] = points[:2]
                roi_dict['entry_point'] = points[2]

                # print(roi_dict)
                # st.info(roi_dict)
                side = hfu.point_side_of_line(roi_dict['line'], roi_dict['entry_point'])
                st.info(side)
                st.info(roi_dict)

                if st.button('Save'):
                    with open(CONFIG_PATH, 'wb') as file_:
                        pickle.dump(roi_dict, file_)

                    
                    img_numpy = np.array(img)
                    LINE = roi_dict['line']
                    ENTRY_POINT = roi_dict['entry_point']
                    cropped_img_numpy_300 = cv2.resize(cropped_img_numpy, (300, 300))
                    cv2.line(cropped_img_numpy_300, LINE[0], LINE[1], (0, 0, 255), 2)
                    cv2.line(cropped_img_numpy_300, (ENTRY_POINT[0]-10, ENTRY_POINT[1]), (ENTRY_POINT[0]+10, ENTRY_POINT[1]), (255, 0, 0), 2)
                    cv2.line(cropped_img_numpy_300, (ENTRY_POINT[0], ENTRY_POINT[1]-10), (ENTRY_POINT[0], ENTRY_POINT[1]+10), (255, 0, 0), 2)

                    cv2.imwrite(f'{SAVE_DIR}02_crop_image_{img_file.name}', cv2.cvtColor(cropped_img_numpy_300, cv2.COLOR_RGB2BGR))

                    cv2.rectangle(img_numpy, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
                    img_numpy[roi_y1:roi_y1+300 , roi_x1:roi_x1+300] = cropped_img_numpy_300

                    cv2.imwrite(f'{SAVE_DIR}01_full_image_{img_file.name}', cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR))


                    # intersect_output = intersects(points[:2], points[3:])
                    # st.info(intersect_output)
                    # status = check_crossed_line_direction(points[:2], points[3:])
                    # st.info(status)
        
