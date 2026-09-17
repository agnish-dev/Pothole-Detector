import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

import base64

# Set page config
st.set_page_config(page_title="Pothole Detection", page_icon="🛣️", layout="wide")

def set_background(image_file):
    # Get file extension for mime type (e.g. webp, jpg, png)
    ext = image_file.split('.')[-1].lower()
    mime_type = "jpeg" if ext in ["jpg", "jpeg"] else ext

    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        /* Hide the Streamlit Deploy button, header, and footer */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .stApp {{
            background-image: url(data:image/{mime_type};base64,{encoded_string});
            background-size: cover;
            background-position: center top; /* Shifted the photo upwards */
            background-repeat: no-repeat;
        }}
        /* Optional: Add a slight dark overlay so the text is still readable */
        .stApp:before {{
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.6);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if os.path.exists('image.webp'):
    set_background('image.webp')
elif os.path.exists('images.jpg'):
    set_background('images.jpg')

st.title("🛣️ Pothole Detector")
st.write("Upload an image or video to detect potholes using YOLOv8.")

# Load the model
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.warning(f"Custom model not found at '{model_path}'. Falling back to default YOLOv8n.")
        return YOLO('yolov8n.pt')

model = load_model()

def process_frame(img_array):
    results = model(img_array, verbose=False)
    result = results[0]
    img_h, img_w = img_array.shape[:2]
    img_area = img_h * img_w
    
    res_plotted = result.orig_img.copy()
    
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        box_area = (x2 - x1) * (y2 - y1)
        ratio = box_area / img_area
        
        if ratio > 0.05:
            severity = "High"
            color = (255, 0, 0) # Red in RGB
        elif ratio > 0.01:
            severity = "Medium"
            color = (255, 255, 0) # Yellow in RGB
        else:
            severity = "Low"
            color = (0, 255, 0) # Green in RGB
            
        cv2.rectangle(res_plotted, (x1, y1), (x2, y2), color, 3)
        
        # Map class IDs to names! (e.g. Alligator Crack instead of just "Pothole")
        class_id = int(box.cls[0])
        class_name = model.names[class_id] if model.names else "Pothole"
        
        label = f"{class_name}: {conf:.2f} ({severity})"
        cv2.putText(res_plotted, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return res_plotted


# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type in ['jpg', 'jpeg', 'png']:
        # Process Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("Detecting...")
        
        img_array = np.array(image)
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        res_plotted = process_frame(img_array)
        st.image(res_plotted, caption='Detected Output', use_container_width=True)
        
    elif file_type in ['mp4', 'mov', 'avi']:
        # Process Video
        st.write("Processing Video...")
        
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB for processing and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference and draw boxes
            res_plotted = process_frame(frame_rgb)
            
            # Display frame dynamically
            stframe.image(res_plotted, channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("Video processing complete!")

st.markdown('<br><hr><center>Developed by Agnish</center>', unsafe_allow_html=True)
