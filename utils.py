import os
import cv2

def check_model_exists(model_path='best.pt'):
    """
    Utility function to verify the YOLOv8 model weights exist 
    before starting the Streamlit application.
    """
    return os.path.exists(model_path)

def get_video_info(video_path):
    """
    Utility function to extract metadata from uploaded videos.
    Returns a tuple of (total_frames, fps, width, height).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return total_frames, fps, width, height

def calculate_severity(box_area, img_area):
    """
    Business logic to determine pothole severity.
    Separated here for clean modular architecture.
    """
    ratio = box_area / img_area
    if ratio > 0.05:
        return "High", (255, 0, 0)
    elif ratio > 0.01:
        return "Medium", (255, 255, 0)
    else:
        return "Low", (0, 255, 0)
