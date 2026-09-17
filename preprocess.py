import os
import xml.etree.ElementTree as ET
import glob
import shutil

# --- Configuration ---
# Update this path once you extract the dataset
RAW_DATASET_DIR = "dataset/raw"  
OUTPUT_DIR = "dataset/yolo_format"

# RDD2022 uses these codes for damage types:
# D00: Longitudinal Crack
# D10: Transverse Crack
# D20: Alligator Crack
# D40: Pothole
# Since your project is "Pothole Detection", we can filter for just D40.
TARGET_CLASSES = {
    "D40": 0  # We map D40 (Pothole) to YOLO class ID 0
}

def convert_to_yolo_format(size, box):
    """
    Converts bounding box from Pascal VOC format (xmin, ymin, xmax, ymax)
    to YOLO format (x_center, y_center, width, height) normalized to 0-1.
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    
    x_center = x_center * dw
    width = width * dw
    y_center = y_center * dh
    height = height * dh
    
    return (x_center, y_center, width, height)

def preprocess_dataset():
    print("Starting Preprocessing and YOLO Annotation Conversion...")
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels/train"), exist_ok=True)
    
    xml_files = glob.glob(os.path.join(RAW_DATASET_DIR, "**/*.xml"), recursive=True)
    
    if not xml_files:
        print(f"No XML files found in {RAW_DATASET_DIR}. Please check your extracted dataset folder.")
        return

    processed_count = 0

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        if size is None:
            continue
            
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        if w == 0 or h == 0:
            continue

        yolo_annotations = []
        has_pothole = False
        
        # Extract bounding boxes
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            
            # If it's a pothole (D40)
            if cls_name in TARGET_CLASSES:
                has_pothole = True
                cls_id = TARGET_CLASSES[cls_name]
                xmlbox = obj.find('bndbox')
                
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                yolo_box = convert_to_yolo_format((w, h), b)
                yolo_annotations.append(f"{cls_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")

        # Only process images that actually contain potholes to save training time
        if has_pothole:
            # 1. Save the new YOLO .txt file
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            txt_path = os.path.join(OUTPUT_DIR, "labels/train", f"{base_name}.txt")
            
            with open(txt_path, 'w') as out_file:
                out_file.write('\n'.join(yolo_annotations))
                
            # 2. Copy the corresponding image to the output folder
            # Assuming image is in the same folder as xml, or replace .xml with .jpg
            image_path = xml_file.replace('.xml', '.jpg')
            if os.path.exists(image_path):
                dest_image_path = os.path.join(OUTPUT_DIR, "images/train", f"{base_name}.jpg")
                shutil.copy(image_path, dest_image_path)
                processed_count += 1
                
    print(f"Pipeline Step Complete! Converted {processed_count} images to YOLO format.")
    print(f"Your ready-to-train dataset is located at: {OUTPUT_DIR}")

if __name__ == '__main__':
    preprocess_dataset()
