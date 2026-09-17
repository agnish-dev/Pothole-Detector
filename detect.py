import cv2
from ultralytics import YOLO

def detect_potholes(image_path, model_path='runs/detect/pothole_detection_model/weights/best.pt'):
    # Load the trained model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}. Please train the model first.")
        print(f"Exception details: {e}")
        return

    # Run inference on an image
    print(f"Running inference on {image_path}...")
    results = model(image_path)
    
    # Process results
    for result in results:
        img = result.orig_img.copy()
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w
        
        for box in result.boxes:
            # Extract coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Severity Estimation (Pipeline step 6)
            # Calculate pothole area relative to image
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / img_area
            
            if ratio > 0.05: # > 5% of image
                severity = "High"
                color = (0, 0, 255) # Red
            elif ratio > 0.01: # > 1% of image
                severity = "Medium"
                color = (0, 255, 255) # Yellow
            else:
                severity = "Low"
                color = (0, 255, 0) # Green
                
            # Draw custom bounding boxes with severity and confidence
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"Pothole: {conf:.2f} ({severity} Severity)"
            cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the image
        cv2.imshow("Pothole Detection", img)
        
        print("Press any key to close the window...")
        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage:
    # Replace 'test_image.jpg' with a real image path once you have one.
    print("Please modify the script with your test image path to run it.")
    # detect_potholes('test_image.jpg')
