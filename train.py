from ultralytics import YOLO

def train_model():
    # Load a pre-trained model (recommended for starting)
    model = YOLO('yolov8n.pt')

    # Train the model
    # Note: You need to specify the path to your data.yaml file once you have your dataset.
    # The dataset folder structure and data.yaml are crucial for YOLO.
    print("Starting training...")
    results = model.train(
        data='data.yaml', # Placeholder! You will replace this with your actual data.yaml
        epochs=50,        # Number of epochs to train for
        imgsz=640,        # Image size
        batch=16,         # Batch size
        name='pothole_detection_model', # Directory name for saved weights
        exist_ok=True     # Overwrite the previous aborted run
    )
    print("Training complete! Model saved in runs/detect/pothole_detection_model/weights/best.pt")

if __name__ == '__main__':
    # Make sure to update data.yaml before running this script
    train_model()
