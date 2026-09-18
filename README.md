# 🛣️ Pothole Detector

Welcome to the **Pothole Detector**! I built this AI-powered computer vision web application to help automatically identify and classify road damage. Poor road conditions are a huge issue for vehicle safety, and manual road inspections are incredibly slow. This project solves that by instantly finding potholes and cracks using both images and video feeds.

I custom-trained a YOLOv8 deep learning model from scratch on a massive 10GB dataset containing over 32,000 images of global roads. Not only does the app find the damage, but it also calculates how severe it is (Low, Medium, or High) so that road maintenance crews can prioritize what needs fixing first!

## ✨ Key Features
- **Smart Object Detection:** Accurately detects 4 different types of road damage: Potholes, Longitudinal Cracks, Transverse Cracks, and Alligator Cracks.
- **Image & Video Processing:** You can upload a static photo of a road or a full dashboard-camera video, and the app handles both seamlessly.
- **Severity Estimation:** The backend runs custom spatial logic to determine the severity of the pothole relative to the frame size.
- **Clean Interface:** Built with Streamlit for a fast, responsive, and easy-to-use web interface.

## 🛠️ Tech Stack
- **Python** for all core logic.
- **Ultralytics YOLOv8** for the heavy lifting (neural network inference).
- **OpenCV** to extract video frames and draw bounding boxes.
- **Streamlit** to host the web application.

## 🚀 How to Run the Project Locally

If you want to run this project on your own machine, everything can be executed directly from your terminal. Just follow these steps!

**1. Clone the repository to your computer:**
```bash
git clone https://github.com/agnish-dev/Pothole-Detector.git
cd Pothole-Detector
```

**2. Create a clean virtual environment (highly recommended):**
```bash
python -m venv venv
# On Windows use: venv\Scripts\activate
# On Mac/Linux use: source venv/bin/activate
```

**3. Install all the necessary dependencies:**
```bash
pip install -r requirements.txt
```

**4. Start the app!**
```bash
streamlit run app.py
```
*Your browser will automatically open a new tab at `http://localhost:8501` where you can start testing out the detector!*

## 🧪 Testing
I've included a simple unit testing script to make sure your environment is set up correctly and the model weights loaded properly before running. You can run the tests directly in your terminal:
```bash
python -m unittest test_app.py
```

## 📸 Screenshots

![Pothole Detector Web App](app_screenshot_v2.png)
