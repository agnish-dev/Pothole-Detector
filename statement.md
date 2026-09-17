# Problem Statement
Road infrastructure degradation, particularly the formation of potholes and cracks, poses a severe risk to road safety, vehicle maintenance, and driver comfort. Manual inspection of vast road networks is highly inefficient, time-consuming, and prone to human error. There is a critical need for an automated, highly accurate system that can detect road damage in real-time to facilitate rapid repairs and prevent accidents.

# Scope of the Project
This project involves the development of an automated computer vision web application using a custom-trained YOLOv8 deep learning model. The system processes both static images and video feeds to instantly detect various types of road damage (Longitudinal Cracks, Transverse Cracks, Alligator Cracks, and Potholes). Furthermore, it estimates the severity of the damage based on the bounding box area relative to the total frame, allowing authorities to prioritize repairs.

# Target Users
* **Municipalities & City Councils:** To monitor road health and prioritize maintenance budgets efficiently.
* **Highway Maintenance Agencies:** For automated scanning of long highway stretches using dashboard cameras.
* **Everyday Drivers / Navigation Apps:** Can be integrated as an API to warn drivers of upcoming severe road damage.

# High-level Features
1. **Multi-Class Damage Detection:** Accurately classifies 4 different types of road degradation.
2. **Video & Image Support:** Processes both static image uploads and video file uploads frame-by-frame.
3. **Severity Estimation:** Automatically calculates the severity (Low, Medium, High) of detected potholes based on spatial ratios.
4. **Interactive Web Interface:** A user-friendly Streamlit web application that requires no technical expertise to operate.
5. **Real-time Processing:** Fast inference using the state-of-the-art YOLOv8 nano architecture.
