# OpenCV Learning & Computer Vision Projects

Welcome to the **OpenCV Learning** repository! This repository contains a collection of scripts, Jupyter Notebooks, and mini-projects that I created while learning computer vision using OpenCV and Python. It covers basics like image processing and object detection, as well as more advanced topics like real-time hand tracking and gesture control.

## 📁 Repository Structure

The repository is organized into three main directories:

### 1. `Learning/`
This folder contains Jupyter Notebooks that cover foundational OpenCV concepts. It's a great place to start if you are new to computer vision.
*   **`intro_to_opencv.ipynb`**: Basics of reading, displaying, and saving images.
*   **`Image_Processing.ipynb` & `Image_filtering.ipynb`**: Applying filters, blurring, and basic transformations to images.
*   **`drawing_shapes.ipynb`**: Drawing lines, rectangles, circles, and putting text on images.
*   **`Contours.ipynb`**: Finding and drawing contours (outlines) of objects in images.
*   **`Face_detection.ipynb`**: Detecting faces in images/videos using Haar Cascades.
*   **`VideoCapturing.ipynb`**: Capturing and displaying video from a webcam.
*   *Includes sample images used in the notebooks (e.g., `cute_puppy.jpg`, `flower.jpg`, `shape.png`, etc.).*

### 2. `HandTrackingProjec/`
This directory contains real-time interactive applications built using OpenCV, MediaPipe (for hand landmark detection), and PyAutoGUI (for system control).
*   **`MouseControl.py`**: A virtual mouse that uses your index finger to move the cursor and tapping gestures to simulate clicks.
*   **`VolumeHandControl.py`**: Controls the system volume based on the distance between your thumb and index finger.

### 3. `Harcascades/`
Contains pre-trained XML cascade classifiers provided by OpenCV, used for object detection in the notebooks.
*   `haarcascade_frontalface_default.xml` (Face detection)
*   `haarcascade_eye.xml` (Eye detection)
*   `haarcascade_smile.xml` (Smile detection)

## 🛠️ Prerequisites

To run the code in this repository, you need to have Python installed along with the following libraries:

```bash
pip install opencv-python numpy mediapipe pyautogui jupyter
```

## 🚀 How to Run

### Jupyter Notebooks (Learning Basics)
1. Navigate to the `Learning` directory.
2. Start Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook
   ```
3. Open any `.ipynb` file and run the cells.

### Hand Tracking Projects
1. Navigate to the `HandTrackingProjec` directory.
2. Run the desired script from your terminal:
   ```bash
   python MouseControl.py
   # OR
   python VolumeHandControl.py
   ```
3. A window will pop up showing your webcam feed. 
   - **For Mouse Control**: Use your index finger to move the mouse. Bring your thumb close to your index finger to click. Press `q` to quit.
   - **For Volume Control**: Pinch your index finger and thumb together or apart to decrease/increase the system volume. Press `q` to quit.

## 🤝 Contributing
Feel free to fork this repository, add your own OpenCV experiments, and submit pull requests. If you find any bugs or have suggestions for improvements, please open an issue.

## 📝 License
This project is open-source and available for educational purposes.
