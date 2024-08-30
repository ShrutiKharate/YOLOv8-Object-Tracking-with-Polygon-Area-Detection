# YOLOv8 Object Tracking with Polygon Area Detection

## Overview
This project demonstrates real-time object tracking in videos using the YOLOv8 model. The primary goal is to detect and track objects within a specified polygonal area in a video. The system highlights detected objects and their tracking IDs, providing a visual representation of object movements and interactions within the defined area.

## Features

~ Real-Time Object Tracking: Uses the YOLOv8 model to track objects in video frames.

~ Polygon Area Detection: Highlights objects that enter a predefined polygonal region.

~ Visual Annotations: Draws bounding boxes around detected objects, marks specific points, and displays tracking IDs and class names.

~ Mouse Interaction: Captures and prints BGR coordinates of the mouse position in the video frame.

### Technologies Used

~ YOLOv8 (Ultralytics): The object detection model used for identifying and tracking objects within video frames.

~ OpenCV: A computer vision library used for video capture, frame processing, and visual annotations.

~ NumPy: A library for handling arrays and performing mathematical operations.

### How It Works

~ Video Capture: The video is read frame by frame using OpenCV.

~ Object Tracking: YOLOv8 performs object detection and tracking on each frame.

~ Polygon Area Check: Objects within a specified polygonal area are highlighted with visual annotations.

~ Visual Output: The processed video frames are displayed in real-time with bounding boxes, tracking IDs, and additional annotations.
