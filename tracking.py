import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Define a callback function to capture mouse movement and print BGR coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

# Create a window named 'RGB' to display the video
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Capture the video from the file
cap = cv2.VideoCapture(r'people.mp4')

# Define the polygon area to track objects within
area = [(267, 396), (337, 447), (944, 359), (837, 331)]

# Check if the video file is successfully opened
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    # Loop to process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break  
    
        # Resize the frame for processing
        frame = cv2.resize(frame, (1020, 500))
    
        # Perform object tracking using the YOLO model
        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")
    
        # Check if any boxes (detections) exist in the results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Extract class IDs, bounding boxes, and tracking IDs from the results
            clss = results[0].boxes.cls
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Loop through detected objects
            for box, track_id, cls in zip(boxes, track_ids, clss):
                px1, py1, px2, py2 = box
                x1, y1, x2, y2 = int(px1), int(py1), int(px2), int(py2)

                # Check if the bottom-left corner of the bounding box is within the polygon area
                result = cv2.pointPolygonTest(np.array(area, np.int32), ((x1, y2)), False)
                if result >= 0:
                    # Draw a red rectangle around the detected object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Mark the bottom-left corner with a blue circle
                    cv2.circle(frame, (x1, y2), 4, (255, 0, 0), -1)
                    # Display the tracking ID and class name
                    cv2.putText(frame, f'{track_id} {results[0].names[0]}', 
                                (int(px1), int(py1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.3, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            print("No detections or tracking IDs in the current frame.")
        
        # Draw the polygon area on the frame
        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)
        # Display the frame in the 'RGB' window
        cv2.imshow("RGB", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
    