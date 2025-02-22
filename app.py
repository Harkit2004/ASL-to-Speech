import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to handle the results
def print_result(result, output_image, timestamp_ms):
    if result.gestures:
        gesture = result.gestures[0][0]
        print(f'Detected gesture: {gesture.category_name}, Score: {gesture.score:.2f}')

# Initialize frame counter and timestamp
frame_timestamp_ms = 0

# Create a gesture recognizer instance
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    # Use OpenCV's VideoCapture to start capturing from the webcam
    cap = cv2.VideoCapture(0)  # Changed from -1 to 0 for default webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                          data=rgb_frame)
        
        # Increment timestamp for each frame
        frame_timestamp_ms += 1
        
        # Process the frame
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        
        # Display the frame
        cv2.imshow('Gesture Recognition', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()