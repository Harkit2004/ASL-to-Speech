import mediapipe as mp
import numpy as np
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
import io
from dotenv import load_dotenv
import os

load_dotenv()

TTS_API_KEY = os.getenv('TTS_API_KEY')
RAPID_API_KEY = os.getenv('RAPID_API_KEY')

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables to track text and current state
current_text = ""
current_gesture = ""
gesture_start_time = 0
GESTURE_HOLD_TIME = 1.2  # Time in seconds to hold gesture before adding to text

# Callback function to handle the results
def process_result(result, output_image, timestamp_ms):
    global current_text, current_gesture, gesture_start_time, button_active
    
    if result.gestures and result.gestures[0]:
        gesture = result.gestures[0][0]
        category = gesture.category_name
        score = gesture.score
        
        if category:
            # If this is a new gesture, reset the timer
            if category != current_gesture:
                current_gesture = category
                gesture_start_time = time.time()
                
            # Check if gesture has been held long enough
            elapsed_time = time.time() - gesture_start_time
            
            # If gesture held long enough, add it to the text
            if elapsed_time >= GESTURE_HOLD_TIME:
                # Handle special cases
                if category.lower() == "del" and current_text:
                    current_text = current_text[:-1]  # Remove last character
                elif category.lower() == "space":
                    current_text += " "  # Add space
                elif len(category) == 1:  # Single letter gestures
                    current_text += category
                
                # Reset timer after adding the character
                gesture_start_time = time.time()
        else:
            # Empty category - reset
            current_gesture = ""
            gesture_start_time = time.time()
    else:
        # No gestures detected - reset
        current_gesture = ""
        gesture_start_time = time.time()
        button_active = False

# Function to draw the text area and button
def draw_text_and_button(frame):
    frame_height, frame_width = frame.shape[:2]
    
    # Draw text area background
    text_area_height = 100
    cv2.rectangle(frame, (0, 0), (frame_width, text_area_height), (0, 0, 0), -1)
    
    # Display entered text
    text_display = current_text
    if not text_display:
        text_display = "Start typing with gestures..."
    
    # Calculate text size for centering
    text_size = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = text_area_height // 2 + text_size[1] // 2
    
    cv2.putText(frame, text_display, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Display current gesture info and progress
    if current_gesture:
        # Calculate progress based on elapsed time
        elapsed_time = time.time() - gesture_start_time
        progress = min(elapsed_time / GESTURE_HOLD_TIME * 100, 100)
        
        gesture_text = f"Current gesture: {current_gesture} (Progress: {progress:.0f}%)"
        cv2.putText(frame, gesture_text, (10, text_area_height + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw progress bar
        bar_width = int(frame_width * 0.8)
        bar_height = 10
        bar_x = (frame_width - bar_width) // 2
        bar_y = text_area_height + 120
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (70, 70, 70), -1)
        
        # Progress fill
        progress_width = int(bar_width * (progress / 100))
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + progress_width, bar_y + bar_height), 
                     (0, 255, 0), -1)

# Initialize timestamp
frame_timestamp_ms = 0

# Create a gesture recognizer instance
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    # Use OpenCV's VideoCapture to start capturing from the webcam
    cap = cv2.VideoCapture(0)  # Default webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
            
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                          data=rgb_frame)
        
        # Increment timestamp for each frame
        frame_timestamp_ms += 1
        
        # Process the frame
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        
        # Draw the text area and button
        draw_text_and_button(frame)
        
        # Display the frame
        cv2.imshow('Gesture Input', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    url = f"https://voicerss-text-to-speech.p.rapidapi.com/?key={TTS_API_KEY}"

    payload = {
        "src": current_text,
        "hl": "en-us",
        "r": "0",
        "c": "mp3",
        "f": "8khz_8bit_mono"
    }
    headers = {
        "x-rapidapi-key": RAPID_API_KEY,
        "x-rapidapi-host": "voicerss-text-to-speech.p.rapidapi.com",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(url, data=payload, headers=headers)

    audio_stream = io.BytesIO(response.content)

    with open('output.mp3', 'wb') as f:
        f.write(audio_stream.read())