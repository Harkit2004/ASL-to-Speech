import mediapipe as mp
import cv2
import time
from elevenlabs import ElevenLabs
import tkinter as tk
from tkinter import ttk
import os
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVEN_LABS_API_KEY")
)

# ==========================
# Fetch top 5 voices
voices = client.voices.get_all().voices
print(voices)
top_voices = voices[:5]
voice_names = [voice.name for voice in top_voices]
voice_id_map = {voice.name: voice.voice_id for voice in top_voices}

# ==========================
# Show Tkinter UI to choose voice
selected_voice_name = None

def show_voice_selector():

    def confirm():
        global selected_voice_name
        nonlocal root
        selected = combo.get()
        if selected:
            selected_voice_name = selected
        else:
            selected_voice_name = voice_names[0]
        root.destroy()

    root = tk.Tk()
    root.title("Select Voice")

    tk.Label(root, text="Choose a voice for TTS:").pack(pady=10)
    combo = ttk.Combobox(root, values=voice_names, state="readonly")
    combo.current(0)
    combo.pack(pady=10)

    tk.Button(root, text="Confirm", command=confirm).pack(pady=10)
    root.mainloop()

show_voice_selector()
selected_voice_id = voice_id_map[selected_voice_name]

# ==========================
# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

current_text = ""
current_gesture = ""
gesture_start_time = 0
GESTURE_HOLD_TIME = 1.2
button_active = False

def process_result(result, output_image, timestamp_ms):
    global current_text, current_gesture, gesture_start_time, button_active

    if result.gestures and result.gestures[0]:
        gesture = result.gestures[0][0]
        category = gesture.category_name
        score = gesture.score

        if category:
            if category != current_gesture:
                current_gesture = category
                gesture_start_time = time.time()

            elapsed_time = time.time() - gesture_start_time

            if category.lower() == "button" and score > 0.7:
                button_active = True
                if elapsed_time >= GESTURE_HOLD_TIME:
                    print(f"Button pressed! Final text: {current_text}")
                    gesture_start_time = time.time()
            else:
                button_active = False
                if elapsed_time >= GESTURE_HOLD_TIME:
                    if category.lower() == "del" and current_text:
                        current_text = current_text[:-1]
                    elif category.lower() == "space":
                        current_text += " "
                    elif len(category) == 1:
                        current_text += category
                    gesture_start_time = time.time()
        else:
            current_gesture = ""
            gesture_start_time = time.time()
    else:
        current_gesture = ""
        gesture_start_time = time.time()
        button_active = False

def draw_text_and_button(frame):
    frame_height, frame_width = frame.shape[:2]
    text_area_height = 100
    cv2.rectangle(frame, (0, 0), (frame_width, text_area_height), (0, 0, 0), -1)

    text_display = current_text if current_text else "Start typing with gestures..."
    text_size = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = text_area_height // 2 + text_size[1] // 2
    cv2.putText(frame, text_display, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if current_gesture:
        height = 50
        y = text_area_height + 20
        elapsed_time = time.time() - gesture_start_time
        progress = min(elapsed_time / GESTURE_HOLD_TIME * 100, 100)
        gesture_text = f"Current gesture: {current_gesture} (Progress: {progress:.0f}%)"
        cv2.putText(frame, gesture_text, (10, y + height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        bar_width = int(frame_width * 0.8)
        bar_height = 10
        bar_x = (frame_width - bar_width) // 2
        bar_y = y + height + 50
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height), (70, 70, 70), -1)
        progress_width = int(bar_width * (progress / 100))
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)

    # Display voice name at the bottom
    cv2.putText(frame, f"Voice: {selected_voice_name}", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# ==========================
# Run the recognizer
frame_timestamp_ms = 0

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms += 1
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        draw_text_and_button(frame)
        cv2.imshow('Gesture Input', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\nFinal text: {current_text}")

            if current_text.strip():
                print(f"Generating speech with voice: {selected_voice_name}, {selected_voice_id}")
                response = client.text_to_speech.convert(
                    voice_id=selected_voice_id,
                    output_format="mp3_44100_128",
                    text=current_text,
                    model_id="eleven_flash_v2_5",
                )
                with open("output.mp3", "wb") as f:
                    for chunk in response:
                        f.write(chunk)
            break

    cap.release()
    cv2.destroyAllWindows()
