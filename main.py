import mediapipe as mp
import cv2
import time
import numpy as np
import sys



BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Store latest results globally
latest_gesture = "No Hand"
latest_landmarks = None  # <-- new

frame_width = 640
frame_height = 480

# Hand connections — which dots to connect with lines
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # Thumb
    (0,5),(5,6),(6,7),(7,8),         # Index
    (0,9),(9,10),(10,11),(11,12),    # Middle
    (0,13),(13,14),(14,15),(15,16),  # Ring
    (0,17),(17,18),(18,19),(19,20),  # Pinky
    (5,9),(9,13),(13,17)             # Palm connections
]

def handle_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture, latest_landmarks

    if result.gestures:
        latest_gesture = result.gestures[0][0].category_name
    else:
        latest_gesture = "No Hand"

    if result.hand_landmarks:
        latest_landmarks = result.hand_landmarks[0]  # first hand's landmarks
    else:
        latest_landmarks = None


def draw_landmarks(frame, landmarks):
    """Draw dots and lines on the hand."""
    h, w, _ = frame.shape

    # Convert normalized (0-1) coordinates to pixel coordinates
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # Draw lines (connections) first so dots appear on top
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)  # green lines

    # Draw dots
    for i, point in enumerate(points):
        cv2.circle(frame, point, 6, (0, 0, 255), -1)   # red filled dot
        cv2.circle(frame, point, 6, (255, 255, 255), 1) # white border


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='models/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result
)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

with GestureRecognizer.create_from_options(options) as recognizer:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Draw landmarks if available
        if latest_landmarks:
            draw_landmarks(frame, latest_landmarks)

        # Show gesture name
        cv2.putText(frame, latest_gesture, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Hand Gesture Recognition", frame)
        print(latest_gesture)
        if(latest_gesture == "victory"):
            sys.exit("Success")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()