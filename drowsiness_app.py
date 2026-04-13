import os
import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import sys

# Audio Libraries
try:
    import winsound
except ImportError:
    winsound = None

try:
    import pyttsx3
    voice_engine = pyttsx3.init()
except ImportError:
    voice_engine = None
    print("Warning: 'pyttsx3' not found. Voice alerts disabled. Install with `pip install pyttsx3`.")

# --- MediaPipe Integration for Mouth Detection ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MEDITPIPE_AVAILABLE = True
except Exception as e:
    MEDITPIPE_AVAILABLE = False
    print(f"Warning: MediaPipe Error ({e}). Mouth/Yawn detection disabled. Install via `pip install mediapipe`.")

# --- Mute TensorFlow Logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
MODEL_PATH = "Models/drowsiness_model_mobilenet.h5"
LABELS = ["Drowsy", "Non-Drowsy"] 
IMG_SIZE = (224, 224)

# Thresholds & Parameters
DROWSY_THRESHOLD = 0.5 
EYE_CLOSED_FRAMES_THRESH = 10  # Consecutive frames with 0 eyes to trigger alert
WARMUP_DURATION = 5.0          # Seconds to wait before alerting
ALERT_COOLDOWN = 3.0           # Seconds between alerts

# Yawn Detection Parameters
YAWN_THRESH = 0.6              # MAR value above which mouth is considered open
YAWN_FRAMES_THRESH = 15        # Consecutive frames to confirm a yawn

# --- State Management ---
alert_active = False 
last_alert_time = 0 
start_time = time.time()
closed_frames = 0
yawn_frames = 0
invert_logic = False 

# --- Helpers ---
def distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_mar(face_landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR).
    MAR = (Vertical Distance) / (Horizontal Distance)
    MediaPipe Indices (approximate for inner lips):
    - Top Lip: 13
    - Bottom Lip: 14
    - Left Corner: 78
    - Right Corner: 308
    """
    # Vertical distance (Top to Bottom)
    # Using inner lips for better precision on open mouth
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    
    # Horizontal distance (Left to Right corners)
    left_corner = face_landmarks.landmark[78]
    right_corner = face_landmarks.landmark[308]
    
    vertical_dist = distance(top_lip, bottom_lip)
    horizontal_dist = distance(left_corner, right_corner)
    
    if horizontal_dist == 0: return 0
    return vertical_dist / horizontal_dist


def calculate_ear(face_landmarks):
    """Calculate Eye Aspect Ratio (EAR) for both eyes."""
    try:
        # Left eye (user's right)
        left_top1 = face_landmarks.landmark[160]
        left_bottom1 = face_landmarks.landmark[144]
        left_top2 = face_landmarks.landmark[158]
        left_bottom2 = face_landmarks.landmark[153]
        left_left = face_landmarks.landmark[33]
        left_right = face_landmarks.landmark[133]

        ear_left = (distance(left_top1, left_bottom1) + distance(left_top2, left_bottom2)) / (2.0 * distance(left_left, left_right) + 1e-6)

        # Right eye (user's left)
        right_top1 = face_landmarks.landmark[385]
        right_bottom1 = face_landmarks.landmark[380]
        right_top2 = face_landmarks.landmark[387]
        right_bottom2 = face_landmarks.landmark[373]
        right_left = face_landmarks.landmark[362]
        right_right = face_landmarks.landmark[263]

        ear_right = (distance(right_top1, right_bottom1) + distance(right_top2, right_bottom2)) / (2.0 * distance(right_left, right_right) + 1e-6)

        return (ear_left + ear_right) / 2.0
    except IndexError:
        return 1.0 # Default to open if landmarks fail

def play_alert_thread(message, beep_enabled=True):
    """
    Handles audio alerts in a separate thread.
    """
    global alert_active
    if alert_active: return
    alert_active = True
    
    try:
        # 1. Beep Sound
        if beep_enabled and winsound:
            winsound.Beep(2500, 1000) 
        
        # 2. Voice Command
        if voice_engine:
            voice_engine.say(message)
            voice_engine.runAndWait()

    except Exception as e:
        print(f"Audio Error: {e}")
    finally:
        alert_active = False

def trigger_alert(message, is_serious):
    global last_alert_time
    current_time = time.time()
    
    # Only alert if warmup is complete
    if (current_time - start_time > WARMUP_DURATION):
        # Check cooldown
        if (current_time - last_alert_time > ALERT_COOLDOWN):
            last_alert_time = current_time
            threading.Thread(target=play_alert_thread, args=(message, is_serious), daemon=True).start()

def main():
    global invert_logic, closed_frames, yawn_frames, MEDITPIPE_AVAILABLE

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print("Loading CNN Model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model Loaded Successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 3. Detectors (OpenCV Cascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
        print("Error: One or more Cascade Classifiers failed to load.")
        return

    # MediaPipe Face Mesh Setup
    face_mesh = None
    if MEDITPIPE_AVAILABLE:
        try:
            face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"MP Init Error: {e}")
            MEDITPIPE_AVAILABLE = False

    print("--- Drowsiness Detection System (CNN + Yawn) ---")
    print("Press 'q' to Quit | 'i' to Invert Logic | 'd' Toggle Debug")
    debug_mode = False

    print("Starting Main Loop...")
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Error: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1) # Mirror
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # For MediaPipe
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # For Cascades
        
        # === A. Face & Eye Detection (Cascade) ===
        try:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        except Exception as e:
            print(f"Face Cascade Error: {e}")
            continue

        # === B. Yawn Detection (MediaPipe) ===
        mar_value = 0.0
        ear_value = 1.0
        is_yawning = False
        
        if MEDITPIPE_AVAILABLE:
            try:
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mar_value = calculate_mar(face_landmarks)
                        
                        # Calculate EAR for Eye Closed Detection
                        ear_value = calculate_ear(face_landmarks)
                        if ear_value < 0.22:
                            closed_frames += 1
                        else:
                            closed_frames = 0
                            
                        if mar_value > YAWN_THRESH:
                            yawn_frames += 1
                            # Draw Landmarks
                            h, w, c = frame.shape
                            t_lip = face_landmarks.landmark[13]
                            b_lip = face_landmarks.landmark[14]
                            cv2.circle(frame, (int(t_lip.x * w), int(t_lip.y * h)), 2, (0, 255, 255), -1)
                            cv2.circle(frame, (int(b_lip.x * w), int(b_lip.y * h)), 2, (0, 255, 255), -1)
                        else:
                            yawn_frames = 0
                            
                        if yawn_frames > YAWN_FRAMES_THRESH:
                            is_yawning = True
            except Exception:
                pass 

        # 2. Fallback: Contour/Cascade Mode (only if MP is completely unavailable)
        pass_fallback = not MEDITPIPE_AVAILABLE
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if pass_fallback:
                try:
                    # ROI for Mouth (Lower 1/3 of face)
                    roi_y_start = y + int(h * 2/3)
                    roi_y_end = y + h
                    
                    if roi_y_start >= roi_y_end: continue

                    mouth_roi_gray = gray[roi_y_start : roi_y_end, x : x + w]
                    mouth_roi_color = frame[roi_y_start : roi_y_end, x : x + w]

                    if mouth_roi_gray.size == 0: continue

                    # Smile Detection to confirm mouth presence
                    # Scale factor 1.7 is high to be fast/strict, minNeighbors 20 to reduce false positives
                    smiles = smile_cascade.detectMultiScale(mouth_roi_gray, 1.7, 20, minSize=(25, 25))
                    
                    # If smile/mouth found, check for open "hole" (Yawn)
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(mouth_roi_color, (sx, sy), (sx+sw, sy+sh), (255, 255, 255), 1)
                        
                        # Analyze specific mouth region
                        inner_mouth = mouth_roi_gray[sy:sy+sh, sx:sx+sw]
                        
                        # Otsu's Thresholding for dynamic lighting
                        _, thresh = cv2.threshold(inner_mouth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        
                        if debug_mode:
                            cv2.imshow("Mouth Thresh", thresh)

                        # Contours
                        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            max_cnt = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(max_cnt)
                            norm_area = area / (sw * sh)
                            
                            # Threshold > 30% of mouth box is dark hole
                            if norm_area > 0.30: 
                                yawn_frames += 1
                                try:
                                    cv2.drawContours(frame[roi_y_start + sy : roi_y_start + sy + sh, x + sx : x + sx + sw], [max_cnt], -1, (0, 255, 255), 1)
                                except: pass
                            else:
                                yawn_frames = max(0, yawn_frames - 1) # Decay
                                
                            if yawn_frames > YAWN_FRAMES_THRESH:
                                is_yawning = True
                except Exception as e:
                    # Print only if debug mode to avoid spam
                    if debug_mode: print(f"Fallback Error: {e}")
                    pass

            # --- Mechanism 1: Geometric Eye Detection (Fallback) ---
            if pass_fallback:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(20, 20))
                
                if len(eyes) == 0:
                    closed_frames += 1
                else:
                    closed_frames = 0
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)

            # --- Mechanism 2: CNN Inference ---
            cnn_status = "Unknown"
            is_cnn_drowsy = False
            
            try:
                face_img = frame[y:y+h, x:x+w]
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(face_img_rgb, IMG_SIZE)
                normalized = tf.keras.applications.mobilenet_v2.preprocess_input(resized.astype('float32'))
                reshaped = np.reshape(normalized, (1, 224, 224, 3))

                prediction = model.predict(reshaped, verbose=0)
                raw_drowsy_score = prediction[0][0]
                raw_non_drowsy_score = prediction[0][1]
                
                is_cnn_drowsy = raw_drowsy_score > raw_non_drowsy_score
                if invert_logic: is_cnn_drowsy = not is_cnn_drowsy
                
                cnn_status = f"DROWSY ({raw_drowsy_score:.2f})" if is_cnn_drowsy else f"Active ({raw_non_drowsy_score:.2f})"

            except Exception as e:
                print(f"CNN Error: {e}")

            # --- Consolidated Alert Logic ---
            final_status = "Scanning..."
            alert_color = (0, 255, 0)
            
            eyes_are_closed = (closed_frames > EYE_CLOSED_FRAMES_THRESH)
            
            if eyes_are_closed and is_yawning:
                final_status = "DROWSY LEVEL: CRITICAL (Closed & Yawning)"
                alert_color = (0, 0, 255)
                trigger_alert("CRITICAL ALERT! Wake up immediately!", is_serious=True)
                
            elif eyes_are_closed:
                final_status = "DROWSY LEVEL: HIGH (Eyes Closed)"
                alert_color = (0, 0, 255)
                trigger_alert("Driver Alert! Your eyes are closed.", is_serious=True)
            
            elif is_cnn_drowsy:
                final_status = f"CNN: {cnn_status}"
                alert_color = (0, 0, 255)
                trigger_alert("Wake up! You are drowsy.", is_serious=True)
                
            elif is_yawning:
                final_status = "YAWNING DETECTED!"
                alert_color = (0, 165, 255) # Orange warning
                trigger_alert("Warning! You are yawning.", is_serious=False)
                
            else:
                final_status = "You are Active"
                alert_color = (0, 255, 0)
                trigger_alert("You are fully attentive in driving stage", is_serious=False)

            # Debug Overlay
            cv2.putText(frame, final_status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            if MEDITPIPE_AVAILABLE:
                cv2.putText(frame, f"MAR: {mar_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, f"Model: {cnn_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Warmup Timer
            current_time = time.time()
            if current_time - start_time < WARMUP_DURATION:
                 cv2.putText(frame, f"WARMUP: {int(WARMUP_DURATION - (current_time - start_time))}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Drowsiness Detector (CNN Hybrid)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('i'): 
            invert_logic = not invert_logic
            print(f"Logic Inverted: {invert_logic}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug Mode: {debug_mode}")
            if not debug_mode: cv2.destroyWindow("Mouth Thresh")

if __name__ == "__main__":
    main()
