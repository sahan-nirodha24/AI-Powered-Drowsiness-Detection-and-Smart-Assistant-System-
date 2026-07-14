import os
import cv2
import numpy as np
import threading
import time
import sys
from collections import Counter

# --- TFLite / LiteRT / TensorFlow Interpreter Setup ---
# Designed to work across lightweight tflite_runtime on Raspberry Pi 4 and full TensorFlow environments
try:
    # 1. Try lightweight tflite_runtime (Best/Standard for Raspberry Pi OS)
    from tflite_runtime.interpreter import Interpreter
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    try:
        # 2. Try new LiteRT package (ai_edge_litert)
        from ai_edge_litert.interpreter import Interpreter
        TFLITE_BACKEND = "ai_edge_litert"
    except ImportError:
        try:
            # 3. Try standard TensorFlow Lite interpreter
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            TFLITE_BACKEND = "tensorflow"
        except ImportError:
            print("Error: No TFLite interpreter found! Please install `tflite-runtime`, `ai-edge-litert`, or `tensorflow` on your Raspberry Pi.")
            sys.exit(1)

# Audio Libraries
try:
    import winsound
except ImportError:
    winsound = None

try:
    import pyttsx3
    voice_engine = pyttsx3.init()
except Exception as e:
    voice_engine = None
    print(f"Warning: Voice alert init failed ({e}). Audio voice alerts disabled.")

# --- MediaPipe Integration for Mouth Detection ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MEDITPIPE_AVAILABLE = True
except Exception as e:
    MEDITPIPE_AVAILABLE = False
    print(f"Warning: MediaPipe Error ({e}). Mouth/Yawn detection disabled. Install via `pip install mediapipe`.")

# --- Configuration ---
MODEL_PATH = "Models/resnet_quantized.tflite"
LABELS = ["distracted", "drowsy", "non-drowsy", "yawn"] 
IMG_SIZE = (224, 224)

# Thresholds & Parameters
DROWSY_THRESHOLD = 0.5 
EYE_CLOSED_FRAMES_THRESH = 8   # Consecutive frames below EAR threshold to trigger alert
WARMUP_DURATION = 5.0          # Seconds to wait before alerting
ALERT_COOLDOWN = 3.0           # Seconds between alerts

# Yawn Detection Parameters
YAWN_THRESH = 0.45             # Scientifically accurate MAR threshold for normal/medium yawns
YAWN_FRAMES_THRESH = 8         # Faster responsiveness (~0.25s of open mouth)

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
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
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

def preprocess_input_resnet(x):
    """
    Pure NumPy implementation of ResNet50 (Caffe mode) preprocessing for Raspberry Pi TFLite.
    Converts RGB to BGR and zero-centers each color channel with respect to ImageNet dataset.
    Works independently of TensorFlow so that `tflite-runtime` alone can run on Raspberry Pi.
    """
    try:
        # If full tensorflow is loaded, use exact built-in Keras preprocessing
        import tensorflow as tf
        return tf.keras.applications.resnet50.preprocess_input(x)
    except ImportError:
        # Pure NumPy fallback for barebone Raspberry Pi TFLite runtime
        x = x.astype('float32')
        # Flip RGB to BGR
        x = x[..., ::-1]
        # Subtract ImageNet mean [B, G, R] = [103.939, 116.779, 123.68]
        mean = np.array([103.939, 116.779, 123.68], dtype='float32')
        x -= mean
        return x

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

    # 1. Load Model (TFLite Quantized for Raspberry Pi 4)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: TFLite model file not found at {MODEL_PATH}")
        return

    print(f"Loading TFLite Quantized Model using '{TFLITE_BACKEND}' backend...")
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ TFLite Model Loaded Successfully for Raspberry Pi 4.")
    except Exception as e:
        print(f"Failed to load TFLite model: {e}")
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

    print("--- Drowsiness Detection System (TFLite Raspberry Pi 4 Hybrid) ---")
    print("Press 'q' to Quit | 'i' to Invert Logic | 'd' Toggle Debug")
    debug_mode = False
    last_box = None
    last_box_timer = 0
    cnn_history = []

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
            if len(faces) > 0:
                last_box = faces[0]
                last_box_timer = 0
            elif last_box is not None and last_box_timer < 30:
                # If face is temporarily occluded by phone/bottle, use last known position
                faces = [last_box]
                last_box_timer += 1
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
                        
                        # Calculate EAR for Eye Closed / Drowsy Detection
                        ear_value = calculate_ear(face_landmarks)
                        if ear_value < 0.25: # 0.25 catches heavy lidded / drowsy eyes reliably
                            closed_frames += 1
                        else:
                            closed_frames = max(0, closed_frames - 1) # Gradual decay prevents flickering
                            
                        if mar_value > YAWN_THRESH:
                            yawn_frames += 1
                            # Draw Landmarks
                            h, w, c = frame.shape
                            t_lip = face_landmarks.landmark[13]
                            b_lip = face_landmarks.landmark[14]
                            cv2.circle(frame, (int(t_lip.x * w), int(t_lip.y * h)), 2, (0, 255, 255), -1)
                            cv2.circle(frame, (int(b_lip.x * w), int(b_lip.y * h)), 2, (0, 255, 255), -1)
                        else:
                            yawn_frames = max(0, yawn_frames - 1) # Gradual decay prevents flickering
                            
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
                    if debug_mode: print(f"Fallback Error: {e}")
                    pass

            # --- Mechanism 1: Geometric Eye Detection (Fallback) ---
            if pass_fallback:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(20, 20))
                
                if len(eyes) == 0:
                    closed_frames += 1
                else:
                    closed_frames = max(0, closed_frames - 1)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)

            # --- Mechanism 2: TFLite Inference (Multi-Class optimized for Raspberry Pi) ---
            cnn_status = "Unknown"
            cnn_status_label = "unknown"
            
            try:
                # Expand bounding box: 45% sideways (for phones at ear) and 60% downwards (for water bottles at chest/neck)
                margin_x = int(w * 0.45)
                margin_y_top = int(h * 0.25)
                margin_y_bottom = int(h * 0.60)
                
                y1 = max(0, y - margin_y_top)
                y2 = min(frame.shape[0], y + h + margin_y_bottom)
                x1 = max(0, x - margin_x)
                x2 = min(frame.shape[1], x + w + margin_x)
                
                face_img = frame[y1:y2, x1:x2]
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(face_img_rgb, IMG_SIZE)
                normalized = preprocess_input_resnet(resized)
                reshaped = np.reshape(normalized, (1, 224, 224, 3))

                # TFLite Interpreter Invocation
                interpreter.set_tensor(input_details[0]['index'], reshaped)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                
                cnn_class_idx = np.argmax(prediction[0])
                confidence = prediction[0][cnn_class_idx]
                
                # 1. Confidence filtering: If model is < 50% sure, default to active (non-drowsy)
                if confidence < DROWSY_THRESHOLD:
                    cnn_status_label = "non-drowsy"
                else:
                    cnn_status_label = LABELS[cnn_class_idx]
                
                if invert_logic: # Only basic inversion for testing
                    if cnn_status_label == "drowsy": cnn_status_label = "non-drowsy"
                    elif cnn_status_label == "non-drowsy": cnn_status_label = "drowsy"
                
                # 2. Prediction Smoothing: 5-frame rolling majority vote prevents 1-frame noise/flickering
                cnn_history.append(cnn_status_label)
                if len(cnn_history) > 5:
                    cnn_history.pop(0)
                cnn_status_label = Counter(cnn_history).most_common(1)[0][0]
                
                cnn_status = f"{cnn_status_label.upper()} ({confidence:.2f})"

            except Exception as e:
                if debug_mode: print(f"TFLite Inference Error: {e}")

            # --- Consolidated Alert Logic (Synergy) ---
            final_status = "Scanning..."
            alert_color = (0, 255, 0)
            
            eyes_are_closed = (closed_frames > EYE_CLOSED_FRAMES_THRESH)
            is_cnn_drowsy = (cnn_status_label == "drowsy")
            is_cnn_yawn = (cnn_status_label == "yawn")
            is_cnn_distracted = (cnn_status_label == "distracted")
            
            # Critical: Closed eyes + Yawning (MediaPipe) or CNN Drowsy + Yawning
            if (eyes_are_closed and is_yawning) or (is_cnn_drowsy and is_cnn_yawn):
                final_status = "DROWSY LEVEL: CRITICAL"
                alert_color = (0, 0, 255)
                trigger_alert("CRITICAL ALERT! Wake up immediately!", is_serious=True)
                
            # High: Eyes closed (MediaPipe) or CNN predicts Drowsy
            elif eyes_are_closed or is_cnn_drowsy:
                final_status = f"DROWSY: HIGH ({'CNN' if is_cnn_drowsy else 'EAR'})"
                alert_color = (0, 0, 255)
                trigger_alert("Driver Alert! You are drowsy.", is_serious=True)
            
            # Warning: Distracted (CNN)
            elif is_cnn_distracted:
                final_status = "WARNING: DISTRACTED!"
                alert_color = (0, 165, 255) # Orange warning
                trigger_alert("Warning! Distraction detected. Focus on the road.", is_serious=False)
                
            # Warning: Yawning (MediaPipe or CNN)
            elif is_yawning or is_cnn_yawn:
                final_status = f"YAWNING ({'CNN' if is_cnn_yawn else 'MAR'})"
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

        cv2.imshow('Drowsiness Detector (TFLite Raspberry Pi 4)', frame)
        
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
