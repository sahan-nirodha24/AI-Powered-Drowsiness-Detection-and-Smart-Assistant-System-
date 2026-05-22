import cv2
import os
import joblib
import numpy as np
import time
import winsound
import threading
from skimage.feature import hog, local_binary_pattern

# --- Configuration ---
MODEL_PATH = 'svm_model.pkl'
PARAMS_PATH = 'hog_parameters.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Detectors
HAAR_FACE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
HAAR_EYE_PATH = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'

# Parameters
EYE_CLOSED_FRAMES_THRESH = 10  # Consecutive frames with 0 eyes to trigger alert
SVM_DROWSY_THRESHOLD = 0.7     # Probability threshold for SVM
WARMUP_DURATION = 5.0          # Seconds to wait before alerting
ALERT_COOLDOWN = 3.0           # Seconds between alerts

# --- Initialization ---
print("Loading model artifacts...")
try:
    if not all(os.path.exists(p) for p in [MODEL_PATH, PARAMS_PATH, LABEL_ENCODER_PATH]):
        raise FileNotFoundError("Model files missing.")
    
    svm_model = joblib.load(MODEL_PATH)
    params = joblib.load(PARAMS_PATH)
    label_map = joblib.load(LABEL_ENCODER_PATH)
    IMG_SIZE = params['image_size']
    MODELS_LOADED = True
    print("SVM Model Loaded Successfully.")
except Exception as e:
    print(f"WARNING: Feature Extraction Models not found ({e}). Running in Eye-Detection Only mode.")
    MODELS_LOADED = False
    IMG_SIZE = (224, 224) # Default

# Load Detectors
face_cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)
eye_cascade = cv2.CascadeClassifier(HAAR_EYE_PATH)

# --- Logic: Feature Extraction (MATCHING TRAINING) ---
def extract_feature_vector(image):
    # Extract HOG
    hog_fd = hog(image, 
                 orientations=params['orientations'], 
                 pixels_per_cell=params['pixels_per_cell'], 
                 cells_per_block=params['cells_per_block'], 
                 visualize=False)
    
    # Extract LBP
    lbp = local_binary_pattern(image, 
                               params['lbp_points'], 
                               params['lbp_radius'], 
                               method='uniform')
    
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # Combine
    combined = np.hstack((hog_fd, lbp_hist))
    return combined

# --- Logic: Audio Alert ---
def play_alert(message, beep_enabled=True):
    try:
        # 1. Beep Sound (Frequency 2500Hz, Duration 1000ms)
        if beep_enabled:
            winsound.Beep(2500, 1000)
        
        # 2. Voice Command (PowerShell TTS)
        # Using a non-blocking calling strategy typically requires threading in Caller, 
        # but os.system is blocking. We'll rely on the thread wrapper.
        safe_msg = message.replace("'", "")
        cmd = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{safe_msg}\');"'
        os.system(cmd)
    except Exception as e:
        print(f"Audio Error: {e}")

# --- Main Application ---
def main():
    cap = cv2.VideoCapture(0)
    
    # State
    closed_frames = 0
    last_alert_time = 0
    start_time = time.time()
    
    print("Starting Hybrid Drowsiness Detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        if len(faces) == 0:
            cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            # Draw Face Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            # --- Mechanism 1: Geometric Eye Detection ---
            face_roi_gray = gray[y:y+h//2, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 4, minSize=(30, 30))
            
            if len(eyes) == 0:
                closed_frames += 1
            else:
                closed_frames = 0
            
            for (ex, ey, ew, eh) in eyes:
                # Draw Eye Boxes (green)
                camera_x = x + ex
                camera_y = y + ey
                cv2.rectangle(frame, (camera_x, camera_y), (camera_x+ew, camera_y+eh), (0, 255, 0), 1)

            # --- Mechanism 2: SVM Analysis ---
            svm_status = "Unknown"
            pred_class_name = "Unknown"
            pred_prob = 0.0
            probs = [0.0, 0.0, 0.0, 0.0]
            
            if MODELS_LOADED:
                try:
                    # The SVM was trained on square 224x224 images.
                    # We must take a center square crop of the webcam frame to preserve the aspect ratio (no squashing)
                    h_cam, w_cam = frame.shape[:2]
                    min_dim = min(h_cam, w_cam)
                    start_x = (w_cam - min_dim) // 2
                    start_y = (h_cam - min_dim) // 2
                    square_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
                    
                    frame_resized = cv2.resize(square_frame, (IMG_SIZE[1], IMG_SIZE[0]))
                    frame_gray_svm = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    
                    # Show the exact image the SVM is looking at for debugging
                    cv2.imshow("SVM Input Debug", frame_gray_svm)
                    
                    features = extract_feature_vector(frame_gray_svm)
                    scores = svm_model.decision_function([features])[0]
                    # Softmax for multi-class pseudo-probabilities
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / np.sum(exp_scores)
                    
                    pred_class_idx = np.argmax(probs)
                    pred_class_name = label_map['classes'][pred_class_idx]
                    pred_prob = probs[pred_class_idx]
                    
                    svm_status = pred_class_name.upper()
                except Exception as e:
                    print(f"SVM Error: {e}")

            # --- Logic & Visuals ---
            final_status = svm_status if MODELS_LOADED else "Monitoring"
            alert_msg = ""
            is_serious_alert = False
            should_alert = False

            # --- SVM + Haar Synergy Logic ---
            is_eyes_closed = (closed_frames > EYE_CLOSED_FRAMES_THRESH)
            
            # Yawn logic: SVM detects yawn
            is_yawning_svm = (pred_class_name.upper() == "YAWN")
            
            # Distracted logic: SVM detects distracted
            is_distracted_svm = (pred_class_name.upper() == "DISTRACTED")
            
            # Drowsy logic: SVM detects drowsy OR eyes closed by Haar
            is_drowsy_svm = (pred_class_name.upper() == "DROWSY")

            if is_eyes_closed:
                final_status = "DROWSY - Eyes Closed"
                status_color = (0, 0, 255) # Red
                should_alert = True
                alert_msg = "Driver Alert: Eyes closed."
                is_serious_alert = True
            elif is_drowsy_svm:
                final_status = "DROWSY - SVM Detected"
                status_color = (0, 0, 255) # Red
                should_alert = True
                alert_msg = "Driver Alert: You are drowsy."
                is_serious_alert = True
            elif is_yawning_svm:
                final_status = "YAWNING - SVM Detected"
                status_color = (0, 165, 255) # Orange
                should_alert = True
                alert_msg = "Warning: Yawning detected."
                is_serious_alert = False
            elif is_distracted_svm:
                final_status = "DISTRACTED - SVM Detected"
                status_color = (255, 0, 255) # Purple
                should_alert = True
                alert_msg = "Warning: Please focus on the road."
                is_serious_alert = False
            else:
                final_status = "ACTIVE"
                status_color = (0, 255, 0) # Green
                should_alert = False
                alert_msg = ""

            # --- UI: Info Panel ---
            # Create a semi-transparent overlay for stats
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y-60), (x+200, y), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Status Text
            cv2.putText(frame, f"STATUS: {final_status}", (x+5, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Closed Frames: {closed_frames}", (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- UI: Probability Bar ---
            if MODELS_LOADED:
                # Bar Background
                bar_x, bar_y, bar_w, bar_h = 10, 50, 200, 20
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50, 50, 50), -1)
                
                # Filled Portion based on Probability
                fill_w = int(pred_prob * bar_w)
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), status_color, -1)
                cv2.putText(frame, f"{final_status}: {int(pred_prob*100)}%", (bar_x, bar_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Trigger Audio
            current_time = time.time()
            if should_alert and (current_time - start_time > WARMUP_DURATION):
                if (current_time - last_alert_time) > ALERT_COOLDOWN:
                    last_alert_time = current_time
                    t = threading.Thread(target=play_alert, args=(alert_msg, is_serious_alert))
                    t.daemon = True
                    t.start()

            # Warmup Timer
            if current_time - start_time < WARMUP_DURATION:
                 cv2.putText(frame, f"WARMUP: {int(WARMUP_DURATION - (current_time - start_time))}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Hybrid Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()