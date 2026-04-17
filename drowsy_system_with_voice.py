# drowsy_system_with_voice.py

import os
import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import speech_recognition as sr

from voice_assistant import AIVoiceAssistant
from dotenv import load_dotenv
load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Detection & Model Settings 
MODEL_PATH                  = "Models/resnet_quantized.tflite"   # TFLite model path
IMG_SIZE                    = (224, 224)                          # model input size

EYE_CLOSED_FRAMES_THRESH    = 15    # frames eyes must be closed to trigger alert
CNN_DROWSY_THRESHOLD        = 0.85  # minimum CNN score to consider drowsy
CNN_DROWSY_CONSEC_THRESH    = 15    # consecutive drowsy frames needed to alert
WARMUP_DURATION             = 5.0   # seconds to wait before allowing any alerts
ALERT_COOLDOWN              = 3.0   # minimum seconds between alerts
MIC_INDEX                   = 1     # microphone device index

# ── Global State ─────────────────────────────────────────────────────────────
closed_frames       = 0       # consecutive frames where eyes were not clearly open
cnn_drowsy_frames   = 0       # consecutive frames CNN predicted drowsy
invert_logic        = False   # toggle to invert CNN prediction (press i)
start_time          = time.time()
alert_active        = False   # whether an alert is currently being spoken
last_alert_time     = 0.0     # timestamp of the last alert
conversation_mode   = False   # whether continuous conversation is active
conversation_thread = None    # reference to conversation background thread

# ── Initialize AI Voice Assistant ────────────────────────────────────────────
assistant = AIVoiceAssistant(
    driver_name="driver",
    use_cloud_assistant=True,
    gemini_model_name="models/gemini-2.5-flash"
)


# ── Load TFLite Model ─────────────────────────────────────────────────────────
def load_tflite_model(path: str):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[System] Input  shape : {input_details[0]['shape']}")
    print(f"[System] Input  dtype : {input_details[0]['dtype']}")
    print(f"[System] Output shape : {output_details[0]['shape']}")
    return interpreter, input_details, output_details


# ── Run TFLite Inference ──────────────────────────────────────────────────────
def tflite_predict(interpreter, input_details, output_details, face_img_bgr):
    # Convert BGR to RGB and resize to model input size
    face_rgb     = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
    resized      = cv2.resize(face_rgb, IMG_SIZE)
    img_float    = resized.astype("float32")

    # Apply ResNet50 preprocessing (mean subtraction)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(img_float)

    # Quantize input if model expects uint8
    if input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        preprocessed = (preprocessed / scale + zero_point).astype(np.uint8)

    input_data = np.expand_dims(preprocessed, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output if model output is uint8
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype("float32") - zero_point) * scale

    # Shape (1,2) → two class scores, shape (1,1) → sigmoid score
    if output.shape[-1] == 2:
        return float(output[0][0]), float(output[0][1])
    else:
        score = float(output[0][0])
        return score, 1.0 - score


# ── Context Helpers (can be connected to GPS / weather API) ───────────────────
def get_speed() -> float:
    return 60.0

def get_weather() -> str:
    return "clear"


# ── Trigger Voice Alert (non-blocking, runs in background thread) ─────────────
def trigger_voice_alert(level, custom_message=None):
    global alert_active, last_alert_time
    now = time.time()

    # Skip alert during warmup period
    if now - start_time < WARMUP_DURATION:
        return

    # Skip if cooldown has not passed since last alert
    if now - last_alert_time < ALERT_COOLDOWN:
        return

    last_alert_time = now

    def _run():
        global alert_active
        if alert_active:
            return
        alert_active = True
        try:
            context = {
                "speed":         get_speed(),
                "weather":       get_weather(),
                "location_hint": "on this route",
            }
            if custom_message and level is None:
                assistant.speak(custom_message)
            elif level is not None:
                assistant.alert_drowsiness(level, context)
            else:
                assistant.speak("Please stay focused on the road.")
        finally:
            alert_active = False

    threading.Thread(target=_run, daemon=True).start()


# ── Listen Once From Microphone ───────────────────────────────────────────────
def listen_once(recognizer, microphone):
    with microphone as source:
        print("[System] Listening to driver...")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"[Driver] {text}")
        return text
    except (sr.UnknownValueError, sr.RequestError):
        return ""


# ── Continuous Conversation Loop (runs in background thread) ──────────────────
def conversation_loop():
    global conversation_mode
    recognizer = sr.Recognizer()

    # Initialize microphone, fall back to default if index fails
    try:
        microphone = sr.Microphone(device_index=MIC_INDEX)
    except:
        microphone = sr.Microphone()

    while conversation_mode:
        text = listen_once(recognizer, microphone)
        if not text:
            continue

        low = text.lower().strip()

        # Stop conversation if driver says exit phrase
        if ("exit assistant" in low) or (low == "exit") or ("stop talking" in low):
            conversation_mode = False
            assistant.speak("Okay, I will stop talking now, but I will keep monitoring.")
            break

        # Send driver input to AI assistant
        try:
            assistant.handle_command_with_nlp_backend(text)
        except SystemExit:
            conversation_mode = False
            break
        except Exception as e:
            print(f"[System] Conversation error: {e}")

    print("[System] Conversation mode ended.")


# ── Start Conversation Mode (only once at a time) ─────────────────────────────
def start_conversation_mode():
    global conversation_mode, conversation_thread
    if conversation_mode:
        return
    conversation_mode = True
    conversation_thread = threading.Thread(target=conversation_loop, daemon=True)
    conversation_thread.start()


# ── Main Application ──────────────────────────────────────────────────────────
def main():
    global closed_frames, cnn_drowsy_frames, invert_logic, start_time

    # Check model file exists before loading
    if not os.path.exists(MODEL_PATH):
        print(f"[System] Error: Model not found → {MODEL_PATH}")
        return

    print("[System] Loading TFLite ResNet model...")
    try:
        interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
        print("[System]  TFLite model loaded.")
    except Exception as e:
        print(f"[System] Failed to load model: {e}")
        return

    # Reset state on startup
    start_time        = time.time()
    closed_frames     = 0
    cnn_drowsy_frames = 0

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[System] Error: Could not open webcam.")
        return

    # Announce system start to driver
    assistant.speak(
        "Drowsiness detection assistant started. "
        "I will warn you if I detect signs of sleepiness."
    )

    # Load Haar cascade classifiers for face and eye detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    print("--- Drowsiness Detector (TFLite + Consecutive Frames) ---")
    print("Press 'q' to Quit | 'i' to Invert CNN logic")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[System] Could not read frame.")
            break

        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize brightness to handle bright/dark lighting conditions
        gray = cv2.equalizeHist(gray)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

        # Reset counters if no face is detected
        if len(faces) == 0:
            closed_frames     = 0
            cnn_drowsy_frames = 0

        eye_count     = 0
        cnn_status    = "Unknown"
        is_cnn_drowsy = False

        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]

            # Equalize eye region separately for better detection in bright light
            roi_eq = cv2.equalizeHist(roi_gray)

            # Detect eyes within the equalized face region
            eyes = eye_cascade.detectMultiScale(
                roi_eq,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(20, 20)   # smaller min size handles bright light conditions
            )
            eye_count = len(eyes)

            # Only reset when both eyes clearly visible
            # 0 or 1 eye detected → treat as closed
            if eye_count >= 2:
                closed_frames = 0
            else:
                closed_frames += 1

            # Draw rectangles around detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    frame,
                    (x+ex, y+ey),
                    (x+ex+ew, y+ey+eh),
                    (0, 255, 0), 1
                )

            # Run CNN drowsiness prediction on face crop
            try:
                face_crop = frame[y:y+h, x:x+w]
                raw_drowsy, raw_non_drowsy = tflite_predict(
                    interpreter, input_details, output_details, face_crop
                )

                # Apply threshold and optional logic inversion
                is_cnn_drowsy = raw_drowsy > CNN_DROWSY_THRESHOLD
                if invert_logic:
                    is_cnn_drowsy = not is_cnn_drowsy

                cnn_status = (
                    f"DROWSY ({raw_drowsy:.2f})"
                    if is_cnn_drowsy
                    else f"Active ({raw_non_drowsy:.2f})"
                )

            except Exception as e:
                print(f"[System] TFLite Error: {e}")
                cnn_status    = "Error"
                is_cnn_drowsy = False

            # Eyes clearly open only when 2 or more detected
            eyes_clearly_open = (eye_count >= 2)

            # ── Combined Alert Logic ──────────────────────────────────────────
            if closed_frames > EYE_CLOSED_FRAMES_THRESH:
                # Eyes closed too long → HIGH alert
                cnn_drowsy_frames = 0
                final_status      = "EYES CLOSED!"
                alert_color       = (0, 0, 255)
                trigger_voice_alert("high")
                start_conversation_mode()

            elif is_cnn_drowsy and not eyes_clearly_open:
                # CNN predicts drowsy and eyes not clearly open → accumulate
                cnn_drowsy_frames += 1
                if cnn_drowsy_frames > CNN_DROWSY_CONSEC_THRESH:
                    # Threshold exceeded → MEDIUM alert
                    final_status = f"DROWSY! ({cnn_drowsy_frames}f)"
                    alert_color  = (0, 0, 255)
                    trigger_voice_alert("medium")
                    start_conversation_mode()
                else:
                    # Still accumulating → show warning only, no alert yet
                    final_status = f"Warning ({cnn_drowsy_frames}f)"
                    alert_color  = (0, 165, 255)

            else:
                # Driver is active → reset CNN consecutive counter
                cnn_drowsy_frames = 0
                final_status      = "Active"
                alert_color       = (0, 255, 0)

            # ── Draw Overlays ─────────────────────────────────────────────────
            cv2.putText(frame, final_status,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

            cv2.putText(frame, f"Closed Frames : {closed_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.putText(frame, f"CNN           : {cnn_status}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.putText(frame, f"Eyes Detected : {eye_count}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.putText(frame, f"CNN Frames    : {cnn_drowsy_frames}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Show warmup countdown if still in warmup period
            now = time.time()
            if now - start_time < WARMUP_DURATION:
                remaining = int(WARMUP_DURATION - (now - start_time))
                cv2.putText(frame, f"WARMUP: {remaining}s",
                            (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Drowsiness Detector + AI Voice", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            # Quit application
            break
        elif key == ord("i"):
            # Toggle CNN logic inversion
            invert_logic = not invert_logic
            print(f"[System] Logic Inverted: {invert_logic}")

    # Release resources on exit
    cap.release()
    cv2.destroyAllWindows()
    print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()