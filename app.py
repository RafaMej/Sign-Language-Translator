# pylint: disable=E1101
import time
import threading

import cv2
import numpy as np

from flask import Flask, render_template, Response, jsonify

from lsm_translator import LSMTranslator

video_capture = None
frame_lock = threading.Lock()
latest_frame = None
detection_results = {
    "current_action": "",
    "probabilities": {}
}
video_streaming = False
camera_thread_instance = None

app = Flask(__name__)
translator = LSMTranslator()

# Estado compartido
video_capture = None
frame_lock = threading.Lock()
latest_frame = None
detection_results = {
    "current_action": "",
    "probabilities": {}
}
video_streaming = False

def camera_thread():
    global video_capture, latest_frame, video_streaming, detection_results

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: No se pudo abrir la cámara.")
        video_streaming = False
        return
    
    sequence = []
    while video_streaming:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = translator.hands.process(rgb_frame)

        # Dibujar landmarks si detecta manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                translator.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, translator.mp_hands.HAND_CONNECTIONS,
                    translator.mp_drawing_styles.get_default_hand_landmarks_style(),
                    translator.mp_drawing_styles.get_default_hand_connections_style())

        # Extraer keypoints y armar secuencia para predicción
        keypoints = translator.extract_hand_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-translator.sequence_length:]

        # Predecir solo si tenemos suficientes frames y modelo cargado
        if len(sequence) == translator.sequence_length and translator.model is not None:
            input_data = np.expand_dims(sequence, axis=0)
            res = translator.model.predict(input_data, verbose=0)[0]
            max_prob = np.max(res)
            if max_prob > translator.threshold:
                idx = np.argmax(res)
                current_action = translator.actions[idx]
            else:
                current_action = ""
            # Actualizar resultados detectados globalmente
            detection_results["current_action"] = current_action
            detection_results["probabilities"] = {
                action: float(prob) for action, prob in zip(translator.actions, res)
            }
        else:
            detection_results["current_action"] = ""
            detection_results["probabilities"] = {}

        # Poner texto en el frame
        cv2.putText(frame, f'Detectado: {detection_results["current_action"]}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Guardar frame para streaming
        with frame_lock:
            latest_frame = frame.copy()
    
    video_capture.release()
    video_streaming = False
    detection_results["current_action"] = ""
    detection_results["probabilities"] = {}

def generate_frames():
    global latest_frame, frame_lock, video_streaming
    while video_streaming:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    yield b''

@app.route('/')
def index():
    actions = translator.actions if hasattr(translator, 'actions') else []
    return render_template('index.html', actions=actions)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global video_streaming, camera_thread_instance
    if video_streaming:
        return jsonify({"status": "Ya está en ejecución"})
    
    if not translator.model_exists:
        return jsonify({"status": "Error: No hay modelo entrenado disponible"})
    
    video_streaming = True
    camera_thread_instance = threading.Thread(target=camera_thread)
    camera_thread_instance.daemon = True
    camera_thread_instance.start()
    
    return jsonify({"status": "Detección iniciada"})

@app.route('/stop_detection')
def stop_detection():
    global video_streaming
    video_streaming = False
    return jsonify({"status": "Detección detenida"})

@app.route('/get_results')
def get_results():
    global detection_results
    return jsonify(detection_results)

@app.route('/collect_data')
def collect_data():
    # Ejecutar colección de datos en thread separado para no bloquear
    def run_collection():
        translator.collect_training_data()
    threading.Thread(target=run_collection).start()
    return jsonify({"status": "Recopilación de datos iniciada"})

@app.route('/train_model')
def train_model():
    sequences, labels = translator.load_training_data()
    if sequences is None or labels is None:
        return jsonify({"status": "Error: No hay datos de entrenamiento disponibles"})
    
    X, y = translator.prepare_data(sequences, labels)
    
    def run_training():
        translator.train_model(X, y)
    threading.Thread(target=run_training).start()
    
    return jsonify({"status": "Entrenamiento iniciado"})

if __name__ == '__main__':
    app.run(debug=True)