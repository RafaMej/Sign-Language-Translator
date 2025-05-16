# pylint: disable=E1101
import os
import time
import threading

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pickle

class LSMTranslator:
    def __init__(self):
        # Configuración general: solo 3 acciones
        self.actions = ['hola', 'gracias', 'por favor']
        self.model_path = 'modelo_lsm.h5'
        self.sequences_path = 'secuencias_lsm.pkl'
        self.sequence_length = 30
        self.threshold = 0.7
        
        # Configuración de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        
        # Variables para la detección
        self.sequence = []
        self.predictions = []
        self.current_action = ""
        self.last_action = ""
        self.last_detection_time = time.time()
        self.cooldown = 2.0  # Tiempo de espera entre detecciones (segundos)
        
        # Cargar o crear modelo
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Modelo cargado de {self.model_path}")
            self.model_exists = True
        else:
            print("Modelo no encontrado. Se creará uno nuevo cuando se entrene.")
            self.model = None
            self.model_exists = False
    
    def extract_hand_keypoints(self, results):
        """Extrae los puntos clave de las manos de los resultados de MediaPipe."""
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = []
                for lm in hand_landmarks.landmark:
                    hand_points.append([lm.x, lm.y, lm.z])
                keypoints.extend(hand_points)
            
            # Si solo detectó una mano, rellenar con ceros para tener consistencia
            if len(results.multi_hand_landmarks) == 1:
                hand_points = [[0.0, 0.0, 0.0]] * 21
                keypoints.extend(hand_points)
        else:
            # No se detectaron manos, rellenar con ceros
            keypoints = [[0.0, 0.0, 0.0]] * 42  # 21 puntos por mano * 2 manos
            
        return np.array(keypoints).flatten()
    
    def create_model(self):
        """Crea un modelo LSTM para el reconocimiento de gestos."""
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length, 126)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.actions), activation='softmax'))
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model
    
    def train_model(self, X_train, y_train, epochs=70):
        """Entrena el modelo con los datos proporcionados."""
        if self.model is None:
            self.model = self.create_model()
        
        tb_callback = TensorBoard(log_dir='logs')
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=1)
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                       validation_split=0.2, callbacks=[tb_callback, checkpoint])
        
        self.model_exists = True
        print(f"Modelo guardado en {self.model_path}")
    
    def collect_training_data(self):
        """Recopila datos de entrenamiento para cada acción (solo las 3 definidas)."""
        sequences = []
        labels = []
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return None, None
        
        for action_idx, action in enumerate(self.actions):
            for sequence_idx in range(30):  # 30 secuencias por acción
                print(f'Recopilando datos para "{action}", secuencia {sequence_idx}')
                
                for countdown in range(5, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: No se pudo leer el fotograma.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return None, None
                    
                    cv2.putText(frame, str(countdown), (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                               6, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.imshow('Recopilación de datos', frame)
                    cv2.waitKey(1000)
                
                sequence_data = []
                for frame_idx in range(self.sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: No se pudo leer el fotograma {frame_idx}, rellenando con ceros.")
                        keypoints = np.zeros(126)
                        sequence_data.append(keypoints)
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    keypoints = self.extract_hand_keypoints(results)
                    if keypoints.shape[0] != 126:
                        print(f"Warning: keypoints inesperados en frame {frame_idx}, rellenando con ceros.")
                        keypoints = np.zeros(126)
                    
                    sequence_data.append(keypoints)
                    
                    cv2.putText(frame, f'Recopilando: {action}, Seq: {sequence_idx}, Frame: {frame_idx}', 
                               (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Recopilación de datos', frame)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return None, None
                
                if len(sequence_data) != self.sequence_length:
                    print(f"Warning: Secuencia incompleta (longitud {len(sequence_data)}), descartando.")
                    continue
                
                sequences.append(sequence_data)
                labels.append(action_idx)
        
        cap.release()
        cv2.destroyAllWindows()
        
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels)
        
        with open(self.sequences_path, 'wb') as f:
            pickle.dump({'sequences': sequences, 'labels': labels, 'actions': self.actions}, f)
        
        return sequences, labels
    
    def prepare_data(self, sequences, labels):
        """Prepara los datos para el entrenamiento."""
        X = sequences
        y = tf.keras.utils.to_categorical(labels, num_classes=len(self.actions))
        return X, y
    
    def load_training_data(self):
        """Carga datos de entrenamiento si existen."""
        if os.path.exists(self.sequences_path):
            with open(self.sequences_path, 'rb') as f:
                data = pickle.load(f)
                sequences = data['sequences']
                labels = data['labels']
                self.actions = data['actions']
                sequences = np.array(sequences, dtype=np.float32)
                labels = np.array(labels)
                return sequences, labels
        return None, None
    
    def detect_signs(self):
        """Detecta los gestos en tiempo real."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return
        
        if not self.model_exists:
            print("Error: No hay modelo entrenado disponible.")
            cap.release()
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el fotograma.")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
            
            keypoints = self.extract_hand_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-self.sequence_length:]
            
            if len(self.sequence) == self.sequence_length and time.time() - self.last_detection_time > self.cooldown:
                res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                
                if np.max(res) > self.threshold:
                    action_idx = np.argmax(res)
                    self.current_action = self.actions[action_idx]
                    
                    if self.current_action != self.last_action:
                        self.last_action = self.current_action
                        self.last_detection_time = time.time()
                        print(f"Detectado: {self.current_action}")
            
            cv2.putText(frame, f'Detectado: {self.last_action}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if len(self.sequence) == self.sequence_length:
                for i, (action, prob) in enumerate(zip(self.actions, res)):
                    cv2.putText(frame, f'{action}: {prob:.2f}', 
                               (10, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            cv2.imshow('Traductor LSM', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Inicia el traductor LSM."""
        sequences, labels = self.load_training_data()
        
        if sequences is None or labels is None:
            print("No se encontraron datos de entrenamiento. Iniciando recopilación de datos...")
            sequences, labels = self.collect_training_data()
            if sequences is None or labels is None:
                print("Error al recopilar datos de entrenamiento.")
                return
        
        X, y = self.prepare_data(sequences, labels)
        
        if not self.model_exists:
            print("Entrenando modelo...")
            self.train_model(X, y)
        
        print("Iniciando detección en tiempo real...")
        self.detect_signs()

def mostrar_menu():
    """Muestra el menú principal de la aplicación."""
    print("\n=== TRADUCTOR DE LENGUA DE SEÑAS MEXICANA ===")
    print("1. Iniciar traductor")
    print("2. Recopilar nuevos datos de entrenamiento")
    print("3. Reentrenar modelo")
    print("4. Salir")
    return input("Seleccione una opción: ")

if __name__ == "__main__":
    traductor = LSMTranslator()
    
    while True:
        opcion = mostrar_menu()
        
        if opcion == '1':
            traductor.run()
        elif opcion == '2':
            sequences, labels = traductor.collect_training_data()
            if sequences is not None and labels is not None:
                print(f"Datos recopilados: {len(sequences)} secuencias")
        elif opcion == '3':
            sequences, labels = traductor.load_training_data()
            if sequences is not None and labels is not None:
                X, y = traductor.prepare_data(sequences, labels)
                traductor.train_model(X, y)
            else:
                print("No hay datos de entrenamiento disponibles.")
        elif opcion == '4':
            print("¡Hasta pronto!")
            break
        else:
            print("Opción no válida. Intente de nuevo.")