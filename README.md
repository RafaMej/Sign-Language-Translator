# 🤟 Sign Language Translator – LSM Real-Time Recognition System

This project implements a real-time translator for **Lengua de Señas Mexicana (LSM)** using computer vision and deep learning. It enables gesture detection through a webcam and translates recognized signs into text on a web interface.

## 📋 Project Report and Presentation

- 📄 [Final Report (PDF)](https://docs.google.com/document/d/19FDSmBPPXUI0ROTCcL36kmV0rueZUmqLKhHbKfeEwdw/edit?usp=sharing)
- 🎞️ [Final Presentation (Slides)](https://docs.google.com/presentation/d/1vK-KaBbdbi6S_aEo1HwLV-jikRa2BG32X25bMqInRjM/edit?usp=sharing)

---

## 🚀 Features

- Detects **real-time hand gestures** (e.g., *hola*, *gracias*, *por favor*) using a webcam.
- Uses **MediaPipe** to extract hand landmarks and **LSTM** for gesture classification.
- Provides an interactive **web interface** built with Flask.
- Supports **data collection** and **retraining** of the model directly from the interface.

---

## ⚙️ Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/RafaMej/Sign-Language-Translator.git
cd Sign-Language-Translator
```

### 2. Create and Activate Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```
o
```bash
pip install flask opencv-python mediapipe tensorflow numpy
```

## ▶️ How to Run the System

### 1. Start the Flask Server

```bash
python app.py
```

### 2. Access the Web Interface

Open your browser and go to:
```bash
http://127.0.0.1:5000
```

## 📈 Example Use Case

1. Click Iniciar Traductor to begin detection.
2. Perform one of the learned gestures.
3. The system displays the most likely detected sign and the associated probabilities.
4. Use Recopilar Datos to expand dataset or Reentrenar Modelo for further training

## 🧑‍🔬 Author

Rafael Mejía
Spring 2025 – Universidad de las Américas Puebla
Artificial Intelligence – Dra. Alejandra Hernández Sánchez
