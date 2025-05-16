import sys
import subprocess
import os

def check_environment():
    """Check the current Python environment"""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"In virtual environment: {in_venv}")
    
    # Get current directory
    print(f"Current working directory: {os.getcwd()}")

def check_packages():
    """Check if required packages are installed"""
    required_packages = ['opencv-python', 'mediapipe', 'tensorflow', 'numpy', 'flask']
    installed_packages = {}
    
    print("\nChecking required packages:")
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                installed_packages[package] = cv2.__version__
            elif package == 'mediapipe':
                import mediapipe as mp
                installed_packages[package] = mp.__version__
            elif package == 'tensorflow':
                import tensorflow as tf
                installed_packages[package] = tf.__version__
            elif package == 'numpy':
                import numpy as np
                installed_packages[package] = np.__version__
            elif package == 'flask':
                import flask
                installed_packages[package] = flask.__version__
            print(f"✅ {package} (version {installed_packages[package]}) is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            
    return installed_packages

def check_camera():
    """Test camera access"""
    try:
        import cv2
        print("\nChecking camera access:")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is accessible")
            ret, frame = cap.read()
            if ret:
                print("✅ Successfully captured a frame")
            else:
                print("❌ Could not capture a frame")
        else:
            print("❌ Camera is not accessible")
        cap.release()
    except ImportError:
        print("❌ Could not test camera - opencv-python not installed")

def install_missing_packages():
    """Install any missing packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "opencv-python", "mediapipe", "tensorflow", "numpy", "flask"])
        print("\n✅ Packages installed successfully")
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install packages")

def main():
    print("=== Python Environment Diagnostic Tool ===\n")
    check_environment()
    installed_packages = check_packages()
    check_camera()
    
    # Ask if user wants to install missing packages
    if len(installed_packages) < 5:
        print("\nSome packages are missing. Would you like to install them? (y/n)")
        response = input().lower()
        if response == 'y':
            install_missing_packages()

if __name__ == "__main__":
    main()