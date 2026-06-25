"""
Face encoding and comparison utilities.
Uses face_recognition library (dlib-based) with OpenCV fallback.
"""
import numpy as np

def encode_face(img_array: np.ndarray):
    """
    Given an RGB numpy array, return face encoding or None if no face found.
    Uses face_recognition library (HOG-based detection).
    """
    try:
        import face_recognition
        encodings = face_recognition.face_encodings(img_array)
        if encodings:
            return encodings[0]
        return None
    except ImportError:
        return _opencv_encode(img_array)
    except Exception:
        return None

def compare_faces(stored_encoding, new_encoding, tolerance: float = 0.6) -> float:
    """Return Euclidean distance between two face encodings. Lower = more similar."""
    try:
        return float(np.linalg.norm(np.array(stored_encoding) - np.array(new_encoding)))
    except Exception:
        return 1.0

def detect_face_in_image(img_array: np.ndarray) -> bool:
    """Return True if at least one face is detected."""
    try:
        import face_recognition
        locations = face_recognition.face_locations(img_array)
        return len(locations) > 0
    except ImportError:
        return _opencv_detect(img_array)
    except Exception:
        return False

def _opencv_encode(img_array: np.ndarray):
    """Fallback: use OpenCV Eigenfaces via LBPH for basic encoding."""
    try:
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (128, 128))
        flat = face_roi.flatten().astype(np.float64)
        flat = flat / np.linalg.norm(flat) if np.linalg.norm(flat) > 0 else flat
        return flat
    except Exception:
        return None

def _opencv_detect(img_array: np.ndarray) -> bool:
    try:
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces) > 0
    except Exception:
        return False
