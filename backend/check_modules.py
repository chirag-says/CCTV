"""Quick check of all vision modules."""
import sys
sys.path.insert(0, ".")

modules = [
    ("Face Detection (InsightFace)", "from insightface.app import FaceAnalysis"),
    ("ANPR (EasyOCR)", "import easyocr"),
    ("Traffic Monitor (YOLO)", "from ultralytics import YOLO"),
    ("Hazard Detection", "from app.vision.hazard_detector import HazardDetector"),
    ("Crowd Detection", "from app.vision.safety_analytics import CrowdDetector"),
    ("Loitering Detection", "from app.vision.safety_analytics import LoiteringDetector"),
    ("Attribute Recognition", "from app.vision.safety_analytics import AttributeRecognizer"),
]

for name, cmd in modules:
    try:
        exec(cmd)
        print(f"  OK  {name}")
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  FAIL {name}: {err}")

# Check ONNX providers
try:
    import onnxruntime
    print(f"\n  ONNX Providers: {onnxruntime.get_available_providers()}")
except:
    print("\n  ONNX Runtime not installed")
