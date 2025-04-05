# fullTrain.py
import os
import argparse
from ultralytics import YOLO

DATA_YAML = "data.yaml"
MODEL_OUTPUT_DIR = "runs/detect/train"
BEST_PT = os.path.join(MODEL_OUTPUT_DIR, "weights", "best.pt")
EPOCHS_FULL = 100  # big run

def ensure_dirs():
    os.makedirs(os.path.join(MODEL_OUTPUT_DIR, "weights"), exist_ok=True)

def train_full():
    """One big YOLO run with all classes, all data."""
    # If there's a prior best, start from it
    if os.path.exists(BEST_PT):
        base_model = BEST_PT
        print(f"[INFO] Starting from existing best: {BEST_PT}")
    else:
        base_model = "yolov8s.pt"
        print("[INFO] Starting from yolov8s.pt (no existing best found)")

    model = YOLO(base_model)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS_FULL,
        imgsz=1920,
        batch=7,
        project="runs/detect",
        name="train",
        exist_ok=True,
        resume=False,
        plots=True   # if you want PR/F1 curves
    )
    print("[INFO] Full training done. Final best => runs/detect/train/weights/best.pt")

def main():
    parser=argparse.ArgumentParser()
    args=parser.parse_args()

    ensure_dirs()
    train_full()

if __name__=="__main__":
    main()
